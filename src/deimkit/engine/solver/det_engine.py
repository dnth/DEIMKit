"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DETR (https://github.com/facebookresearch/detr/blob/main/engine.py)
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""


import sys
import math
from typing import Iterable, Dict, Optional, List, Tuple
from tqdm import tqdm

import torch
import torch.amp
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils

import matplotlib.pyplot as plt
import numpy as np


def train_one_epoch(self_lr_scheduler, lr_scheduler, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, **kwargs):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    print_freq = kwargs.get('print_freq', 10)
    writer :SummaryWriter = kwargs.get('writer', None)

    ema :ModelEMA = kwargs.get('ema', None)
    scaler :GradScaler = kwargs.get('scaler', None)
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)

    cur_iters = epoch * len(data_loader)
    
    # Add progress bar
    pbar = tqdm(total=len(data_loader), desc=f'Epoch {epoch}', leave=True)

    for i, (samples, targets) in enumerate(data_loader):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader))

        if scaler is not None:
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)

            if torch.isnan(outputs['pred_boxes']).any() or torch.isinf(outputs['pred_boxes']).any():
                print(outputs['pred_boxes'])
                state = model.state_dict()
                new_state = {}
                for key, value in model.state_dict().items():
                    # Replace 'module' with 'model' in each key
                    new_key = key.replace('module.', '')
                    # Add the updated key-value pair to the state dictionary
                    state[new_key] = value
                new_state['model'] = state
                dist_utils.save_on_master(new_state, "./NaN.pth")

            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            loss = sum(loss_dict.values())
            scaler.scale(loss).backward()

            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)

            loss : torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # ema
        if ema is not None:
            ema.update(model)

        if self_lr_scheduler:
            optimizer = lr_scheduler.step(cur_iters + i, optimizer)
        else:
            if lr_warmup_scheduler is not None:
                lr_warmup_scheduler.step()

        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if writer and dist_utils.is_main_process() and global_step % 10 == 0:
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)

        # Update progress bar with loss info
        pbar.set_postfix({'loss': f'{loss_value:.4f}'})
        pbar.update()

    pbar.close()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def create_pr_curve_plot(
    precisions: np.ndarray,
    recalls: np.ndarray,
    labels: List[str],
    colors: List[str],
    title: str,
    figsize: Tuple[int, int] = (10, 10)
) -> plt.Figure:
    """
    Create a precision-recall curve plot.
    
    Args:
        precisions: Array of precision values for each curve, shape (num_curves, num_points)
        recalls: Array of recall values, shape (num_points,)
        labels: List of labels for each curve
        colors: List of colors for each curve
        title: Plot title
        figsize: Figure size in inches
    
    Returns:
        matplotlib.figure.Figure: The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    for precision, label, color in zip(precisions, labels, colors):
        ax.plot(recalls, precision, color=color, label=label, linewidth=2)
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True)
    ax.legend(loc='lower left')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    return fig

def get_precision_recall_data(
    eval_result,
    iou_thresh: Optional[float] = None,
    area_idx: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract precision-recall data from COCO evaluation results.
    
    Args:
        eval_result: COCO evaluation result object
        iou_thresh: IoU threshold value. If None, average over all IoU thresholds
        area_idx: Area index (0: all, 1: small, 2: medium, 3: large). If None, use all areas
    
    Returns:
        Tuple of (precision array, recall array)
    """
    recalls = eval_result.params.recThrs

    if iou_thresh is not None:
        iou_index = eval_result.params.iouThrs == iou_thresh
        # Make sure to get the same shape as recalls
        precision = eval_result.eval['precision'][iou_index, :, :, 0, -1].mean(axis=1).squeeze()
    else:
        # Average over IoU thresholds and categories
        precision = eval_result.eval['precision'][:, :, :, 0, -1].mean(axis=2).mean(axis=0)
    
    if area_idx is not None:
        precision = eval_result.eval['precision'][:, :, :, area_idx, -1].mean(axis=2).mean(axis=0)
    
    # Ensure precision has the same shape as recalls
    if precision.shape != recalls.shape:
        precision = np.full_like(recalls, np.nan)
    
    return precision, recalls

def log_pr_curves(
    coco_evaluator: CocoEvaluator,
    writer: SummaryWriter,
    global_step: int,
    iou_types: List[str]
) -> None:
    """
    Log precision-recall curves to TensorBoard.
    """
    if writer is None or not dist_utils.is_main_process():
        return

    iou_thresholds = [0.5, 0.75]
    area_labels = ['all', 'small', 'medium', 'large']
    
    for iou_type in iou_types:
        eval_result = coco_evaluator.coco_eval[iou_type]
        recalls = eval_result.params.recThrs
        
        # IoU threshold based curves
        precisions = []
        labels = []
        colors = ['b', 'r', 'g']  # Colors for IoU=0.5, 0.75, and mean
        
        # Get PR curves for specific IoU thresholds
        for iou_thresh in iou_thresholds:
            precision, _ = get_precision_recall_data(eval_result, iou_thresh=iou_thresh)
            precisions.append(precision)
            labels.append(f'IoU={iou_thresh:.2f}')
        
        # Add mean PR curve (IoU=0.50:0.95)
        precision, _ = get_precision_recall_data(eval_result)
        precisions.append(precision)
        labels.append('IoU=0.50:0.95')
        
        # Stack precisions into a 2D array
        precisions = np.stack(precisions)
        
        # Create and log IoU threshold based plot
        fig = create_pr_curve_plot(
            precisions,
            recalls,
            labels,
            colors,
            f'Precision-Recall Curves ({iou_type})'
        )
        writer.add_figure(f'metrics-PR/{iou_type}/precision_recall_curve', fig, global_step)
        plt.close(fig)
        
        # Area based curves
        precisions = []
        colors = ['g', 'b', 'r', 'c']
        
        # Get PR curves for different areas
        for area_idx in range(4):
            precision, _ = get_precision_recall_data(eval_result, area_idx=area_idx)
            precisions.append(precision)
        
        # Stack precisions into a 2D array
        precisions = np.stack(precisions)
        
        # Create and log area based plot
        fig = create_pr_curve_plot(
            precisions,
            recalls,
            [f'area={label}' for label in area_labels],
            colors,
            f'Precision-Recall Curves by Area ({iou_type})'
        )
        writer.add_figure(f'metrics-PR/{iou_type}/precision_recall_curve_by_area', fig, global_step)
        plt.close(fig)

def calculate_f1_score(precision: float, recall: float) -> float:
    """
    Calculate F1 score from precision and recall values.
    
    Args:
        precision: Precision value (AP)
        recall: Recall value (AR)
    
    Returns:
        float: F1 score if valid, float('nan') if invalid
    """
    if precision <= 0 or recall <= 0:
        return float('nan')
    
    return 2 * (precision * recall) / (precision + recall)

@torch.no_grad()
def evaluate(model: torch.nn.Module, criterion: torch.nn.Module, postprocessor, data_loader, 
            coco_evaluator: CocoEvaluator, device, writer: Optional[SummaryWriter] = None, 
            global_step: Optional[int] = None):
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    iou_types = coco_evaluator.iou_types

    # Add progress bar for evaluation
    pbar = tqdm(total=len(data_loader), desc='Evaluating', leave=True)

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        results = postprocessor(outputs, orig_target_sizes)

        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        # Update progress bar
        pbar.update()

    pbar.close()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
        
        # Log PR curves
        log_pr_curves(coco_evaluator, writer, global_step, iou_types)

    stats = {}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            bbox_stats = coco_evaluator.coco_eval['bbox'].stats
            # Add top-level metrics for quick overview
            if writer is not None and dist_utils.is_main_process() and global_step is not None:
                # Primary metrics at top level
                writer.add_scalar('top-level-metrics/mAP_50_95', bbox_stats[0], global_step)  
                
                # Top-level recall metrics
                writer.add_scalar('top-level-metrics/mAR_50_95', bbox_stats[8], global_step)     
                
                # Calculate and log F1 scores at top level
                f1_50_95 = calculate_f1_score(bbox_stats[0], bbox_stats[8])
                
                if f1_50_95 is not None:
                    writer.add_scalar('top-level-metrics/F1_50_95', f1_50_95, global_step)

            # Continue with existing detailed metrics logging
            if writer is not None and dist_utils.is_main_process() and global_step is not None:
                # Average Precision metrics (indices 0-5)
                writer.add_scalar('metrics-AP/IoU_0.50-0.95_area_all_maxDets_100', bbox_stats[0], global_step)
                writer.add_scalar('metrics-AP/IoU_0.50_area_all_maxDets_100', bbox_stats[1], global_step)
                writer.add_scalar('metrics-AP/IoU_0.75_area_all_maxDets_100', bbox_stats[2], global_step)
                writer.add_scalar('metrics-AP/IoU_0.50-0.95_area_small_maxDets_100', bbox_stats[3], global_step)
                writer.add_scalar('metrics-AP/IoU_0.50-0.95_area_medium_maxDets_100', bbox_stats[4], global_step)
                writer.add_scalar('metrics-AP/IoU_0.50-0.95_area_large_maxDets_100', bbox_stats[5], global_step)
                # Average Recall metrics (indices 6-11)
                writer.add_scalar('metrics-AR/IoU_0.50-0.95_area_all_maxDets_1', bbox_stats[6], global_step)
                writer.add_scalar('metrics-AR/IoU_0.50-0.95_area_all_maxDets_10', bbox_stats[7], global_step)
                writer.add_scalar('metrics-AR/IoU_0.50-0.95_area_all_maxDets_100', bbox_stats[8], global_step)
                writer.add_scalar('metrics-AR/IoU_0.50-0.95_area_small_maxDets_100', bbox_stats[9], global_step)
                writer.add_scalar('metrics-AR/IoU_0.50-0.95_area_medium_maxDets_100', bbox_stats[10], global_step)
                writer.add_scalar('metrics-AR/IoU_0.50-0.95_area_large_maxDets_100', bbox_stats[11], global_step)

                # Calculate and log F1 scores only when valid
                # For IoU 0.50:0.95
                f1_50_95 = calculate_f1_score(bbox_stats[0], bbox_stats[8])
                if f1_50_95 is not None:
                    writer.add_scalar('metrics-F1/IoU_0.50-0.95_area_all_maxDets_100', f1_50_95, global_step)

                # For IoU 0.50
                f1_50 = calculate_f1_score(bbox_stats[1], bbox_stats[8])
                if f1_50 is not None:
                    writer.add_scalar('metrics-F1/IoU_0.50_area_all_maxDets_100', f1_50, global_step)

                # For IoU 0.75
                f1_75 = calculate_f1_score(bbox_stats[2], bbox_stats[8])
                if f1_75 is not None:
                    writer.add_scalar('metrics-F1/IoU_0.75_area_all_maxDets_100', f1_75, global_step)

                # Small
                f1_small = calculate_f1_score(bbox_stats[3], bbox_stats[9])
                if f1_small is not None:
                    writer.add_scalar('metrics-F1/IoU_0.50-0.95_area_small_maxDets_100', f1_small, global_step)

                # Medium
                f1_medium = calculate_f1_score(bbox_stats[4], bbox_stats[10])
                if f1_medium is not None:
                    writer.add_scalar('metrics-F1/IoU_0.50-0.95_area_medium_maxDets_100', f1_medium, global_step)

                # Large
                f1_large = calculate_f1_score(bbox_stats[5], bbox_stats[11])
                if f1_large is not None:
                    writer.add_scalar('metrics-F1/IoU_0.50-0.95_area_large_maxDets_100', f1_large, global_step)
            
            if 'segm' in iou_types:
                segm_stats = coco_evaluator.coco_eval['segm'].stats
                # Average Precision metrics (indices 0-5)
                writer.add_scalar('metrics-AP/IoU_0.50-0.95_area_all_maxDets_100', segm_stats[0], global_step)
                writer.add_scalar('metrics-AP/IoU_0.50_area_all_maxDets_100', segm_stats[1], global_step)
                writer.add_scalar('metrics-AP/IoU_0.75_area_all_maxDets_100', segm_stats[2], global_step)
                writer.add_scalar('metrics-AP/IoU_0.50-0.95_area_small_maxDets_100', segm_stats[3], global_step)
                writer.add_scalar('metrics-AP/IoU_0.50-0.95_area_medium_maxDets_100', segm_stats[4], global_step)
                writer.add_scalar('metrics-AP/IoU_0.50-0.95_area_large_maxDets_100', segm_stats[5], global_step)
                # Average Recall metrics (indices 6-11)
                writer.add_scalar('metrics-AR/IoU_0.50-0.95_area_all_maxDets_1', segm_stats[6], global_step)
                writer.add_scalar('metrics-AR/IoU_0.50-0.95_area_all_maxDets_10', segm_stats[7], global_step)
                writer.add_scalar('metrics-AR/IoU_0.50-0.95_area_all_maxDets_100', segm_stats[8], global_step)
                writer.add_scalar('metrics-AR/IoU_0.50-0.95_area_small_maxDets_100', segm_stats[9], global_step)
                writer.add_scalar('metrics-AR/IoU_0.50-0.95_area_medium_maxDets_100', segm_stats[10], global_step)
                writer.add_scalar('metrics-AR/IoU_0.50-0.95_area_large_maxDets_100', segm_stats[11], global_step)

    return stats, coco_evaluator
