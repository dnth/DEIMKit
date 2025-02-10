"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..'))

import torch
import torch.nn as nn

from engine.core import YAMLConfig


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # NOTE load train mode state -> convert to deploy mode
        cfg.model.load_state_dict(state)

    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print('not load model.state_dict, use default init state dict...')

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs)
            return outputs

    model = Model()

    h, w = args.size
    data = torch.rand(1, 3, h, w)
    size = torch.tensor([[h, w]])
    _ = model(data, size)

    dynamic_axes = {
        'images': {0: 'N', },
        'label_xyxy_score': {0: 'N', 1: '1250', 2: '6'},
    }

    output_file = f'{os.path.splitext(os.path.basename(args.config))[0]}_{args.query}query.onnx'

    if not args.dynamic_batch:
        torch.onnx.export(
            model,
            (data, size),
            output_file,
            input_names=['images'],
            output_names=['label_xyxy_score'],
            dynamic_axes=None,
            opset_version=17,
        )
    else:
        torch.onnx.export(
            model,
            (data, size),
            f'{os.path.splitext(os.path.basename(output_file))[0]}_n_batch.onnx',
            input_names=['images'],
            output_names=['label_xyxy_score'],
            dynamic_axes=dynamic_axes,
            opset_version=17,
        )

    if args.check:
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print('Check export onnx model done...')

    if args.simplify:
        import onnx
        import onnxsim

        if not args.dynamic_batch:
            onnx_model_simplify, check = onnxsim.simplify(output_file)
            onnx.save(onnx_model_simplify, output_file)
            print(f'Simplify onnx model {check}...')
        else:
            onnx_model_simplify, check = onnxsim.simplify(f'{os.path.splitext(os.path.basename(output_file))[0]}_n_batch.onnx')
            onnx.save(onnx_model_simplify, f'{os.path.splitext(os.path.basename(output_file))[0]}_n_batch.onnx')
            print(f'Simplify onnx model {check}...')

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='configs/deim_dfine/deim_hgnetv2_x_wholebody28.yml', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--size', '-s', nargs="*", default=[640,640], type=int, )
    parser.add_argument('--check',  action='store_true', default=True,)
    parser.add_argument('--simplify',  action='store_true', default=True,)
    parser.add_argument('--dynamic_batch',  action='store_true', default=False,)
    parser.add_argument('--query', '-q', type=int, default=1250)
    args = parser.parse_args()
    main(args)
