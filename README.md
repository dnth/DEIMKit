# This fork `wholebody28` branch only has some parameters customized for learning Wholebody28. Also, ONNX/TensorRT optimization.

---

<h2 align="center">
  DEIM: DETR with Improved Matching for Fast Convergence
</h2>

<p align="center">
    <a href="https://github.com/ShihuaHuang95/DEIM/blob/master/LICENSE">
        <img alt="license" src="https://img.shields.io/badge/LICENSE-Apache%202.0-blue">
    </a>
    <a href="https://arxiv.org/abs/2412.04234">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2412.04234-red">
    </a>
   <a href="https://www.shihuahuang.cn/DEIM/">
        <img alt="project webpage" src="https://img.shields.io/badge/Webpage-DEIM-purple">
    </a>
    <a href="https://github.com/ShihuaHuang95/DEIM/pulls">
        <img alt="prs" src="https://img.shields.io/github/issues-pr/ShihuaHuang95/DEIM">
    </a>
    <a href="https://github.com/ShihuaHuang95/DEIM/issues">
        <img alt="issues" src="https://img.shields.io/github/issues/ShihuaHuang95/DEIM?color=olive">
    </a>
    <a href="https://github.com/ShihuaHuang95/DEIM">
        <img alt="stars" src="https://img.shields.io/github/stars/ShihuaHuang95/DEIM">
    </a>
    <a href="mailto:shihuahuang95@gmail.com">
        <img alt="Contact Us" src="https://img.shields.io/badge/Contact-Email-yellow">
    </a>
</p>

<p align="center">
    DEIM is an advanced training framework designed to enhance the matching mechanism in DETRs, enabling faster convergence and improved accuracy. It serves as a robust foundation for future research and applications in the field of real-time object detection. 
</p>

---


<div align="center">
  <a href="http://www.shihuahuang.cn">Shihua Huang</a><sup>1</sup>,
  <a href="https://scholar.google.com/citations?user=tIFWBcQAAAAJ&hl=en">Zhichao Lu</a><sup>2</sup>,
  <a href="https://vinthony.github.io/academic/">Xiaodong Cun</a><sup>3</sup>,
  Yongjun Yu<sup>1</sup>,
  Xiao Zhou<sup>4</sup>, 
  <a href="https://xishen0220.github.io">Xi Shen</a><sup>1*</sup>
</div>

  
<p align="center">
<i>
1. Intellindust AI Lab &nbsp; 2. City University of Hong Kong &nbsp; 3. Great Bay University &nbsp; 4. Hefei Normal University
</i>
</p>

<p align="center">
  **📧 Corresponding author:** <a href="mailto:shenxiluc@gmail.com">shenxiluc@gmail.com</a>
</p>

<p align="center">
    <a href="https://paperswithcode.com/sota/real-time-object-detection-on-coco?p=deim-detr-with-improved-matching-for-fast">
    <img alt="sota" src="https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/deim-detr-with-improved-matching-for-fast/real-time-object-detection-on-coco">
    </a>
</p>

<p align="center">
<strong>If you like our work, please give us a ⭐!</strong>
</p>


<p align="center">
  <img src="./figures/teaser_a.png" alt="Image 1" width="49%">
  <img src="./figures/teaser_b.png" alt="Image 2" width="49%">
</p>

</details>

 
  
## 🚀 Updates
- [x] **\[2025.03.12\]** The Object365 Pretrained [DEIM-D-FINE-X](https://drive.google.com/file/d/1RMNrHh3bYN0FfT5ZlWhXtQxkG23xb2xj/view?usp=drive_link) model is released, which achieves 59.5% AP after fine-tuning 24 COCO epochs.
- [x] **\[2025.03.05\]** The Nano DEIM model is released.
- [x] **\[2025.02.27\]** The DEIM paper is accepted to CVPR 2025. Thanks to all co-authors.
- [x] **\[2024.12.26\]** A more efficient implementation of Dense O2O, achieving nearly a 30% improvement in loading speed (See [the pull request](https://github.com/ShihuaHuang95/DEIM/pull/13) for more details). Huge thanks to my colleague [Longfei Liu](https://github.com/capsule2077).
- [x] **\[2024.12.03\]** Release DEIM series. Besides, this repo also supports the re-implmentations of [D-FINE](https://arxiv.org/abs/2410.13842) and [RT-DETR](https://arxiv.org/abs/2407.17140).

## Table of Content
* [1. Model Zoo](https://github.com/ShihuaHuang95/DEIM?tab=readme-ov-file#1-model-zoo)
* [2. Quick start](https://github.com/ShihuaHuang95/DEIM?tab=readme-ov-file#2-quick-start)
* [3. Usage](https://github.com/ShihuaHuang95/DEIM?tab=readme-ov-file#3-usage)
* [4. Tools](https://github.com/ShihuaHuang95/DEIM?tab=readme-ov-file#4-tools)
* [5. Citation](https://github.com/ShihuaHuang95/DEIM?tab=readme-ov-file#5-citation)
* [6. Acknowledgement](https://github.com/ShihuaHuang95/DEIM?tab=readme-ov-file#6-acknowledgement)
  
  
## 1. Model Zoo

### DEIM-D-FINE
| Model | Dataset | AP<sup>D-FINE</sup> | AP<sup>DEIM</sup> | #Params | Latency | GFLOPs | config | checkpoint
| :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: 
**N** | COCO | **42.8** | **43.0** | 4M | 2.12ms | 7 | [yml](./configs/deim_dfine/deim_hgnetv2_n_coco.yml) | [ckpt](https://drive.google.com/file/d/1ZPEhiU9nhW4M5jLnYOFwTSLQC1Ugf62e/view?usp=sharing) |
**S** | COCO | **48.7** | **49.0** | 10M | 3.49ms | 25 | [yml](./configs/deim_dfine/deim_hgnetv2_s_coco.yml) | [ckpt](https://drive.google.com/file/d/1tB8gVJNrfb6dhFvoHJECKOF5VpkthhfC/view?usp=drive_link) |
**M** | COCO | **52.3** | **52.7** | 19M | 5.62ms | 57 | [yml](./configs/deim_dfine/deim_hgnetv2_m_coco.yml) | [ckpt](https://drive.google.com/file/d/18Lj2a6UN6k_n_UzqnJyiaiLGpDzQQit8/view?usp=drive_link) |
**L** | COCO | **54.0** | **54.7** | 31M | 8.07ms | 91 | [yml](./configs/deim_dfine/deim_hgnetv2_l_coco.yml) | [ckpt](https://drive.google.com/file/d/1PIRf02XkrA2xAD3wEiKE2FaamZgSGTAr/view?usp=drive_link) | 
**X** | COCO | **55.8** | **56.5** | 62M | 12.89ms | 202 | [yml](./configs/deim_dfine/deim_hgnetv2_x_coco.yml) | [ckpt](https://drive.google.com/file/d/1dPtbgtGgq1Oa7k_LgH1GXPelg1IVeu0j/view?usp=drive_link) | 


### DEIM-RT-DETRv2
| Model | Dataset | AP<sup>RT-DETRv2</sup> | AP<sup>DEIM</sup> | #Params | Latency | GFLOPs | config | checkpoint
| :---: | :---: | :---: | :---: |  :---: | :---: | :---: | :---: | :---: 
**S** | COCO | **47.9** | **49.0** | 20M | 4.59ms | 60 | [yml](./configs/deim_rtdetrv2/deim_r18vd_120e_coco.yml) | [ckpt](https://drive.google.com/file/d/153_JKff6EpFgiLKaqkJsoDcLal_0ux_F/view?usp=drive_link) | 
**M** | COCO | **49.9** | **50.9** | 31M | 6.40ms | 92 | [yml](./configs/deim_rtdetrv2/deim_r34vd_120e_coco.yml) | [ckpt](https://drive.google.com/file/d/1O9RjZF6kdFWGv1Etn1Toml4r-YfdMDMM/view?usp=drive_link) | 
**M*** | COCO | **51.9** | **53.2** | 33M | 6.90ms | 100 | [yml](./configs/deim_rtdetrv2/deim_r50vd_m_60e_coco.yml) | [ckpt](https://drive.google.com/file/d/10dLuqdBZ6H5ip9BbBiE6S7ZcmHkRbD0E/view?usp=drive_link) | 
**L** | COCO | **53.4** | **54.3** | 42M | 9.15ms | 136 | [yml](./configs/deim_rtdetrv2/deim_r50vd_60e_coco.yml) | [ckpt](https://drive.google.com/file/d/1mWknAXD5JYknUQ94WCEvPfXz13jcNOTI/view?usp=drive_link) | 
**X** | COCO | **54.3** | **55.5** | 76M | 13.66ms | 259 | [yml](./configs/deim_rtdetrv2/deim_r101vd_60e_coco.yml) | [ckpt](https://drive.google.com/file/d/1BIevZijOcBO17llTyDX32F_pYppBfnzu/view?usp=drive_link) | 


## 2. Quick start

### Setup

```shell
conda create -n deim python=3.11.9
conda activate deim
pip install -r requirements.txt
```


### Data Preparation

<details>
<summary> COCO2017 Dataset </summary>

1. Download COCO2017 from [OpenDataLab](https://opendatalab.com/OpenDataLab/COCO_2017) or [COCO](https://cocodataset.org/#download).
1. Modify paths in [coco_detection.yml](./configs/dataset/coco_detection.yml)

    ```yaml
    train_dataloader:
        img_folder: /data/COCO2017/train2017/
        ann_file: /data/COCO2017/annotations/instances_train2017.json
    val_dataloader:
        img_folder: /data/COCO2017/val2017/
        ann_file: /data/COCO2017/annotations/instances_val2017.json
    ```

</details>

<details>
<summary>Custom Dataset</summary>

To train on your custom dataset, you need to organize it in the COCO format. Follow the steps below to prepare your dataset:

1. **Set `remap_mscoco_category` to `False`:**

    This prevents the automatic remapping of category IDs to match the MSCOCO categories.

    ```yaml
    remap_mscoco_category: False
    ```

2. **Organize Images:**

    Structure your dataset directories as follows:

    ```shell
    dataset/
    ├── images/
    │   ├── train/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    │   ├── val/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── ...
    └── annotations/
        ├── instances_train.json
        ├── instances_val.json
        └── ...
    ```

    - **`images/train/`**: Contains all training images.
    - **`images/val/`**: Contains all validation images.
    - **`annotations/`**: Contains COCO-formatted annotation files.

3. **Convert Annotations to COCO Format:**

    If your annotations are not already in COCO format, you'll need to convert them. You can use the following Python script as a reference or utilize existing tools:

    ```python
    import json

    def convert_to_coco(input_annotations, output_annotations):
        # Implement conversion logic here
        pass

    if __name__ == "__main__":
        convert_to_coco('path/to/your_annotations.json', 'dataset/annotations/instances_train.json')
    ```

4. **Update Configuration Files:**

    Modify your [custom_detection.yml](./configs/dataset/custom_detection.yml).

    ```yaml
    task: detection

    evaluator:
      type: CocoEvaluator
      iou_types: ['bbox', ]

    num_classes: 777 # your dataset classes
    remap_mscoco_category: False

    train_dataloader:
      type: DataLoader
      dataset:
        type: CocoDetection
        img_folder: /data/yourdataset/train
        ann_file: /data/yourdataset/train/train.json
        return_masks: False
        transforms:
          type: Compose
          ops: ~
      shuffle: True
      num_workers: 4
      drop_last: True
      collate_fn:
        type: BatchImageCollateFunction

    val_dataloader:
      type: DataLoader
      dataset:
        type: CocoDetection
        img_folder: /data/yourdataset/val
        ann_file: /data/yourdataset/val/ann.json
        return_masks: False
        transforms:
          type: Compose
          ops: ~
      shuffle: False
      num_workers: 4
      drop_last: False
      collate_fn:
        type: BatchImageCollateFunction
    ```

</details>


## 3. Usage
<details open>
<summary> COCO2017 </summary>

1. Training
    ```shell
    pip install -r requirements.txt

    rm -rf tools/dataset/wholebody28
    mv tools/dataset/wholebody28_1 tools/dataset/wholebody28

    ####################################################### X
    CUDA_VISIBLE_DEVICES=0 torchrun \
    --master_port=7777 \
    --nproc_per_node=1 \
    train.py \
    -c configs/deim_dfine/deim_hgnetv2_x_wholebody28.yml \
    --use-amp \
    --seed=0 &> log.txt 2>&1 &
    
    CUDA_VISIBLE_DEVICES=0 torchrun \
    --master_port=7777 \
    --nproc_per_node=1 \
    train.py \
    -c configs/deim_dfine/deim_hgnetv2_x_wholebody28.yml \
    -r outputs/deim_hgnetv2_x_wholebody28/last.pth \
    --use-amp --seed=0 &> log.txt 2>&1 &

    ####################################################### S
    CUDA_VISIBLE_DEVICES=0 torchrun \
    --master_port=7777 \
    --nproc_per_node=1 \
    train.py \
    -c configs/deim_dfine/deim_hgnetv2_s_wholebody28.yml \
    --use-amp \
    --seed=0 &> log.txt 2>&1 &
    
    CUDA_VISIBLE_DEVICES=0 torchrun \
    --master_port=7777 \
    --nproc_per_node=1 \
    train.py \
    -c configs/deim_dfine/deim_hgnetv2_s_wholebody28.yml \
    -r outputs/deim_hgnetv2_s_wholebody28/last.pth \
    --use-amp --seed=0 &> log.txt 2>&1 &

    ####################################################### N
    CUDA_VISIBLE_DEVICES=0 torchrun \
    --master_port=7777 \
    --nproc_per_node=1 \
    train.py \
    -c configs/deim_dfine/deim_hgnetv2_n_wholebody28.yml \
    --use-amp \
    --seed=0 &> log.txt 2>&1 &
    
    CUDA_VISIBLE_DEVICES=0 torchrun \
    --master_port=7777 \
    --nproc_per_node=1 \
    train.py \
    -c configs/deim_dfine/deim_hgnetv2_n_wholebody28.yml \
    -r outputs/deim_hgnetv2_n_wholebody28/last.pth \
    --use-amp --seed=0 &> log.txt 2>&1 &

    #######################################################
    tail -n 20 -f log.txt
    
    ps aux | grep torchrun
    kill xxxxx
    ```

<!-- <summary>2. Testing </summary> -->
2. Testing
    ```shell
    ####################################################### X
    CUDA_VISIBLE_DEVICES=0 torchrun \
    --master_port=7777 \
    --nproc_per_node=1 \
    train.py \
    -c configs/deim_dfine/deim_hgnetv2_x_wholebody28.yml \
    --test-only \
    -r outputs/deim_hgnetv2_x_wholebody28/best_stg2.pth
    
    ####################################################### S
    CUDA_VISIBLE_DEVICES=0 torchrun \
    --master_port=7777 \
    --nproc_per_node=1 \
    train.py \
    -c configs/deim_dfine/deim_hgnetv2_s_wholebody28.yml \
    --test-only \
    -r outputs/deim_hgnetv2_s_wholebody28/best_stg2.pth

    ####################################################### N
    CUDA_VISIBLE_DEVICES=0 torchrun \
    --master_port=7777 \
    --nproc_per_node=1 \
    train.py \
    -c configs/deim_dfine/deim_hgnetv2_n_wholebody28.yml \
    --test-only \
    -r outputs/deim_hgnetv2_n_wholebody28/best_stg2.pth
    ```

<!-- <summary>3. Tuning </summary> -->
3. Tuning
    ```shell
    rm -rf tools/dataset/wholebody28
    mv tools/dataset/wholebody28_2 tools/dataset/wholebody28
    ```
    Uncomment.
    https://github.com/PINTO0309/DEIM/blob/c4029e6e569f13f5bb7d7c7a5f8c3b640706aa9a/engine/solver/det_solver.py#L157-L159
    ```
    ####################################################### X
    CUDA_VISIBLE_DEVICES=0 torchrun \
    --master_port=7777 \
    --nproc_per_node=1 \
    train.py \
    -c configs/deim_dfine/deim_hgnetv2_x_wholebody28_ft.yml \
    --use-amp \
    --seed=0 \
    -t outputs/deim_hgnetv2_x_wholebody28/best_stg2.pth &> log.txt 2>&1 &
    
    CUDA_VISIBLE_DEVICES=0 torchrun \
    --master_port=7777 \
    --nproc_per_node=1 train.py \
    -c configs/deim_dfine/deim_hgnetv2_x_wholebody28_ft.yml \
    -r outputs/deim_hgnetv2_x_wholebody28_ft/last.pth \
    --use-amp --seed=0 &> log.txt 2>&1 &

    ####################################################### S
    CUDA_VISIBLE_DEVICES=0 torchrun \
    --master_port=7777 \
    --nproc_per_node=1 \
    train.py \
    -c configs/deim_dfine/deim_hgnetv2_s_wholebody28_ft.yml \
    --use-amp \
    --seed=0 \
    -t outputs/deim_hgnetv2_s_wholebody28/best_stg2.pth &> log.txt 2>&1 &
    
    CUDA_VISIBLE_DEVICES=0 torchrun \
    --master_port=7777 \
    --nproc_per_node=1 train.py \
    -c configs/deim_dfine/deim_hgnetv2_s_wholebody28_ft.yml \
    -r outputs/deim_hgnetv2_s_wholebody28_ft/last.pth \
    --use-amp --seed=0 &> log.txt 2>&1 &

    ####################################################### N
    CUDA_VISIBLE_DEVICES=0 torchrun \
    --master_port=7777 \
    --nproc_per_node=1 \
    train.py \
    -c configs/deim_dfine/deim_hgnetv2_n_wholebody28.yml \
    --use-amp \
    --seed=0 \
    -t outputs/deim_hgnetv2_n_wholebody28_1/best_stg2.pth &> log.txt 2>&1 &
    
    CUDA_VISIBLE_DEVICES=0 torchrun \
    --master_port=7777 \
    --nproc_per_node=1 train.py \
    -c configs/deim_dfine/deim_hgnetv2_n_wholebody28.yml \
    -r outputs/deim_hgnetv2_n_wholebody28/last.pth \
    --use-amp --seed=0 &> log.txt 2>&1 &
    #######################################################
    tail -n 20 -f log.txt
    
    ps aux | grep torchrun
    kill xxxxx
    ```
</details>

<details>
<summary> Customizing Batch Size </summary>

For example, if you want to double the total batch size when training D-FINE-L on COCO2017, here are the steps you should follow:

1. **Modify your [dataloader.yml](./configs/base/dataloader.yml)** to increase the `total_batch_size`:

    ```yaml
    train_dataloader:
        total_batch_size: 64  # Previously it was 32, now doubled
    ```

2. **Modify your [deim_hgnetv2_l_coco.yml](./configs/deim_dfine/deim_hgnetv2_l_coco.yml)**. Here’s how the key parameters should be adjusted:

    ```yaml
    optimizer:
    type: AdamW
    params:
        -
        params: '^(?=.*backbone)(?!.*norm|bn).*$'
        lr: 0.000025  # doubled, linear scaling law
        -
        params: '^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$'
        weight_decay: 0.

    lr: 0.0005  # doubled, linear scaling law
    betas: [0.9, 0.999]
    weight_decay: 0.0001  # need a grid search

    ema:  # added EMA settings
        decay: 0.9998  # adjusted by 1 - (1 - decay) * 2
        warmups: 500  # halved

    lr_warmup_scheduler:
        warmup_duration: 250  # halved
    ```

</details>


<details>
<summary> Customizing Input Size </summary>

If you'd like to train **DEIM** on COCO2017 with an input size of 320x320, follow these steps:

1. **Modify your [dataloader.yml](./configs/base/dataloader.yml)**:

    ```yaml

    train_dataloader:
    dataset:
        transforms:
            ops:
                - {type: Resize, size: [320, 320], }
    collate_fn:
        base_size: 320
    dataset:
        transforms:
            ops:
                - {type: Resize, size: [320, 320], }
    ```

2. **Modify your [dfine_hgnetv2.yml](./configs/base/dfine_hgnetv2.yml)**:

    ```yaml
    eval_spatial_size: [320, 320]
    ```

</details>

## 4. Tools
<details open>
<summary> Deployment </summary>

<!-- <summary>4. Export onnx </summary> -->
1. Setup
    ```shell
    pip install onnx onnxsim
    ```

2. Converting Model Weights (Extract and save only the EMA weights.)
    ```shell
    python tools/reference/convert_weight.py \
    outputs/deim_hgnetv2_x_wholebody28
    ```
    ![image](https://github.com/user-attachments/assets/d93dc98d-9b7a-4fbb-b50d-0bf9186ffdd7)

3. Export onnx
    ```shell
    python tools/deployment/export_onnx.py \
    --check \
    --query 1250 \
    -c configs/deim_dfine/deim_hgnetv2_x_wholebody28.yml \
    -r outputs/deim_hgnetv2_x_wholebody28/best_stg2_converted.pth
    
    python tools/deployment/export_onnx.py \
    --check \
    --query 1250 \
    --dynamic_batch \
    -c configs/deim_dfine/deim_hgnetv2_x_wholebody28.yml \
    -r outputs/deim_hgnetv2_x_wholebody28/best_stg2_converted.pth
    ```

4. Variable batch size pre-processing and post-processing integration
    ```shell
    python tools/deployment/make_prep.py
    ```

5. Export [tensorrt](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)
    ```shell
    trtexec --onnx="model.onnx" --saveEngine="model.engine" --fp16
    ```

6. Export TFLite/TensorFlow.js
    ```shell
    pip install \
    onnx2tf \
    tensorflow \
    tensorflowjs \
    onnx \
    onnxruntime \
    simple_onnx_processing_tools \
    onnxsim==0.4.33

    onnx2tf -i deim_hgnetv2_n_wholebody28_1250query.onnx

    tensorflowjs_converter \
    --input_format tf_saved_model \
    --output_format tfjs_graph_model \
    --weight_shard_size_bytes 1048576000 \
    saved_model \
    tfjs_model

    python -m http.server
    ```
    Access to `http://localhost:8000/index_tfjs.html`

    ![image](https://github.com/user-attachments/assets/566792df-16c8-4efa-9383-2a3e5acc2a87)

</details>

<details>
<summary> Inference (Visualization) </summary>


1. Setup
```shell
pip install -r tools/inference/requirements.txt
```


<!-- <summary>5. Inference </summary> -->
2. Inference (onnxruntime / tensorrt / torch)

Inference on images and videos is now supported.
```shell
python tools/inference/onnx_inf.py --onnx model.onnx --input image.jpg  # video.mp4
python tools/inference/trt_inf.py --trt model.engine --input image.jpg
python tools/inference/torch_inf.py -c configs/deim_dfine/deim_hgnetv2_${model}_coco.yml -r model.pth --input image.jpg --device cuda:0
```
</details>

<details>
<summary> Benchmark </summary>

1. Setup
```shell
pip install -r tools/benchmark/requirements.txt
```

<!-- <summary>6. Benchmark </summary> -->
2. Model FLOPs, MACs, and Params
```shell
python tools/benchmark/get_info.py -c configs/deim_dfine/deim_hgnetv2_${model}_coco.yml
```

2. TensorRT Latency
```shell
python tools/benchmark/trt_benchmark.py --COCO_dir path/to/COCO2017 --engine_dir model.engine
```
</details>

<details>
<summary> Fiftyone Visualization  </summary>

1. Setup
```shell
pip install fiftyone
```
4. Voxel51 Fiftyone Visualization ([fiftyone](https://github.com/voxel51/fiftyone))
```shell
python tools/visualization/fiftyone_vis.py -c configs/deim_dfine/deim_hgnetv2_${model}_coco.yml -r model.pth
```
</details>

<details>
<summary> Others </summary>

1. Auto Resume Training
```shell
bash reference/safe_training.sh
```
</details>


## 5. Citation
If you use `DEIM` or its methods in your work, please cite the following BibTeX entries:
<details open>
<summary> bibtex </summary>

```latex
@misc{huang2024deim,
      title={DEIM: DETR with Improved Matching for Fast Convergence},
      author={Shihua Huang, Zhichao Lu, Xiaodong Cun, Yongjun Yu, Xiao Zhou, and Xi Shen},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year={2025},
}
```
</details>

## 6. Acknowledgement
Our work is built upon [D-FINE](https://github.com/Peterande/D-FINE) and [RT-DETR](https://github.com/lyuwenyu/RT-DETR).

✨ Feel free to contribute and reach out if you have any questions! ✨
