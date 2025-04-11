"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

# from ._dataset import DetDataset
from .coco_dataset import CocoDetection
from .coco_dataset import (
    mscoco_category2name,
    mscoco_category2label,
    mscoco_label2category,
)
from .coco_eval import CocoEvaluator
from .coco_utils import get_coco_api_from_dataset
from .voc_detection import VOCDetection
from .voc_eval import VOCEvaluator

# PlantDoc dataset support
from .plantdoc_dataset import PlantDocDetection
from .plantdoc_dataset import (
    plantdoc_category2name,
    plantdoc_category2label,
    plantdoc_to_coco_mapping,
)
from .plantdoc_eval import PlantDocEvaluator
