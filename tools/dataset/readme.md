- yolov9 structure to train.txt,val.txt

```
python yolov9_dataset_to_txt.py --dataset_name wholebody28
```

- yolo to coco

https://github.com/open-mmlab/mmyolo/blob/8c4d9dc503dc8e327bec8147e8dc97124052f693/tools/dataset_converters/yolo2coco.py

```
cd tools/dataset

python yolo2coco.py wholebody28

Start to load existing images and annotations from wholebody28
All necessary files are located at wholebody28
Checking if train.txt, val.txt, and test.txt are in wholebody28
Found train.txt
Found val.txt
Need to organize the data accordingly.
Start to read train dataset definition
Start to read val dataset definition
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 12186/12186, 475.5 task/s, elapsed: 26s, ETA:     0s
Saving converted results to wholebody28/annotations/train.json ...
Saving converted results to wholebody28/annotations/val.json ...
Process finished! Please check at wholebody28/annotations .
Number of images found: 12186, converted: 12186, and skipped: 0. Total annotation count: 530268.
```
