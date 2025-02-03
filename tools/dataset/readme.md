- Copy datasets
    - from: `yolov9/dataset/images` to: `DEIM/tools/dataset/wholebody28/images`
    - from: `yolov9/dataset/labels` to: `DEIM/tools/dataset/wholebody28/labels`
- Make `DEIM/tools/dataset/wholebody28/classes.txt`
    ```
    body
    adult
    child
    male
    female
    body_with_wheelchair
    body_with_crutches
    head
    front
    right-front
    right-side
    right-back
    back
    left-back
    left-side
    left-front
    face
    eye
    nose
    mouth
    ear
    shoulder
    elbow
    hand
    hand_left
    hand_right
    knee
    foot
    ```
- yolov9 structure to train.txt,val.txt

    ```
    cd tools/dataset

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
    [>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 12186/12186, 463.5 task/s, elapsed: 26s, ETA:     0s
    Saving converted results to wholebody28/annotations/train.json ...
    Saving converted results to wholebody28/annotations/val.json ...
    Process finished! Please check at wholebody28/annotations .
    Number of images found: 12186, converted: 12186, and skipped: 0. Total annotation count: 717921.
    ```
