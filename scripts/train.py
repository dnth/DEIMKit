# from deimkit import Trainer, Config, configure_dataset

# conf = Config.from_model_name("deim_hgnetv2_m")

# conf = configure_dataset(
#     config=conf,
#     image_size=(416, 416),
#     train_ann_file="/home/mohamed/DEIMKit/data/plantdoc.coco/train/_annotations.coco.json",
#     train_img_folder="/home/mohamed/DEIMKit/data/plantdoc.coco/train",
#     val_ann_file="/home/mohamed/DEIMKit/data/plantdoc.coco/test/_annotations.coco.json",
#     val_img_folder="/home/mohamed/DEIMKit/data/plantdoc.coco/test",
#     train_batch_size=16,
#     val_batch_size=16,
#     num_classes=31,
#     output_dir="./outputs/plantdoc/deim_hgnetv2_m_30ep_416x416",
# )

# trainer = Trainer(conf)

# trainer.fit(epochs=20, save_best_only=True)


from deimkit import Trainer, Config, configure_dataset
import os

# Use absolute paths
BASE_DIR = "/home/mohamed/DEIMKit"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs/plantdoc/deim_hgnetv2_m_30ep_416x416")

conf = Config.from_model_name("deim_hgnetv2_m")  # Changed to x to match the output directory

conf = configure_dataset(
    config=conf,
    image_size=(416, 416),
    train_ann_file=os.path.join(BASE_DIR, "data/plantdoc.coco/train/_annotations.coco.json"),
    train_img_folder=os.path.join(BASE_DIR, "data/plantdoc.coco/train"),
    val_ann_file=os.path.join(BASE_DIR, "data/plantdoc.coco/test/_annotations.coco.json"),
    val_img_folder=os.path.join(BASE_DIR, "data/plantdoc.coco/test"),
    train_batch_size=16,
    val_batch_size=16,
    num_classes=31,
    output_dir=OUTPUT_DIR,
)

trainer = Trainer(conf)

trainer.fit(epochs=20, save_best_only=True)