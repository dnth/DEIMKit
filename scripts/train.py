from deimkit import Trainer, Config, configure_dataset

conf = Config.from_model_name("deim_hgnetv2_s")

conf = configure_dataset(
    config=conf,
    image_size=(416, 416),
    train_ann_file="/home/mohamed/datasets/plantdoc.coco/train/_annotations.coco.json",
    train_img_folder="/home/mohamed/datasets/plantdoc.coco/train",
    val_ann_file="/home/mohamed/datasets/plantdoc.coco/test/_annotations.coco.json",
    val_img_folder="/home/mohamed/datasets/plantdoc.coco/test",
    train_batch_size=16,
    val_batch_size=16,
    num_classes=31,
    output_dir="./outputs/csgo-videogame/deim_hgnetv2_s_30ep_416x416",
)

trainer = Trainer(conf)

trainer.fit(epochs=5, save_best_only=True)
