from deimkit import Trainer, Config, configure_dataset

conf = Config.from_model_name("deim_hgnetv2_s")

conf = configure_dataset(
    config=conf,
    image_size=[320, 320],
    train_ann_file="/home/dnth/Downloads/vehicles.v2-release.coco/train/_annotations.coco.json",
    train_img_folder="/home/dnth/Downloads/vehicles.v2-release.coco/train",
    val_ann_file="/home/dnth/Downloads/vehicles.v2-release.coco/valid/_annotations.coco.json",
    val_img_folder="/home/dnth/Downloads/vehicles.v2-release.coco/valid",
    train_batch_size=8,
    val_batch_size=8,
    num_classes=13,
    output_dir="./outputs/deim_hgnetv2_s_vehicles_50epochs_2",
)


trainer = Trainer(conf)

trainer.fit(epochs=50)
