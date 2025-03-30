from deimkit import Trainer, Config, configure_dataset, configure_model

conf = Config.from_model_name("deim_hgnetv2_s")

conf = configure_model(conf, num_queries=100)

conf = configure_dataset(
    config=conf,
    image_size=(640, 640),
    train_ann_file="/home/dnth/Desktop/DEIMKit/dataset_collections/aquarium-combined-gjvb.v1i.coco/train/_annotations.coco.json",
    train_img_folder="/home/dnth/Desktop/DEIMKit/dataset_collections/aquarium-combined-gjvb.v1i.coco/train",
    val_ann_file="/home/dnth/Desktop/DEIMKit/dataset_collections/aquarium-combined-gjvb.v1i.coco/valid/_annotations.coco.json",
    val_img_folder="/home/dnth/Desktop/DEIMKit/dataset_collections/aquarium-combined-gjvb.v1i.coco/valid",
    train_batch_size=16,
    val_batch_size=16,
    num_classes=8,
    output_dir="./outputs/aquarium/deim_hgnetv2_s_30ep_640px_num_queries_100_no_aug_epoch_15",
)

trainer = Trainer(conf)

trainer.fit(epochs=30, save_best_only=True, no_aug_epoch=15)
