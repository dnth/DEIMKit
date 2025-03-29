from deimkit import Trainer, Config, configure_dataset, configure_model

conf = Config.from_model_name("deim_hgnetv2_s")

conf = configure_model(conf, num_queries=50, freeze_at=3)

conf = configure_dataset(
    config=conf,
    image_size=(640, 640),
    train_ann_file="/home/dnth/Desktop/DEIMKit/dataset_collections/Rock Paper Scissors SXSW.v14i.coco/train/_annotations.coco.json",
    train_img_folder="/home/dnth/Desktop/DEIMKit/dataset_collections/Rock Paper Scissors SXSW.v14i.coco/train",
    val_ann_file="/home/dnth/Desktop/DEIMKit/dataset_collections/Rock Paper Scissors SXSW.v14i.coco/valid/_annotations.coco.json",
    val_img_folder="/home/dnth/Desktop/DEIMKit/dataset_collections/Rock Paper Scissors SXSW.v14i.coco/valid",
    train_batch_size=16,
    val_batch_size=16,
    num_classes=4,
    output_dir="./outputs/rock-paper-scissors/deim_hgnetv2_s_50ep_640px_num_queries_50_freeze_at_3",
)

trainer = Trainer(conf)

trainer.fit(epochs=50, save_best_only=True)
