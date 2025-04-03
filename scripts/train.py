from deimkit import Config, Trainer, configure_dataset, configure_model

conf = Config.from_model_name("deim_hgnetv2_s")

conf = configure_model(conf, num_queries=100, freeze_at=0, pretrained=True)

conf = configure_dataset(
    config=conf,
    image_size=(800, 800),
    train_ann_file="/home/dnth/Desktop/DEIMKit/dataset_collections/Rock Paper Scissors SXSW.v14i.coco/train/_annotations.coco.json",
    train_img_folder="/home/dnth/Desktop/DEIMKit/dataset_collections/Rock Paper Scissors SXSW.v14i.coco/train",
    val_ann_file="/home/dnth/Desktop/DEIMKit/dataset_collections/Rock Paper Scissors SXSW.v14i.coco/valid/_annotations.coco.json",
    val_img_folder="/home/dnth/Desktop/DEIMKit/dataset_collections/Rock Paper Scissors SXSW.v14i.coco/valid",
    train_batch_size=16,
    val_batch_size=16,
    num_classes=4,
    remap_mscoco=False,
    output_dir="./outputs/rock-paper-scissors/deim_hgnetv2_s_30ep_640px_num_queries_resume_3",
)

trainer = Trainer(conf)

trainer.load_checkpoint("/home/dnth/Desktop/DEIMKit/outputs/rock-paper-scissors/deim_hgnetv2_s_30ep_640px_num_queries_resume_2/best.pth")
trainer.fit(epochs=10, save_best_only=True, lr=0.00004)
