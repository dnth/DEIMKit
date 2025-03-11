from deimkit import Trainer, Config, configure_dataset

conf = Config.from_model_name("deim_hgnetv2_s")

conf = configure_dataset(
    config=conf,
    image_size=[640, 640],
    train_ann_file="/home/dnth/Downloads/PCB Holes.v4i.coco/train/_annotations.coco.json",
    train_img_folder="/home/dnth/Downloads/PCB Holes.v4i.coco/train",
    val_ann_file="/home/dnth/Downloads/PCB Holes.v4i.coco/valid/_annotations.coco.json",
    val_img_folder="/home/dnth/Downloads/PCB Holes.v4i.coco/valid",
    train_batch_size=16,
    val_batch_size=16,
    num_classes=2,
    remap_mscoco=False,
    output_dir="./outputs/deim_hgnetv2_s_pcb_100e_warm50",
)


trainer = Trainer(conf)
# trainer.fit(epochs=60, flat_epoch=30, no_aug_epoch=3, warmup_iter=100, ema_warmups=100)
trainer.fit(
    epochs=100,
    flat_epoch=50,
    no_aug_epoch=10,
    warmup_iter=50,
    ema_warmups=50,
    # verbose=False,
)

conf.save("my_train_config.yml")
