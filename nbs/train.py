from deimkit import Trainer, Config, configure_dataset

conf = Config.from_model_name("deim_hgnetv2_s")

conf = configure_dataset(
    config=conf,
    image_size=[320, 320],
    train_ann_file="/home/dnth/Downloads/vehicles.v2-release.coco/train/_annotations.coco.json",
    train_img_folder="/home/dnth/Downloads/vehicles.v2-release.coco/train",
    val_ann_file="/home/dnth/Downloads/vehicles.v2-release.coco/valid/_annotations.coco.json",
    val_img_folder="/home/dnth/Downloads/vehicles.v2-release.coco/valid",
    train_batch_size=10,
    val_batch_size=10,
    num_classes=13,
    output_dir="./outputs/deim_hgnetv2_s_vehicles",
)

# def update_learning_rate(config: Config, lr: float = 0.0007) -> Config:
#     """
#     Update the learning rate in the configuration for both optimizer params and base lr.
    
#     Args:
#         config: The configuration object
#         lr: The new learning rate value (default: 0.0007)
    
#     Returns:
#         Config: The updated configuration object
#     """
#     # Get the current params list
#     params_list = config.get('yaml_cfg.optimizer.params')
    
#     # Update the learning rate for backbone parameters (first entry in the list)
#     params_list[0]['lr'] = lr
    
#     # Set the updated params list back to the config
#     config.set('yaml_cfg.optimizer.params', params_list)
#     config.set('yaml_cfg.optimizer.lr', lr)
    
#     return config

# # Update the learning rate using the new function
# conf = update_learning_rate(conf, lr=0.0007)

trainer = Trainer(conf)
# trainer.fit(epochs=60, flat_epoch=30, no_aug_epoch=3, warmup_iter=100, ema_warmups=100)
trainer.fit(
    epochs=100,
    flat_epoch=50,
    no_aug_epoch=3,
    warmup_iter=50,
    ema_warmups=50,
    # lr=0.0005
)

conf.save("my_train_config.yml")
