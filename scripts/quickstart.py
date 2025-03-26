import requests
import zipfile
import os
from deimkit import Trainer, Config, configure_dataset

# Simple function to download and extract a zip file
def download_and_unzip(url, extract_to="./"):
    """Download a zip file and extract its contents"""
    # Download the file
    print(f"Downloading {url}...")
    response = requests.get(url)
    zip_path = os.path.join(extract_to, os.path.basename(url))
    
    # Save the zip file
    os.makedirs(extract_to, exist_ok=True)
    with open(zip_path, 'wb') as f:
        f.write(response.content)
    
    # Extract the zip file
    print(f"Extracting to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    print("Download and extraction complete!")
    return extract_to

download_and_unzip("https://github.com/dnth/DEIMKit/releases/download/v0.1.0/coco8.zip", "./data")

conf = Config.from_model_name("deim_hgnetv2_n")

conf = configure_dataset(
    config=conf,
    image_size=(320, 320),
    train_ann_file="./data/coco8-converted/train_coco.json",
    train_img_folder="./data/coco8-converted",
    val_ann_file="./data/coco8-converted/val_coco.json",
    val_img_folder="./data/coco8-converted",
    train_batch_size=2,
    val_batch_size=2,
    num_classes=81,
    output_dir="./outputs/coco8/deim_hgnetv2_n_10ep",
)

trainer = Trainer(conf)

trainer.fit(epochs=10, save_best_only=True)
