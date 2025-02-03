import os
import glob
import argparse
import shutil

# コマンドライン引数を設定する
parser = argparse.ArgumentParser(description="Generate train.txt and val.txt from image directories.")
parser.add_argument("--dataset_name", type=str, default='wholebody28', help="Dataset folder name (e.g., 'wholebody28')")
args = parser.parse_args()

# コマンドライン引数から dataset_name を取得
dataset_name = args.dataset_name

# 実行中のPythonスクリプトのフォルダパスを取得
current_directory = os.path.dirname(os.path.abspath(__file__))

# ファイルパスとテキストファイル名のマッピングを設定
folders = {
    f"{current_directory}/{dataset_name}/images/train": f"{current_directory}/{dataset_name}/train.txt",
    f"{current_directory}/{dataset_name}/images/val": f"{current_directory}/{dataset_name}/val.txt"
}

# 各フォルダに対して処理を行う
for folder, txt_file in folders.items():
    # フォルダが存在するかを確認し、存在しない場合はスキップ
    if not os.path.exists(folder):
        print(f"Warning: Folder '{folder}' does not exist. Skipping.")
        continue

    # フォルダ内の全ての画像ファイル（jpg, png, bmpなど）を取得
    image_files = glob.glob(os.path.join(folder, "*.*"))

    # 画像ファイル名を抽出してテキストファイルに追記
    with open(txt_file, 'a') as f:
        for image_file in image_files:
            file_name = os.path.basename(image_file)  # ファイル名のみを取得
            f.write(file_name + "\n")  # ファイル名を1行に1つずつ追記

    print(f"Processed {len(image_files)} files in {folder} and written to {txt_file}")

# 画像とラベルファイルの移動を行う関数を定義
def move_files_to_parent_directory(source_dir, destination_dir):
    # ソースディレクトリ内の全ファイルを取得
    files = glob.glob(os.path.join(source_dir, "*.*"))

    # 各ファイルをディスティネーションディレクトリへ移動
    for file_path in files:
        # ファイルの移動先パスを決定
        file_name = os.path.basename(file_path)
        destination_path = os.path.join(destination_dir, file_name)

        # ファイルを移動
        shutil.move(file_path, destination_path)
        print(f"Moved: {file_path} -> {destination_path}")

# 画像ファイルの移動 (images/train/* と images/val/* を images/ 直下に移動)
images_train_dir = f"{current_directory}/{dataset_name}/images/train"
images_val_dir = f"{current_directory}/{dataset_name}/images/val"
images_dest_dir = f"{current_directory}/{dataset_name}/images"

# ディレクトリが存在する場合のみファイルを移動
if os.path.exists(images_train_dir):
    move_files_to_parent_directory(images_train_dir, images_dest_dir)
if os.path.exists(images_val_dir):
    move_files_to_parent_directory(images_val_dir, images_dest_dir)

# ラベルファイルの移動 (labels/train/* と labels/val/* を labels/ 直下に移動)
labels_train_dir = f"{current_directory}/{dataset_name}/labels/train"
labels_val_dir = f"{current_directory}/{dataset_name}/labels/val"
labels_dest_dir = f"{current_directory}/{dataset_name}/labels"

# ディレクトリが存在する場合のみファイルを移動
if os.path.exists(labels_train_dir):
    move_files_to_parent_directory(labels_train_dir, labels_dest_dir)
if os.path.exists(labels_val_dir):
    move_files_to_parent_directory(labels_val_dir, labels_dest_dir)

# 空になったフォルダを削除する関数を定義
def remove_empty_directory(directory):
    # ディレクトリが存在し、かつ空である場合に削除
    if os.path.exists(directory) and not os.listdir(directory):  # 空のディレクトリは `os.listdir()` が空のリストを返す
        os.rmdir(directory)
        print(f"Removed empty directory: {directory}")

# 空のフォルダを削除 (images/train, images/val, labels/train, labels/val)
remove_empty_directory(images_train_dir)
remove_empty_directory(images_val_dir)
remove_empty_directory(labels_train_dir)
remove_empty_directory(labels_val_dir)

print("All files have been successfully moved and empty directories have been deleted.")
