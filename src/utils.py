import os
import pandas as pd
import shutil

def convert_to_yolo_format(df: pd.DataFrame, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    df["x_center"] = ((df["x1"] + df["x2"]) / 2) / df["image_width"]
    df["y_center"] = ((df["y1"] + df["y2"]) / 2) / df["image_height"]
    df["width"] = (df["x2"] - df["x1"]) / df["image_width"]
    df["height"] = (df["y2"] - df["y1"]) / df["image_height"]

    # convert class object to 0
    df["class"] = df["class"].replace("object", 1)
    df["class"] = df["class"].replace("empty", 0)

    for img_name, group in df.groupby("image_name"):
        label_path = os.path.join(output_dir, os.path.splitext(img_name)[0] + ".txt")
        group[["class", "x_center", "y_center", "width", "height"]].to_csv(
            label_path, sep=" ", header=False, index=False
        )
    
    print(f"Conversion complete. Files saved in {output_dir}")


def order_images(destination_dir: str):

    for subset in ["test", "train", "val"]:
        os.makedirs(os.path.join(destination_dir, subset), exist_ok=True)

    for img_file in os.listdir(destination_dir):
        full_path = os.path.join(destination_dir, img_file)

        if not os.path.isfile(full_path) or not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        prefix = img_file.split("_")[0].lower()

        if prefix not in ["train", "val", "test"]:
            print(f"⚠️ Prefijo desconocido: {prefix} — se omite {img_file}")
            continue

        dst = os.path.join(destination_dir, prefix, img_file)

        shutil.move(full_path, dst)

    print("Images ordered succesfully.")