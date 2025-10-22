import os
import shutil
import time
import logging
import pandas as pd
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


logging.basicConfig(
    filename="cleaning_report.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def validate_images_and_clean_csv(image_dir, ann_csv_path, output_csv_path=None):
    """
    Validates images in a directory, removes corrupted ones, and cleans annotation CSV.

    Args:
        image_dir (str): Path to the folder containing images.
        ann_csv_path (str): Path to the annotations CSV.
        output_csv_path (str, optional): Path to save the cleaned CSV.
                                         If None, overwrites the original.
    """
    print(f"\nValidating images for {os.path.basename(ann_csv_path)}...")
    logging.info(f"Starting validation for {ann_csv_path}")

    annotations = pd.read_csv(ann_csv_path)
    corrupted_images = []

    # Validate images
    for img_name in tqdm(os.listdir(image_dir), desc="Checking images"):
        img_path = os.path.join(image_dir, img_name)
        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception as e:
            corrupted_images.append(img_name)
            logging.warning(f"Corrupted image removed: {img_name} | {e}")
            try:
                os.remove(img_path)
            except Exception as e2:
                logging.error(f"Failed to remove {img_path}: {e2}")

    # Filter out corrupted images from CSV
    before_count = len(annotations)
    cleaned_annotations = annotations[~annotations['image_name'].isin(corrupted_images)]
    after_count = len(cleaned_annotations)
    removed_rows = before_count - after_count

    # Save cleaned CSV
    output_csv_path = output_csv_path or ann_csv_path
    cleaned_annotations.to_csv(output_csv_path, index=False)

    # Report
    print(f"{len(corrupted_images)} corrupted images removed.")
    print(f"{removed_rows} rows removed from CSV.")
    print(f"Cleaned annotations saved to: {output_csv_path}")

    # Detailed logging
    if corrupted_images:
        logging.info(f"Removed {len(corrupted_images)} corrupted images for {ann_csv_path}:")
        for img in corrupted_images:
            logging.info(f" - {img}")
    else:
        logging.info("No corrupted images found.")

    return corrupted_images, removed_rows


def organize_images_into_subfolders(base_path):
    """
    Organizes images into train, test, val folders based on filename prefix.
    """
    print("\nOrganizing images into subfolders...")

    subfolders = ["train", "test", "val"]
    for folder in subfolders:
        os.makedirs(os.path.join(base_path, folder), exist_ok=True)

    start_time = time.time()
    count = 0

    for filename in tqdm(os.listdir(base_path), desc="Moving images"):
        file_path = os.path.join(base_path, filename)
        if not os.path.isfile(file_path):
            continue

        lower_name = filename.lower()
        dest_folder = None

        if lower_name.startswith("train"):
            dest_folder = "train"
        elif lower_name.startswith("test"):
            dest_folder = "test"
        elif lower_name.startswith("val"):
            dest_folder = "val"

        if dest_folder:
            shutil.move(file_path, os.path.join(base_path, dest_folder, filename))
            count += 1

    elapsed_time = time.time() - start_time

    print(f"{count} images moved to their respective folders.")
    print(f"Time taken: {elapsed_time:.2f} seconds.")


def process_chunk(chunk, split, base_dir):
    """Genera archivos .txt por imagen a partir de un chunk del CSV limpio."""
    labels_dir = os.path.join(base_dir, "labels", split)
    os.makedirs(labels_dir, exist_ok=True)

    # Tus CSV no tienen cabecera, así que asignamos columnas manualmente
    chunk.columns = ["image_name", "class_id", "x", "y", "w", "h"]

    for img_name, group in chunk.groupby("image_name"):
        label_path = os.path.join(labels_dir, img_name.replace(".jpg", ".txt"))
        with open(label_path, "w") as f:
            for _, row in group.iterrows():
                f.write(f"{int(row['class_id'])} {row['x']} {row['y']} {row['w']} {row['h']}\n")


def process_csv(csv_path, split, base_dir, chunksize=100_000):
    """Procesa un CSV grande en partes (chunking)."""
    print(f"\nConvirtiendo {split} → formato YOLO ...")
    for chunk in tqdm(pd.read_csv(csv_path, chunksize=chunksize, header=None)):
        process_chunk(chunk, split, base_dir)


def convert_all_to_yolo():
    """Convierte todos los CSV limpios (train, val, test) al formato YOLO."""
    base_dir = "data/SKU110K_dataset"
    csvs = {
        "train": os.path.join(base_dir, "annotations", "annotations_train_clean.csv"),
        "val": os.path.join(base_dir, "annotations", "annotations_val_clean.csv"),
        "test": os.path.join(base_dir, "annotations", "annotations_test_clean.csv"),
    }

    # Crear carpetas destino
    for split in csvs.keys():
        os.makedirs(os.path.join(base_dir, "labels", split), exist_ok=True)

    # Procesar en paralelo (usa 3 procesos)
    with Pool(min(3, cpu_count())) as pool:
        pool.starmap(process_csv, [(csvs[k], k, base_dir) for k in csvs])

    print("\nConversión completada Todos los CSV limpios ahora están en formato YOLO.")


if __name__ == "__main__":
    # === CONFIG ===
    BASE_PATH = "SKU110K_dataset/images"
    ANN_CSV_PATHS = [
        "../data/SKU110K_dataset/annotations/annotations_test_xywh.csv",
        "../data/SKU110K_dataset/annotations/annotations_train_xywh.csv",
        "../data/SKU110K_dataset/annotations/annotations_val_xywh.csv"
    ]
    OUTPUT_CSV_PATHS = [
        "../data/SKU110K_dataset/annotations/annotations_test_clean.csv",
        "../data/SKU110K_dataset/annotations/annotations_train_clean.csv",
        "../data/SKU110K_dataset/annotations/annotations_val_clean.csv"
    ]

    total_removed_images = 0
    total_removed_rows = 0

    # === 1. Validate and clean annotations ===
    for ann_csv, out_csv in zip(ANN_CSV_PATHS, OUTPUT_CSV_PATHS):
        corrupted, removed_rows = validate_images_and_clean_csv(BASE_PATH, ann_csv, out_csv)
        total_removed_images += len(corrupted)
        total_removed_rows += removed_rows

    # === 2. Organize images ===
    organize_images_into_subfolders(BASE_PATH)
    convert_all_to_yolo()
    print("\nFINAL REPORT")
    print(f"Total corrupted images removed: {total_removed_images}")
    print(f"Total annotation rows removed: {total_removed_rows}")
    print("Detailed log saved to: cleaning_report.log")
    print("\nDataset cleaning and organization complete!")

