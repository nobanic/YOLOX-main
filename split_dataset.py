import os
import json
import shutil
import random
from PIL import Image

# ==== VSTUP ====
YOLO_DIR = "images"  # adresář se vstupními .jpg a .txt
OUT_DIR = "datasets/yolox_dataset"  # výstupní složka
VAL_RATIO = 0.2  # podíl validačních dat

# ==== TŘÍDY ====
CLASSES = [
    "hlava"
]

# ==== SLOŽKY ====
TRAIN_IMG_DIR = os.path.join(OUT_DIR, "images/train2017")
VAL_IMG_DIR = os.path.join(OUT_DIR, "images/val2017")
ANNOT_DIR = os.path.join(OUT_DIR, "annotations")

os.makedirs(TRAIN_IMG_DIR, exist_ok=True)
os.makedirs(VAL_IMG_DIR, exist_ok=True)
os.makedirs(ANNOT_DIR, exist_ok=True)

# ==== PÁRY JPG + TXT ====
all_images = [f for f in os.listdir(YOLO_DIR) if f.endswith(".jpg")]
random.shuffle(all_images)

val_size = int(len(all_images) * VAL_RATIO)
val_images = set(all_images[:val_size])
train_images = set(all_images[val_size:])

# ==== Pomocná funkce ====
def process_set(image_set, output_dir, json_path):
    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name, "supercategory": "none"} for i, name in enumerate(CLASSES)]
    }

    image_id = 1
    annotation_id = 1

    for img_name in sorted(image_set):
        basename = os.path.splitext(img_name)[0]
        img_path = os.path.join(YOLO_DIR, img_name)
        txt_path = os.path.join(YOLO_DIR, basename + ".txt")

        if not os.path.exists(txt_path):
            print(f"⚠️ Přeskočeno (chybí anotace): {txt_path}")
            continue

        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except:
            print(f"⚠️ Neplatný obrázek: {img_path}")
            continue

        shutil.copy(img_path, os.path.join(output_dir, img_name))

        coco["images"].append({
            "id": image_id,
            "file_name": img_name,
            "width": width,
            "height": height
        })

        with open(txt_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cls_id, x_center, y_center, w, h = map(float, parts)
            cls_id = int(cls_id)

            x = (x_center - w / 2) * width
            y = (y_center - h / 2) * height
            w_box = w * width
            h_box = h * height

            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": cls_id,
                "bbox": [x, y, w_box, h_box],
                "area": w_box * h_box,
                "iscrowd": 0
            })
            annotation_id += 1

        image_id += 1

    with open(json_path, "w") as f:
        json.dump(coco, f, indent=2)
    print(f"✅ JSON uložen: {json_path} ({len(coco['images'])} obrázků)")

# ==== Vytvoření dat ====
process_set(train_images, TRAIN_IMG_DIR, os.path.join(ANNOT_DIR, "instances_train2017.json"))
process_set(val_images, VAL_IMG_DIR, os.path.join(ANNOT_DIR, "instances_val2017.json"))
