import os
import json
import shutil
from PIL import Image

# ===== CESTY =====
YOLO_INPUT_DIR = "images"  # vstupní složka s YOLO .txt a .jpg
YOLOX_ROOT = os.path.join("datasets", "yolox_dataset")
IMAGE_OUT_DIR = os.path.join(YOLOX_ROOT, "images", "train2017")
ANNOT_OUT_PATH = os.path.join(YOLOX_ROOT, "annotations", "instances_train2017.json")

# ===== TVOJE TŘÍDY =====
CLASSES = ["hlava"]  # rozšiř podle potřeby

# ===== Inicializace COCO struktury =====
coco = {
    "images": [],
    "annotations": [],
    "categories": []
}

for i, class_name in enumerate(CLASSES):
    coco["categories"].append({
        "id": i,
        "name": class_name,
        "supercategory": "none"
    })

# ===== Vytvoření výstupních složek =====
os.makedirs(IMAGE_OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(ANNOT_OUT_PATH), exist_ok=True)

annotation_id = 1
image_id = 1
skipped_missing = 0
skipped_invalid = 0

for filename in os.listdir(YOLO_INPUT_DIR):
    if not filename.endswith(".jpg"):
        continue

    basename = os.path.splitext(filename)[0]
    txt_path = os.path.join(YOLO_INPUT_DIR, basename + ".txt")
    img_path = os.path.join(YOLO_INPUT_DIR, filename)

    # === Kontrola existence anotací ===
    if not os.path.exists(txt_path):
        print(f"[⚠️ CHYBÍ ANOTACE] {txt_path}")
        skipped_missing += 1
        continue

    try:
        with Image.open(img_path) as img:
            width, height = img.size
    except Exception as e:
        print(f"[❌ NEJDE NAČÍST OBRÁZEK] {img_path} – {e}")
        skipped_invalid += 1
        continue

    # === Přesun obrázku ===
    shutil.copy(img_path, os.path.join(IMAGE_OUT_DIR, filename))

    # === Zápis obrázku do COCO ===
    coco["images"].append({
        "id": image_id,
        "file_name": filename,
        "width": width,
        "height": height
    })

    # === Zpracuj .txt anotace ===
    with open(txt_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls_id, x_center, y_center, w, h = map(float, parts)
        cls_id = int(cls_id)

        # Převod YOLO -> COCO box
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

# === Ulož JSON ===
with open(ANNOT_OUT_PATH, "w") as f:
    json.dump(coco, f, indent=2)

print("\n✅ HOTOVO!")
print(f"📸 Vygenerováno obrázků: {len(coco['images'])}")
print(f"🔍 Vygenerováno anotací: {len(coco['annotations'])}")
print(f"⚠️ Přeskočeno chybějících anotací: {skipped_missing}")
print(f"❌ Přeskočeno špatných obrázků: {skipped_invalid}")
print(f"📂 Dataset uložen do: {ANNOT_OUT_PATH}")
