import os
import json
from PIL import Image

# Cesty
ann_dir = "datasets/yolox_dataset/annotations"
img_dir = "datasets/yolox_dataset/images/train2017"
ann_files = ["instances_train2017.json", "instances_val2017.json"]

for ann_file in ann_files:
    ann_path = os.path.join(ann_dir, ann_file)
    with open(ann_path, "r") as f:
        coco = json.load(f)

    existing_images = []
    existing_anns = []

    available_files = set(os.listdir(img_dir))
    img_id_set = set()

    for img in coco["images"]:
        if img["file_name"] in available_files:
            existing_images.append(img)
            img_id_set.add(img["id"])
        else:
            print(f"❌ Chybí obrázek: {img['file_name']}")

    for ann in coco["annotations"]:
        if ann["image_id"] in img_id_set:
            existing_anns.append(ann)

    coco["images"] = existing_images
    coco["annotations"] = existing_anns

    with open(ann_path, "w") as f:
        json.dump(coco, f)

    print(f"✅ Vyčištěno: {ann_path} (obrázky: {len(existing_images)}, anotace: {len(existing_anns)})")
