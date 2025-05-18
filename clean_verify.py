import json
import os

def clean_annotations(json_path, images_dir):
    with open(json_path, "r") as f:
        data = json.load(f)

    image_ids_to_keep = []
    valid_images = []
    missing_files = []

    existing_files = set(os.listdir(images_dir))

    for img in data["images"]:
        if img["file_name"] in existing_files:
            valid_images.append(img)
            image_ids_to_keep.append(img["id"])
        else:
            missing_files.append(img["file_name"])

    valid_annotations = [ann for ann in data["annotations"] if ann["image_id"] in image_ids_to_keep]

    data["images"] = valid_images
    data["annotations"] = valid_annotations

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ {json_path}: {len(missing_files)} chybějících obrázků odstraněno")

# Cesty
clean_annotations(
    "datasets/yolox_dataset/annotations/instances_train2017.json",
    "datasets/yolox_dataset/images/train2017"
)

clean_annotations(
    "datasets/yolox_dataset/annotations/instances_val2017.json",
    "datasets/yolox_dataset/images/train2017"
)
