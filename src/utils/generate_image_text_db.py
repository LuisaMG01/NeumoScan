import os
import json

BASE_DIR = "../../data/splitted/train"
IMAGE_JSON = "../../data/image_text_db.json"

descriptions = {
    "NORMAL": "Chest X-ray with no signs of pneumonia. Clear lungs, no opacities or infiltrates.",
    "BACTERIAL": "Presence of alveolar consolidations in the lower lobes, typical of bacterial infection.",
    "VIRAL": "Bilateral diffuse infiltrates, reticulonodular pattern consistent with viral infection."
}

image_text_db = {}

for class_name in os.listdir(BASE_DIR):
    class_dir = os.path.join(BASE_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue

    for filename in os.listdir(class_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            key = filename
            desc = descriptions.get(class_name.upper(), "Chest X-ray without annotation.")
            image_text_db[key] = desc

with open(IMAGE_JSON, "w", encoding="utf-8") as f:
    json.dump(image_text_db, f, indent=2, ensure_ascii=False)

print("Image_text_db.json file generated successfully.")
