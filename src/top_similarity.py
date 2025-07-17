import os
import base64
import numpy as np

from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from constants.model import IMG_SIZE, DATASET_EMBEDDINGS, IMAGE_PATHS


def preprocess_image_pil(image: Image.Image) -> np.ndarray:
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    if len(image.shape) == 2:
        image = np.stack((image,) * 3, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image

def get_top_k_similar_images(query_embedding, k=5, predicted_class=None):
    sims = cosine_similarity(query_embedding, DATASET_EMBEDDINGS)[0]

    filtered_indices = range(len(IMAGE_PATHS))

    if predicted_class:
        filtered_indices = [
            i for i, path in enumerate(IMAGE_PATHS)
            if predicted_class.lower() in os.path.basename(path).lower()
        ]

    filtered_sims = [(i, sims[i]) for i in filtered_indices]
    filtered_sims = sorted(filtered_sims, key=lambda x: x[1], reverse=True)[:k]

    top_k = []
    for idx, sim_score in filtered_sims:
        img_path = IMAGE_PATHS[idx]
        with open(img_path, "rb") as img_file:
            img_bytes = img_file.read()
            img_base64 = base64.b64encode(img_bytes).decode("utf-8")
        
        class_name = os.path.basename(os.path.dirname(img_path)).upper()

        print(f"Image: {os.path.basename(img_path)}, Class: {class_name}, Similarity: {sim_score:.4f}")
        top_k.append({
            "path": os.path.basename(img_path),
            "similarity": float(sim_score),
            "image_base64": img_base64,
            "class": class_name
        })
    return top_k
