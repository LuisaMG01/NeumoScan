import os
import json

import numpy as np

from keras.models import load_model, Model


def full_model():
    return load_model("./model/model_keras.keras")

def embedding_model():
    model_full = full_model()
    return Model(inputs=model_full.input, outputs=model_full.layers[-2].output)

def get_descriptions(similar_images):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.abspath(os.path.join(base_dir, "../../data/image_text_db.json"))

    with open(json_path, "r", encoding="utf-8") as f:
        image_descriptions = json.load(f)

    contexts = []
    for img in similar_images:
        path = img["path"]
        if path in image_descriptions:
            contexts.append(image_descriptions[path])

    return contexts