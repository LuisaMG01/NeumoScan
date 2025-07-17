import numpy as np

from keras.models import load_model, Model

def full_model():
    return load_model("./model/model_keras.keras")

def embedding_model():
    model_full = full_model()
    return Model(inputs=model_full.input, outputs=model_full.layers[-2].output)
