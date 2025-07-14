from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from PIL import Image
import tensorflow as tf

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("./model/model_keras.keras")
class_labels = ['NORMAL', 'BACTERIAL', 'VIRAL']

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    if len(image.shape) == 2:
        image = np.stack((image,)*3, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    input_array = preprocess_image(image)
    prediction = model.predict(input_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    return {"class": predicted_class, "confidence": confidence}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
