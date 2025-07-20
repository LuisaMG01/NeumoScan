import numpy as np
import uvicorn
from PIL import Image
import base64

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from top_similarity import preprocess_image_pil, get_top_k_similar_images
from constants.model import CLASS_LABELS
from utils.utils import full_model, embedding_model
from medical_explanation import generate_medical_explanation


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_full = full_model()
model_embedding = embedding_model()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    input_array = preprocess_image_pil(image)
    
    pred = model_full.predict(input_array)
    predicted_class = CLASS_LABELS[np.argmax(pred)]
    confidence = float(np.max(pred))
    embedding = model_embedding.predict(input_array)
    similar_images = get_top_k_similar_images(embedding, k=5, predicted_class=predicted_class)
    medical_explanation = generate_medical_explanation(predicted_class, similar_images)

    print(medical_explanation)
    return {
        "class": predicted_class,
        "confidence": confidence,
        "similar_images": similar_images,
        "medical_explanation": medical_explanation
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
