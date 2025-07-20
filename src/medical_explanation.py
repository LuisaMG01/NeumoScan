#This module generates a clear medical explanation about a prediction, based on clinical descriptions of similar images, using a language model (default: Llama 3).
import ollama

from utils.utils import get_descriptions

def generate_medical_explanation(predicted_class, similar_images, model='llama3'):
    contexts = get_descriptions(similar_images)
    prompt = f"""
        The predicted class is {predicted_class.upper()}. Below are clinical descriptions of similar images:

        {chr(10).join(contexts)}

        Based on this evidence, write a clear and precise medical explanation of why this prediction was made.
    """

    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response['message']['content']
