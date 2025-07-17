"""
Template de Integración para Modelo Real - Semana 3
==================================================
Este archivo muestra cómo integrar el modelo real de Member 1 con Grad-CAM

Uso:
1. Recibir multiclass_model.py 
2. Cargar modelo entrenado
3. Ejecutar evaluación con Grad-CAM
"""

from semana_3_multiclass_gradcam import MulticlassEvaluator, GradCAMExplainer
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import os

def integrate_real_model():
    """
    Integrar modelo real con framework Grad-CAM
    """
    print("Integrando modelo real con Grad-CAM...")
    
    # Paso 1: Cargar modelo entrenado
    model_path = "models/multiclass_pneumonia_model.h5"  # Ajustar según Member 1
    
    try:
        trained_model = load_model(model_path)
        print(f"Modelo cargado exitosamente desde: {model_path}")
        trained_model.summary()
    except Exception as e:
        print(f"Error cargando modelo: {e}")
        return
    
    # Paso 2: Configurar Grad-CAM con modelo real
    class_names = ['Normal', 'Bacterial Pneumonia', 'Viral Pneumonia']
    real_gradcam = GradCAMExplainer(trained_model, class_names)
    
    print(f"Grad-CAM configurado con modelo real")
    print(f"Última capa conv: {real_gradcam.last_conv_layer_name}")
    
    # Paso 3: Cargar dataset de test
    multiclass_df = pd.read_csv("results/multiclass_gradcam/multiclass_dataset.csv")
    test_samples = multiclass_df.sample(20, random_state=42)
    
    print(f"Cargadas {len(test_samples)} muestras de prueba")
    
    # Paso 4: Generar explicaciones reales
    output_dir = "results/multiclass_gradcam/real_explanations/"
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, row in test_samples.iterrows():
        image_path = row['image_path']
        true_label = row['label']
        
        # Cargar y procesar imagen
        try:
            # Aquí se usaría el preprocesamiento del modelo real
            # image_array = preprocess_image(image_path)  # Implementar según modelo
            
            # Por ahora, imagen dummy para template
            import numpy as np
            image_array = np.random.rand(224, 224, 3)
            
            # Generar explicación
            save_path = os.path.join(output_dir, f"real_explanation_{idx}_true_{true_label}.png")
            predictions, predicted_class = real_gradcam.explain_prediction(
                image_array, save_path=save_path
            )
            
            print(f"Explicación {idx}: Verdadero={true_label}, Predicho={predicted_class}, Confianza={predictions[predicted_class]:.3f}")
            
        except Exception as e:
            print(f"Error procesando imagen {idx}: {e}")
    
    print("Integración completada!")

if __name__ == "__main__":
    integrate_real_model()
