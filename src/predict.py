"""
predict.py

Fonctions pour prédire la classe d'une image avec un modèle entraîné.
"""

import numpy as np
import cv2
from tensorflow.keras.models import load_model


def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """Charge une image, la redimensionne et la normalise."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)


def predict_image(model_path, image_path, class_names):
    """Prédit la classe d'une image et retourne le nom de la classe avec la confiance."""
    model = load_model(model_path)
    img_array = load_and_preprocess_image(image_path)
    predictions = model.predict(img_array, verbose=0)[0]
    pred_class = np.argmax(predictions)
    confidence = predictions[pred_class]
    return class_names[pred_class], confidence