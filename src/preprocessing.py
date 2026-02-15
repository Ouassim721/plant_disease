import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def preprocess_image(image_path, target_size=(224, 224)):
    """Lit une image, redimensionne et normalise"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return img


# Exemple Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)
