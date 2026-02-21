"""
evaluate.py

Fonctions d'évaluation des modèles : métriques, matrice de confusion, rapport.
"""

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', cmap='Blues'):
    """Affiche une matrice de confusion."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap,
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


def evaluate_model(model, test_generator):
    """Évalue le modèle sur le générateur de test et affiche les métriques."""
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Prédictions
    test_generator.reset()
    predictions = model.predict(test_generator)
    pred_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    class_labels = list(test_generator.class_indices.keys())

    # Matrice de confusion
    cm = confusion_matrix(true_classes, pred_classes)
    plot_confusion_matrix(cm, class_labels, title='Confusion Matrix')

    # Rapport de classification
    print("\nClassification Report:")
    print(classification_report(true_classes, pred_classes, target_names=class_labels))