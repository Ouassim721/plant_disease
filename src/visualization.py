import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2


def plot_training_history(history):
    """Trace les courbes loss et accuracy train/val."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Époque')
    axes[0].legend()

    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Époque')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('reports/training_curves.png', dpi=150)
    plt.show()


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Génère une heatmap Grad-CAM pour une image donnée."""
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def display_gradcam(img_path, model, last_conv_layer_name, class_names, target_size=(224, 224)):
    """Affiche l'image originale et sa heatmap Grad-CAM superposée."""
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)

    img_array = np.expand_dims(img_resized / 255.0, axis=0).astype(np.float32)
    preds = model.predict(img_array)
    pred_index = np.argmax(preds[0])
    pred_class = class_names[pred_index]
    confidence = preds[0][pred_index]

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index)
    heatmap_resized = cv2.resize(heatmap, target_size)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    superimposed = (0.4 * heatmap_color + 0.6 * img_resized).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_resized); axes[0].set_title('Image originale'); axes[0].axis('off')
    axes[1].imshow(heatmap, cmap='jet'); axes[1].set_title('Grad-CAM'); axes[1].axis('off')
    axes[2].imshow(superimposed); axes[2].set_title(f'Prédiction: {pred_class}\n({confidence:.1%})'); axes[2].axis('off')
    plt.tight_layout()
    plt.savefig('reports/gradcam_example.png', dpi=150)
    plt.show()
