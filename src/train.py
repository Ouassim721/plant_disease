from src.model import create_cnn_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def train_model(train_dir, val_dir, epochs=5, batch_size=32, num_classes=5):
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(train_dir, target_size=(
        224, 224), batch_size=batch_size, class_mode='sparse')
    val_gen = val_datagen.flow_from_directory(val_dir, target_size=(
        224, 224), batch_size=batch_size, class_mode='sparse')

    model = create_cnn_model(num_classes=num_classes)
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)

    model.save("models/baseline_model.h5")
    return model, history
