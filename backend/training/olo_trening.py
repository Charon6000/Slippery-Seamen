import os
import pathlib

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D, RandomFlip, RandomRotation,
                                     RandomZoom, Rescaling)
from tensorflow.keras.models import Sequential, load_model

base_dir = pathlib.Path(__file__).resolve().parent.parent

training_dir = base_dir / 'data' / "training"
validation_dir = base_dir / 'data' / "validation"
testing_dir = base_dir / 'data' / "testing"

IMAGE_SIZE = (600, 600)
BATCH_SIZE = 32
EPOCHS = 1
NUM_CLASSES = 6
MODEL_SAVE_PATH = 'models/olov2.keras'

CATEGORIES = [
    "0_mouse_bite", 
    "1_spur", 
    "2_missing_hole", 
    "3_short", 
    "4_open_circuit", 
    "5_spurious_copper"
]

for dir_path in [training_dir, validation_dir, testing_dir]:
    if not dir_path.exists():
        print(f"Błąd: Nie znaleziono katalogu danych: {dir_path}")
        exit()

print("Ładowanie zbiorów treningowych, walidacyjnych i testowych...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    training_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    testing_dir,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2),                   
    RandomZoom(0.1),                       
], name="data_augmentation")


def create_model():
    print("\nTworzenie modelu MobileNetV2 (Transfer Learning)...")
    
    base_model = MobileNetV2(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False, 
        weights='imagenet'
    )
    
    base_model.trainable = False

    model = Sequential([
        Rescaling(1./127.5, offset=-1), 
        
        data_augmentation,
        
        base_model,
        
        tf.keras.layers.GlobalAveragePooling2D(),
        
        Dropout(0.2),
        
        Dense(NUM_CLASSES, activation='softmax')
    ], name='MobileNetV2_Classifier')

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model

if os.path.exists(MODEL_SAVE_PATH):
    print(f"\nZnaleziono zapisany model: {MODEL_SAVE_PATH}. Ładowanie...")
    model = load_model(MODEL_SAVE_PATH)
        
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
else:
    print("\nNie znaleziono zapisanego modelu. Tworzenie nowego...")
    model = create_model()


print(f"\nRozpoczynanie treningu na {EPOCHS} epok...")

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', 
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping]
)

print("\nTrening zakończony.")

print("\nOcena modelu na zbiorze testowym...")
loss, acc = model.evaluate(test_ds)
print(f"Dokładność na zbiorze testowym: {acc * 100:.2f}%")

model.save(MODEL_SAVE_PATH)
model.summary()
print(f"Nowy/Ulepszony model zapisany jako: {MODEL_SAVE_PATH}")


if len(test_ds.take(1)) > 0:
    print("\nPrzykład użycia na pojedynczym obrazie z zestawu testowego:")
    
    # Pobranie jednej partii danych do testowania
    for images, labels in test_ds.take(1):
        test_image = images[0]
        test_label_index = tf.argmax(labels[0]).numpy()
        test_label_name = CATEGORIES[test_label_index]
        
        img_array = tf.expand_dims(test_image, 0) 
        
        predictions = model.predict(img_array)

        preds = np.asarray(predictions)
        if preds.ndim == 1:
            score = preds
        else:
            score = preds[0]

        if not np.allclose(np.sum(score), 1.0, atol=1e-3):
            score = tf.nn.softmax(score).numpy()
        else:
            if isinstance(score, tf.Tensor):
                score = score.numpy()

        predicted_index = int(np.argmax(score))
        confidence = float(np.max(score))

        print(f"Prawdziwa wada: {test_label_name}")
        print(f"Przewidziana wada: {CATEGORIES[predicted_index]}")
        print(f"Pewność przewidywania: {100 * confidence:.2f}%")
        break