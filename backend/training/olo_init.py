import os
import pathlib
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, Dense, Flatten, MaxPooling2D,
                                     Rescaling)
from tensorflow.keras.models import Sequential

base_dir = Path(__file__).resolve().parent.parent
training_dir = base_dir / 'data' / "training"
validation_dir = base_dir / 'data' / "validation"
testing_dir = base_dir / 'data' / "testing"

IMAGE_SIZE = (600, 600) 
BATCH_SIZE = 32
EPOCHS = 10 
NUM_CLASSES = 6

CATEGORIES = [
    "0_mouse_bite", 
    "1_spur", 
    "2_missing_hole", 
    "3_short", 
    "4_open_circuit", 
    "5_spurious_copper"
]

training_dir_path = pathlib.Path(str(training_dir))
if not training_dir_path.exists():
    print(f"Błąd: Nie znaleziono katalogu danych: {training_dir}")
    print("Proszę zaktualizować zmienną DATA_DIR na poprawną ścieżkę.")
    exit()


print("Ładowanie zbioru danych...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    training_dir_path,
    validation_split=0.2, # 20% danych zostanie użyte jako walidacyjne
    subset="training",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical' 
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    training_dir_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


model = Sequential([
    Rescaling(1./255, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    
    Conv2D(128, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    
    Flatten(),
    
    Dense(256, activation='relu'),
    
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

model.summary()


print("\nRozpoczynanie treningu...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

print("\nTrening zakończony.")


MODEL_SAVE_PATH = 'models/olov2.keras'
model.save(MODEL_SAVE_PATH)
print(f"Model zapisany jako: {MODEL_SAVE_PATH}")


if len(val_ds) > 0:
    print("\nPrzykład użycia na pojedynczym obrazie z zestawu walidacyjnego:")
    
    for images, labels in val_ds.take(1):
        test_image = images[0]
        test_label_index = tf.argmax(labels[0]).numpy()
        test_label_name = CATEGORIES[test_label_index]
        
        img_array = tf.expand_dims(test_image, 0) 
        
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        predicted_index = tf.argmax(score).numpy()
        
        print(f"Prawdziwa wada: {test_label_name}")
        print(f"Przewidziana wada: {CATEGORIES[predicted_index]}")
        print(f"Pewność przewidywania: {100 * tf.reduce_max(score):.2f}%")
        break