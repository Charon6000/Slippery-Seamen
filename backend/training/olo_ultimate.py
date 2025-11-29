import os
import pathlib
import re

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import MobileNetV2  # Używamy MobileNetV2
from tensorflow.keras.layers import (  # Dodano GlobalAveragePooling2D
    BatchNormalization, Conv2D, Dense, Dropout, Flatten,
    GlobalAveragePooling2D, MaxPooling2D, RandomFlip, RandomRotation,
    RandomZoom, Rescaling)
from tensorflow.keras.models import Sequential, load_model

base_dir = pathlib.Path(__file__).resolve().parent.parent

DATA_DIR = base_dir / 'data' / "data_contrast" 

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
NUM_CLASSES = 6
MODEL_SAVE_PATH = 'models/pcb_defect_classifier_mobilenetv2.keras'

# Kategorie wad
CATEGORIES = [
    "0_mouse_bite", 
    "1_spur", 
    "2_missing_hole", 
    "3_short", 
    "4_open_circuit", 
    "5_spurious_copper"
]

CATEGORY_TO_INDEX = {category.split('_')[0]: i for i, category in enumerate(CATEGORIES)}

INDEX_TO_NAME = {i: category for i, category in enumerate(CATEGORIES)}

if not DATA_DIR.exists():
    print(f"Błąd: Nie znaleziono katalogu danych: {DATA_DIR}")
    exit()

def load_data_from_filenames(data_dir, categories_map):
    """Skanuje folder, wyodrębnia etykiety z nazw plików i tworzy listy ścieżek/etykiet."""
    
    all_image_paths = sorted(list(data_dir.glob('*.*')))
    all_image_paths = [str(path) for path in all_image_paths if path.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    all_labels = []
    
    label_parts = []
    for i, category in enumerate(CATEGORIES):
        category_name_part = category.split('_', 1)[1]
        label_parts.append((i, category_name_part))

    # Sortowanie etykiet od najdłuższej do najkrótszej
    sorted_label_parts = sorted(label_parts, key=lambda x: len(x[1]), reverse=True)
    
    print(f"Kolejność sprawdzania etykiet (długość malejąco): {[part[1] for part in sorted_label_parts]}")
    
    valid_image_paths = all_image_paths[:]
    
    for path in all_image_paths:
        filename = pathlib.Path(path).name.lower()
        
        found_label = False
        
        for i, category_name_part in sorted_label_parts:
            if category_name_part in filename:
                all_labels.append(tf.keras.utils.to_categorical(i, num_classes=NUM_CLASSES))
                found_label = True
                break
        
        if not found_label:
            print(f"Ostrzeżenie: pominięto plik bez pasującej etykiety w nazwie: {filename}")
            valid_image_paths.remove(path)
            
    return valid_image_paths, np.array(all_labels)

all_paths, all_labels = load_data_from_filenames(DATA_DIR, CATEGORIES)

if len(all_paths) == 0:
    print("Błąd: Nie znaleziono żadnych plików obrazów z poprawnymi etykietami. Sprawdź ścieżkę i nazwy plików.")
    exit()
    
data_size = len(all_paths)
train_split = int(0.8 * data_size)
val_split = int(0.1 * data_size)

indices = np.arange(data_size)
np.random.shuffle(indices)

train_indices = indices[:train_split]
val_indices = indices[train_split:train_split + val_split]
test_indices = indices[train_split + val_split:]

train_paths = [all_paths[i] for i in train_indices]
train_labels = all_labels[train_indices]

val_paths = [all_paths[i] for i in val_indices]
val_labels = all_labels[val_indices]

test_paths = [all_paths[i] for i in test_indices]
test_labels = all_labels[test_indices]

print(f"Rozmiary zbiorów: Treningowy={len(train_paths)}, Walidacyjny={len(val_paths)}, Testowy={len(test_paths)}")

#normalizacja
def decode_img(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    # Zwraca tensor w zakresie [0, 255]
    return img, label

train_ds_raw = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
val_ds_raw = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
test_ds_raw = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds_raw.map(decode_img, num_parallel_calls=AUTOTUNE) \
                       .cache() \
                       .shuffle(1000) \
                       .batch(BATCH_SIZE) \
                       .prefetch(buffer_size=AUTOTUNE)

val_ds = val_ds_raw.map(decode_img, num_parallel_calls=AUTOTUNE) \
                   .cache() \
                   .batch(BATCH_SIZE) \
                   .prefetch(buffer_size=AUTOTUNE)

test_ds = test_ds_raw.map(decode_img, num_parallel_calls=AUTOTUNE) \
                    .cache() \
                    .batch(BATCH_SIZE) \
                    .prefetch(buffer_size=AUTOTUNE)

#augumentacja
data_augmentation = Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.2),
    RandomZoom(0.1),
], name="data_augmentation")

def create_model():
    print("\nTworzenie modelu MobileNetV2 (Transfer Learning)...")
    
    base_model = MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False, 
        weights='imagenet' 
    )

    base_model.trainable = True #False

    model = Sequential([
        #augumentacja
        data_augmentation, 
        #normalizacja
        Rescaling(1./255), 
        #model bazowy
        base_model,
        #uśrednianie
        GlobalAveragePooling2D(),
        #gąszcz
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dense(NUM_CLASSES, activation='softmax')
    ], name='MobileNetV2_Classifier')
    
    custom_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    
    model.compile(
        optimizer=custom_optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model


model = None
if os.path.exists(MODEL_SAVE_PATH):
    print(f"\nZnaleziono zapisany model: {MODEL_SAVE_PATH}. Ładowanie...")
    try:
        model = load_model(
            MODEL_SAVE_PATH, 
            custom_objects={'data_augmentation': data_augmentation}
        )
        custom_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
        model.compile(
            optimizer=custom_optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    except Exception as e:
        print(f"Błąd podczas ładowania modelu: {e}. Tworzenie nowego modelu.")
        model = create_model()
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

y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred_probs = model.predict(test_ds)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true_labels = np.argmax(y_true, axis=1)

target_names = [name.split('_', 1)[1] for name in CATEGORIES]

print("\nRaport Klasyfikacji:")
print(classification_report(y_true_labels, y_pred, target_names=target_names))

cm = confusion_matrix(y_true_labels, y_pred)
print("\nMacierz Pomyłek (Confusion Matrix):\n", cm)

try:
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=target_names,
                yticklabels=target_names)
    plt.title('Macierz Pomyłek dla Zestawu Testowego')
    plt.ylabel('Prawdziwa Etykieta')
    plt.xlabel('Przewidziana Etykieta')
    plt.show() 
except Exception as e:
    print(f"\nBłąd podczas generowania wizualizacji Macierzy Pomyłek (wymaga Matplotlib i Seaborn): {e}")

pathlib.Path('models').mkdir(exist_ok=True)
model.save(MODEL_SAVE_PATH)
model.summary()
print(f"Forged: {MODEL_SAVE_PATH}")


try:
    if len(test_ds) > 0:
        print("\nPrzykład pojedynczy")
        
        for images, labels in test_ds.take(1):
            test_image = images[0]
            test_label_index = tf.argmax(labels[0]).numpy()
            test_label_name = INDEX_TO_NAME[test_label_index]
            
            img_array = tf.expand_dims(test_image, 0) 
            
            predictions = model.predict(img_array)

            preds = np.asarray(predictions)
            score = preds[0] if preds.ndim > 1 else preds

            if not np.allclose(np.sum(score), 1.0, atol=1e-3):
                score = tf.nn.softmax(score).numpy()
            else:
                if isinstance(score, tf.Tensor):
                    score = score.numpy()

            predicted_index = int(np.argmax(score))
            confidence = float(np.max(score))

            print(f"Prawdziwa wada: {test_label_name}")
            print(f"Przewidziana wada: {INDEX_TO_NAME[predicted_index]}")
            print(f"Pewność przewidywania: {100 * confidence:.2f}%")
            break
except Exception as e:
    print(f"Błąd podczas testowania pojedynczego obrazu: {e}")

print("Rozkład y_true:", np.bincount(y_true_labels))