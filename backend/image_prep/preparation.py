import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras import layers, mixed_precision, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import (CSVLogger, EarlyStopping,
                                        ModelCheckpoint, ReduceLROnPlateau)

try:
    from image_display import show_images_from_dataset
    HAS_DISPLAY = True
except ImportError:
    HAS_DISPLAY = False

print("TensorFlow:", tf.__version__)

try:
    mixed_precision.set_global_policy('mixed_float16')
except:
    pass

training_dir = Path(__file__).resolve().parent.parent / "data" / "training"
validation_dir = Path(__file__).resolve().parent.parent / "data" / "validation"
testing_dir = Path(__file__).resolve().parent.parent / "data" / "testing"

os.makedirs('models', exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
AUTOTUNE = tf.data.AUTOTUNE


train_ds_raw = tf.keras.utils.image_dataset_from_directory(
    training_dir,
    labels='inferred',
    label_mode='int',
    image_size=IMG_SIZE,
    batch_size=None,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    labels='inferred',
    label_mode='int',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    testing_dir,
    labels='inferred',
    label_mode='int',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds_raw.class_names
num_classes = len(class_names)
print(f'Class names: {class_names}')

if HAS_DISPLAY:
    show_images_from_dataset(train_ds_raw.batch(8), num=8)

print("Creating balanced dataset...")
class_datasets = []
for i in range(num_classes):
    ds_i = train_ds_raw.filter(lambda x, y: tf.equal(y, i))
    class_datasets.append(ds_i)

train_ds_balanced = tf.data.Dataset.sample_from_datasets(
    class_datasets, 
    weights=[1/num_classes] * num_classes,
    stop_on_empty_dataset=False
)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
], name='data_augmentation')

def preprocess_data(x, y):
    x = tf.cast(x, tf.float32)
    x = preprocess_input(x)
    return x, y

train_ds = train_ds_balanced.batch(BATCH_SIZE)
train_ds = train_ds.map(preprocess_data, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
train_ds = train_ds.prefetch(AUTOTUNE)

val_ds = val_ds.map(preprocess_data, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
test_ds = test_ds.map(preprocess_data, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

inputs = layers.Input(shape=(*IMG_SIZE, 3))
base_model = EfficientNetB0(include_top=False, input_tensor=inputs, weights='imagenet')

base_model.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x) 
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x) # Ensure float32 output for mixed precision

model = models.Model(inputs, outputs)

print('\n=== Phase 1: Training Head ===')

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    steps_per_epoch=200 # Optional: Limit steps since we are oversampling infinitely
)

print('\n=== Phase 2: Fine-Tuning ===')

base_model.trainable = True
# Freeze all layers except the last 20
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    ModelCheckpoint('models/best_model.keras', save_best_only=True, monitor='val_loss'),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7),
    CSVLogger('models/training.log')
]

history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    steps_per_epoch=200, 
    callbacks=callbacks
)

print('\n=== Evaluation ===')

try:
    model.load_weights('models/best_model.keras')
except:
    print("Could not load best weights, using current weights.")

results = model.evaluate(test_ds)
print(f'Test Loss: {results[0]}, Test Acc: {results[1]}')

y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)
predictions = model.predict(test_ds)
y_pred = np.argmax(predictions, axis=-1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

model.save('models/final_model.keras')
print("Model saved to models/final_model.keras")
