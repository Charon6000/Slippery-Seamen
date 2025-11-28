import importlib.util
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from image_display import show_images_from_dataset
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models  # type: ignore
from tensorflow.keras.applications import EfficientNetB0  # type: ignore
from tensorflow.keras.applications.efficientnet import preprocess_input

sorting_path = Path(__file__).resolve().parent / "sorting.py"
spec = importlib.util.spec_from_file_location("sorting", sorting_path)
sorting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sorting)
categories = sorting.categories

training_dir = Path(__file__).resolve().parent.parent / "data" / "training"

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))

os.makedirs('models', exist_ok=True)

IMG_SIZE = (224, 224) #224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

print("Categories:")
i = 0
for cat in categories:
    file_path = training_dir / cat
    print(file_path)
    i+=1
    #ologej for file in file_path.iterdir():
    #     print(file)
        
train_ds = tf.keras.utils.image_dataset_from_directory(
    training_dir,
    labels='inferred',
    label_mode='int',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset='training',
    seed=123
)

val_ds = tf.keras.utils.image_dataset_from_directory(
       training_dir,
    labels='inferred',
    label_mode='int',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset='validation',
    seed=123
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    training_dir,
    labels='inferred',
    label_mode='int',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

show_images_from_dataset(train_ds, num=8)

def _apply_preprocess(x, y):
    x = tf.cast(x, tf.float32)
    x = preprocess_input(x)
    return x, y

train_ds = train_ds.map(_apply_preprocess, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(_apply_preprocess, num_parallel_calls=AUTOTUNE)

train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

print(train_ds)
print(val_ds)

num_classes = 6
inputs = layers.Input(shape=(*IMG_SIZE, 3))
x = inputs

base_model = EfficientNetB0(include_top=False, input_shape=(*IMG_SIZE, 3), weights='imagenet')
base_model.trainable = False
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs, outputs)
                     
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
callbacks = [
    tf.keras.callbacks.ModelCheckpoint('models/best_model.h5', save_best_only=True, monitor='val_loss'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
]
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks
)
base_model.trainable = True

for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

fine_history = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=callbacks)

test_ds = test_ds.map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y), num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

y_true = []
y_pred = []
for x, y in test_ds.unbatch():
    preds = model.predict(tf.expand_dims(x, 0))
    y_pred.append(np.argmax(preds, axis=-1)[0])
    y_true.append(int(y.numpy()))

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))

model.save('models/saved_model.keras')
model.summary()
print('Trainable params:', sum(p.numpy().size for p in model.trainable_weights))

    