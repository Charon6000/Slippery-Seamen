import importlib.util
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from image_display import show_images_from_dataset
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, mixed_precision, models  # type: ignore
from tensorflow.keras.applications import EfficientNetB0  # type: ignore
from tensorflow.keras.applications.efficientnet import preprocess_input

training_dir = Path(__file__).resolve().parent.parent / "data" / "training"
validation_dir = Path(__file__).resolve().parent.parent / "data" / "validation"
testing_dir = Path(__file__).resolve().parent.parent / "data" / "testing"

categories = ["0_mouse_bite", "5_spurious_copper", "1_spur", "2_missing_hole", "3_short", "4_open_circuit"]

print("TensorFlow:", tf.__version__)

gpus = tf.config.list_physical_devices('GPU')
print("Physical GPUs:", gpus)
gpu_present = len(gpus) > 0 and tf.test.is_built_with_cuda()
if gpus:
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception as e:
            print("failed memory growth: ", e)

if gpu_present:
    mixed_precision.set_global_policy('mixed_float16')
    tf.config.optimizer.set_jit(True)
else:
    print('GPU not available')

os.makedirs('models', exist_ok=True)

IMG_SIZE = (224, 224)
DEFAULT_BATCH = 128 if gpu_present else 32
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', DEFAULT_BATCH))
AUTOTUNE = tf.data.AUTOTUNE

print("Categories:")
i = 0
for cat in categories:
    file_path = training_dir / cat
    print(file_path)
    i+=1
        
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
    validation_dir,
    labels='inferred',
    label_mode='int',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset='validation',
    seed=123
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    testing_dir,
    labels='inferred',
    label_mode='int',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

#display zone
class_names = train_ds.class_names
print('Class names:', class_names)
show_images_from_dataset(train_ds, num=8)
    

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
], name='data_augmentation')

train_ds = train_ds.map(lambda x, y: (tf.cast(x, tf.float32), y), num_parallel_calls=AUTOTUNE)
train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=AUTOTUNE)

val_ds = val_ds.map(lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y), num_parallel_calls=AUTOTUNE)


num_classes = len(class_names)
counts = {}
for i, cname in enumerate(class_names):
    p = training_dir / cname
    counts[cname] = len([f for f in p.iterdir() if f.is_file()])
print('Class counts:', counts)
total = sum(counts.values()) if counts else 0
if total == 0:
    raise RuntimeError('No training images found; check `training_dir`.')
class_weights = {i: total / (num_classes * counts[class_names[i]]) for i in range(num_classes)}
print('Class weights:', class_weights)

class_weights_list = [class_weights[i] for i in range(num_classes)]
class_weights_tensor = tf.constant(class_weights_list, dtype=tf.float32)

train_ds = train_ds.map(
    lambda x, y: (x, y, tf.gather(class_weights_tensor, tf.cast(y, tf.int32))),
    num_parallel_calls=AUTOTUNE
)

train_ds = train_ds.shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

print(train_ds)
print(val_ds)


inputs = layers.Input(shape=(*IMG_SIZE, 3))
x = data_augmentation(inputs)

base_model = EfficientNetB0(include_top=False, input_shape=(*IMG_SIZE, 3), weights='imagenet')
base_model.trainable = False
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(num_classes)(x)
outputs = layers.Activation('softmax', dtype='float32')(outputs)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
callbacks = [
    tf.keras.callbacks.ModelCheckpoint('models/best_weights.weights.h5', save_best_only=True, save_weights_only=True, monitor='val_loss'),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
]
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks,
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

y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)
y_scores = model.predict(test_ds)
y_pred = np.argmax(y_scores, axis=-1)

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))

try:
    model.save('models/saved_model.keras')
except Exception:
    model.save_weights('models/final_weights.weights.h5')

model.summary()
print('Trainable params:', sum(int(tf.size(w).numpy()) for w in model.trainable_weights))

    