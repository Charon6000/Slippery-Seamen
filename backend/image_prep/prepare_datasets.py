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

IMG_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32

base_dir = Path(__file__).resolve().parent.parent
training_dir = base_dir / "data" / "training"
validation_dir = base_dir / "data" / "validation"
testing_dir = base_dir / "data" / "testing"

def prepare_datasets(img_size=IMG_SIZE):

    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        training_dir,
        labels='inferred',
        label_mode='int',
        image_size=img_size,
        batch_size=None,
        shuffle=True
    )

    class_names = train_ds_raw.class_names
    num_classes = len(class_names)

    class_datasets = [train_ds_raw.filter(lambda x, y, idx=i: tf.equal(y, idx)) for i in range(num_classes)]
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

    def _preprocess(x, y):
        x = tf.cast(x, tf.float32)
        x = preprocess_input(x)
        return x, y

    train_ds = train_ds_balanced.batch(BATCH_SIZE)
    train_ds = train_ds.map(_preprocess, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.prefetch(AUTOTUNE)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        validation_dir,
        labels='inferred',
        label_mode='int',
        image_size=img_size,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    val_ds = val_ds.map(_preprocess, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        testing_dir,
        labels='inferred',
        label_mode='int',
        image_size=img_size,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    test_ds = test_ds.map(_preprocess, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names
