import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from prepare_datasets import prepare_datasets
from sklearn.metrics import classification_report
from tensorflow.keras import layers, mixed_precision, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import (CSVLogger, EarlyStopping,
                                        ModelCheckpoint, ReduceLROnPlateau)

IMG_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32



def load_and_train_final_model(input_model='models/final_model.keras',
                               epochs_head=3,
                               epochs_finetune=5,
                               steps_per_epoch=100):

    candidate_paths = [input_model, 'models/saved_model.keras', 'models/best_model.h5', 'models/best_model.keras']
    chosen = None
    for p in candidate_paths:
        if Path(p).exists():
            chosen = p
            break
    if chosen is None:
        raise FileNotFoundError(f"None of the candidate model files were found: {candidate_paths}")

    print(f"Loading model from: {chosen}")
    model = tf.keras.models.load_model(chosen)

    train_ds, val_ds, test_ds, class_names = prepare_datasets()
    for layer in model.layers:
        layer.trainable = False
    for layer in model.layers[-5:]:
        layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print('\n=== Phase 1: Training head ===')
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_head,
        steps_per_epoch=steps_per_epoch
    )

    for layer in model.layers[-20:]:
        layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        ModelCheckpoint('models/retrained_best.keras', save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7),
        CSVLogger('models/retrain.log')
    ]

    print('\n=== Phase 2: Fine-tuning ===')
    history_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_finetune,
        steps_per_epoch=steps_per_epoch,
        callbacks=callbacks
    )

    print('\n=== Evaluation ===')
    try:
        model.load_weights('models/retrained_best.keras')
    except Exception:
        print('Could not load best retrained weights, using current weights.')

    test_results = model.evaluate(test_ds)
    print(f'Test Loss: {test_results[0]}, Test Acc: {test_results[1]}')

    y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)
    predictions = model.predict(test_ds)
    y_pred = np.argmax(predictions, axis=-1)
    print('\nClassification Report:')
    print(classification_report(y_true, y_pred, target_names=class_names))

    out_path = 'models/final_model_retrained.keras'
    model.save(out_path)
    print(f'Model saved to {out_path}')

    return model, history_head, history_ft, test_results

load_and_train_final_model()
