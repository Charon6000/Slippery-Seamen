from image_prep.preparation import prepare
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore
from tensorflow.keras.models import Sequential


base_path = Path(__file__).resolve().parent / "training_data"
IMG_SIZE=(32, 32)
train_ds, val_ds, categories = prepare(base_path, IMG_SIZE)

num_classes = len(categories)

train_ds, val_ds, categories = prepare(base_path, IMG_SIZE)

num_classes = len(categories)
model = models.load_model(Path(__file__).resolve().parent / "models/spur_model.h5")

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=100
)
model.save(Path(__file__).resolve().parent / "models/spur_model.h5")