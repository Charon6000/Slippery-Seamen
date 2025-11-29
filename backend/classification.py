from image_prep.preparation import prepare
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore
from tensorflow.keras.models import Sequential


base_path = Path("training_data")
IMG_SIZE=(224, 224)
train_ds, val_ds, categories = prepare(base_path, IMG_SIZE)

num_classes = len(categories)

train_ds, val_ds, categories = prepare(base_path, IMG_SIZE)

num_classes = len(categories)


model = Sequential([
  layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(*IMG_SIZE, 3)),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.1),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=1
)
model.save(Path("models/model.h5"))