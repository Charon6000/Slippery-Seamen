from image_prep.preparation import prepare
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore
from tensorflow.keras.models import Sequential

base_path = Path("training_data")
IMG_SIZE=(224, 224)
train_ds, val_ds, categories = prepare(base_path, IMG_SIZE)

num_classes = len(categories)

model = Sequential([
  # usuwamy Rescaling tutaj — dane już znormalizowane w prepare()
  layers.Conv2D(2, 3, padding='same', activation='relu', input_shape=(*IMG_SIZE, 3)),
  layers.MaxPooling2D(),
  layers.Conv2D(4, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(8, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(16, activation='relu'),
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