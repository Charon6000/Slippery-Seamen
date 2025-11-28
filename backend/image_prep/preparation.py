import importlib.util
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore

sorting_path = Path(__file__).resolve().parent / "sorting.py"
spec = importlib.util.spec_from_file_location("sorting", sorting_path)
sorting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sorting)
categories = sorting.categories

base_path = Path(__file__).resolve().parent.parent / "data" / "training"

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))


IMG_SIZE = (600, 600) #224
BATCH_SIZE = 32


print("Categories:")
i = 0
for cat in categories:
    file_path = base_path / cat
    print(file_path)
    i+=1
    #ologej for file in file_path.iterdir():
    #     print(file)
        
train_ds = tf.keras.utils.image_dataset_from_directory(
    base_path,
    labels='inferred',
    label_mode='categorical',        # 'int' lub 'categorical'
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

normalization = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x,y: (normalization(x), y), num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

print(train_ds)
def show_images_from_dataset(dataset, num=8, ncols=4):
    images = []
    labels = []
    for img, lbl in dataset.unbatch().take(num):
        # jeśli dataset zawiera float w [0,1], przeskaluj do 0-255 do display
        arr = img.numpy()
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255.0, 0, 255).astype('uint8')
        images.append(arr)
        try:
            labels.append(int(lbl.numpy()))
        except Exception:
            labels.append(None)

    # rysuj
    n = len(images)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*3, nrows*3))
    axes = axes.flatten()
    for i in range(n):
        axes[i].imshow(images[i])
        if labels[i] is not None:
            axes[i].set_title(str(labels[i]))
        axes[i].axis('off')
    for ax in axes[n:]:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

show_images_from_dataset(train_ds, num=8)
# fig, axes = plt.subplots(1,10, figsize=(10,10))
# for i in range(10):
#     image = train_ds[i]
#     denormalized = (image + 1)/2
#     axes[i].imshow(denormalized)
#     axes[i].axis('off')
    