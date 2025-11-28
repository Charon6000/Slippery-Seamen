import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models  # type: ignore

sorting_path = Path(__file__).resolve().parent / "sorting.py"
spec = importlib.util.spec_from_file_location("sorting", sorting_path)
sorting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sorting)
categories = sorting.categories

training_dir = Path(__file__).resolve().parent.parent / "data" / "training"

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))

print("Categories:")
for cat in categories:
    print(" -", cat)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
