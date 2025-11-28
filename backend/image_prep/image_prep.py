import os
from pathlib import Path

# import numpy as np
# import tensorflow as tf
# from tensorflow import karas

here = Path(__file__).resolve().parent.parent / "data"
sort_dir = here / "to_sort"
training_dir = here / "training"

for filename in os.listdir(sort_dir):
    if(filename.__contains__("bite")):
        print(filename)