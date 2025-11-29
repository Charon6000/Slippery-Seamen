from pathlib import Path
# from image_prep.preparation import prepare
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import models
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image

def check_test(img_path):
    PIL.image(image=img_path).show()
            
    # base_path = Path(path)
    # IMG_SIZE=(224, 224)
    # train_ds, val_ds, categories = prepare(base_path, IMG_SIZE)

    # model = models.load_model(Path("models/model.h5"))
    # predictions = model.predict(
    #     x=val_ds
    #     , batch_size=10
    #     , verbose=0
    # )  

    # for i in predictions:
    #     print(i)
    # return predictions


check_test("training_data/")
