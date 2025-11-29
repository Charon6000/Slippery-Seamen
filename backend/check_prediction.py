from pathlib import Path
from image_prep.preparation import prepare
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import models
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import pathlib
from classify_many import classify_and_categorize


def check_test(path):
    model_path = Path("models/model.h5")
    model = load_model(model_path)
    categories = [
            "mouse_bite", 
            "spur", 
            "missing_hole", 
            "short", 
            "open_circuit", 
            "spurious_copper"
        ]

    correct_results = {a:0 for a in categories}
    incorrect_results = {a:0 for a in categories}

    for image in pathlib.Path(path).glob("*/*.jpg"):
        img = Image.open(image)
        img = img.resize((224, 224))
        img_array = np.array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)
        
        category = ""
        for i in categories:
            if i in image.name:
                category = i

        if categories[predicted_class[0]] == category:
            correct_results[category] += 1
        else:
            incorrect_results[category]+=1

    plt.scatter(correct_results.keys(), [correct_results[k] for k in correct_results.keys()], color='g', label='Correct')
    plt.xlabel('Categories')
    plt.ylabel('Quality of Prediction')
    plt.show()
    return predictions


check_test("training_data/")
