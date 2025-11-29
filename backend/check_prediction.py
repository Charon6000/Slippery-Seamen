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
from classify_many import classify_and_categorize, categories as CATEGORIES
import seaborn as sns

def check_test(path):
    model_path = Path("models/model.h5")
    model = load_model(model_path)
    # re-use categories from classify_many so labels/index mapping matches training
    categories = list(CATEGORIES)

    # Containers to build actual vs predicted label lists
    y_true = []
    y_pred = []

    for image in pathlib.Path(path).glob("*/*.jpg"):
        img = Image.open(image)
        img = img.resize((224, 224))
        img_array = np.array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        predicted_idx = int(np.argmax(predictions, axis=1)[0])

        # determine true label from parent folder name (training data is in subfolders)
        true_label = image.parent.name
        if true_label in categories:
            true_idx = categories.index(true_label)
        else:
            # fallback: try to infer from filename if parent folder doesn't match
            match_idx = None
            for idx, c in enumerate(categories):
                if c in image.name:
                    match_idx = idx
                    break
            if match_idx is None:
                # unknown — skip this sample
                print(f"Skipping unknown label for {image}")
                continue
            true_idx = match_idx

        y_true.append(true_idx)
        y_pred.append(predicted_idx)

    # If no labeled images were found, exit early
    if not y_true:
        print("No labeled images were found, nothing to plot.")
        return

    # Build confusion matrix without external dependencies
    n = len(categories)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # create readable tick labels (strip numeric prefix like '0_')
    tick_labels = [c.split('_', 1)[-1] if '_' in c else c for c in categories]

    try:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=tick_labels, yticklabels=tick_labels)
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.title('Confusion matrix for dataset')
        plt.show()
    except Exception as e:
        print(f"Error generating heatmap: {e}")


check_test("training_data/")
