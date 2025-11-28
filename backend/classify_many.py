
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.utils import img_to_array, load_img

MODEL_INPUT_SIZE = (224, 224) 
TEMP_DIR_NAME = 'tempFiles'

categories = [
    "0_mouse_bite", 
    "1_spur", 
    "2_missing_hole", 
    "3_short", 
    "4_open_circuit", 
    "5_spurious_copper"
]

def classify_and_categorize(temp_dir_name=TEMP_DIR_NAME, model_path='models/final_model.keras', cleanup=True):
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    base_dir = Path(__file__).resolve().parent.parent
    model_path_abs = base_dir / model_path
    temp_dir_abs = base_dir / 'backend' / temp_dir_name

    if not temp_dir_abs.is_dir():
        logging.error(f"Temporary directory not found at: {temp_dir_abs}")
        return

    model = tf.keras.models.load_model(model_path_abs)
    logging.info(f"Model loaded from: {model_path_abs}")
    
    image_extensions = ('.jpg', '.jpeg', '.png')
    processed_count = 0
    
    for img_path in temp_dir_abs.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in image_extensions:
            processed_count += 1
            logging.info(f"Processing image: {img_path.name}")
            
            try:
                img = load_img(img_path, target_size=MODEL_INPUT_SIZE)
                x = img_to_array(img).astype('float32')
                x = preprocess_input(x)
                x = np.expand_dims(x, 0)
                preds = model.predict(x, verbose=0)[0]
                idxs = np.argsort(preds)[::-1][:len(categories)]
                results = []
                for i in idxs:
                    label = categories[i] if i < len(categories) else f"Unknown_Class_{i}"
                    results.append((int(i), label, float(preds[i])))
                
                logging.info(f"--- Results for {img_path.name} ---")
                for index, label, confidence in results:
                    logging.info(f"  {index} {label}: {confidence:.4f}")
                
            except Exception as e:
                logging.error(f"Error processing {img_path.name}: {e}")
        
    logging.info(f"Classification complete. Processed {processed_count} images.")
    
    if cleanup and temp_dir_abs.exists():
        for child in temp_dir_abs.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
        logging.info(f"Successfully removed contents of temporary directory: {temp_dir_abs}")

classify_and_categorize()
