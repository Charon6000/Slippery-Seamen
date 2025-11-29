from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras import layers, models 
from pathlib import Path
from flask import send_file
from tensorflow.keras.models import load_model
import os
from classify_many import classify_and_categorize


app = Flask(__name__)
CORS(app)
@app.route("/", methods=["POST"])
def upload_data():
    file = request.files.get("files")

    if not file:
        return jsonify({"error": "no file uploaded"}), 400

    filename = file.filename
    save_path = os.path.join("tempFiles", filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    file.save(save_path)

    # [(index, label, confidence), ...]
    predictions = classify_and_categorize(temp_dir_name="tempFiles", model_path='models/model.h5', cleanup=True)

    if not predictions:
        return jsonify({"error": "no prediction returned"}), 500

    pred_map = {label: float(conf) for (_idx, label, conf) in predictions}

    result = {
        "file_name": filename,
        "mouse_bite": round(pred_map.get("0_mouse_bite", 0.0),2),
        "spur": round(pred_map.get("1_spur", 0.0),2),
        "missing_hole": round(pred_map.get("2_missing_hole", 0.0),2),
        "short": round(pred_map.get("3_short", 0.0),2),
        "opencircut": round(pred_map.get("4_open_circuit", 0.0),2),
        "spurious_copper": round(pred_map.get("5_spurious_copper", 0.0),2),
    }

    return jsonify(result)