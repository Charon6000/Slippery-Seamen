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

    if file:
        filename = file.filename
        save_path = os.path.join("tempFiles/", filename)
        file.save(save_path)
    else:
        return

    
    predictions = classify_and_categorize(temp_dir_name = "tempFiles", model_path = 'models/model.h5', cleanup=True)
    print(predictions)

    return jsonify({
        "results": [{
                "file_name": request.files.get("files").filename,
                "mouse_bite": predictions[0][2],
                "spur": predictions[1][2],
                "missing_hole": predictions[2][2],
                "short": predictions[3][2],
                "opencircut": predictions[4][2],
                 "spurious_copper": predictions[5][2],
             }]})