from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras import layers, models 

app = Flask(__name__)
CORS(app)
@app.route("/", methods=["POST"])
def upload_data():
    return "Hello, World!"
    
    # return jsonify({"mouse_bite": {models.load_model("mouse_bite_model.h5").predict(request.files.get("files"))},
    #                 "spur": {models.load_model("spur_model.h5").predict(request.files.get("files"))},
    #                 "missing_hole": {models.load_model("missing_hole_model.h5").predict(request.files.get("files"))},
    #                 "short": {models.load_model("short_model.h5").predict(request.files.get("files"))},
    #                 "opencircut": {models.load_model("opencircut_model.h5").predict(request.files.get("files"))},
    #                 "spurious_copper": {models.load_model("spurious_copper_model.h5").predict(request.files.get("files"))},})