import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RESULT_FOLDER'] = 'static/results'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Load Model
model = load_model("models/hibiscus_model.h5")

# Class Names
CLASS_NAMES = ["Early_Mild_Spotting", "Healthy", "Wrinkled_Leaf"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}_{file.filename}"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(256,256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    pred = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(pred)]
    confidence = round(float(np.max(pred) * 100), 2)

    # Annotated output image
    annotated_path = os.path.join(app.config['RESULT_FOLDER'], f"result_{filename}")
    img_original = Image.open(filepath)
    draw = ImageDraw.Draw(img_original)
    text = f"{predicted_class} ({confidence}%)"
    draw.text((10,10), text, fill=(255,0,0))
    img_original.save(annotated_path)

    return render_template(
        "index.html",
        file_path=filepath,
        annotated_path=annotated_path,
        pred_class=predicted_class,
        confidence=confidence
    )

if __name__ == "__main__":
    app.run(debug=True)
