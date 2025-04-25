from flask import Flask, render_template, request
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("sign_language_mobilenetv2_improved.keras")

labels = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
    5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E',
    15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O',
    25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y',
    35: 'Z', 36: 'del', 37: 'nothing', 38: 'space', 39: 'unknown'
}

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def preprocess_image(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)  # Convert back to 3 channels
    thresh = cv2.resize(thresh, (128, 128))
    thresh = thresh / 255.0
    return np.expand_dims(thresh, axis=0)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    filename = None
    if request.method == "POST":
        file = request.files["file"]
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)

        img = preprocess_image(filename)
        pred = model.predict(img)
        prediction = labels[np.argmax(pred)]

    return render_template("index.html", prediction=prediction, filename=filename)

if __name__ == "__main__":
    app.run(debug=True)
