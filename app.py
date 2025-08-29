from flask import Flask, request, render_template
import numpy as np, cv2, tensorflow as tf
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
MODEL = tf.keras.models.load_model("leaf_model.h5")  # replace with your model
LABELS = ["Healthy", "Blight", "Rust", "Mildew"]     # adjust classes

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224,224)) / 255.0
    img = np.expand_dims(img, axis=0)
    preds = MODEL.predict(img)[0]
    cls = int(np.argmax(preds))
    conf = float(np.max(preds))
    return LABELS[cls], conf

@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        file = request.files["file"]
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        label, conf = predict_image(filepath)
        return render_template("index.html", filename=filename, label=label, conf=round(conf*100,2))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
