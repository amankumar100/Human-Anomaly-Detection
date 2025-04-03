from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Register 'mse' loss explicitly
custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}

# Load trained model with custom_objects
try:
    model = tf.keras.models.load_model(r"C:\H.A.D\backend\model.h5", custom_objects=custom_objects)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Class labels
classes = [
    "Abuse", "Arrest", "Arson", "Assault", "Burglary",
    "Explosion", "Fighting", "Normal Videos", "Road Accidents",
    "Robbery", "Shooting", "Shoplifting", "Stealing", "Vandalism"
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]
        img = image.load_img(file, target_size=(64, 64))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        result = {"prediction": classes[class_index], "confidence": float(np.max(prediction))}
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
