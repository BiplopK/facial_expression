import os
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np

model_path = 'facial_expression.h5'
model = load_model(model_path)

app = Flask(__name__)

UPLOAD_FOLDER = '../datasets'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        # Perform predictions using your ML model here
        prediction = predict(filename)

        return jsonify({"prediction": prediction})

def predict(image_path):
    test_img = cv2.imread(image_path)
    test_img = cv2.resize(test_img, (150, 150))
    test_input = test_img.reshape((1, 150, 150, 3))
    prediction = model.predict(test_input)

    plt.imshow(test_img)

    plt.show()

    # Find the class with the highest probability
    predicted_class = np.argmax(prediction)

    if predicted_class == 0:
        Predicted_Result = 'Angry'
    elif predicted_class == 1:
        Predicted_Result = 'Fear'
    elif predicted_class == 2:
        Predicted_Result = 'Happy'
    elif predicted_class == 3:
        Predicted_Result = 'Neutral'
    elif predicted_class == 4:
        Predicted_Result = 'Sad'
    elif predicted_class == 5:
        Predicted_Result = 'Suprise'
    
    
    return Predicted_Result

if __name__ == '__main__':
    app.run(debug=True)
