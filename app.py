import os
import numpy as np
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import cv2
import shutil
from detection_function import *

# Define a flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # delete the files in ./uploads
        for root, dirs, files in os.walk('uploads'):
            for file in files:
                os.unlink(os.path.join(root, file))

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = image_detection(file_path)

        print(preds)
        return preds
    return None


if __name__ == '__main__':
    app.run()
