import os
import numpy as np
from PIL import Image, ImageOps
import re
import base64
from io import BytesIO

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect

#tensorflow
import tensorflow as tf
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image

cwd = os.getcwd()

Model_json = os.path.join(cwd, 'GenderClassifier.json')
Model_weigths = os.path.join(cwd, 'GenderClassifier.h5')


# Declare a flask app
app = Flask(__name__)

def get_ImageClassifierModel():
    model_json = open(Model_json, encoding="utf-8", errors="ignore")
    loaded_model_json = model_json.read()
    model_json.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(Model_weigths)

    return model  

def model_predict(img_path, model):
    '''
    Prediction Function for model.
    Arguments: 
        img: is address to image
        model : image classification model
    '''
    im = Image.open(img_path)
    im = im.resize((50,50))
    im = np.array(im)/255
    im = im.reshape(1,50,50,3)
    prob = model.predict(im)[0][0]
    
    return prob
    
def classify(prob):
	if prob >= 0.5:
		return 'male'
	else:
		return 'female'

@app.route('/', methods=['GET'])
def index():
    '''
    Render the main page
    '''
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    '''
    predict function to predict the image
    Api hits this function when someone clicks submit.
    '''
    if request.method == 'POST':
        # Get the image from post request
        image_data = re.sub('^data:image/.+;base64,', '', request.json)
        img_path = BytesIO(base64.b64decode(image_data))
        
        # initialize model
        model = get_ImageClassifierModel()

        # Make prediction
        prob = model_predict(img_path, model)
        result = classify(prob)
        
        # Serialize the result, you can add additional fields
        return jsonify(result=result)
    return None


if __name__ == "__main__":
    app.run(host = "127.0.0.1" ,port = 5000)
