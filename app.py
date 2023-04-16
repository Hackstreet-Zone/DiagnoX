import tensorflow as tf
from util.model import get_model, labels
from util.allowed import allowed_file
from util.cloud import upload_file
from util.gradcam import compute_gradcam, auc_rocs

import os
os.environ['KERAS_BACKEND'] = 'theano'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('SVG')

from flask import Flask, request, jsonify, render_template, redirect
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

graph = tf.compat.v1.get_default_graph()
tf.compat.v1.disable_eager_execution()

df = pd.read_csv("data/train-small.csv")
labels_to_show = np.take(labels, np.argsort(auc_rocs)[::-1])
app = Flask(__name__)
app.secret_key = "DiagnoX"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['DATA_FOLDER'] = 'data/images'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/docs')
def docs():
    return redirect()

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    global model
    global graph
    global sess
    if 'user_file' not in request.files:
        return {'error': 'No file part'}, 400

    file = request.files['user_file']
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    image = tf.keras.utils.load_img(save_path)
    input_arr = tf.keras.utils.img_to_array(image)
    input_arr = np.array([input_arr])
    with graph.as_default():
        model = get_model()
        model._make_predict_function()
        predictions = model.predict(input_arr)
        compute_gradcam(model, filename, app.config['UPLOAD_FOLDER'], app.config['DATA_FOLDER'], df, labels, labels_to_show, predictions=predictions, layer_name='bn')


    upload_url = upload_file(os.path.join(app.config['OUTPUT_FOLDER'], filename))
    return jsonify({'url': upload_url, 'predictions': predictions.tolist()}), 200

if __name__ == '__main__':
    app.run(port=8080, debug=True)
