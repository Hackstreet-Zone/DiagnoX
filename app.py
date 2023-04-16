import tensorflow as tf
from util.model import get_model
from util.allowed import allowed_file
from util.firebase import upload_file
from util.gradcam import compute_gradcam

import os
import numpy as np

from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

model = get_model()
app = Flask(__name__)
app.secret_key = "DiagnoX"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    global model
    if 'user_file' not in request.files:
        return {'error': 'No file part'}, 400

    file = request.files['user_file']
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    image = tf.keras.utils.load_img(save_path)
    input_arr = tf.keras.utils.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)

    compute_gradcam(model, filename, app.config['UPLOAD_FOLDER'], df, labels, labels_to_show, predictions=prediction, layer_name='bn')
    upload_url = upload_file(filename)
    return jsonify({'url': upload_url, predictions: prediction}), 200

if __name__ == '__main__':
    app.run(port=8080, debug=True)
