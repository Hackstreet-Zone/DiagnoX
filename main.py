import tensorflow as tf
from util.model import get_model
from util.allowed import allowed_file

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

model = get_model()
app = Flask(__name__)
app.secret_key = "DiagnoX"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'

@app.route('/predict', methods=['POST'])
def predict():
    if 'user_file' not in request.files:
        return {'error': 'No file part'}, 400

    img = request.files['user_file']
    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img.save(save_path)

    image = tf.keras.utils.load_img(save_path)
    input_arr = tf.keras.utils.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)

    prediction = model.predict(img)
    
    return jsonify(str(prediction))

if __name__ == '__main__':
    app.run(port=8080, debug=True)