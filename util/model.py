import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

labels = ['Cardiomegaly',
          'Emphysema',
          'Effusion',
          'Hernia',
          'Infiltration',
          'Mass',
          'Nodule',
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening',
          'Pneumonia',
          'Fibrosis',
          'Edema',
          'Consolidation']

def get_model():
    base_model = DenseNet121(
        weights='models/nih/densenet.hdf5', include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(len(labels), activation="sigmoid")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model


if __name__ == "__main__":
    model = get_model()
    save_path = 'uploads/00000020_001.png'
    image = tf.keras.utils.load_img(save_path)
    input_arr = tf.keras.utils.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    print(predictions)
