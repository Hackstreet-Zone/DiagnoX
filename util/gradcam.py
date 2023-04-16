import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from tensorflow.keras.utils import load_img, img_to_array
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.compat.v1.logging import INFO, set_verbosity
import os

random.seed(a=None, version=2)
set_verbosity(INFO)

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

auc_rocs = [ 0.896108108108108,
 0.7963598901098901,
 0.7620173769986118,
 0.7741081081081082,
 0.6515798863796423,
 0.8193055555555556,
 0.6556871078729002,
 0.7857407407407407,
 0.8020921544209215,
 0.7115164793293961,
 0.6775135135135135,
 0.7282980958034613,
 0.856054054054054,
 0.7516323068222713
]

IMAGE_DIR = "data/images/"


def get_mean_std_per_batch(image_dir, data_dir, df, H=320, W=320):
    sample_data = []
    for img in df.sample(100)["Image"].values:
        image_path = os.path.join(data_dir, img)
        sample_data.append(
            np.array(load_img(image_path, target_size=(H, W))))
    mean = np.mean(sample_data, axis=(0, 1, 2, 3))
    std = np.std(sample_data, axis=(0, 1, 2, 3), ddof=1)
    return mean, std


def load_image(img, image_dir, data_dir, df, preprocess=True, H=320, W=320):
    """Load and preprocess image."""
    mean, std = get_mean_std_per_batch(image_dir, data_dir, df, H=H, W=W)
    img_path = os.path.join(image_dir, img)
    x = load_img(img_path, target_size=(H, W))
    x = img_to_array(x)
    if preprocess:
        x -= mean
        x /= std
        x = np.expand_dims(x, axis=0)
    return x


def grad_cam(input_model, image, cls, layer_name, H=320, W=320):
    """GradCAM method for visualizing input saliency."""
    y_c = input_model.output[0, cls]
    conv_output = input_model.get_layer(layer_name).output
    grads = K.gradients(y_c, conv_output)[0]

    gradient_function = K.function([input_model.input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis=(0, 1))
    cam = np.dot(output, weights)

    # Process CAM
    cam = cv2.resize(cam, (W, H), cv2.INTER_LINEAR)
    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam


def compute_gradcam(model, img, image_dir, data_dir, df, labels, selected_labels,
                    predictions, layer_name='bn'):
    global labels_to_show
    global auc_rocs
    preprocessed_input = load_image(img, image_dir, data_dir, df)

    print("Loading original image")
    plt.figure(figsize=(30, 50))
    plt.subplot(15,1,1)
    plt.title("Original")
    plt.axis('off')
    tmp = load_image(img, image_dir, data_dir, df, preprocess=False)
    tmp = tmp.astype(np.uint8)
    plt.imshow(tmp, cmap='gray')
    # plt.close(tmp)

    j = 1
    for i in range(len(labels)):
        if labels[i] in selected_labels:
            print(f"Generating gradcam for class {labels[i]}")
            gradcam = grad_cam(model, preprocessed_input, i, layer_name)
            plt.subplot(15,1,j+1)
            plt.title(f"{labels[i]}: p={predictions[0][i]:.3f}")
            plt.axis('off')
            tmp = load_image(img, image_dir, data_dir, df, preprocess=False)
            tmp = tmp.astype(np.uint8)
            # plt.close(tmp)
            plt.imshow(tmp, cmap='gray')
            plt.imshow(gradcam, cmap='jet', alpha=min(0.5, predictions[0][i]))
            j += 1
    plt.savefig(os.path.join('outputs', img),  bbox_inches='tight')


def get_roc_curve(labels, predicted_vals, generator):
    auc_roc_vals = []
    for i in range(len(labels)):
        try:
            gt = generator.labels[:, i]
            pred = predicted_vals[:, i]
            auc_roc = roc_auc_score(gt, pred)
            auc_roc_vals.append(auc_roc)
            fpr_rf, tpr_rf, _ = roc_curve(gt, pred)
            plt.figure(1, figsize=(10, 10))
            plt.plot([0, 1], [0, 1], 'k--')
            plt.plot(fpr_rf, tpr_rf,
                     label=labels[i] + " (" + str(round(auc_roc, 3)) + ")")
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')
        except:
            print(
                f"Error in generating ROC curve for {labels[i]}. "
                f"Dataset lacks enough examples."
            )
    plt.show()
    return auc_roc_vals
