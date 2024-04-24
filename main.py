import numpy as np
import os
import pathlib

from flask import Flask, request, jsonify
from pathlib import Path

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf

from utils.score_utils import mean_score, std_score

with tf.device('/CPU:0'):
    base_model = MobileNet((None, None, 3), alpha=1, include_top=False, pooling='avg', weights=None)
    x = Dropout(0.75)(base_model.output)
    x = Dense(10, activation='softmax')(x)

    model = Model(base_model.input, x)
    model.load_weights('weights/mobilenet_weights.h5')

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello():
    return jsonify({'result': 'HELLO WORLD'}) 

@app.route('/evaluate', methods=['POST'])
def evaluate_image():
    images = request.files.getlist('image')

    score_list = []
    for idx, img in enumerate(images):
        img_path = os.path.join('img/', f'temp_{idx}.jpg')

        img.save(img_path)

        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        x = preprocess_input(x)

        scores = model.predict(x, batch_size=1, verbose=0)[0]

        mean = mean_score(scores)
        std = std_score(scores)

        file_name = Path(img_path).name.lower()
        score_list.append((file_name, mean))

        print("Evaluating : ", img_path)
        print("NIMA Score : %0.3f +- (%0.3f)" % (mean, std))

    score_list = sorted(score_list, key=lambda x: x[1], reverse=True)
    for i, (name, score) in enumerate(score_list):
        print("%d)" % (i + 1), "%s : Score = %0.5f" % (name, score))
    
    return jsonify({'result': score_list})

if __name__ == '__main__':
    app.run(debug=True)
    
