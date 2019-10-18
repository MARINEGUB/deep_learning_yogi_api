import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np

upload_folder = '/tmp'
allowed_extensions = {'png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def main():
    classifier = tf.keras.models.load_model('./models')
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = upload_folder

    @app.route('/', methods=['POST'])
    def predict():
        if 'image' not in request.files:
            return jsonify(
                message='Image required'
            )
        file = request.files['image']
        if file.filename == '':
            return jsonify(
                message='Image filename required'
            )
        if not file or not allowed_file(file.filename):
            return jsonify(
                message='Invalid file, image required'
            )
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        image = tf.io.read_file(app.config['UPLOAD_FOLDER'] + "/" + filename)
        image = tf.io.decode_image(image)
        image = tf.image.resize(image, [192, 192])
        image = image / 255.0
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = np.expand_dims(image, axis=0)
        predictions = classifier.predict(image)
        return jsonify(
            predictions=predictions.tolist()[0]
        )


    app.run()


if __name__ == '__main__':
    main()
