from flask import Flask
from flask.globals import request
import requests
from flask.json import jsonify
import http
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator

lb = LabelBinarizer()


MODEL_PATH = '/home/assem/Desktop/.myfiles/myProjects/api/model/VGG16_Garbage_Classifier.h5'
model = load_model(MODEL_PATH)


def predict(imagePath):
    classes = ['metal', 'glass', 'paper', 'trash', 'cardboard', 'plastic']
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    proba = model.predict(image)[0]
    idx = np.argmax(proba)
    label = classes[idx]

    return label


app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ['png', 'jpg', 'jpeg', 'gif']


@app.route('/', methods=['POST'])
def index():
    if request.method == 'POST':
        try:
            image = request.files['image']
            filenameOb = image.filename
            image.save(
                '/home/assem/Desktop/.myfiles/myProjects/api/image/'+str(filenameOb))
            if not image:
                return jsonify({
                    "error": "you didn't send an image "
                }), http.HTTPStatus.BAD_REQUEST
            if not allowed_file(image.filename):
                return jsonify({
                    "error": "the type of file image is not suitable ",
                    "suitable extensions": "png, jpg, jpeg, gif"
                }), http.HTTPStatus.BAD_REQUEST
            preds = predict(
                '/home/assem/Desktop/.myfiles/myProjects/api/image/'+str(filenameOb))
            a = requests.post('http://127.0.0.1:5000/api',
                              json={'input': preds})
            return a.json(), http.HTTPStatus.OK
        except Exception as exp:
            import pdb
            pdb.set_trace()
            return jsonify({
                'error': 'something went wrong'
            }), http.HTTPStatus.INTERNAL_SERVER_ERROR


@app.route('/test', methods=['POST'])
def test():
    message = request.json.get("input", None)
    if message:
        return jsonify({'msg': 'your pic was of  '+str(message)}), http.HTTPStatus.OK
    else:
        return jsonify({
            'error': 'something went wrong'
        }), http.HTTPStatus.INTERNAL_SERVER_ERROR


if __name__ == '__main__':
    app.run(debug=True)
