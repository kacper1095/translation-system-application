from flask import Flask
from flask_restful import Resource, Api, request
from flask_cors import CORS, cross_origin
from PIL import Image
from binascii import a2b_base64
import json
import io
import cv2
import re
import numpy as np

app = Flask(__name__)
CORS(app, origins='http://localhost:8000')
api = Api(app)


def load_img(img_str):
    image_data = re.sub('^data:image/.+;base64,', '', img_str)
    buffer = io.BytesIO()
    buffer.write(a2b_base64(image_data))
    img = Image.open(buffer)
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    return img.shape


@api.resource('/')
class Classifier(Resource):
    def get(self):
        pass

    def post(self):
        img_array_json = request.form.get('img_array')
        img_array = json.loads(img_array_json)
        return {'result': str(load_img(img_array[0])) + ' '}


if __name__ == '__main__':
    app.run(debug=True)
