from flask import Flask
from flask_restful import Resource, Api, request
from flask_cors import CORS
from pipeline import evaluate
import json
import cv2
import base64
import urllib

app = Flask(__name__)
CORS(app, origins='http://localhost:8000')
api = Api(app)


@api.resource('/')
class Classifier(Resource):

    @staticmethod
    def convert_img(img):
        img = cv2.flip(img,1)
        to_png = cv2.imencode('.png', img)[1]
        bytes = 'data:image/png;base64,{}'.format(urllib.quote(base64.encodestring(to_png).rstrip('\n')))
        return bytes

    def get(self):
        pass

    def post(self):
        img_array_json = request.form.get('img_array')
        img_array = json.loads(img_array_json)
        evaluated = evaluate(img_array)
        return {'result': str(evaluated[0][-1].shape)+ '\n', 'resized': Classifier.convert_img(evaluated[1][-1])}


if __name__ == '__main__':
    app.run()
