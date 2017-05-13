import os
os.environ['THEANO_FLAGS'] = 'floatX=float32,mode=FAST_RUN'

from flask import Flask
from flask_restful import Resource, Api, request
from flask_cors import CORS
from pipeline import evaluate, load_transformers
import json
import cv2
import base64
import urllib
import matplotlib.pyplot as plt
from io import BytesIO

app = Flask(__name__)
CORS(app, origins='http://localhost:8000')
api = Api(app)


@api.resource('/')
class Classifier(Resource):
    @staticmethod
    def generate_plot(plot_results, title):
        fig = plt.figure()
        x = range(0, plot_results.shape[0])
        plt.plot(x, plot_results)
        plt.title(title)
        figfile = BytesIO()
        plt.savefig(figfile, format='png')
        figfile.seek(0)
        figdata_png = base64.b64encode(figfile.getvalue())
        bytes = 'data:image/png;base64,{}'.format(urllib.quote(figdata_png))
        plt.close(fig)
        del fig
        return bytes

    @staticmethod
    def convert_img(img):
        img = cv2.flip(img, 1)
        to_png = cv2.imencode('.png', img)[1]
        bytes = 'data:image/png;base64,{}'.format(urllib.quote(base64.b64encode(to_png)))
        return bytes

    def get(self):
        pass

    def post(self):
        img_array_json = request.form.get('img_array')
        debug = bool(request.form.get('debug'))
        img_array = json.loads(img_array_json)
        evaluated = evaluate(img_array)
        if debug:
            return {'result': str(evaluated['init'][-1].shape) + '\n',
                    'resized': Classifier.convert_img(evaluated['hands'][-1][0]),
                    'predictedChars': Classifier.generate_plot(evaluated['chars'], 'chars'),
                    'classifiedGestures': Classifier.generate_plot(evaluated['gesture'], 'gesture'),
                    'finallyPredicted': Classifier.generate_plot(evaluated['prediction_selection'], 'selection')}
        else:
            return {'result': str(evaluated['init'][-1].shape)}


load_transformers()
app.run()
