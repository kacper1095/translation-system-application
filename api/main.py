from flask import Flask
from flask_restful import Resource, Api, request
from flask_cors import CORS
from pipeline import evaluate
import json
import cv2
import base64
import urllib
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from io import BytesIO

app = Flask(__name__)
CORS(app, origins='http://localhost:8000')
api = Api(app)


@api.resource('/')
class Classifier(Resource):

    @staticmethod
    def generate_plot(plot_results):
        fig = plt.figure()
        x = range(0, plot_results.shape[0])
        plt.plot(x, plot_results)
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
        return {'result': str(evaluated['init'][-1].shape)+ '\n', 'resized': Classifier.convert_img(evaluated['hands'][-1]),
                'predictedChars': Classifier.generate_plot(evaluated['chars']),
                'classifiedGestures': Classifier.generate_plot(evaluated['gesture']),
                'finallyPredicted': Classifier.generate_plot(evaluated['prediction_selection'])}


if __name__ == '__main__':
    app.run()
