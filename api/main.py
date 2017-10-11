import os

os.environ['THEANO_FLAGS'] = 'floatX=float32,mode=FAST_RUN'

from flask import Flask
from flask_restful import Resource, Api, request
from flask_cors import CORS
# from flask_sockets import Sockets
from pipeline import evaluate, load_transformers, convert_last_output_to_ascii, convert_hand_tracker_output_to_readable
from src.utils.Logger import Logger
from urllib.parse import quote
import json
import cv2
import os
import base64
import matplotlib.pyplot as plt
from io import BytesIO

app = Flask(__name__)
CORS(app, origins='http://localhost:8000')
api = Api(app)

COMBINED_PREDICTION_INDEX = 0
DICTIONARY_PREDICTION_INDEX = 1


def clear_log():
    if os.path.exists("log.txt"):
        os.remove("log.txt")


@api.resource('/')
class Classifier(Resource):
    @staticmethod
    def generate_plot(plot_results, title):
        if plot_results is None:
            return None
        fig = plt.figure()
        x = range(0, plot_results.shape[0])
        plt.plot(x, plot_results)
        plt.title(title)
        figfile = BytesIO()
        plt.savefig(figfile, format='png')
        figfile.seek(0)
        figdata_png = base64.b64encode(figfile.getvalue())
        bytes = 'data:image/png;base64,{}'.format(quote(figdata_png))
        plt.close(fig)
        del fig
        return bytes

    @staticmethod
    def convert_img(img):
        img = cv2.flip(img, 1)
        to_png = cv2.imencode('.png', img)[1]
        bytes = 'data:image/png;base64,{}'.format(quote(base64.b64encode(to_png)))
        return bytes

    def get(self):
        pass

    def post(self):
        with Logger("all"):
            img_array_json = request.form.get('img_array')
            debug = request.form.get('debug') == 'true'
            img_array = json.loads(img_array_json)
            evaluated = evaluate(img_array)
            ascii_output_from_last_layer = convert_last_output_to_ascii(evaluated['prediction_selection'], which_selection=COMBINED_PREDICTION_INDEX)
            if debug:
                localized_hands_output = Classifier.convert_img(
                    convert_hand_tracker_output_to_readable(evaluated['hands'][-1]))
                predicted_chars = Classifier.generate_plot(evaluated['chars'], 'chars')
                classified_gestures = Classifier.generate_plot(evaluated['gesture'], 'gesture')
                finally_predicted = Classifier.generate_plot(evaluated['prediction_selection'][COMBINED_PREDICTION_INDEX] if evaluated['prediction_selection'] is not None else None, 'selection')
                dictionary_prediction = convert_last_output_to_ascii(evaluated['prediction_selection'], which_selection=DICTIONARY_PREDICTION_INDEX)[0]
                return {
                    'result': ascii_output_from_last_layer[0],
                    'resized': localized_hands_output,
                    'predictedChars': predicted_chars,
                    'classifiedGestures': classified_gestures,
                    'finallyPredicted': finally_predicted,
                    'nearestPredictions': ascii_output_from_last_layer,
                    'dictionaryPrediction': dictionary_prediction
                }
            else:
                return {'result': ascii_output_from_last_layer[0]}

clear_log()
load_transformers()
app.run()
