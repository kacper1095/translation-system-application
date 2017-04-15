from flask import Flask
from flask_restful import Resource, Api, request
from flask_cors import CORS
from pipeline import evaluate
import json

app = Flask(__name__)
CORS(app, origins='http://localhost:8000')
api = Api(app)


@api.resource('/')
class Classifier(Resource):
    def get(self):
        pass

    def post(self):
        img_array_json = request.form.get('img_array')
        img_array = json.loads(img_array_json)
        return {'result': str(evaluate(img_array)[0].shape)}


if __name__ == '__main__':
    app.run()
