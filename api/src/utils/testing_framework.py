import os

os.environ['THEANO_FLAGS'] = 'floatX=float32,mode=FAST_RUN'

import cv2
import tqdm
import json
import datetime
from sphinx.versioning import levenshtein_distance

from pipeline import load_transformers, convert_last_output_to_ascii

from src.common import *
from src.utils.Transformers.eval_transformer_pipeline import eval_transformer_pipeline

TIMESTAMP = datetime.datetime.now().strftime('%H_%M_%d_%m_%y')


class Tester(object):
    def __init__(self):
        transformers = load_transformers()
        self.transformers = transformers
        self.text_transformer = None

    def load_data(self, label_path):
        with open(label_path) as f:
            label_dict = json.load(f)
        labels = []
        video_paths = []
        for data in label_dict['data']:
            labels.append(data['label'])
            video_paths.append(data['filename'])
        return video_paths, labels

    def load_video(self, video_path):
        print(video_path)
        transformers = load_transformers()
        self.transformers = transformers
        capture = cv2.VideoCapture(os.path.abspath(video_path))
        while capture.isOpened():
            ret, img = capture.read()
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if not ret:
                break
            yield np.array(img)
        capture.release()

    def test(self):
        print('Testing')
        report_path = os.path.join(REPORTS_FOLDER, TIMESTAMP)
        ensure_dir(report_path)
        distances = []
        report_file_content = ['distance,true_label,predicted_label']
        video_file_paths, labels = self.load_data(os.path.join(TESTING_VIDEO_FOLDER, 'labels.txt'))
        for i, (file_name, label) in enumerate(tqdm.tqdm(zip(video_file_paths, labels))):
            video = self.load_video(os.path.join(TESTING_VIDEO_FOLDER, file_name))
            predicted = ''
            import pdb
            pdb.set_trace()
            for chunk in video:
                if chunk is None:
                    break
                evaluation = eval_transformer_pipeline(np.array([chunk]), self.transformers)
                if evaluation is not None:
                    predicted += convert_last_output_to_ascii(evaluation, number_of_predictions=1)[0]
            del video
            # predicted = self.text_transformer.transform(predicted)
            distances.append(levenshtein_distance(label, predicted))
            report_file_content.append('{},{},{}'.format(distances[-1], label, predicted))
            print('Predicted: {}\nExpected: {}\nDistance: {}'.format(predicted, label, distances[-1]))

        with open(os.path.join(report_path, 'report.csv'), 'w') as f:
            f.write(''.join(report_file_content))

        with open(os.path.join(report_path, 'means_stats.txt'), 'w') as f:
            f.write('mean: {}\n'.format(np.mean(distances)))
            f.write('std: {}'.format(np.std(distances)))
        return np.mean(distances)


if __name__ == '__main__':
    Tester().test()
    # print(levenshtein_distance('cos', 'cod'))
