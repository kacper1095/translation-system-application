import os

os.environ['THEANO_FLAGS'] = 'floatX=float32,mode=FAST_RUN'

import cv2
import tqdm
import json
import datetime
import skvideo.io
from sphinx.versioning import levenshtein_distance

from pipeline import load_transformers, convert_last_output_to_ascii

from src.common import *
from src.utils.Transformers.eval_transformer_pipeline import eval_transformer_pipeline

TIMESTAMP = datetime.datetime.now().strftime('%H_%M_%d_%m_%y')
SAVE_OUTPUT_PATH = os.path.join(REPORTS_FOLDER, TIMESTAMP, 'outputs')


class Tester(object):
    def __init__(self, transformer_index_to_save_output=-1):
        transformers = load_transformers()
        self.transformers = transformers
        self.transformer_index_to_save_output = transformer_index_to_save_output
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
        capture = skvideo.io.vread(os.path.abspath(video_path))
        for frame in capture:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32)
            yield np.array(frame)

    def test(self):
        print('Testing')
        report_path = os.path.join(REPORTS_FOLDER, TIMESTAMP)
        ensure_dir(report_path)
        distances = []
        report_file_content = ['distance,true_label,predicted_label']
        video_file_paths, labels = self.load_data(os.path.join(TESTING_VIDEO_FOLDER, 'labels.txt'))
        for i, (file_name, label) in enumerate(tqdm.tqdm(zip(video_file_paths, labels))):
            # predicted = self.text_transformer.transform(predicted)
            predicted = self.__get_letters_from_video_path(os.path.join(TESTING_VIDEO_FOLDER, file_name))

            # predicted_bgr = self.__get_letters_from_video_path_in_bgr(os.path.join(TESTING_VIDEO_FOLDER, file_name))
            #
            # orig_lev_distance = levenshtein_distance(label, predicted)
            # bgr_lev_distance = levenshtein_distance(label, predicted_bgr)
            #
            # if bgr_lev_distance < orig_lev_distance:
            #     predicted = predicted_bgr
            #
            # distances.append(levenshtein_distance(label, predicted))
            # report_file_content.append('{},{},{}'.format(distances[-1], label, predicted))
            # print('\nPredicted: {}\nExpected: {}\nDistance: {}'.format(predicted, label, distances[-1]))

        with open(os.path.join(report_path, 'report.csv'), 'w') as f:
            f.write(''.join(report_file_content))

        with open(os.path.join(report_path, 'means_stats.txt'), 'w') as f:
            f.write('mean: {}\n'.format(np.mean(distances)))
            f.write('std: {}'.format(np.std(distances)))
        return np.mean(distances)

    def __get_letters_from_video_path(self, video_path):
        video = self.load_video(video_path)
        label = os.path.basename(video_path[:-4])
        predicted = ''
        for i, chunk in enumerate(video):
            evaluation = eval_transformer_pipeline(np.array([chunk]), self.transformers)
            if 0 < self.transformer_index_to_save_output < len(self.transformers):
                outputs = self.transformers[self.transformer_index_to_save_output].output
                folder_label = str(i) + '_' + label
                folder_output = os.path.join(SAVE_OUTPUT_PATH, folder_label)
                if not os.path.exists(folder_output):
                    os.makedirs(folder_output)
                if outputs is not None:
                    for output in outputs:
                        nb = len(os.listdir(folder_output))
                        cv2.imwrite(os.path.join(folder_output, str(nb) + '.png'), output)
            if evaluation is not None:
                predicted += convert_last_output_to_ascii(evaluation, number_of_predictions=1)[0]
        del video
        return predicted

    def __get_letters_from_video_path_in_bgr(self, video_path):
        video = self.load_video(video_path)
        predicted = ''
        for chunk in video:
            chunk = cv2.cvtColor(chunk, cv2.COLOR_BGR2RGB)
            evaluation = eval_transformer_pipeline(np.array([chunk]), self.transformers)

            if evaluation is not None:
                predicted += convert_last_output_to_ascii(evaluation, number_of_predictions=1)[0]
        del video
        return predicted


class FrameExtractor(object):
    def __init__(self):
        pass

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
        capture = skvideo.io.vread(os.path.abspath(video_path))
        for frame in capture:
            frame = frame.astype(np.float32)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield np.array(frame)

    def extract(self):
        print('Extracting')
        video_file_paths, labels = self.load_data(os.path.join(TESTING_VIDEO_FOLDER, 'labels.txt'))
        for i, (file_name, label) in enumerate(tqdm.tqdm(zip(video_file_paths, labels))):
            video_frames = self.load_video(os.path.join(TESTING_VIDEO_FOLDER, file_name))
            video_name_without_ext = file_name[:-4]
            frame_folder = os.path.join(FRAME_SAVE_PATH, video_name_without_ext + ' - ' + label)
            if not os.path.exists(frame_folder):
                os.makedirs(frame_folder)
            for i, frame in enumerate(video_frames):
                if i % 4 == 0:
                    cv2.imwrite(os.path.join(frame_folder, str(i) + '.jpg'), frame)


if __name__ == '__main__':
    Tester(4).test()
    # FrameExtractor().extract()
    # print(levenshtein_distance('cos', 'cod'))
