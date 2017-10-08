import glob
import os

import cv2
import numpy as np
import tqdm
from sphinx.versioning import levenshtein_distance

from src.common import *
from src.utils.Transformers.eval_transformer_pipeline import eval_transformer_pipeline


class Tester:

    def __init__(self, transformers):
        self.transformers = transformers
        self.frame_skip = 2
        self.text_transformer = None
        self.nb_of_frames = 5

    @staticmethod
    def load_data(path):
        capture = cv2.VideoCapture(os.path.abspath(path))
        frame_list = []
        while capture.isOpened():
            ret, img = capture.read()
            if not ret:
                break
            frame_list.append(img)
        capture.release()
        tensor = np.array(frame_list)
        tensor = np.expand_dims(tensor, 1)
        file_name = os.path.basename(path)[:-len(TESTING_VIDEO_FOLDER)]
        return tensor, file_name

    def video_to_chunks(self, video):
        for frame_nb in range(0, len(video) - (self.frame_skip * self.frame_skip), self.frame_skip):
            frames = video[frame_nb:frame_nb + (self.frame_skip * self.frame_skip):self.frame_skip]
            yield frames

    def test(self):
        distances = []
        for file in tqdm.tqdm(glob.glob(os.path.join(TESTING_VIDEO_FOLDER, '*'))):
            video, text = Tester.load_data(file)
            predicted = ''
            for chunk in self.video_to_chunks(video):
                predicted += eval_transformer_pipeline(np.array([chunk]), self.transformers)[-1]
            predicted = self.text_transformer.transform(predicted)
            distances.append(levenshtein_distance(text, predicted))
        return np.mean(distances)


if __name__ == '__main__':
    print(levenshtein_distance('cos', 'cod'))
