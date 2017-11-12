import time
import datetime
import cv2
import os


class Logger(object):
    def __init__(self, name=''):
        self.start_time = 0
        self.end_time = 0
        self.name = name
        self.file = None

    @staticmethod
    def log(key='', message=''):
        with open('log.txt', 'a') as f:
            f.write(key + ': ' + str(message))
            f.write('\n')

    @staticmethod
    def log_img(img):
        img = img[:]
        if img.max() < 127:
            img *= 255
        cv2.imwrite(os.path.join('tmp', Logger.get_time_stamp() + '.png'), img)

    def log_time(self, key=''):
        with open('log.txt', 'a') as f:
            f.write(key + ': ' + str(self.end_time - self.start_time))
            f.write('\n')

    @staticmethod
    def get_time_stamp():
        timestamp = datetime.datetime.now().strftime('%H_%M_%S_%m_%d')
        return timestamp

    def __enter__(self):
        self.start_time = time.time()
        self.file = open("log.txt", 'a')
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.file.write(self.name + ': ' + str(self.end_time - self.start_time))
        self.file.write('\n')
        self.file.close()