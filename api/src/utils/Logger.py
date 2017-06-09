import time


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

    def log_time(self, key=''):
        with open('log.txt', 'a') as f:
            f.write(key + ': ' + str(self.end_time - self.start_time))
            f.write('\n')

    def __enter__(self):
        self.start_time = time.time()
        self.file = open("log.txt", 'a')
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.file.write(self.name + ': ' + str(self.end_time - self.start_time))
        self.file.write('\n')
        self.file.close()