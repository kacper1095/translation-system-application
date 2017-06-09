class Logger(object):

    @staticmethod
    def log(key='', message=''):
        with open('log.txt', 'a') as f:
            f.write(key + ': ' + message)
            f.write('\n')