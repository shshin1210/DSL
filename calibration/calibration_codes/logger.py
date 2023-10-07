import constants


class LogSaver:
    def __init__(self, fn):
        # self.stream = stream
        self.f = open(fn, 'w')
        print('log file: %s'%fn)

    def __del__(self):
        self.f.close()

    def write(self, str):
        # self.stream.write(str)
        # self.stream.flush()
        self.f.write(str + '\n')
        self.f.flush()
        print(str)


log_saver = LogSaver(fn=constants.LOG_FN)
