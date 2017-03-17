import sys
from datetime import datetime

from utils import system


class Tee(object):
    __message__ = ""

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "w")

    def write(self, message):
        self.__message__ += message
        if '\n' in self.__message__:
            current_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self.__message__ = current_time + " - " + self.__message__
            self.terminal.write(self.__message__)
            self.logfile.write(self.__message__)
            self.__message__ = ""

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


def log_to_files(prefix="log"):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    system.mkdir('logs')
    sys.stdout = Tee('logs/' + prefix + '_' + current_time + '.log')
