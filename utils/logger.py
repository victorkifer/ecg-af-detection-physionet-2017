import sys
from datetime import datetime

from utils import system


class Tee(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()


def log_to_files():
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    system.mkdir('logs')
    sys.stdout = Tee('logs/log_' + current_time + '.log')
