import os
import shutil

from keras.callbacks import Callback


class RecentModelRenamer(Callback):

    def __init__(self, model_file):
        self.model_file = os.path.abspath(model_file)
        self.model_file_base = self.model_file.split('.')[0].split('/')[-1]
        self.dirname = os.path.dirname(model_file)

    def on_epoch_begin(self, epoch, logs):
        epoch_search = epoch - 1
        for file in os.listdir(self.dirname):
            name_split = file.split('.')
            if len(name_split) != 5:
                continue
            file_base = name_split[0]
            epoch_curr = int(name_split[1].split('-')[0])
            if file_base == self.model_file_base and epoch_curr == epoch_search:
                full_path = os.path.join(self.dirname, file)
                shutil.copy(full_path, self.model_file)
