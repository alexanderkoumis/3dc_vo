import os
import shutil

from keras.callbacks import Callback


class RecentModelRenamer(Callback):
    """Copies the last saved model using the simpler template name

    For example, if the last saved model is: model_yaw.2994-0.000181-0.000248.h5,
    it will copy this model to model_yaw.h5 in the same directory.

    """
    def __init__(self, model_file):
        model_file_tpl = self.get_model_tpl(model_file)
        self.model_file = os.path.abspath(model_file_tpl)
        self.model_file_base = self.model_file.split('.')[0].split('/')[-1]
        self.dirname = os.path.dirname(model_file_tpl)

    def get_model_tpl(self, model_file_full):
        model_file_full = os.path.abspath(model_file_full)
        model_file = model_file_full.split('/')[-1]
        directory = '/'.join(model_file_full.split('/')[:-1])
        base, ext = model_file.split('.')
        model_tpl = base + '.{epoch:04d}-{loss:.6f}-{val_loss:.6f}.' + ext
        return os.path.join(directory, model_tpl)

    def on_epoch_begin(self, epoch, logs):
        """This callback is executed by Keras... at the beginning of each epoch.
        It finds the specified epoch and renames it to the model_file_base.

        Args:
            epoch (int): Epoch number
            logs (?): ?
        """
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
