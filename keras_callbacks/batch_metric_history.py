import os
import time
import datetime

from shutil import copyfile

import pickle

from keras.callbacks import Callback

from basics.base import Base
import basics.base_utils as _


class BatchMetricHistory(Base, Callback):

    def __init__(self,
                 model_path="../trained-models/",
                 base_filename=time.strftime("%d-%m-%Y_%H-%M-%S"),
                 save_period=2000,
                 history=None,
                 **kwargs):
        """

        :param model_path:
        :param base_filename:
        :param save_period:
        :param history: previously saved history
                        It is assumed that the initial epoch is given to the Keras train function
        """

        super().__init__(**kwargs)

        self._model_path = model_path
        self._base_filename = base_filename
        self._save_period = save_period

        self._history = {}

        self._current_epoch = 0

        self._global_iter = -1
        self._epoch_iter = -1

        self._log.info("Save history period : %d" % self._save_period)

        self._set_init_history(history)

    def on_epoch_begin(self, epoch, logs=None):
        self._current_epoch = epoch
        self._epoch_iter = -1

    def on_batch_end(self, batch, logs=None):
        self._global_iter += 1
        self._epoch_iter += 1

        t = int(round(time.time() * 1000))
        d = str(datetime.datetime.fromtimestamp(t/1000.0))
        
        for k, v in logs.items():
            self._history.setdefault(k, []).append(v)

        self._history.setdefault("epoch", []).append(self._current_epoch)
        self._history.setdefault("global_iter", []).append(self._global_iter)
        self._history.setdefault("epoch_iter", []).append(self._epoch_iter)
        self._history.setdefault("time_stamp", []).append(t)
        self._history.setdefault("date", []).append(d)

        if self._global_iter == 0:
            return

        if (self._save_period > 0) and (self._global_iter % self._save_period == 0):
            self._save_history()

    def on_epoch_end(self, epoch, logs=None):
        self._save_history()

    def _set_init_history(self, history):
        if not _.is_dict(history):
            self._log.debug('No initial history given, starting with a clean slate ...')
            return

        self._log.debug("Using given initial history.")
        try:
            self._history = history.copy()

            self._current_epoch = self._history['epoch'][-1]
            self._global_iter = self._history['global_iter'][-1]
            self._epoch_iter = self._history['epoch_iter'][-1]

            self._log.debug("Global iter : %d" % self._global_iter)

            self._log.debug("Current epoch : %d" % self._current_epoch)
            self._log.debug("Epoch iter : %d" % self._epoch_iter)
        except Exception as e:
            _.log_exception(self._log, "Unable to set initial training history", e)

    def _save_history(self):
        try:
            fname = self._history_file_name()

            # only backup if exists
            if os.path.isfile(fname):
                backup_success = self._copy(fname, "%s.backup" % fname)
                if not backup_success:
                    self._log.error("Backing up history file unsuccessful, will not override history file with new data.")
                    return

            self._log.debug("Saving training history to [%s]" % fname)
            with open(fname, 'wb') as f:
                pickle.dump({
                    "history" : self._history
                }, f)

        except Exception as e:
            _.log_exception(self._log, "Unable to save training history", e)

    def _history_file_name(self):
        return os.path.join(self._model_path, '%s.history' % self._base_filename)

    def _copy(self, source_fname, dest_fname):
        success = True

        try:
            self._log.debug("Copying file: [%s] ==> [%s]" % (source_fname, dest_fname))

            copyfile(source_fname, dest_fname)
        except Exception as e:
            _.log_exception(self._log, "Unable to copy [%s]" % source_fname, e)
            success = False

        return success
