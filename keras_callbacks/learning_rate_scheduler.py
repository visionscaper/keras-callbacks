import numpy as np

from keras.callbacks import Callback

from basics.base import Base


class LearningRateState():
    def __init__(self, lr):
        self.lr = lr


class LearningRateScheduler(Base, Callback):
    def __init__(self, session, init_iter = -1, log_period=2000, **kwargs):
        super().__init__(**kwargs)

        self._sess = session

        self._iter = init_iter

        self._log_period = log_period

        self._optimizer = None
        self._lr_state = None

    def set_model(self, model):
        super().set_model(model)

        self._optimizer = model.optimizer

    def on_batch_begin(self, batch, logs=None):
        self._iter += 1

        if self._lr_state is None:
            self._lr_state = self._calc_init_lr_state()

        if not hasattr(self._optimizer, 'lr'):
            self._log.error('Optimizer must have a "lr" attribute, unable to update the learning rate')
            return

        self._lr_state = self._update_lr_state(self._lr_state)

        self._optimizer.lr.load(self._lr_state.lr, self._sess)

        logs['learning_rate'] = np.float32(self._lr_state.lr)

        if (self._log_period >= 0) and (self._iter % self._log_period == 0):
            self._log.info('Iter. : %d, learning rate : %0.3e' % (self._iter, self._optimizer.lr.eval(self._sess)))

    def _calc_init_lr_state(self):
        """
        Calculates the initial learning rate state

        :return: learning rate state instance
        """
        self._log.error('Please implement this methiod in you child class')

    def _update_lr_state(self, lr_state):
        """
        Returns the updated learning rate state as a new state instance

        :param lr_state: current learning rate state instance
        :return: new, updated, learning rate state instance
        """
        self._log.error('Please implement this methiod in you child class')
