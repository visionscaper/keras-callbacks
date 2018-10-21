from keras.callbacks import Callback

from basics.base import Base


class PerformanceLoggerBase(Base, Callback):
    """

    Abstract method to log epoch level metrics (e.g. for a validation set)
    Please implement:

    _calc_metrics()
    _inspect() # Only if you want to inspect results during training

    """
    def __init__(self,
                 generator,
                 metrics_name_postfix="unknown",
                 inspect_period=-1,
                 **kwargs):
        super().__init__(**kwargs)

        self._generator = generator
        self._metrics_name_postfix = metrics_name_postfix

        self._inspect_period = inspect_period

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        metrics = self._calc_metrics()
        self._log_metrics(metrics, logs)

        if (epoch > 0) and \
            (self._inspect_period > 0) and \
            (epoch % self._inspect_period == 0):
            self._inspect()

    def _calc_metrics(self):
        self._log.error("Please implement this method")

    # TODO : FS : Factor out in to a inspection callback
    def _inspect(self):
        self._log.error("Please implement this method")

    def _log_metrics(self, metrics, logs):
        for metric, value in metrics.items():
            logs[metric] = value
