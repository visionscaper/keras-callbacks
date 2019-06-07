from tensorflow.keras.callbacks import Callback

from basics.base import Base


class BatchPerformanceLoggerBase(Base, Callback):
    """

    Abstract method to log batch level metrics (e.g. for a validation set)
    Please implement:

    _predict_model(batch_data)
    _calc_performance(batch_data)

    # Only if you want to inspect results during training
    _inspect_data( generated_batch_data, predicted_batch_data, batch_metrics_data)

    """
    def __init__(self,
                 batch_generator,
                 metrics_name_postfix="unknown",
                 inspect_period = 2000,
                 init_iter = -1,
                 **kwargs):
        super().__init__(**kwargs)

        self._batch_generator = batch_generator
        self._metrics_name_postfix = metrics_name_postfix

        self._inspect_period = inspect_period
        self._iter = init_iter

    def on_batch_end(self, batch, logs=None):
        self._iter += 1
        # 1) generate new batch
        # 2) predict model
        # 3) calculate performance
        # 4) log in logs object
        # 5) inspect data if inspect_period > 0 given

        # 1) generate new batch
        generated_batch_data = self._generate_batch()
        # 2) predict model
        predicted_batch_data = self._predict_model(generated_batch_data)
        # 3) calculate performance
        batch_metrics_data = self._calc_performance(generated_batch_data, predicted_batch_data)
        # 4) log in logs object
        self._log_metrics(batch_metrics_data, logs)

        # 5) log data to console if log_data_period given
        if (self._iter > 0) and \
            (self._inspect_period > 0) and \
            (self._iter % self._inspect_period == 0):
            self._inspect_data(generated_batch_data, predicted_batch_data, batch_metrics_data)

    def _generate_batch(self):
        return next(self._batch_generator)

    def _predict_model(self, batch_data):
        self._log.error("Please implement this method")

    def _calc_performance(self, generated_batch_data, predicted_batch_data):
        self._log.error("Please implement this method")

    # TODO : FS : Factor out in to a inspection callback
    def _inspect_data(self, generated_batch_data, predicted_batch_data, batch_metrics_data):
        self._log.error("Please implement this method")

    def _log_metrics(self, batch_metrics_data, logs):
        for metric, value in batch_metrics_data.items():
            logs[metric] = value

