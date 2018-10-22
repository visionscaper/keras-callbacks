import tensorflow as tf
from keras.callbacks import TensorBoard as KerasTensorBoard

from basics.base import Base

import basics.validation_utils as _u


class TensorBoard(Base, KerasTensorBoard):
    def __init__(self, metric_mapping=None, init_iter=-1, batch_level=True, **kwargs):
        super().__init__(**kwargs)
        self._metric_mapping = metric_mapping
        self._iter = init_iter

        self._batch_level = batch_level

    def on_batch_end(self, batch, logs=None):
        if self._batch_level:
            self._iter += 1
            self._write_mapped_logs(self._iter, logs)

        return super().on_batch_end(batch, logs)

    def on_epoch_end(self, epoch, logs=None):
        if not self._batch_level:
            self._write_mapped_logs(epoch, logs)

            # Set logs to None such that no new data is written
            logs = None

        return super().on_epoch_end(epoch, logs)

    def _write_mapped_logs(self, iter, logs):
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue

            log_metric_name = None
            if _u.is_dict(self._metric_mapping):
                if name in self._metric_mapping:
                    log_metric_name = self._metric_mapping[name]
            else:
                log_metric_name = name

            if log_metric_name is None:
                continue

            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()

            summary_value.tag = log_metric_name
            self.writer.add_summary(summary, iter)
        self.writer.flush()


