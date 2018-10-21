import tensorflow as tf
from keras.callbacks import TensorBoard as KerasTensorBoard

from basics.base import Base

import basics.validation_utils as _u


class TensorBoard(Base, KerasTensorBoard):
    def __init__(self, metric_mapping=None, init_iter=-1, **kwargs):
        super().__init__(**kwargs)
        self._metric_mapping = metric_mapping
        self._iter = init_iter

    def on_batch_end(self, batch, logs=None):
        self._iter += 1
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
            self.writer.add_summary(summary, self._iter)
        self.writer.flush()

        super().on_batch_end(batch, logs)