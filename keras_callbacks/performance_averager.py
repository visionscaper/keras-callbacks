import numpy as np
import math

from tensorflow.keras.callbacks import Callback

from basics.base import Base
import basics.base_utils as _

from keras_callbacks.utils.sliding_window import SlidingWindow


class PerformanceAverager(Base, Callback):
    """

    Calculates averages of performance values with certain name postfix

    """
    def __init__(self, window_length, metrics_name_postfix="unknown", window_values=None, **kwargs):
        super().__init__(**kwargs)

        self._log.debug("Averaging performance for %s metrics over %d iterations" % (metrics_name_postfix, window_length))

        self._window_length = window_length
        self._metrics_name_postfix = metrics_name_postfix

        self._window_values = window_values

        self._window = dict()

    def on_batch_end(self, batch, logs=None):
        if not _.is_dict(logs):
            self._log.error("No logs dict given, unable to average metrics")
            return

        average = dict()
        for metric, value in logs.items():
            if not metric.endswith(self._metrics_name_postfix):
                continue

            if metric not in self._window:
                self._log.debug("Creating sliding window for metric [%s]" % metric)
                has_init_values = _.is_dict(self._window_values) and metric in self._window_values
                init_window_values = self._window_values[metric] if has_init_values else None
                self._window[metric] = SlidingWindow(self._window_length, init_window_values)
                
            self._window[metric].slide(value)

            m = PerformanceAverager.average(self._window[metric])
            if not (m is None):
                if m == math.inf or m == math.nan:
                    self._log.warn("Mean value for %s is %s" % (metric, m))
                    self._log.warn("Window values : \n%s\n\n" % str(self._window[metric].get_window()))
                average['mean_%s' % metric] = m

        logs.update(average)

    @staticmethod
    def average(sliding_window):
        if sliding_window.is_empty():
            return None

        # Needs to be at least float32 because mean() can lead to inf for float16 input values
        window = np.array(sliding_window.get_window(), dtype='float32')

        return np.mean(window)

