import numpy as np
from tensorflow.keras import backend as K
from keras_callbacks.batch_performance_logger_base import BatchPerformanceLoggerBase
from tensorflow.keras.losses import categorical_crossentropy, sparse_categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy, sparse_categorical_accuracy

# From 1.14.0-rc0 : weighted_masked_objective not available anymore, was replaced by call_metric_function
from tensorflow.python.keras.engine.training_utils import call_metric_function

import basics.base_utils as _


class DefaultBatchPerformanceLogger(BatchPerformanceLoggerBase):

    def __init__(self,
                 session,
                 max_seq_length,
                 num_symbols,
                 batch_generator,
                 metrics_name_postfix="unknown",
                 inspector=None,
                 num_samples_to_inspect=5,
                 one_hot_encoding=True,
                 **kwargs):
        super().__init__(batch_generator, metrics_name_postfix, **kwargs)

        self._sess = session

        self._one_hot_encoding = one_hot_encoding

        if self._one_hot_encoding:
            self._target_batch_placeholder = K.placeholder((None, max_seq_length + 2, num_symbols))
        else:
            self._target_batch_placeholder = K.placeholder((None, max_seq_length + 2, 1))

        self._output_batch_placeholder = K.placeholder((None, max_seq_length+2, num_symbols))
        self._sample_weights_batch_placeholder = K.placeholder((None, max_seq_length+2))

        loss = categorical_crossentropy if self._one_hot_encoding else sparse_categorical_crossentropy
        self._weighted_categorical_cross_entropy_op = call_metric_function(loss,
                                                                           self._target_batch_placeholder,
                                                                           self._output_batch_placeholder,
                                                                           self._sample_weights_batch_placeholder)

        metric = categorical_accuracy if self._one_hot_encoding else sparse_categorical_accuracy
        self._weighted_categorical_accuracy_op = call_metric_function(metric,
                                                                      self._target_batch_placeholder,
                                                                      self._output_batch_placeholder,
                                                                      self._sample_weights_batch_placeholder)

        self._inspector = inspector
        self._num_samples_to_inspect = num_samples_to_inspect

    def _predict_model(self, batch_data):
        inputs_batch = batch_data[0]
        return self.model.predict_on_batch(inputs_batch)

    def _calc_performance(self, generated_batch_data, predicted_batch_data):
        target_batch = generated_batch_data[1]
        sample_weights_batch = generated_batch_data[2]

        output_batch = predicted_batch_data

        with self._sess.as_default():
            cross_entropy = self._weighted_categorical_cross_entropy_op.eval(
                feed_dict={
                    self._target_batch_placeholder: target_batch,
                    self._output_batch_placeholder: output_batch,
                    self._sample_weights_batch_placeholder: sample_weights_batch
                })

            accuracy = self._weighted_categorical_accuracy_op.eval(
                feed_dict={
                    self._target_batch_placeholder: target_batch,
                    self._output_batch_placeholder: output_batch,
                    self._sample_weights_batch_placeholder: sample_weights_batch
                })

        pf = self._metrics_name_postfix

        return {
            ('batch_cross_entropy_%s' % pf): cross_entropy,
            ('batch_perplexity_%s' % pf): np.exp(cross_entropy),
            ('batch_accuracy_%s' % pf): accuracy
        }

    def _inspect_data(self, generated_batch_data, predicted_batch_data, batch_metrics_data):
        if not (hasattr(self._inspector, "inspect") and _.is_callable(self._inspector.inspect)):
            self._log.error("No valid inspector given, unable to inspect random samples")
            return

        self._log.info("Current batch performance metrics : \n%s")
        for metric, value in batch_metrics_data.items():
            self._log.info("%s : %f" % (metric, value))

        self._inspector.inspect(generated_batch_data, predicted_batch_data, self._num_samples_to_inspect)
