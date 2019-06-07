import numpy as np

from tensorflow.keras.callbacks import Callback

from basics.base import Base


class TrainingPerplexityLogger(Base, Callback):

    def on_batch_end(self, batch, logs=None):

        if not 'loss' in logs:
            return

        loss = logs['loss']

        logs['batch_perplexity_training'] = np.exp(loss)
