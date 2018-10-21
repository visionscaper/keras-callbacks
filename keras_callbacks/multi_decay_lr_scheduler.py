import numpy as np

from keras_callbacks.learning_rate_scheduler import LearningRateState, LearningRateScheduler

_DEFAULT_LR_DECAY_SETUP = [[0, 0.0009], [10000, 0.00009], [110000, 0.0000075], [1211000, 0]]


class LRDecayRegime(LearningRateState):
    def __init__(self, lr=None, init_lr=None, start_step=None):
        super().__init__(lr)

        self.init_lr = init_lr
        self.start_step = start_step


class MultiDecayLRScheduler(LearningRateScheduler):
    def __init__(self, session,
                 init_lr=0.001,
                 lr_decay_setup=None,
                 min_lr=1e-6,
                 init_iter=-1,
                 log_period=2000,
                 **kwargs):
        super().__init__(session, init_iter, log_period, **kwargs)

        self._init_lr = init_lr
        self._lr_decay_setup = lr_decay_setup or _DEFAULT_LR_DECAY_SETUP
        self._min_lr = min_lr

    def _calc_init_lr_state(self):
        _lr = self._init_lr
        _init_lr = self._init_lr
        _start_step = 0

        def __log(regime_idx, at_end, new_start_step, new_init_lr, new_lr):
            self._log.debug(
                "Calc. init. LR-regime: "
                "Regime : %d : "
                "END of regime : %r : "
                "start_step : %d : "
                "lr : %0.3e : "
                "init_lr : %0.3e" % (regime_idx, at_end, new_start_step, new_lr, new_init_lr))

        ready = False
        for i in range(1, len(self._lr_decay_setup)):
            prev_entry = self._lr_decay_setup[i - 1]
            entry = self._lr_decay_setup[i]

            start_step = entry[0]
            prev_start_step = prev_entry[0]
            prev_decay = prev_entry[1]

            _start_step = prev_start_step

            if prev_start_step <= self._iter < start_step:
                _lr = _init_lr * (1. / (1. + prev_decay * (self._iter - prev_start_step)))
                ready = True
            else:
                _init_lr = _init_lr * (1. / (1. + prev_decay * ((start_step - 1) - prev_start_step)))
                _lr = _init_lr

            __log(i - 1, not ready, _start_step, _init_lr, _lr)

            if ready:
                break

        if not ready:
            # We are in last regime
            entry = self._lr_decay_setup[-1]

            start_step = entry[0]
            decay = entry[1]

            _start_step = start_step
            _lr = _init_lr * (1. / (1. + decay * (self._iter - start_step)))

            __log(len(self._lr_decay_setup) - 1, False, _start_step, _init_lr, _lr)

        return LRDecayRegime(_lr, _init_lr, _start_step)

    def _get_decay_regime(self):
        decay = None
        start_step = None
        regime_index = None

        for index, entry in enumerate(self._lr_decay_setup):
            if self._iter >= entry[0]:
                start_step = entry[0]
                decay = entry[1]
                regime_index = index

        return decay, start_step, regime_index

    def _update_lr_state(self, lr_state):
        decay, start_step, regime_idx = self._get_decay_regime()
        # print("Regime : %d : decay = %f : start_step = %d" % (regime_idx, decay, start_step))

        init_lr = lr_state.init_lr
        if start_step > lr_state.start_step:
            init_lr = lr_state.lr

        lr = init_lr * (1. / (1. + decay * (self._iter - start_step)))

        lr = lr if lr > self._min_lr else self._min_lr

        return LRDecayRegime(lr, init_lr, start_step)
