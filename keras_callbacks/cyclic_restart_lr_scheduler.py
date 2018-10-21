import math

from keras_callbacks.learning_rate_scheduler import LearningRateState, LearningRateScheduler


class LRCycleState(LearningRateState):
    def __init__(self, lr, in_cycle_iter, max_lr, min_lr, period, cycle):
        super().__init__(lr)

        self.in_cycle_iter = in_cycle_iter
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.period = period
        self.cycle = cycle

    def log_with(self, logger):
        logger.debug("LRCycleState : ")
        logger.debug("  in_cycle_iter : %d" % self.in_cycle_iter)
        logger.debug("  max_lr : %f" % self.max_lr)
        logger.debug("  min_lr : %f" % self.min_lr)
        logger.debug("  period : %d" % self.period)
        logger.debug("  cycle : %d" % self.cycle)


class CyclicRestartLRScheduler(LearningRateScheduler):
    def __init__(self,
                 session,
                 init_iter=-1,
                 max_lr=0.001,
                 min_lr=0.000001,
                 restart_period=50000,
                 elongation_factor=1.41421,
                 decay_factor=0.125,
                 **kwargs):
        super().__init__(session, init_iter, **kwargs)

        self._max_lr = max_lr
        self._min_lr = min_lr
        self._restart_period = restart_period
        self._elongation_factor = elongation_factor
        self._decay_factor = decay_factor

    def _calc_init_lr_state(self):
        current_cycle, in_cycle_iter, current_restart_period = self._calc_cycle_state(self._iter)

        current_max_lr = self._calc_max_lr(self._max_lr, self._min_lr, current_cycle)

        lr = CyclicRestartLRScheduler.cosine_annealing_lr(in_cycle_iter,
                                                       current_restart_period,
                                                       self._min_lr,
                                                       current_max_lr)

        state = LRCycleState(lr,
                             in_cycle_iter,
                             current_max_lr,
                             self._min_lr,
                             current_restart_period,
                             current_cycle)

        self._log.debug("Initial LR state calculated")
        self._log.debug("Global iter : %d" % self._iter)
        state.log_with(self._log)

        return state

    def _calc_cycle_state(self, iter):
        iter_count = 0
        cycle = 0
        restart_period = self._restart_period
        while True:
            _iter_count = iter_count + restart_period

            # if they are equal, iter is at the first iteration of the new cycle
            if _iter_count > iter:
                break

            cycle += 1

            iter_count = _iter_count

            restart_period = int(restart_period * self._elongation_factor)

        in_cycle_iter = iter - iter_count

        return cycle, in_cycle_iter, restart_period

    def _update_lr_state(self, lr_state):
        new_period = lr_state.period
        new_max_lr = lr_state.max_lr
        new_cycle = lr_state.cycle

        restart = False
        if lr_state.in_cycle_iter >= lr_state.period:
            restart = True

            new_in_cycle_iter = 0
            new_period = int(new_period * self._elongation_factor)
            new_cycle += 1

            new_max_lr = self._calc_max_lr(self._max_lr, self._min_lr, new_cycle)
        else:
            # update iteration afterwards, because it already starts at 0 at the first call to _update_lr_state
            new_in_cycle_iter = lr_state.in_cycle_iter + 1

        new_lr = CyclicRestartLRScheduler.cosine_annealing_lr(new_in_cycle_iter,
                                                              new_period,
                                                              lr_state.min_lr,
                                                              new_max_lr)

        state = LRCycleState(new_lr,
                             new_in_cycle_iter,
                             new_max_lr,
                             lr_state.min_lr,
                             new_period,
                             new_cycle)

        if restart:
            self._log.debug("LR restart occurred : ")
            self._log.debug("Global iter : %d" % self._iter)
            state.log_with(self._log)

        return state

    def _calc_max_lr(self, max_lr, min_lr, cycle):
        max_lr = max_lr * 1. / (1. + self._decay_factor * cycle)
        max_lr = max_lr if max_lr >= min_lr else min_lr

        return max_lr

    @staticmethod
    def cosine_annealing_lr(iter, period_length, min_lr, max_lr):
        cos_factor = 0.5*math.cos(iter*math.pi/period_length)+0.5
        lr = (max_lr - min_lr) * cos_factor + min_lr

        return lr

    @staticmethod
    def calc_schedule_params(epoch_length,
                             batch_size,
                             init_period=None,
                             max_lr=0.0005,
                             target_max_lr=None,
                             target_num_epochs=3):
        """

        Returns elongation_factor, decay_factor such that :
         * After target_num_epochs the restart period is epoch_length
         * After target_num_epochs the max_lr is target_max_lr

        The method also returns the number of restart periods reuired to het to the given target values

        if init_period is not given, a proposed value is calculated based on the batch_size

        :param batch_size:
        :param epoch_length:
        :param init_period:
        :param target_max_lr:
        :param target_num_epochs:
        :return: (init_period, elongation_factor, decay_factor, num_restart_periods)
        """

        if init_period is None:
            init_period = int(2000 * (32/batch_size))+1

        if target_max_lr is None:
            target_max_lr = max_lr/25

        r = float(epoch_length)/float(init_period)
        num_restart_periods = math.log(r)/math.log(1 - (1-r)/(target_num_epochs*r))

        num_restart_periods = int(num_restart_periods) + 1

        elongation_factor = math.pow(r, (1/num_restart_periods))

        decay_factor = ((max_lr/target_max_lr) - 1.)/float(num_restart_periods)

        return init_period, elongation_factor, decay_factor, num_restart_periods
