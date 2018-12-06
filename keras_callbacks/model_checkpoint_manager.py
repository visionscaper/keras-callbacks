import os
import time
import pickle
import tempfile
from shutil import copyfile
import json

from keras.callbacks import Callback

from basics.base import Base
import basics.base_utils as _


class ModelCheckpointManager(Base, Callback):

    def __init__(self,
                 model,
                 model_path="../trained-models/",
                 base_filename=time.strftime("%d-%m-%Y_%H-%M-%S"),
                 metric_to_monitor="mean_batch_perplexity_validation",
                 metric_opt_mode='min',
                 metric_monitor_period=2000,
                 early_good_model_delta=0.5,  # percentage
                 create_checkpoint_every=2000,
                 archive_last_checkpoint_every=20000,
                 save_best_per_epoch=True,
                 simulation_mode=False,
                 debug_mode=False,
                 checkpoint_state=None,
                 **kwargs):
        """

        :param model: model to save checkpoints for
        :param model_path:
        :param base_filename:
        :param metric_to_monitor:
        :param metric_opt_mode: 'max', 'min'
        :param metric_monitor_period:
        :param early_good_model_delta: earliest model that is within this percentage of current best model
        :param create_checkpoint_every: period in number of batch-wise training iterations before saving next checkpoint
                                        (will be overridden after each period)
        :param archive_last_checkpoint_every: period in number of batch-wise training iterations before the
                                              last available checkpoint is archived.
                                              Period must be multiple of create_checkpoint_every period
        :param save_best_per_epoch:
        :param simulation_mode: Set to true to disable saving weights and copying or removing of files
                                Only logs are generated to simulate the management function.
        :param debug_mode: Set to true to log all actions
        :param checkpoint_state Dict with saved checkpoint state to continue tracking latest and earliest good model
        """

        super().__init__(**kwargs)

        self._model = model
        self._model_path = model_path
        self._base_filename = base_filename
        self._metric_to_monitor = metric_to_monitor
        self._metric_opt_mode = metric_opt_mode
        self._metric_monitor_period = metric_monitor_period
        self._early_good_model_delta = early_good_model_delta
        self._create_checkpoint_every = create_checkpoint_every
        self._archive_last_checkpoint_every = archive_last_checkpoint_every
        self._save_best_per_epoch = save_best_per_epoch
        self._simulation_mode = simulation_mode
        self._debug_mode = debug_mode

        self._model_quality = dict()
        self._model_iter = dict()
        self._best_model = None
        self._best_model_quality = float('Inf') if self._metric_opt_mode == 'min' else -float('Inf')

        self._earliest_good_model = None
        self._earliest_good_model_iter = None

        self._iter = -1

        self._log.info("Metric monitor period : %d" % self._metric_monitor_period)
        self._log.info("Archive last checkpoint every %d iterations" % self._archive_last_checkpoint_every)

        self._setup_temp_model_path()

        self._check_settings()

        self._set_state(checkpoint_state)

    def on_batch_end(self, batch, logs=None):
        self._iter += 1

        if self._iter == 0:
            return

        checkpoint_fname = None
        if (self._create_checkpoint_every > 0) and (self._iter % self._create_checkpoint_every == 0):
            checkpoint_fname = self._save_checkpoint()

        if (self._archive_last_checkpoint_every > 0) and (self._iter % self._archive_last_checkpoint_every == 0):
            self._copy(self.latest_model_file_name(), self.current_model_file_name())

        if (self._metric_monitor_period > 0) and (self._iter % self._metric_monitor_period != 0):
            return

        if self._metric_to_monitor not in logs:
            self._log.info("Metric to monitor [%s] not found, unable to create model checkpoints" % self._metric_to_monitor)
            return

        model_quality = logs[self._metric_to_monitor].item()

        model_improved = ((self._metric_opt_mode == 'min') and (model_quality < self._best_model_quality)) or \
                         ((self._metric_opt_mode == 'max') and (model_quality > self._best_model_quality))

        if model_improved:
            if self._simulation_mode or self._debug_mode:
                self._log.debug("Iter : %s : Model improved : %s : %3e " % (self._iter,
                                                                            self._metric_to_monitor,
                                                                            model_quality))

            model_fname = self._save_current_model_as_temp(checkpoint_fname)
            if model_fname:
                self._best_model = model_fname
                self._best_model_quality = model_quality
                self._model_quality[model_fname] = model_quality
                self._model_iter[model_fname] = self._iter

                self._copy(model_fname, self.latest_best_model_file_name())

                earliest_good_model_new, earliest_good_model_iter_new = self._prune_models()

                if earliest_good_model_new:
                    if earliest_good_model_new != self._earliest_good_model:
                        if self._simulation_mode or self._debug_mode:
                            self._log.debug("New earliest good model : %s " % earliest_good_model_new)

                        self._earliest_good_model = earliest_good_model_new
                        self._earliest_good_model_iter = earliest_good_model_iter_new

                        self._copy(earliest_good_model_new, self.earliest_good_model_file_name())
                    else:
                        if self._simulation_mode or self._debug_mode:
                            self._log.debug("Current earliest good model remains earliest good, noting to do")
                else:
                    if self._simulation_mode or self._debug_mode:
                        self._log.debug("No earliest good model available")

                self._save_checkpoint_state()
            else:
                self._log.error("Unable to save improved model to temp. file, "
                                "model not incorporated in analysis and no checkpoint created.")

    def on_epoch_end(self, epoch, logs=None):
        self._save_checkpoint()

    def reset(self):
        if self._simulation_mode or self._debug_mode:
            self._log.debug("Resetting ...")

        for fname in self._model_quality.keys():
            self._remove_model(fname)

        self._model_quality = dict()
        self._model_iter = dict()
        self._best_model = None
        self._best_model_quality = float('Inf') if self._metric_opt_mode == 'min' else -float('Inf')

        self._earliest_good_model = None
        self._earliest_good_model_iter = None

    def checkpoint_state_file_name(self):
        return os.path.join(self._model_path, '%s-checkpoint.state' % self._base_filename)

    def current_model_file_name(self):
        return os.path.join(self._model_path, '%s-%s.model' % (self._base_filename, str(self._iter)))

    def latest_model_file_name(self):
        return os.path.join(self._model_path, '%s-latest.model' % self._base_filename)

    def latest_best_model_file_name(self):
        return os.path.join(self._model_path, '%s-best-latest.model' % self._base_filename)

    def earliest_good_model_file_name(self):
        return os.path.join(self._model_path, '%s-earliest-good.model' % self._base_filename)

    def _setup_temp_model_path(self):
        try:
            self._temp_model_path = os.path.join(self._model_path, 'temp-models')
            if not os.path.exists(self._temp_model_path):
                os.makedirs(self._temp_model_path)
        except Exception as e:
            _.log_exception(self._log, "Unable to use temp. model path: %s" % self._temp_model_path, e)

    def _set_state(self, checkpoint_state):
        if not _.is_dict(checkpoint_state):
            if self._simulation_mode or self._debug_mode:
                self._log.debug("No initial checkpoint state given, starting with clean slate ...")
            return

        if self._simulation_mode or self._debug_mode:
            self._log.debug("Using given initial checkpoint state: \n\n%s" % json.dumps(checkpoint_state, indent=4))

        try:
            self._model_quality = checkpoint_state['model_quality']
            self._model_iter = checkpoint_state['model_iter']
            self._best_model = checkpoint_state['best_model']
            self._best_model_quality = checkpoint_state['best_model_quality']
            self._earliest_good_model = checkpoint_state['earliest_good_model']
            self._earliest_good_model_iter = checkpoint_state['earliest_good_model_iter']
            self._iter = checkpoint_state['iter']
        except Exception as e:
            _.log_exception(self._log, "Unable to set checkpoint state", e)

    def _prune_models(self):
        if self._simulation_mode or self._debug_mode:
            self._log.debug("Pruning models ... ")

        if self._metric_opt_mode == 'min':
            good_model_boundary = self._best_model_quality*(1 + (self._early_good_model_delta/100))
        elif self._metric_opt_mode == 'max':
            good_model_boundary = self._best_model_quality*(1 - (self._early_good_model_delta/100))
        else:
            self._log.error("Unknown metric optimization mode : [%s]. Unable to prune models" % self._metric_opt_mode)
            return None

        if self._simulation_mode or self._debug_mode:
            self._log.debug("Good model boundary : %3e " % good_model_boundary)

        earliest_good_model_new = None
        earliest_good_model_iter_new = None

        model_quality_new = self._model_quality.copy()
        model_iter_new = self._model_iter.copy()

        def __pop(model_name):
            self._log.debug("Removing model : %s" % model_name)

            model_quality_new.pop(model_name)
            model_iter_new.pop(model_name)
            self._remove_model(model_name)

        for fname, q in self._model_quality.items():
            if fname == self._best_model:
                # always keep the best model
                continue

            if self._simulation_mode or self._debug_mode:
                self._log.debug("Analyzing model : %s " % fname)
                self._log.debug("Model quality : %3e " % q)

            model_is_good = ((self._metric_opt_mode == 'min') and (q < good_model_boundary)) or \
                            ((self._metric_opt_mode == 'max') and (q > good_model_boundary))

            new_earliest_good_model = False
            if model_is_good:
                if self._simulation_mode or self._debug_mode:
                    self._log.debug("Model is good.")

                if earliest_good_model_new is None:
                    earliest_good_model_new = fname
                    earliest_good_model_iter_new = self._model_iter[fname]

                    new_earliest_good_model = True
                else:
                    model_iter = self._model_iter[fname]
                    if model_iter < earliest_good_model_iter_new:
                        if self._simulation_mode or self._debug_mode:
                            self._log.debug("Model is earliest, remove previous earliest model")

                        earliest_good_model_new = fname
                        earliest_good_model_iter_new = model_iter

                        new_earliest_good_model = True
                    else:
                        if self._simulation_mode or self._debug_mode:
                            self._log.debug("Model is not earliest")

                if new_earliest_good_model and (self._simulation_mode or self._debug_mode):
                    self._log.debug("New earliest good model : %s" % earliest_good_model_new)
                    self._log.debug("New earliest good model iter : %s" % earliest_good_model_iter_new)

            else:
                __pop(fname)

        if earliest_good_model_new is None:
            earliest_good_fname = self.earliest_good_model_file_name()
            if self._simulation_mode or self._debug_mode:
                self._log.debug("No earliest good model found, removing ...")

            self._remove_model(earliest_good_fname)

        self._model_quality = model_quality_new
        self._model_iter = model_iter_new

        if self._simulation_mode or self._debug_mode:
            self._log.debug("Models registered:")
            for fname, q in self._model_quality.items():
                self._log.debug("%s: %d: %3e" % (fname,
                                                 self._model_iter[fname],
                                                 q))

        return earliest_good_model_new, earliest_good_model_iter_new

    def _save_checkpoint_state(self):
        try:
            fname = self.checkpoint_state_file_name()

            # only backup if exists
            if os.path.isfile(fname):
                backup_success = self._copy(fname, "%s.backup" % fname)
                if not backup_success:
                    self._log.error("Backing up checkpoint state unsuccessful, "
                                    "will not override checkpoint state file with new state.")
                    return

            if self._simulation_mode or self._debug_mode:
                self._log.debug("Saving checkpoint state to [%s]" % fname)

            if not self._simulation_mode:
                with open(fname, 'wb') as f:
                    pickle.dump({
                        "model_quality": self._model_quality,
                        "model_iter": self._model_iter,
                        "best_model": self._best_model,
                        "best_model_quality": self._best_model_quality,
                        "earliest_good_model": self._earliest_good_model,
                        "earliest_good_model_iter": self._earliest_good_model_iter,
                        "iter": self._iter
                    }, f)
        except Exception as e:
            _.log_exception(self._log, "Unable to save checkpoint state", e)

    def _save_checkpoint(self):
        try:
            fname = self.latest_model_file_name()
            if self._simulation_mode or self._debug_mode:
                self._log.debug("Saving checkpoint as [%s]" % fname)

            if not self._simulation_mode:
                self._model.save_weights(fname)

            return fname
        except Exception as e:
            _.log_exception(self._log, "Unable to save current model", e)

        return None

    def _copy(self, source_fname, dest_fname):
        success = True

        try:
            if self._simulation_mode or self._debug_mode:
                self._log.debug("Copying model: [%s] ==> [%s]" % (source_fname, dest_fname))

            if not self._simulation_mode:
                copyfile(source_fname, dest_fname)
        except Exception as e:
            _.log_exception(self._log, "Unable to copy [%s]" % source_fname, e)
            success = False

        return success

    def _save_current_model_as_temp(self, checkpoint_fname):
        try:
            fname = tempfile.mkstemp(dir=self._temp_model_path)
            fname = fname[1]

            if checkpoint_fname is None:
                if self._simulation_mode or self._debug_mode:
                    self._log.debug("Saving current model weights to temp. file [%s]" % fname)

                if not self._simulation_mode:
                    self._model.save_weights(fname)

                return fname
            else:
                if self._simulation_mode or self._debug_mode:
                    self._log.debug("Saving current model to temp. file: "
                                    "Copying model checkpoint [%s] to temp. file [%s]" % (checkpoint_fname, fname))

                if not self._simulation_mode:
                    if self._copy(checkpoint_fname, fname):
                        # Success
                        return fname
                    else:
                        return None

        except Exception as e:
            _.log_exception(self._log, "Unable to save or copy current model as temp. model", e)
            return None

    def _remove_model(self, model_fname):
        try:
            if self._simulation_mode or self._debug_mode:
                self._log.debug("Removing model: [%s]" % model_fname)

            if not os.path.isfile(model_fname):
                if self._simulation_mode or self._debug_mode:
                    self._log.debug("File does not exists, will not remove ...")

                return

            if not self._simulation_mode:
                os.remove(model_fname)
        except Exception as e:
            _.log_exception(self._log, "Unable to remove file [%s]" % model_fname, e)

    def _check_settings(self):
        if not (hasattr(self._model, 'save_weights') and _.is_callable(self._model.save_weights)):
            self._log.error("No valid model provided, creating checkpoints will fail ...")

        if (self._create_checkpoint_every < 0) and (self._archive_last_checkpoint_every > 0):
            self._log.error("archive_last_checkpoint_every can't be > 0 while _create_checkpoint_every < 0, "
                            "disabling archiving ... ")
            self._archive_last_checkpoint_every = -1

        if (self._create_checkpoint_every > 0) and \
           (self._archive_last_checkpoint_every > 0) and \
           (self._archive_last_checkpoint_every % self._create_checkpoint_every != 0):

            self._archive_last_checkpoint_every = 10 * self._create_checkpoint_every
            self._log.error("archive_last_checkpoint_every must be exact multiple of _create_checkpoint_every, "
                            "changing archive_last_checkpoint_every to [%d]" % self._archive_last_checkpoint_every)
