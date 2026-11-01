"""
Created by Zehao Li (Takuho Ri)
Created on 2025-05-31 (Sat)  17:02:39 (+09:00)

pre-training Trainer script
"""

import contextlib
import json
import logging
import os
import shutil
from typing import Optional

import numpy as np
import torch
import wandb
from schedulefree import RAdamScheduleFree
from torch.utils.data import DataLoader
from transformers import EarlyStoppingCallback, Trainer, TrainerCallback
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint
from transformers.utils import logging as tf_logging
from zclip import ZClip

from pan_core.common.data_handler import (
    DynamicBucketDistributedSampler,
    PhasedDistributedSampler,
)
from pan_core.common.utils import attach_run_log

logger = tf_logging.get_logger(__name__)
logger.setLevel(logging.INFO)


def _group_grad_norm(grads: list[torch.Tensor]) -> torch.Tensor | float:
    """L2 norm over a group of gradient tensors, for logging.

    The norm of the per-tensor norms equals the norm of the concatenation, so the
    gradients never have to be copied into one buffer -- which, for the paper-size
    configuration, meant allocating and copying ~318 MiB every step to produce two
    numbers that only reach the log. ``torch._foreach_norm`` is the same fused
    primitive ``clip_grad_norm_`` uses internally.

    Summation order differs from a single concatenated norm, so the value can move
    in its last digits. Nothing consumes it beyond logging.

    The result is left on the device; ``_pull_to_host`` reads it back together
    with the other norms of the same step.
    """
    if not grads:
        return 0.0
    return torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads)))


def _pull_to_host(*values: torch.Tensor | float) -> tuple[float, ...]:
    """Reads a handful of device scalars back in a single transfer.

    Every ``.item()`` is its own synchronisation, and these run on each training
    step for numbers that go nowhere but the log. Stacking them first makes it
    one. The widening to float64 is exact for the float32 norms this is called
    with, so each value matches what ``.item()`` returned. Values that are already
    on the host pass through.

    PRECONDITION: every tensor passed in sits on the same device, which holds
    while the model does. Reading each scalar with its own ``.item()`` had no
    such requirement, so a model split across two GPUs would raise here. Nothing
    in this project is large enough to need that split; grouping by device first
    is the fix if one ever does.
    """
    on_device = [v for v in values if isinstance(v, torch.Tensor)]
    if not on_device:
        return tuple(float(v) for v in values)
    pulled = iter(torch.stack([v.detach().reshape(()).double() for v in on_device]).tolist())
    return tuple(next(pulled) if isinstance(v, torch.Tensor) else float(v) for v in values)


KL_ACTIVE_THRESHOLD = 0.01


def _kl_dim_metrics(kl_per_dim, prefix: str) -> dict:
    """Collapse diagnostics the summed KL cannot show.

    A latent whose signal sits in five of 512 dimensions reports the same total KL
    as one that spreads it evenly, so the total alone cannot tell a healthy
    posterior from a mostly collapsed one.

    Args:
        kl_per_dim (torch.Tensor): Per-dimension KL in nats. Shape [E].
        prefix (str): Metric name prefix, "train" or the eval prefix.

    Returns:
        dict: ``<prefix>_kl_active_units`` (dimensions above the threshold) and
        ``<prefix>_kl_per_dim_max``.
    """
    return {
        f"{prefix}_kl_active_units": int((kl_per_dim > KL_ACTIVE_THRESHOLD).sum()),
        f"{prefix}_kl_per_dim_max": float(kl_per_dim.max()),
    }


def _mean_of_batch_scalars(values: list[torch.Tensor]) -> float:
    """Mean over per-batch scalar losses held on the device.

    ``tolist()`` moves every value in one transfer and yields Python floats, so the
    mean is taken in float64 over exactly the numbers a per-batch ``.item()`` would
    have produced.
    """
    return float(np.mean(torch.stack(values).tolist()))


class SetEpochCallback(TrainerCallback):
    """
    Callback to set epoch for DynamicBucketDistributedSampler at the start of each epoch.

    ``state.epoch`` may remain fractional after resuming from a mid-epoch
    checkpoint, which can repeat or skip sampler epochs. Track the next sampler
    epoch explicitly instead, deriving its initial value from the restored
    global step.
    """
    def __init__(self, trainer):
        self.trainer = trainer
        self._next_epoch: int | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        """Clears the epoch counter so a resumed run recomputes it from global_step."""
        self._next_epoch = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Advances the epoch counter and passes it to the curriculum sampler.

        The count is derived from ``global_step`` on the first call rather than read
        from ``state.epoch``, so a resumed run restarts the curriculum where it
        stopped.
        """
        if self._next_epoch is None:
            sampler = getattr(self.trainer, "sampler", None)
            if sampler is not None:
                steps_per_epoch = max(1, len(sampler))
                self._next_epoch = state.global_step // steps_per_epoch
            else:
                self._next_epoch = max(0, int(state.epoch))

        epoch = self._next_epoch
        self._next_epoch += 1
        # The epoch's iterator starts after this call; saving counts trained batches from here.
        self.trainer._epoch_start_step = state.global_step

        logger.info(f"***** Epoch[{epoch}] *****")
        if hasattr(self.trainer, "sampler") and self.trainer.sampler is not None:
            self.trainer.sampler.set_epoch(epoch)
            if isinstance(self.trainer.sampler, DynamicBucketDistributedSampler):
                logger.info(f"sampling prop: {self.trainer.sampler.dynamic_sampling_probabilities[epoch]}")

class PhasedLogCallback(TrainerCallback):
    """
    Callback to adjust logging steps with an offset for phased training.
    """
    def __init__(self, offset:int=0):
        self.offset = offset

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Adds ``visible_step`` (offset by earlier phases) beside ``inner_step``."""
        if logs is not None:
            current_inner_step = state.global_step
            logs["visible_step"] = current_inner_step + self.offset
            logs["inner_step"] = current_inner_step
            logs["phase_offset"] = self.offset

class HF_Trainer(Trainer):
    """
    HuggingFace Trainer customized for this project.
    - schedule-free optimizer support (eval/train switching)
    - Dynamic Curriculum Learning (custom dataloader) support
    - ZClip support (via callback)
    - Dynamic Gradient Accumulation support
    """

    def __init__(self, *args, **kwargs):
        """Initializes HF_Trainer with custom extensions.

        Sets up WandB, ZClip, dynamic gradient accumulation mapping, phased logging,
        early stopping, and the SetEpochCallback for curriculum samplers.
        All standard HuggingFace Trainer parameters are accepted via *args/**kwargs.
        """
        self.setup_wandb(kwargs.get("args"))
        super().__init__(*args, **kwargs)

        # logging to run.log
        attach_run_log(self.args.output_dir)

        # --- Adjust logging for phased training ---
        self._phased_log_callback = None
        if self.args.training_strategy_type == "phased":
            self.global_step_offset = 0
            self._phased_log_callback = PhasedLogCallback(offset=0)
            self.add_callback(self._phased_log_callback)

        # Metric history is not tracked here: the HuggingFace Trainer already
        # records it in the checkpoint's trainer_state.json.

        # --- sampler related ---
        self.sampler = None
        self.add_callback(SetEpochCallback(self))

        # --- data skip config (if dynamic-sampling strategy) ---
        if self.args.training_strategy_type in ["dynamic", "phased"]:
            if not self.args.ignore_data_skip:
                logger.warning(f"{self.args.training_strategy_type} Strategy: Forcing 'ignore_data_skip=True'.")
                self.args.ignore_data_skip = True

        # --- setup each phase epochs if phased strategy (inner epoch limit) ---
        if self.args.training_strategy_type == "phased":
            # In phased training, num_train_epochs is per-phase
            if hasattr(self.args, "phased_phase_epochs"):
                self.args.num_train_epochs = self.args.phased_phase_epochs

        # --- Initialize ZClip ---
        if self.args.zclip_enabled:
            self.args.max_grad_norm = None  # Disable default gradient clipping
            self.zclip = ZClip(
                alpha=self.args.zclip_alpha,
                z_thresh=self.args.zclip_thresh,
                # Always None by the line above: ZClip replaces fixed-norm clipping
                # rather than layering on top of it.
                max_grad_norm=self.args.max_grad_norm,
                warmup_steps=self.args.zclip_warmup_steps,
                clip_factor=self.args.zclip_clip_factor
            )

        # --- Dynamic Accumulation: acc_step is injected per-batch by the sampler ---
        if self.args.gradient_accumulation_steps != 1:
            logger.warning("Forcing 'gradient_accumulation_steps=1' for Trainer, as dynamic accumulation is handled manually in training_step.")
            self.args.gradient_accumulation_steps = 1

        # --- prevent duplicated validation ---
        self._last_eval_global_step = -1
        self._last_eval_metrics = None

        # --- per-step metrics (populated in forward pass, consumed in training_step) ---
        self._last_step_metrics: dict = {}

        # --- earlystopping ---
        if getattr(self.args, "early_stopping_enabled", False):
            # Remove default if exists to prevent duplication
            self.remove_callback(EarlyStoppingCallback)
            self.add_callback(EarlyStoppingCallback(
                early_stopping_patience=self.args.early_stopping_patience,
                early_stopping_threshold=self.args.early_stopping_delta
            ))
            logger.info(f"Early Stopping Enabled: Patience={self.args.early_stopping_patience}, Delta={self.args.early_stopping_delta}")

    def setup_wandb(self, args):
        """
        Sets up Weights & Biases (WandB) for experiment tracking before Trainer initialization.

        Args:
            args: TrainingArguments object containing WandB configuration.
        """
        rank = int(os.environ.get("RANK", 0))

        if (not args.wandb_enabled) or (rank != 0):
            return

        if args.wandb_api_key_path:
            try:
                with open(args.wandb_api_key_path) as f:
                    wandb_api_key = f.readline().strip()
                wandb.login(key=wandb_api_key)
                logger.info("Successfully logged in to WandB using API key.")
            except OSError as e:
                logger.error(f"Failed to load WandB API key from {args.wandb_api_key_path}: {e}")

        init_kwargs = {
            "project": args.wandb_project_name if args.wandb_project_name else "huggingface",
            "name": args.wandb_exp_name,   # run_name
            "id": args.wandb_exp_id,       # run_id
            "resume": "allow",             # if run_id exists, resume the run, else create new run
            "config": args.to_dict(),      # log all training args
        }
        wandb.init(**init_kwargs)
        logger.info(f"WandB initialized on Rank 0: {wandb.run.name} (ID: {wandb.run.id})")

    def create_optimizer(self):
        """
        override to initialize RAdamScheduleFree if selected.
        """
        if self.args.optimizer_type == "schedulefree":
            if self.optimizer is None:
                logger.info("Using RAdamScheduleFree optimizer.")

                decay_parameters = get_parameter_names(self.model, [torch.nn.LayerNorm])
                # Membership is tested once per parameter tensor in each of the two
                # groups below; a list makes that a linear scan.
                decay_parameters = {name for name in decay_parameters if "bias" not in name}

                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in self.model.named_parameters() if (n in decay_parameters and p.requires_grad)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if (n not in decay_parameters and p.requires_grad)],
                        "weight_decay": 0.0,
                    }
                ]

                self.optimizer = RAdamScheduleFree(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate,
                    r=getattr(self.args, "schedulefree_r", 0.0),
                    weight_decay=self.args.weight_decay
                )
            return self.optimizer
        else:
            return super().create_optimizer()

    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        """
        Override to return dummy scheduler for ScheduleFree optimizer.
        """
        if self.args.optimizer_type == "schedulefree":
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1.0)
            return self.lr_scheduler
        else:
            return super().create_scheduler(num_training_steps, optimizer)

    def _save_sampler_state(self, output_dir):
        """
        Saves the sampler state directly into the checkpoint directory.
        Filename: sampler_rank{rank}.pth
        """
        if not hasattr(self.sampler, "state_dict"):
            return

        state_path = os.path.join(output_dir, f"sampler_rank{self.args.process_index}.pth")

        try:
            if isinstance(self.sampler, DynamicBucketDistributedSampler):
                # One batch per optimizer step; the live counters also hold the prefetched batches.
                start = getattr(self, "_epoch_start_step", None)
                consumed = None if start is None else self.state.global_step - start
                sampler_state = self.sampler.state_dict(consumed=consumed)
            else:
                sampler_state = self.sampler.state_dict()
            torch.save(sampler_state, state_path)
        except Exception as e:
            logger.error(f"Failed to save sampler state: {e}")

    def _save_strategy_state(self, output_dir):
        """
        Saves training strategy metadata (phase index, global step offset, etc.).
        Filename: strategy_state.json
        """
        state = {}

        # Phased Training Info
        if hasattr(self, "phased_current_phase_idx"):
            state["phased_current_phase_idx"] = self.phased_current_phase_idx

        if hasattr(self, "global_step_offset"):
            state["global_step_offset"] = self.global_step_offset

        # Dynamic Strategy Info (if any special state needs saving)
        if self.args.training_strategy_type == "dynamic":
             state["strategy_type"] = "dynamic"

        if state:
            path = os.path.join(output_dir, "strategy_state.json")
            try:
                with open(path, "w") as f:
                    json.dump(state, f, indent=4)
            except Exception as e:
                logger.error(f"Failed to save strategy state: {e}")

    def _save_checkpoint(self, model, trial):
        """
        Orchestrator for saving checkpoints.
        Injects step offset logic and calls sub-savers.
        """
        is_schedulefree = (self.args.optimizer_type == "schedulefree" and self.optimizer is not None)
        if is_schedulefree:
            self.optimizer.eval()

        try:
            # 1. Call Parent (creates the folder, saves model/optimizer/rng)
            super()._save_checkpoint(model, trial)

            if not self.args.save_only_model:
                # 2. Determine Output Directory
                run_dir = self._get_output_dir(trial=trial)
                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                output_dir = os.path.join(run_dir, checkpoint_folder)

                # 3. Save Custom Artifacts
                if self.sampler is not None:
                    self._save_sampler_state(output_dir)

                if self.is_world_process_zero:
                    self._save_strategy_state(output_dir)

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

        finally:
            if is_schedulefree:
                self.optimizer.train()

    def _save_checkpoint_without_optimizer_and_scheduler(self):
        """Saves model weights, sampler state, and strategy state without optimizer/scheduler.

        Used at phase boundaries in phased training to persist a lightweight checkpoint
        that carries the model and curriculum metadata into the next phase.
        """
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)

        is_schedulefree = (self.args.optimizer_type == "schedulefree" and self.optimizer is not None)
        if is_schedulefree:
            self.optimizer.eval()

        try:
            # 1. save model
            self.save_model(output_dir)

            # 2. Save Custom Artifacts
            if self.sampler is not None:
                self._save_sampler_state(output_dir)

            if self.is_world_process_zero:
                self._save_strategy_state(output_dir)

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

        finally:
            if is_schedulefree:
                self.optimizer.train()

    def _load_sampler_state(self, sampler, resume_path):
        """
        Loads sampler state from the checkpoint directory.
        """
        state_path = os.path.join(resume_path, f"sampler_rank{self.args.process_index}.pth")

        if not os.path.isfile(state_path):
            return # No saved state found

        try:
            # weights_only=False is needed for some complex objects, depends on pytorch version
            states = torch.load(state_path, weights_only=False)
            sampler.load_state_dict(states)
            logger.info(f"Sampler state loaded from {state_path}")
        except Exception as e:
            logger.error(f"Failed to load sampler state: {e}")

    def _load_strategy_state(self, resume_path):
        """
        Loads strategy metadata (phase info, offsets).
        """
        path = os.path.join(resume_path, "strategy_state.json")
        if not os.path.isfile(path):
            return

        try:
            with open(path) as f:
                state = json.load(f)

            # Restore Phase Info
            if "phased_current_phase_idx" in state:
                self.phased_current_phase_idx = state["phased_current_phase_idx"]
                logger.info(f"Restored Phase Index: {self.phased_current_phase_idx+1}")

            if "global_step_offset" in state:
                self.global_step_offset = state["global_step_offset"]
                logger.info(f"Restored Global Step Offset: {self.global_step_offset}")

                if hasattr(self, "_phased_log_callback") and self._phased_log_callback is not None:
                    self._phased_log_callback.offset = self.global_step_offset

        except Exception as e:
            logger.error(f"Failed to load strategy state: {e}")

    def _load_custom_state(self, resume_path):
        """
        Orchestrator for loading custom states.
        Order matters: Strategy -> Sampler -> Metrics
        """
        if resume_path is None:
            return

        logger.info(f"Loading custom states from {resume_path}")

        # 1. Load Strategy First (to get correct offsets/phases)
        self._load_strategy_state(resume_path)

        # 2. Load Sampler
        if self.sampler is not None:
            self._load_sampler_state(self.sampler, resume_path)

    def _load_optimizer_and_scheduler(self, checkpoint: str):
        """Loads optimizer/scheduler from checkpoint and then restores custom states (sampler, strategy).

        Args:
            checkpoint (str): Path to the checkpoint directory.
        """
        super()._load_optimizer_and_scheduler(checkpoint)
        self._load_custom_state(checkpoint)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Override to ensure ALL states (Sampler, Metrics, Strategy) are saved alongside the model weights.
        This allows 'final_model' to act as a pseudo-checkpoint for metadata retrieval.
        """

        # Not caught: a failed save is the end of the run's value. Swallowing it let
        # training finish, report success and leave an output directory holding only
        # config.json -- days of compute with nothing to load.
        super().save_model(output_dir, _internal_call)

    def _resolve_strategy_from_path(self, resume_path):
        """
        Determines the starting phase and offset by inspecting checkpoints.

        Args:
            resume_path: could be a specific checkpoint path, the root output dir, or None/True.

        Returns:
            (start_phase, current_offset, found_checkpoint_path)
            - start_phase (int): The phase index to start/resume loop from.
            - current_offset (int): The global step offset retrieved from the state.
            - found_checkpoint_path (str or None): The specific path to resume from (for the first loop).
        """
        # Default values
        start_phase = 0
        current_offset = 0
        target_checkpoint = None

        # 1. if the path is not specified or is a bool, start searching from args.output_dir
        search_root = self.args.output_dir
        if isinstance(resume_path, str) and os.path.isdir(resume_path):
            # If "checkpoint-xxx" or "final_model" is directly specified, trust it
            if "checkpoint-" in resume_path or "final_model" in resume_path:
                search_root = None # Skip search, use this directly
                target_checkpoint = resume_path
            else:
                search_root = resume_path

        # 2. Automatic search logic (if root directory is specified)
        if search_root:
            # Check from Phase N down to 1 to find the latest checkpoint
            max_phases = getattr(self.args, "phased_phases", 1)
            for p in range(max_phases, 0, -1):
                phase_dir = os.path.join(search_root, f"phase_{p}")
                if os.path.isdir(phase_dir):
                    # get last checkpoint in this phase
                    last_ckpt = get_last_checkpoint(phase_dir)
                    if last_ckpt:
                        target_checkpoint = last_ckpt
                        start_phase = p - 1
                        break # Stop once the latest phase is found

        if target_checkpoint and os.path.isdir(target_checkpoint):
            strategy_path = os.path.join(target_checkpoint, "strategy_state.json")
            if os.path.isfile(strategy_path):
                try:
                    with open(strategy_path) as f:
                        state = json.load(f)
                    json_phase = state.get("phased_current_phase_idx", start_phase)
                    if json_phase != start_phase:
                         logger.warning(f"Phase mismatch! Directory implies index {start_phase}, but JSON says {json_phase}. Using Directory index.")
                    current_offset = state.get("global_step_offset", 0)
                    logger.info(f"Resolved strategy from {target_checkpoint}: Phase {start_phase+1}, Offset {current_offset}")
                except Exception as e:
                    logger.error(f"Failed to read strategy state from {strategy_path}: {e}")
            else:
                pass

        return start_phase, current_offset, target_checkpoint

    def set_phase_context(self, phase_idx: int, step_offset: int):
        """
        Sets the current phase context for phased training.
        Adjusts global step offset and updates phased log callback.
        """
        self.phased_current_phase_idx = phase_idx

        self.global_step_offset = step_offset
        if self._phased_log_callback is not None:
            self._phased_log_callback.offset = step_offset
            logger.info(f"Updated phased log callback offset to {step_offset} for phase {phase_idx+1}")

    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None, **kwargs):
        """
        Override to accept custom resume path for loading sampler state.
        """
        if self.args.wandb_enabled and self.is_world_process_zero():
            wandb.watch(self.model, log="all", log_freq=100)

        if self.args.training_strategy_type == "phased":
            # 1. Determine starting phase and offset
            start_phase, cumulative_offset, initial_checkpoint = self._resolve_strategy_from_path(resume_from_checkpoint)
            num_phases = self.args.phased_phases
            base_output_dir = self.args.output_dir
            logger.info(f"Phased Loop Planning: Start Phase {start_phase+1}, Offset-step {cumulative_offset}")

            # 2. Loop over phases
            for phase in range(start_phase, num_phases):
                logger.info(f"Starting Phase {phase+1} / {num_phases}")
                phase_output_dir = os.path.join(base_output_dir, f"phase_{phase+1}")
                self.args.output_dir = phase_output_dir

                if phase == start_phase:
                    self.set_phase_context(phase, cumulative_offset)

                # reset earlystpping
                if getattr(self.args, "early_stopping_enabled", False):
                    self.remove_callback(EarlyStoppingCallback)
                    self.add_callback(EarlyStoppingCallback(
                        early_stopping_patience=self.args.early_stopping_patience,
                        early_stopping_threshold=self.args.early_stopping_delta
                    ))

                current_resume_checkpoint = initial_checkpoint if phase == start_phase else None
                if current_resume_checkpoint:
                    logger.info(f"Phase {phase+1}: Resuming from {current_resume_checkpoint}")
                else:
                    logger.info(f"Phase {phase+1}: Starting fresh (inheriting weights in memory)")
                # execute training for this phase
                result = super().train(resume_from_checkpoint=current_resume_checkpoint, trial=trial, **kwargs)
                # update cumulative offset
                cumulative_offset += result.global_step
                # prepare for next phase
                if PREFIX_CHECKPOINT_DIR+"-0" in os.listdir(phase_output_dir):
                    shutil.rmtree(os.path.join(phase_output_dir, PREFIX_CHECKPOINT_DIR+"-0"))
                if (self.args.optimizer_type == "schedulefree" and self.optimizer is not None):
                    self.optimizer.eval()
                self.state.global_step = 0
                self.state.epoch = 0
                self.state.best_metric = None
                self.state.best_model_checkpoint = None
                self.sampler = None
                self.optimizer = None
                self.scheduler = None
                self.args.output_dir = os.path.join(base_output_dir, f"phase_{phase+2}")
                self.set_phase_context(phase+1, cumulative_offset)
                self._save_checkpoint_without_optimizer_and_scheduler()

                initial_checkpoint = None

            return result

        else:
            return super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)

    def get_train_dataloader(self):
        """Builds and returns the training DataLoader with the appropriate curriculum sampler.

        - "dynamic" strategy: wraps ``DynamicBucketDistributedSampler`` (interpolated phase probs).
        - "phased" strategy: wraps ``PhasedDistributedSampler`` (single-phase slice with replay).
        - Otherwise: falls back to the default HuggingFace implementation.

        Returns:
            torch.utils.data.DataLoader: Training DataLoader.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        if self.args.training_strategy_type == "dynamic":
            logger.info("Using DynamicBucketDistributedSampler for training dataloader.")

            if self.args.train_data_config_list is None:
                raise ValueError("Dynamic training strategy requires train_data_config_list.")

            num_data_list = [d.get("num_data", 0) for d in self.args.train_data_config_list]

            logger.info(f"train data: {sum(num_data_list)}")

            sampler = DynamicBucketDistributedSampler(
                batch_sizes=self.args.curriculum_batch_sizes,
                accumulate_steps=self.args.curriculum_accumulate_steps,
                num_replicas=self.args.world_size if self.args.world_size is not None else 1,
                rank=self.args.process_index if self.args.process_index is not None else 0,
                max_epochs=int(self.args.num_train_epochs),
                total_optimizer_steps=self.args.max_steps,
                num_data=num_data_list,
                start_sampling_probabilities=self.args.dynamic_start_probs,
                end_sampling_probabilities=self.args.dynamic_end_probs,
                sampling_update_last_epoch=self.args.dynamic_update_last_epoch,
                sampling_interpolation_type=self.args.dynamic_interpolation_type,
                sampling_interpolation_steepness=self.args.dynamic_sigmoid_steepness,
                sampling_interpolation_delay_factor=self.args.dynamic_sigmoid_delay,
                shuffle=True,
                seed=self.args.seed
            )

            self.sampler = sampler

            return DataLoader(
                dataset=self.train_dataset,
                batch_sampler=self.sampler,
                collate_fn=self.data_collator,
                **self._worker_kwargs(),
            )

        elif self.args.training_strategy_type == "phased":
            logger.info("Using PhasedDistributedSampler for training dataloader.")

            if self.args.train_data_config_list is None:
                raise ValueError("Phased training strategy requires train_data_config_list.")

            num_data_list = [d.get("num_data", 0) for d in self.args.train_data_config_list]
            current_phase_idx = getattr(self, "phased_current_phase_idx", 0)

            sampler = PhasedDistributedSampler(
                current_phase_idx=current_phase_idx,
                batch_size=self.args.curriculum_batch_sizes[current_phase_idx],
                accumulate_steps=self.args.curriculum_accumulate_steps[current_phase_idx],
                epoch=self.state.epoch,
                num_data_list=num_data_list,
                replay_ratio=self.args.phased_replay_ratio,
                num_replicas=self.args.world_size if self.args.world_size is not None else 1,
                rank=self.args.process_index if self.args.process_index is not None else 0,
                shuffle=True,
                drop_last=False,
                seed=self.args.seed
            )

            self.sampler = sampler

            return DataLoader(
                dataset=self.train_dataset,
                batch_sampler=self.sampler,
                collate_fn=self.data_collator,
                **self._worker_kwargs(),
            )

        else:
            return super().get_train_dataloader()

    def _worker_kwargs(self) -> dict:
        """DataLoader worker settings shared by the curriculum loaders.

        The default loader of the Trainer reads persistent_workers and
        prefetch_factor from the arguments; these hand-built ones used to pass
        only num_workers, so both settings were silently ignored. torch rejects
        either one without worker processes, hence the guard.
        """
        kwargs = {"num_workers": self.args.dataloader_num_workers, "pin_memory": True}
        if self.args.dataloader_num_workers > 0:
            kwargs["persistent_workers"] = self.args.dataloader_persistent_workers
            if self.args.dataloader_prefetch_factor is not None:
                kwargs["prefetch_factor"] = self.args.dataloader_prefetch_factor
        return kwargs

    def _advance_kl_warmup(self):
        """Sets the KL weight for this optimizer step.

        The weight rises linearly from 0 to ``config.beta`` over
        ``config.beta_warmup_steps``. Starting at the full weight lets a strong
        autoregressive decoder drive the reconstruction loss down while the KL term
        pushes the posterior onto the prior, so the decoder learns to ignore the
        latent before it ever learns to use it. Held in a buffer and written in
        place so the value can change every step without rebuilding any graph.
        """
        warmup = getattr(self.model.config, "beta_warmup_steps", 0)
        if not self.model.config.vae or warmup <= 0:
            return
        scale = min(1.0, self.state.global_step / warmup)
        self.model.kl_beta_scale.fill_(scale)

    def training_step(self, model, inputs: dict, _num_items_in_batch):
        """
        Override to handle Dynamic Gradient Accumulation based on batch size.
        """
        logger.debug(f"[step{self.state.global_step}] Running training step...")
        model.train()
        if (self.args.optimizer_type == "schedulefree" and self.optimizer is not None):
            self.optimizer.train()

        # pop the meta data (if exist)
        acc_step = inputs.pop("accumulate_step", 1)
        inputs = self._prepare_inputs(inputs)

        self._advance_kl_warmup()

        # Determine accumulation steps based on batch size
        if "input_ids" in inputs:
            current_macro_bs = inputs["input_ids"].size(0)
        elif "latents" in inputs:
            current_macro_bs = inputs["latents"].size(0)
        else:
            current_macro_bs = self.args.per_device_train_batch_size

        if current_macro_bs % acc_step != 0:
            # Should not happen if Sampler logic is correct
            raise ValueError(f"Macro BS {current_macro_bs} not divisible by Acc {acc_step} !")
        else:
            micro_bs = current_macro_bs // acc_step

        # Manual Accumulation loop
        total_loss = torch.tensor(0.0, device=model.device, requires_grad=False)

        logger.debug(f"[step{self.state.global_step}] Running accumulation loop... (acc_step: {acc_step})")
        for i in range(acc_step):
            start_idx = i * micro_bs
            end_idx = start_idx + micro_bs

            micro_inputs = {
                k: self._slice_input(v, start_idx, end_idx) for k, v in inputs.items()
            }

            # Only the last micro-batch all-reduces. The Trainer's own
            # gradient_accumulation_steps is pinned to 1 (see __init__), so
            # Accelerate treats every backward as a sync step, and DDP's autograd
            # hooks would all-reduce the full gradient once per micro-batch: acc_step
            # times per optimizer step instead of once. no_sync defers it; the sum
            # is then averaged across ranks once, which is the same value in exact
            # arithmetic (floating-point summation order differs).
            # DDP decides whether to sync at forward time, so the context must cover
            # the forward as well as the backward. `model` is the wrapped model
            # (self.model_wrapped); self.model is the bare module and has no no_sync.
            # Off DDP, accelerator.no_sync is a null context.
            sync_ctx = (self.accelerator.no_sync(model) if i < acc_step - 1
                        else contextlib.nullcontext())
            with sync_ctx:
                with self.compute_loss_context_manager():
                    # No KV cache: only the loss is read, and the cache would keep
                    # this micro-batch's K/V alive into the next micro-batch's forward.
                    outputs = model(**micro_inputs, use_cache=False)
                    loss = outputs.loss
                    # Every iteration used to overwrite this dict, so only the last
                    # micro-batch's numbers ever reached the log -- and each of the
                    # discarded ones still cost a .item(), which drains the device
                    # queue. Built once, on the iteration whose values are kept.
                    if i == acc_step - 1:
                        self._last_step_metrics = {
                            "train_celoss": outputs.celoss.item() if outputs.celoss is not None else None,
                            "train_klloss": outputs.klloss.item() if outputs.klloss is not None else None
                        }
                        if outputs.kl_per_dim is not None:
                            self._last_step_metrics.update(
                                _kl_dim_metrics(outputs.kl_per_dim, prefix="train"))
                            self._last_step_metrics["train_kl_beta_scale"] = float(
                                self.model.kl_beta_scale)

                # if not DDP, such as DataParallel etc...
                if self.args.n_gpu > 1:
                    loss = loss.mean()

                loss_to_backward = loss / acc_step

                if getattr(self.accelerator, "scaler", None) is not None:
                    # AMP scaling enabled
                    self.accelerator.scaler.scale(loss_to_backward).backward()
                else:
                    # No AMP → normal backward
                    self.accelerator.backward(loss_to_backward)

            total_loss += loss_to_backward.detach()

        # gradient clipping + per-step logging
        grad_norm, encoder_norm, decoder_norm = self._gradient_clipping(self.model)
        logs = {
            "train_grad_norm": grad_norm,
            "train_encoder_grad_norm": encoder_norm,
            "train_decoder_grad_norm": decoder_norm,
            **{k: v for k, v in self._last_step_metrics.items() if v is not None},
        }
        self._last_step_metrics = {}
        self.log(logs)

        return total_loss

    def _gradient_clipping(self, model) -> tuple[float, float, float]:
        """Clips gradients and computes per-component norms.

        Returns:
            Tuple[float, float, float]: (grad_norm, encoder_norm, decoder_norm).
        """
        logger.debug(f"[step{self.state.global_step}] Running gradient clipping...")
        encoder_params, decoder_params = self._norm_param_groups(model)
        encoder_grads = [p.grad.detach() for p in encoder_params if p.grad is not None]
        decoder_grads = [p.grad.detach() for p in decoder_params if p.grad is not None]

        encoder_norm = _group_grad_norm(encoder_grads)
        decoder_norm = _group_grad_norm(decoder_grads)

        if hasattr(self, "zclip"):
            grad_norm = self.zclip.step(model)
        elif self.args.max_grad_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
        else:
            # No clipping — compute norm for logging without modifying gradients.
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))

        return _pull_to_host(grad_norm, encoder_norm, decoder_norm)

    def _norm_param_groups(self, model):
        """Encoder and decoder parameters, split once and kept.

        The split is by parameter name and the names do not change during a run,
        so walking `named_parameters()` on every step only rebuilt the same two
        lists. The order is the one that walk produced, which is what
        `_group_grad_norm` sums over.

        The cache is keyed on the identity of every parameter, not on how many
        there are. ``resize_token_embeddings`` and ``set_input_embeddings``
        replace the embedding module and leave the count unchanged, so a count
        would still match while the cached lists held dead ``Parameter`` objects
        whose ``.grad`` is always None -- the embedding's gradient would silently
        drop out of the logged encoder norm. Walking the parameters to collect
        their ids costs the same walk the count did.
        """
        key = tuple(map(id, model.parameters()))
        cached = getattr(self, "_norm_param_groups_cache", None)
        if cached is not None and cached[0] is model and cached[1] == key:
            return cached[2], cached[3]
        encoder_params, decoder_params = [], []
        for name, param in model.named_parameters():
            if "encoder" in name:
                encoder_params.append(param)
            elif "decoder" in name:
                decoder_params.append(param)
        self._norm_param_groups_cache = (model, key, encoder_params, decoder_params)
        return encoder_params, decoder_params

    @staticmethod
    def _slice_input(data, start: int, end: int):
        """Slices a tensor along dim 0; passes non-tensor values through unchanged."""
        if isinstance(data, torch.Tensor):
            return data[start:end]
        return data

    def _phase_means(self, values: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Splits *values* into equal curriculum phases and averages each one.

        Args:
            values (torch.Tensor): Per-sample values ordered by dataset index. CE losses
                arrive as [N, L], KL losses as [N].
            mask (torch.Tensor, optional): Boolean selector of the same shape, applied
                before averaging. None counts every element.

        Returns:
            Tuple[Dict[int, float], float | None]: 1-based phase index to mean (a phase
                with nothing selected is left out), and the mean over all phases, which
                is None when the mask selects nothing.
        """
        total_samples = values.size(0)
        num_phases = self.args.dynamic_phases if self.args.training_strategy_type == 'dynamic' else self.args.phased_phases
        samples_per_phase = total_samples // num_phases

        per_phase = {}
        for p in range(1, num_phases + 1):
            start = samples_per_phase * (p - 1)
            end = min(samples_per_phase * p, total_samples)
            if start >= total_samples:
                break

            chunk = values[start:end]
            if mask is None:
                per_phase[p] = float(chunk.mean().item())
                continue
            chunk_mask = mask[start:end]
            if chunk_mask.sum() > 0:
                per_phase[p] = float(chunk[chunk_mask].mean().item())

        if mask is None:
            overall = float(values.mean().item())
        else:
            overall = float(values[mask].mean().item()) if mask.sum() > 0 else None
        return per_phase, overall

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """
        Override to switch optimizer to eval mode (swap model weights) before evaluation and switch back to train mode after.
        """
        # --- prevent duplicate ---
        current_step = self.state.global_step

        if current_step == self._last_eval_global_step and self._last_eval_metrics is not None:
            return self._last_eval_metrics

        is_schedulefree = (self.args.optimizer_type == "schedulefree" and self.optimizer is not None)

        if is_schedulefree:
            self.optimizer.eval()

        try:
            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            model = self.model
            model.eval()

            all_total_loss = []
            all_ce_loss = []
            all_kl_loss = []
            all_kl_per_dim = []

            # raw tensors to be gathered
            all_ce_raw = []
            all_kl_raw = []
            all_mask = []
            all_indices = []

            logger.info(f"[step{self.state.global_step}] Running Evaluation ...")

            for batch in eval_dataloader:
                batch = self._prepare_inputs(batch)

                # if curriculum learning, set reduction to "none" to get raw losses
                batch["reduction"] = "none" if self.args.training_strategy_type != "normal" else "mean"

                # add collected indices for proper ordering later
                if "dataset_indices" in batch:
                    all_indices.append(batch["dataset_indices"])

                # use_cache=False: the decoder's K/V for every layer would otherwise be
                # held for the whole forward (most of its peak at batch 2048, length 402).
                with torch.no_grad():
                    outputs = model(**batch, use_cache=False)

                    # Scalars stay on the device and are read in one pass below;
                    # .item() here would synchronise three times per batch.
                    all_total_loss.append(outputs.loss.detach())

                    if outputs.celoss is not None:
                        all_ce_loss.append(outputs.celoss.detach())
                    if outputs.klloss is not None:
                        all_kl_loss.append(outputs.klloss.detach())
                    if outputs.celoss_raw is not None:
                        all_ce_raw.append(outputs.celoss_raw.transpose(0, 1).cpu())
                    if outputs.klloss_raw is not None:
                        all_kl_raw.append(outputs.klloss_raw.cpu())
                    if outputs.kl_per_dim is not None:
                        all_kl_per_dim.append(outputs.kl_per_dim.detach())
                    all_mask.append((batch["labels"][:, 1:] != self.model.config.pad_token_id).cpu())

            # Aggregation
            metrics = {}

            if len(all_total_loss) > 0:
                metrics[f"{metric_key_prefix}_loss"] = _mean_of_batch_scalars(all_total_loss)
            if len(all_ce_loss) > 0:
                metrics[f"{metric_key_prefix}_celoss"] = _mean_of_batch_scalars(all_ce_loss)
            if len(all_kl_loss) > 0:
                metrics[f"{metric_key_prefix}_klloss"] = _mean_of_batch_scalars(all_kl_loss)
            if len(all_kl_per_dim) > 0:
                metrics.update(_kl_dim_metrics(
                    torch.stack(all_kl_per_dim).mean(dim=0), prefix=metric_key_prefix))
                metrics[f"{metric_key_prefix}_kl_beta_scale"] = float(self.model.kl_beta_scale)

            # --- Gather CE Raw ---
            full_ce = None
            if len(all_ce_raw) > 0 and len(all_indices) > 0:
                full_ce = self.gather_and_sort(all_ce_raw, all_indices, is_seq=True, pad_val=0)

            # --- Gather Mask ---
            full_mask = None
            if len(all_mask) > 0 and len(all_indices) > 0:
                # pad_val for mask should be False (0) or ignored, using 0 here
                full_mask = self.gather_and_sort(all_mask, all_indices, is_seq=True, pad_val=0)

            # --- Gather KL Raw ---
            full_kl = None
            if len(all_kl_raw) > 0 and len(all_indices) > 0:
                full_kl = self.gather_and_sort(all_kl_raw, all_indices, is_seq=False)

            if full_ce is not None and full_mask is not None:
                # ensure size matches
                assert full_ce.size() == full_mask.size()

                per_phase, overall = self._phase_means(full_ce, full_mask)
                for p, value in per_phase.items():
                    metrics[f"{metric_key_prefix}_ce_phase{p}"] = value
                # overall CE
                if overall is not None:
                    metrics[f"{metric_key_prefix}_celoss"] = overall

            if full_kl is not None:
                per_phase, overall = self._phase_means(full_kl)
                for p, value in per_phase.items():
                    metrics[f"{metric_key_prefix}_kl_phase{p}"] = value
                # overall KL (overwrites the batch-mean estimate with the token-exact value)
                metrics[f"{metric_key_prefix}_klloss"] = overall

            self.log(metrics)

            self._last_eval_global_step = current_step
            self._last_eval_metrics = metrics

            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            self._memory_tracker.stop_and_update_metrics(metrics)

            return metrics

        finally:
            if is_schedulefree:
                self.optimizer.train()

    def gather_and_sort(self, local_tensors, local_indices, is_seq=False, pad_val=0):
        """Gathers tensors across all DDP processes, sorts by original index, and deduplicates.

        Args:
            local_tensors (List[torch.Tensor]): Per-batch tensors from this process.
                Shape per tensor: [B] or [B, L] depending on ``is_seq``.
            local_indices (List[torch.Tensor]): Corresponding dataset indices, shape [B] per batch.
            is_seq (bool, optional): If True, treats tensors as sequences and pads along dim=1
                before gathering. Defaults to False.
            pad_val (int, optional): Value used for sequence padding. Defaults to 0.

        Returns:
            torch.Tensor | None: Deduplicated and index-sorted tensor of shape [N] or [N, L],
                or None if ``local_tensors`` is empty.
        """
        if len(local_tensors) == 0:
            return None

        # Concatenate locally first
        # Handle sequence padding locally if needed
        if is_seq:
            max_len = max(t.size(1) for t in local_tensors)
            padded_list = []
            for t in local_tensors:
                if t.size(1) < max_len:
                    pad = torch.full((t.size(0), max_len - t.size(1)), pad_val, dtype=t.dtype, device=t.device)
                    t = torch.cat([t, pad], dim=1)
                padded_list.append(t)
            local_cat = torch.cat(padded_list, dim=0)
        else:
            local_cat = torch.cat(local_tensors, dim=0)
        local_idx_cat = torch.cat(local_indices, dim=0)
        # A bool tensor (the validation token mask) deadlocks DDP validation.
        # pad_across_processes builds the padded copy as new_zeros(...) + pad_index,
        # and bool + int promotes to int64, while the rank that already holds the
        # longest sequence is returned unpadded as bool. The ranks then all_gather
        # tensors of different dtypes and NCCL never returns (seen on 2 and 4
        # GH200 nodes: the float CE gather went through, the mask gather hung until
        # the 1800 s collective timeout). uint8 + int stays uint8 on every rank.
        was_bool = local_cat.dtype == torch.bool
        if was_bool:
            local_cat = local_cat.to(torch.uint8)
        # Move to current device for accelerator.gather
        local_cat = local_cat.to(self.args.device)
        local_idx_cat = local_idx_cat.to(self.args.device)
        # Pad across processes (Essential for DDP if seq_len differs across GPUs)
        if is_seq:
            local_cat = self.accelerator.pad_across_processes(local_cat, dim=1, pad_index=pad_val)

        # Gather from all GPUs
        full_data = self.accelerator.gather(local_cat)
        full_indices = self.accelerator.gather(local_idx_cat)
        # Sort by Index (Recover original order)
        sorted_indices = torch.argsort(full_indices)
        full_data_sorted = full_data[sorted_indices]
        full_indices_sorted = full_indices[sorted_indices]
        # Deduplicate (remove DDP padding duplicates).
        # Indices are already sorted, so duplicates are always consecutive —
        # a position is unique iff it differs from its predecessor.
        unique_mask = torch.ones(full_indices_sorted.size(0), dtype=torch.bool, device=full_indices_sorted.device)
        unique_mask[1:] = full_indices_sorted[1:] != full_indices_sorted[:-1]

        result = full_data_sorted[unique_mask]
        return result.bool() if was_bool else result
