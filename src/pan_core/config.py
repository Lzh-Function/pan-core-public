"""
Created by Zehao Li (Takuho Ri)
Created on 2025-07-26 (Sat)  17:29:45 (+09:00)

setting configures
"""

import difflib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import yaml
from transformers import PretrainedConfig, TrainingArguments


class ModelConfig(PretrainedConfig):
    """
    Configuration class for the PanCORE model.
    Initialization with yaml_dict.
    """
    model_type = "pan-core"

    def __init__(
        self,
        # --- architecture args ---
        vae: bool = False,

        # --- common args ---
        vocab_size: int = 185,
        dropout: float = 0.2,
        embedding_dim: int = 512,
        tanh_latent: bool = False,
        latent_noising: bool = False,

        # --- vae specific args ---
        beta: float = 0.01,
        beta_warmup_steps: int = 0,

        # --- Transformer specific args ---
        n_positions: int = 2048,
        n_layer: int = 8,
        n_head: int = 8,
        bias: bool = True,
        weight_sharing: bool = False,
        layer_norm_epsilon: float = 1.0e-5,
        gpt2style:  bool = False,
        # RoPE
        encoder_rope: bool = True,
        decoder_rope: bool = True,
        base_freq: int = 10000,
        # latent
        pooling_mode: str = "MHA",
        cross_attn: str|None = None,
        latent_conditioning: str|None = "add_once",

        # --- HF standard args ---
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,

        **kwargs # ignore other args
    ):
        """
        Initializes the PanCORE configuration.

        Args:
            vae (bool): Whether to use VAE.
            vocab_size (int): Size of the vocabulary. (can be overwritten by tokenizer)
            dropout (float): Dropout probability.
            embedding_dim (int): Dimension of the embeddings.
            tanh_latent (bool): Whether to apply tanh activation to latent variables.
            latent_noising (bool): Whether to add noise to latent variables during training.
            beta (float): KL weight, in nats of KL per token of reconstruction.
            beta_warmup_steps (int): Optimizer steps over which the KL weight rises
                linearly from 0 to ``beta``. 0 applies ``beta`` from the first step.
            n_positions (int): Maximum number of positions for Transformer.
            n_layer (int): Number of layers in Transformer.
            n_head (int): Number of attention heads in Transformer.
            bias (bool): Whether to use bias in Transformer layers.
            weight_sharing (bool): Whether to share weights between embedding and output layers in Transformer decoder.
            layer_norm_epsilon (float): Epsilon value for layer normalization in Transformer.
            gpt2style (bool): Whether to use GPT-2 style TransformerBlock (SwiGLU -> GPT2MLP, Linear -> Conv1D).
            encoder_rope (bool): Whether to use Rotary Positional Embeddings in the Transformer encoder.
            decoder_rope (bool): Whether to use Rotary Positional Embeddings in the Transformer decoder.
            base_freq (int): Base frequency for Rotary Positional Embeddings.
            pooling_mode (str): Pooling mode for latent variables.
            cross_attn (str|None): Cross-attention mode.
            latent_conditioning (str|None): Latent conditioning mode.
            pad_token_id (int): Padding token ID.
            bos_token_id (int): Beginning-of-sequence token ID.
            eos_token_id (int): End-of-sequence token ID.
            **kwargs: Additional keyword arguments (will be ignored).
        """
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id
        )
        self.vae = vae
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.tanh_latent = tanh_latent
        self.latent_noising = latent_noising
        self.beta = beta
        self.beta_warmup_steps = beta_warmup_steps
        self.n_positions = n_positions
        self.n_layer = n_layer
        self.n_head = n_head
        self.bias = bias
        self.weight_sharing = weight_sharing
        self.layer_norm_epsilon = layer_norm_epsilon
        self.gpt2style = gpt2style
        self.encoder_rope = encoder_rope
        self.decoder_rope = decoder_rope
        self.base_freq = base_freq
        self.pooling_mode = pooling_mode
        self.cross_attn = cross_attn
        self.latent_conditioning = latent_conditioning

# Accepted values of training.optimizer.type. The name also selects the nested
# settings block of the same name inside training.optimizer, so it is matched exactly.
SUPPORTED_OPTIMIZERS = frozenset({"schedulefree", "AdamW"})


@dataclass
class TrainArgs(TrainingArguments):
    """
    Extended TrainingArguments.

    NOTE:
    - Standard HF arguments (output_dir, learning_rate, seed, etc.) are inherited.
    - ONLY custom parameters are defined below.
    """

    # ==========================================================
    # [CUSTOM] System & Logging Extras
    # ==========================================================
    # experiment environment
    environment: str = field(default="normal", metadata={"help": "Execution environment: normal or hpc"})

    # HF has 'report_to' and 'run_name', but API keys/Project names are custom handling
    wandb_enabled: bool = field(default=False, metadata={"help": "Whether to use WandB logging"})
    wandb_api_key_path: Optional[str] = field(default=None, metadata={"help": "Path to WandB API key"})
    wandb_project_name: Optional[str] = field(default=None, metadata={"help": "WandB project name"})
    wandb_exp_name: Optional[str] = field(default=None, metadata={"help": "WandB experiment name"})
    wandb_exp_id: Optional[str] = field(default=None, metadata={"help": "WandB experiment ID"})

    # ==========================================================
    # [CUSTOM] Training Strategy
    # ==========================================================
    init_model_path: Optional[str] = field(default=None, metadata={"help": "Path to initial model checkpoint"})
    save_only_model: bool = field(default=False, metadata={"help": "Whether to save only the model weights without optimizer/scheduler states"})
    training_strategy_type: str = field(default="normal", metadata={"help": "normal, phased, or dynamic"})

    # Strategy: Normal / Phased
    phased_replay_ratio: float = field(default=0.0, metadata={"help": "Replay ratio for normal/phased training"})
    phased_phases: int = field(default=5, metadata={"help": "Number of phases for phased training"})
    phased_phase_epochs: int = field(default=10, metadata={"help": "Number of epochs per phase for phased training"})

    # Strategy: Dynamic CPT
    dynamic_phases: int = field(default=5, metadata={"help": "Dynamic CPT phase"})
    dynamic_start_probs: Optional[List[float]] = field(default=None, metadata={"help": "Dynamic CPT start probabilities"})
    dynamic_end_probs: Optional[List[float]] = field(default=None, metadata={"help": "Dynamic CPT end probabilities"})
    dynamic_update_last_epoch: int = field(default=50, metadata={"help": "Epoch to finish updating sampling probabilities"})
    dynamic_interpolation_type: str = field(default="sigmoid", metadata={"help": "Dynamic CPT interpolation type, e.g., linear, sigmoid"})
    dynamic_sigmoid_steepness: float = field(default=12.0, metadata={"help": "Dynamic CPT sigmoid steepness"})
    dynamic_sigmoid_delay: float = field(default=0.8, metadata={"help": "Dynamic CPT sigmoid delay factor for later phase data"})

    # ==========================================================
    # [CUSTOM] Optimizer & Scheduler Extras
    # ==========================================================
    # HF supports "AdamW", but "schedulefree" is custom. Spelling is matched exactly;
    # see SUPPORTED_OPTIMIZERS.
    optimizer_type: str = field(default="schedulefree", metadata={"help": "Optimizer type: AdamW or schedulefree"})

    # ScheduleFree specific
    schedulefree_r: float = field(default=0.5, metadata={"help": "ScheduleFree optimizer parameter r"})

    # ==========================================================
    # [CUSTOM] Z-Clip
    # ==========================================================
    # Z-Clip is fully custom
    zclip_enabled: bool = field(default=True, metadata={"help": "Whether to enable Z-Clip"})
    zclip_alpha: float = field(default=0.97, metadata={"help": "Alpha parameter for Z-Clip"})
    zclip_thresh: float = field(default=2.5, metadata={"help": "Threshold for Z-Clip"})
    zclip_warmup_steps: int = field(default=25, metadata={"help": "Warmup steps for Z-Clip"})
    zclip_clip_factor: float = field(default=1.0, metadata={"help": "Clip factor for Z-Clip"})

    # ==========================================================
    # [CUSTOM] Early Stopping
    # ==========================================================
    # HF has no built-in early stopping configuration in args, usually done via callbacks
    early_stopping_enabled: bool = field(default=True, metadata={"help": "Whether to enable early stopping"})
    early_stopping_patience: int = field(default=10000, metadata={"help": "Early stopping patience"})
    early_stopping_delta: float = field(default=0.0, metadata={"help": "Early stopping delta"})

    # ==========================================================
    # [CUSTOM] Evaluation Extras
    # ==========================================================
    # not teacher forcing validation but generation-based evaluation
    evaluation_maxlen: int = field(default=500, metadata={"help": "Maximum length for evaluation generation"})
    evaluation_batch_size: int = field(default=1024, metadata={"help": "Batch size for evaluation"})

    # ==========================================================
    # [CUSTOM] Data Config
    # ==========================================================
    dataset_mode: str = field(default="on_the_fly", metadata={"help": "Dataset loading mode: 'on_the_fly' or 'memmap'"})
    csv_column_name: Optional[Dict[str, str]] = field(default=None, metadata={"help": "Column names for source and target in CSV files (used if mode is 'on_the_fly')"})
    token_file_path: Optional[str] = field(default=None, metadata={"help": "Path to token file"})

    # Curriculum Lists (HF only takes single int for batch_size)
    curriculum_batch_sizes: Optional[List[int]] = field(default=None, metadata={"help": "List of batch sizes for curriculum learning"})
    curriculum_accumulate_steps: Optional[List[int]] = field(default=None, metadata={"help": "List of accumulation steps for curriculum learning"})

    # Raw Configs for DataHandler
    train_data_config_list: Optional[List[Dict[str, Any]]] = field(default=None, metadata={"help": "List of training data configurations"})
    valid_data_config: Optional[Dict[str, Any]] = field(default=None, metadata={"help": "Validation data configuration"})
    eval_data_config: Optional[Dict[str, Any]] = field(default=None, metadata={"help": "Evaluation data configuration"})


# load two config from yaml
class _Section(dict):
    """One YAML mapping, looked up without regard to case and recording what was read.

    Two failure modes motivated this. Key case is inconsistent in the schema
    (``Transformer`` beside ``common`` and ``latent``), and a key the loader does not
    recognise -- wrong case, a typo, a plural -- was simply never read, leaving the
    setting at its default with nothing said. Writing ``transformer`` instead of
    ``Transformer`` silently trained the default 8-layer, 512-dim architecture.

    Lookups here are case-insensitive, and every key that is read is remembered so
    :func:`_report_unread_keys` can object to the ones that were not.
    """

    def __init__(self, data, path, registry):
        super().__init__(data)
        self.path = path
        self._by_lower = {}
        for key in data:
            self._by_lower.setdefault(str(key).lower(), str(key))
        self._read = set()
        self.asked: list = []
        registry.append(self)

    def get(self, key, default=None):
        self.asked.append(str(key))
        actual = self._by_lower.get(str(key).lower())
        if actual is None:
            return default
        self._read.add(actual)
        return dict.__getitem__(self, actual)

    def mark_read(self, *keys):
        """Records a key as intentionally consumed elsewhere."""
        for key in keys:
            actual = self._by_lower.get(str(key).lower())
            if actual is not None:
                self._read.add(actual)

    def unread(self):
        return [k for k in self if k not in self._read]


def _section(parent, key, path, registry=None):
    """Reads a nested YAML mapping, naming the key when it is not one.

    A scalar where a mapping belongs (``rope: true`` instead of
    ``rope: {encoder: true}``) otherwise surfaces as
    ``AttributeError: 'bool' object has no attribute 'get'`` from whichever line
    happens to read it first, which names neither the key nor the file.

    Args:
        parent (dict): The enclosing mapping.
        key (str): Key to read.
        path (str): Dotted path used in the error message.
        registry (list, optional): Collects the sections for the unread-key report.

    Returns:
        _Section | dict: The section, empty when the key is absent.

    Raises:
        ValueError: If the key is present but is not a mapping.
    """
    value = parent.get(key, {})
    if value is None:
        value = {}
    if not isinstance(value, dict):
        raise ValueError(
            f"{path} must be a mapping of settings, got {type(value).__name__} "
            f"({value!r}). Check the indentation under this key."
        )
    if registry is None:
        return value
    return _Section(value, path, registry)


def _report_unread_keys(registry, yaml_path):
    """Raises if the YAML carries a key the loader never looked at.

    An unrecognised key is not harmless: the setting it was meant to carry stays at
    its default and the run proceeds, so a typo costs a whole training run before
    anyone notices the architecture is not the one that was asked for.

    Args:
        registry (list): The sections created during this load.
        yaml_path (str): Path quoted in the error message.

    Raises:
        ValueError: Listing every unread key with the closest valid name.
    """
    problems = []
    for section in registry:
        unread = section.unread()
        if not unread:
            continue
        # Suggest from the keys this section's defaults define, which is what the
        # loader asked for and did not find.
        candidates = sorted(set(section.asked) - set(unread))
        for key in unread:
            close = difflib.get_close_matches(str(key).lower(),
                                              [c.lower() for c in candidates],
                                              n=1, cutoff=0.6)
            hint = ""
            if close:
                match = next(c for c in candidates if c.lower() == close[0])
                hint = f" Did you mean {match!r}?"
            where = f"{section.path}.{key}" if section.path else str(key)
            problems.append(f"  {where} was never read.{hint}")
    if problems:
        raise ValueError(
            f"{yaml_path} contains settings this version does not recognise, so they "
            "would have been silently ignored:\n" + "\n".join(problems)
        )


def _validate_paths(training_args, yaml_path):
    """Raises if a path the run cannot proceed without is unset or missing.

    Without this the first symptom of an unfilled ``sample/config.yml`` is a
    ``TypeError`` raised several frames deep inside the tokenizer, which does not
    say which key is empty.

    Args:
        training_args (TrainArgs): The parsed training arguments.
        yaml_path (str): Path to the YAML file, quoted in the error message.

    Only the vocabulary is checked here. Whether train, valid or eval data is
    required depends on the mode, which ``load_config`` does not know, so those
    paths are checked by the dataset builders that need them.

    Raises:
        ValueError: If ``token_file_path`` is unset or does not exist.
    """
    if not training_args.token_file_path:
        raise ValueError(
            f"training.data.token_file_path is not set in {yaml_path}. "
            "Point it at a vocabulary file, one token per line "
            "(e.g. sample/data/normal_tokens.txt)."
        )
    if not os.path.exists(training_args.token_file_path):
        raise ValueError(
            f"training.data.token_file_path points to {training_args.token_file_path!r}, "
            f"which does not exist (from {yaml_path})."
        )


def load_config(yaml_path: str, output_dir: str) -> Tuple[ModelConfig, TrainArgs]:
    """Loads a YAML config file and constructs ModelConfig and TrainArgs.

    Args:
        yaml_path (str): Path to the YAML configuration file.
        output_dir (str): Directory for saving checkpoints and logs; passed as
                          ``output_dir`` to TrainArgs.

    Returns:
        Tuple[ModelConfig, TrainArgs]:
            - config: Model architecture configuration.
            - training_args: Training hyperparameters and data configuration.
    """
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    # ==========================================================
    # 1. Parsing Model Config
    # ==========================================================
    sections: list = []
    cfg = _Section(cfg, "", sections)
    m_sec = _section(cfg, "model", "model", sections)
    m_common = _section(m_sec, "common", "model.common", sections)

    # loss section
    loss_sec = _section(m_sec, "loss", "model.loss", sections)

    # Transformer-specific section
    m_spec = _section(m_sec, "Transformer", "model.Transformer", sections)

    # Extract nested sections (Transformer specifics)
    m_rope = _section(m_spec, "rope", "model.Transformer.rope", sections)
    m_latent = _section(m_spec, "latent", "model.Transformer.latent", sections)

    config = ModelConfig(
        # --- Common ---
        vocab_size=m_common.get("vocab_size", 185),
        dropout=m_common.get("dropout", 0.2),
        embedding_dim=m_common.get("embedding_dim", 512),
        tanh_latent=m_common.get("tanh_latent", False),
        latent_noising=m_common.get("latent_noising", False),
        vae=m_sec.get("use_vae", False),

        # --- loss ---
        beta=loss_sec.get("beta", 0.01),
        beta_warmup_steps=loss_sec.get("beta_warmup_steps", 0),

        # --- Transformer Specific ---
        n_positions=m_spec.get("n_positions", 2048),
        n_layer=m_spec.get("n_layer", 8),
        n_head=m_spec.get("n_head", 8),
        bias=m_spec.get("bias", True),
        weight_sharing=m_spec.get("weight_sharing", False),
        layer_norm_epsilon=float(m_spec.get("layer_norm_epsilon", 1e-5)),
        gpt2style=m_spec.get("gpt2style", False),

        # RoPE
        encoder_rope=m_rope.get("encoder", True),
        decoder_rope=m_rope.get("decoder", True),
        base_freq=m_rope.get("base_freq", 10000),  # YAML key: model.Transformer.rope.base_freq

        # Latent
        pooling_mode=m_latent.get("pooling_mode", "MHA"),
        cross_attn=m_latent.get("cross_attn", None),
        latent_conditioning=m_latent.get("latent_conditioning", "add_once"),
    )

    # ==========================================================
    # 2. Training Arguments Parsing
    # ==========================================================
    sys_sec = _section(cfg, "system", "system", sections)
    log_sec = _section(cfg, "logging", "logging", sections)
    wandb_sec = _section(log_sec, "wandb", "logging.wandb", sections)
    tqdm_sec = _section(log_sec, "tqdm", "logging.tqdm", sections)
    t_sec = _section(cfg, "training", "training", sections)

    # Sub-sections
    strat_sec = _section(t_sec, "strategy", "training.strategy", sections)
    optim_sec = _section(t_sec, "optimizer", "training.optimizer", sections)
    zclip_sec = _section(t_sec, "zclip", "training.zclip", sections)
    # "chekpoint" is a misspelling that shipped in older config files; kept so they load.
    t_sec.mark_read("chekpoint", "checkpoint")
    ckpt_sec = _section(t_sec, "chekpoint", "training.chekpoint", sections) \
        or _section(t_sec, "checkpoint", "training.checkpoint", sections)
    valid_sec = _section(t_sec, "validation", "training.validation", sections)
    eval_sec = _section(t_sec, "evaluation", "training.evaluation", sections)
    data_sec = _section(t_sec, "data", "training.data", sections)

    # Complex Logic Extraction
    strat_type = strat_sec.get("type", "normal")
    # All three strategy blocks may be present; only the selected one is checked for
    # unrecognised keys, because the other two are deliberately inert.
    strat_sec.mark_read("normal", "phased", "dynamic")
    specs = {
        name: _section(strat_sec, name, f"training.strategy.{name}",
                       sections if name == strat_type else None)
        for name in ("normal", "phased", "dynamic")
    }
    strat_spec = specs.get(strat_type) or _section(
        strat_sec, strat_type, f"training.strategy.{strat_type}", sections)
    phased_spec = specs["phased"]
    dynamic_spec = specs["dynamic"]

    num_epochs = float(strat_sec.get("num_epochs", 50))
    max_steps = strat_spec.get("max_steps", -1)

    if strat_type == "dynamic" and (not isinstance(max_steps, int) or max_steps <= 0):
        raise ValueError(
            f"training.strategy.dynamic.max_steps must be a positive integer, got {max_steps!r}. "
            "The dynamic sampler pre-calculates per-epoch step counts from this value; "
            "without it, steps-per-epoch = -1 // num_epochs ≤ 0 and training silently produces no steps."
        )

    optim_type = optim_sec.get("type", "schedulefree")
    if optim_type not in SUPPORTED_OPTIMIZERS:
        raise ValueError(
            f"training.optimizer.type must be one of {sorted(SUPPORTED_OPTIMIZERS)}, got {optim_type!r}. "
            "The name is matched exactly, both to pick the nested settings block below and in "
            "HF_Trainer.create_optimizer; an unrecognised spelling silently falls back to the "
            "HuggingFace default optimizer with no warmup."
        )
    optim_sec.mark_read("schedulefree", "AdamW", "adamw")
    sched_spec = _section(optim_sec, optim_type, f"training.optimizer.{optim_type}", sections)
    es_spec = _section(strat_spec, "early_stopping", f"training.strategy.{strat_type}.early_stopping", sections)

    # Determine Start Batch Size
    bs_list = data_sec.get("batch_size", [1024])
    if isinstance(bs_list, int):
        bs_list = [bs_list]
    train_bs = bs_list[0] if isinstance(bs_list, list) and len(bs_list) > 0 else 1024

    acc_list = data_sec.get("accumulate_steps", [1])
    if isinstance(acc_list, int):
        acc_list = [acc_list]
    grad_acc = acc_list[0] if isinstance(acc_list, list) and len(acc_list) > 0 else 1

    # ----------------------------------------------------------
    # GROUP A: Hugging Face Native Arguments
    # ----------------------------------------------------------
    hf_args = {
        "output_dir": output_dir,

        # System
        "seed": sys_sec.get("seed", 42),
        "dataloader_num_workers": sys_sec.get("num_workers", 2),
        # Off / torch's default unless set. persistent_workers keeps the workers
        # (and their open memmaps) across epochs instead of re-forking them every
        # epoch; prefetch_factor is batches queued per worker. Both need workers.
        "dataloader_persistent_workers": sys_sec.get("persistent_workers", False),
        "dataloader_prefetch_factor": sys_sec.get("prefetch_factor", None),
        "bf16": sys_sec.get("use_bf16", False),

        # Logging
        "disable_tqdm": not tqdm_sec.get("enabled", True),
        "report_to": "wandb" if wandb_sec.get("enabled", False) else "none",
        "run_name": wandb_sec.get("exp_name", None),
        "logging_steps": log_sec.get("log_interval_steps", 100),

        # Training Loop Control
        "num_train_epochs": num_epochs,
        "max_steps": max_steps,

        # Batch Size & Accumulation
        "per_device_train_batch_size": train_bs,
        "gradient_accumulation_steps": grad_acc,

        # Optimization (Standard)
        "learning_rate": optim_sec.get("max_lr", 3e-4),
        "weight_decay": optim_sec.get("weight_decay", 0.0),

        # Scheduler (Warmup for AdamW only, handled by ScheduleFree otherwise)
        "warmup_steps": _section(sched_spec, "cosinedecay", f"training.optimizer.{optim_type}.cosinedecay", sections).get("warmup_steps", 0) if optim_type == "AdamW" else 0,

        # Checkpointing
        "save_only_model": t_sec.get("save_only_model", False),
        "save_strategy": "steps",
        "save_steps": ckpt_sec.get("step", 5000),
        "save_total_limit": ckpt_sec.get("limit", None),

        # Validation
        "do_eval": valid_sec.get("enabled", False),
        "eval_strategy": "steps", # Enforce steps
        "eval_steps": valid_sec.get("interval_steps", 100),
        "per_device_eval_batch_size": valid_sec.get("batch_size", 1024),

        # EarlyStopping
        "load_best_model_at_end": es_spec.get("enabled", False),
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
    }

    # ----------------------------------------------------------
    # GROUP B: PanCORE Custom Arguments
    # ----------------------------------------------------------
    custom_args = {
        # System environment (normal vs hpc)
        "environment": sys_sec.get("environment", "normal"),
        # WandB Extras
        "wandb_enabled": wandb_sec.get("enabled", False),
        "wandb_api_key_path": wandb_sec.get("api_key_path", None),
        "wandb_project_name": wandb_sec.get("project_name", ""),
        "wandb_exp_name": wandb_sec.get("exp_name", ""),
        "wandb_exp_id": wandb_sec.get("exp_id", ""),

        # Strategy & Init
        "init_model_path": t_sec.get("init_model_path", None),
        "training_strategy_type": strat_type,

        # Normal / Phased
        "phased_replay_ratio": phased_spec.get("replay_ratio", 0.0),
        "phased_phases": phased_spec.get("phases", 5),
        "phased_phase_epochs": phased_spec.get("phase_epochs", 10),

        # Dynamic Strategy
        "dynamic_phases": dynamic_spec.get("phases", 1),
        "dynamic_start_probs": dynamic_spec.get("start_sampling_probabilities", []),
        "dynamic_end_probs": dynamic_spec.get("end_sampling_probabilities", []),
        "dynamic_update_last_epoch": dynamic_spec.get("sampling_update_last_epoch", 50),
        "dynamic_interpolation_type": dynamic_spec.get("sampling_interpolation_type", "sigmoid"),
        "dynamic_sigmoid_steepness": _section(dynamic_spec, "sigmoid", "training.strategy.dynamic.sigmoid").get("steepness", 12.0),
        "dynamic_sigmoid_delay": _section(dynamic_spec, "sigmoid", "training.strategy.dynamic.sigmoid").get("delay_factor", 0.8),

        # Optimizer Extras
        "optimizer_type": optim_type,
        "schedulefree_r": sched_spec.get("r", 0.5), # if schedulefree

        # Scheduler Extras (Cosine)

        # Early Stopping (Custom implementation)
        "early_stopping_enabled": es_spec.get("enabled", False),
        "early_stopping_patience": es_spec.get("patience", 10000),
        "early_stopping_delta": es_spec.get("delta", 0.0),

        # Z-Clip
        "zclip_enabled": zclip_sec.get("enabled", False),
        "zclip_alpha": zclip_sec.get("alpha", 0.97),
        "zclip_thresh": zclip_sec.get("z_thresh", 2.5),
        "zclip_warmup_steps": zclip_sec.get("warmup_steps", 25),
        "zclip_clip_factor": zclip_sec.get("clip_factor", 1.0),

        # Evaluation (Generation)
        "evaluation_batch_size": eval_sec.get("batch_size", 1024),
        "evaluation_maxlen": eval_sec.get("maxlen", 500),

        # Data Configs
        "dataset_mode": data_sec.get("mode", "on_the_fly"),
        "csv_column_name": data_sec.get("csv_column_name", {}),
        "curriculum_batch_sizes": bs_list,
        "curriculum_accumulate_steps": acc_list,
        "token_file_path": data_sec.get("token_file_path") or "",

        # Raw Config Lists
        "train_data_config_list": data_sec.get("train_data", []),
        "valid_data_config": data_sec.get("valid_data", {}),
        "eval_data_config": data_sec.get("eval_data", {})
    }

    # ==========================================================
    # 3. Combine & Initialize
    # ==========================================================
    # Merging both dictionaries to initialize the class
    training_args = TrainArgs(**hf_args, **custom_args)
    _report_unread_keys(sections, yaml_path)
    _validate_paths(training_args, yaml_path)

    # S-T hands the decoder the encoder's hidden states and never pools a latent, so
    # there is no distribution for the VAE head to parameterise. The combination
    # currently dies several frames later on a tensor-shape mismatch.
    if config.beta_warmup_steps < 0:
        raise ValueError(
            f"model.loss.beta_warmup_steps must be >= 0, got {config.beta_warmup_steps}"
        )

    if config.vae and config.cross_attn == "S-T":
        raise ValueError(
            "model.use_vae is true and Transformer.latent.cross_attn is 'S-T', which "
            "cannot be combined: S-T feeds the decoder the encoder hidden states "
            "directly, so no latent is pooled and there is nothing for the VAE to "
            "sample. Use 'L-T' or leave cross_attn blank."
        )

    if training_args.environment == "hpc":
        localdir = os.environ.get("LOCALDIR")
        if localdir is None:
            raise ValueError("LOCALDIR is not set in HPC environment")

        def rewrite_path(cfg):
            """Repoints one data entry at $LOCALDIR, keeping only the basename."""
            if cfg and "path" in cfg:
                cfg["path"] = os.path.join(localdir, os.path.basename(cfg["path"]))
            return cfg

        # train
        training_args.train_data_config_list = [
            rewrite_path(cfg) for cfg in training_args.train_data_config_list
        ]

        # valid
        training_args.valid_data_config = rewrite_path(training_args.valid_data_config)

        # eval
        training_args.eval_data_config = rewrite_path(training_args.eval_data_config)

    return config, training_args
