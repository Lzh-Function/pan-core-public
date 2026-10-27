"""Command-line entry point and the four pipeline functions.

Each mode is a plain function (``run_training``, ``run_generation_evaluation``,
``run_accuracy_progress``, ``get_hidden_outputs``) that can also be imported and
called directly; :func:`main` only parses arguments and dispatches to them.

Logging is configured in :func:`main`, never on import, so importing this module
does not change the host application's logging.
"""
import logging
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import set_seed

from pan_core.common.data_handler import (
    GeneralDataCollator,
    build_eval_dataset,
    build_latent_dataset,
    build_train_dataset,
    build_valid_dataset,
)
from pan_core.common.tokenizer import SMILESTokenizer
from pan_core.common.utils import attach_run_log, log_model_summary, move_to_cpu
from pan_core.config import ModelConfig, TrainArgs, load_config
from pan_core.models import PanCoreTransformerModel
from pan_core.train import HF_Trainer

logger = logging.getLogger(__name__)


def configure_cli_logging(level=logging.DEBUG):
    """Sends log records to stdout. Called by the CLI, never on import.

    Importing a library must not reconfigure the host application's logging, so
    this is kept out of module scope: ``logging.basicConfig`` sets the level and
    handler of the *root* logger, which would make every other library in the
    caller's process emit at the same level. Callers using pan_core as a library
    configure logging themselves, or call this helper explicitly.

    Args:
        level (int, optional): Root log level. Defaults to ``logging.DEBUG``.
    """
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level,
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def main():
    """Parses the command line and runs the selected mode.

    Exactly one mode flag is expected; ``--output_dir`` defaults to
    ``<config dir>/output/``.
    """
    import argparse
    # The collator pads each batch to its own maximum length, so the caching
    # allocator sees thousands of block shapes across the five buckets and cannot
    # reuse freed blocks; with validation on top it reserved 92-94 of 95 GiB on a
    # GH200 against about 41 GiB actually allocated. Expandable segments bring the
    # reservation down to the allocation (40.6 GiB, same run) and change no
    # computed value. Read when CUDA first allocates, so it has to be set before
    # that; an explicit PYTORCH_CUDA_ALLOC_CONF from the caller wins.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    configure_cli_logging()
    parser = argparse.ArgumentParser(description="PanCORE Experiment Runner")

    parser.add_argument("--config", type=str, required=True, help="Path to Config.yml")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for checkpoints/logs/results")

    # Mode flags
    parser.add_argument("--do_train", action="store_true", help="Run training mode")
    parser.add_argument("--do_eval_gen", action="store_true", help="Run evaluation generation mode")
    parser.add_argument("--get_hidden", action="store_true", help="Run model-forward to get all hidden outputs")
    parser.add_argument("--get_acc_progress", action="store_true", help="Run evaluation for all checkpoints, get accuracy progress")

    # Evaluation specific
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for evaluation or hidden output extracting")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Path to directory contained all checkpoints")
    # encode and generate
    parser.add_argument("--encode_only", action="store_true", help="Only run encoding to get latent vectors without decoding")
    parser.add_argument("--latent_decode", action="store_true", help="Decode with latent vector arrays")
    parser.add_argument("--teacher_forcing", action="store_true", help="Run generation with teacher forcing")
    parser.add_argument("--csv_output", type=str, default=None, help="Output file for generation results")
    parser.add_argument("--encode_output", type=str, default=None, help="Output file for encode results (latents arrays) if not 'latent_decode'")
    parser.add_argument("--need_logits", action="store_true", help="Saving all output logits when generating")
    parser.add_argument("--logits_output", type=str, default=None)
    # get_hidden_outputs
    parser.add_argument("--hidden_output", type=str, default=None, help="Output file for all hidden states (dict)")

    args = parser.parse_args()

    if args.get_acc_progress and args.checkpoint_dir is None:
        # Without it run_accuracy_progress would list the current directory.
        parser.error("--get_acc_progress requires --checkpoint_dir")

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.config), "output")
    if args.checkpoint is None:
        args.checkpoint = os.path.join(args.output_dir, "final_model")
    if args.csv_output is None:
        args.csv_output = os.path.join(args.output_dir, "gen_results.csv")
    if args.encode_output is None:
        args.encode_output = os.path.join(args.output_dir, "encode_latents.npy")
    if args.logits_output is None:
        args.logits_output = os.path.join(args.output_dir, "gen_output_logits.npy")
    if args.hidden_output is None:
        args.hidden_output = os.path.join(args.output_dir, "hidden_output.pkl")

    os.makedirs(args.output_dir, exist_ok=True)

    attach_run_log(args.output_dir)

    if args.do_train:
        run_training(
            config_path=args.config,
            output_dir=args.output_dir
        )

    if args.do_eval_gen:
        run_generation_evaluation(
            config=args.config,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint,
            encode_only=args.encode_only,
            latent_decode=args.latent_decode,
            teacher_forcing=args.teacher_forcing,
            output_file=args.csv_output,
            encode_output_file=args.encode_output,
            need_logits=args.need_logits,
            logits_output_file=args.logits_output
        )

    if args.get_acc_progress:
        run_accuracy_progress(
            config_path=args.config,
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
        )

    if args.get_hidden:
        get_hidden_outputs(
            config=args.config,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint,
            output_file=args.hidden_output
        )

    if not any([args.do_train, args.do_eval_gen, args.get_acc_progress, args.get_hidden]):
        logger.warning("No mode specified. Use --do_train, --do_eval_gen, --get_acc_progress, or --get_hidden.")


def run_training(config_path: str, output_dir: str):
    """Executes the full training pipeline: load config → tokenizer → model → data → train → save.

    Args:
        config_path (str): Path to the YAML configuration file.
        output_dir (str): Directory for checkpoints, logs, and the final saved model.
    """
    logger.info(f"Loading configuration from {config_path}...")
    model_config, training_args = load_config(config_path, output_dir)

    # 1. Set Seed
    set_seed(training_args.seed)

    # 2. Tokenizer
    tokenizer = SMILESTokenizer(vocab_file=training_args.token_file_path)
    logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    # Update config vocab size just in case
    model_config.vocab_size = tokenizer.vocab_size

    # 3. Model Initialization
    logger.info("Initializing Model...")
    if training_args.init_model_path:
        logger.info(f"Loading weights from {training_args.init_model_path}")
        model = PanCoreTransformerModel.from_pretrained(training_args.init_model_path, config=model_config)
    else:
        logger.info("Initializing from scratch")
        model = PanCoreTransformerModel(model_config)
    log_model_summary(model, logger)


    # 4. Data Preparation
    logger.info("Building Datasets...")
    train_dataset, num_data_list = build_train_dataset(training_args, tokenizer)
    for i, d in enumerate(training_args.train_data_config_list):
        d["num_data"] = num_data_list[i]
    valid_dataset = build_valid_dataset(training_args, tokenizer)

    data_collator = GeneralDataCollator(pad_token_id=model_config.pad_token_id)

    # 5. Trainer Initialization
    logger.info("Initializing Trainer...")
    trainer = HF_Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
    )

    # 6. Start Training
    logger.info("Starting Training...")

    # Handle Resume logic
    resume_from_checkpoint = None
    if os.path.isdir(output_dir):
        # Check for existing checkpoints
        from transformers.trainer_utils import get_last_checkpoint
        last_ckpt = get_last_checkpoint(output_dir)
        if last_ckpt:
            logger.info(f"Found checkpoint: {last_ckpt}. Resuming.")
            resume_from_checkpoint = last_ckpt
        elif training_args.training_strategy_type == "phased":
             # For phased, Trainer handles complex resume logic via boolean/path
             resume_from_checkpoint = output_dir

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # 7. Final Save
    logger.info("Saving final model...")
    trainer.save_model(os.path.join(output_dir, "final_model"))
    tokenizer.save_vocabulary(os.path.join(output_dir, "final_model"))


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    """Moves all tensors in a batch dict to *device*; non-tensor values pass through unchanged."""
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _concat_ragged_logits(chunks: list[np.ndarray]) -> np.ndarray:
    """Joins per-batch logits ``[B, L, V]`` along the batch axis.

    The collator pads each batch to its own longest sequence, so ``L`` differs
    between chunks. Every chunk is right-padded with zeros up to the longest ``L``
    before the join, which reproduces what an incremental pad-and-concatenate
    produces -- without re-copying the accumulated array once per batch.

    The pad keeps each chunk's own dtype on purpose: a bare ``np.zeros`` is
    float64 and would promote the whole result, doubling the saved .npy.
    """
    max_len = max(c.shape[1] for c in chunks)
    padded = []
    for c in chunks:
        if c.shape[1] < max_len:
            pad = np.zeros((c.shape[0], max_len - c.shape[1], c.shape[2]), dtype=c.dtype)
            c = np.concatenate([c, pad], axis=1)
        padded.append(c)
    return np.concatenate(padded, axis=0)


def run_generation_evaluation(
        config: str|tuple[ModelConfig, TrainArgs],
        output_dir: str,
        checkpoint_path: str,
        encode_only: bool,
        latent_decode: bool,
        teacher_forcing: bool = False,
        output_file: str | None = None,
        encode_output_file: str | None = None,
        need_logits: bool = False,
        logits_output_file: str | None = None
    ):
    """Runs greedy decoding evaluation and saves generated sequences.

    Args:
        config (str | Tuple[ModelConfig, TrainArgs]): Path to the YAML config file, or a
            pre-loaded (ModelConfig, TrainArgs) tuple.
        output_dir (str): Directory for logs and intermediate outputs.
        checkpoint_path (str): Path to the model checkpoint to load.
        encode_only (bool): If True, only run the encoder to produce latent vectors
            without decoding.
        latent_decode (bool): If True, load pre-computed latent arrays from
            ``eval_data_config["path"]`` and decode them directly (skips encoding).
        teacher_forcing (bool, optional): If True, use teacher-forced forward pass
            instead of autoregressive greedy decoding. Defaults to False.
        output_file (str, optional): CSV file path for saving generation results.
            Defaults to None (results returned but not written).
        encode_output_file (str, optional): .npy file path for saving latent arrays.
            Defaults to None.
        need_logits (bool, optional): If True, also collect and save full output logits.
            Defaults to False.
        logits_output_file (str, optional): .npy file path for saving logits.
            Defaults to None.

    Returns:
        Tuple[pd.DataFrame | None, TrainArgs]: A DataFrame with columns
            ``[judge, input, generated, target]`` when not ``encode_only`` (else None),
            and the resolved ``TrainArgs`` used during evaluation.
    """
    if isinstance(config, (str, os.PathLike)):
        config_path = config
        logger.info(f"Loading configuration from {config_path}...")
        # We use a dummy output dir for loading args, as we won't train
        model_config, training_args = load_config(config_path, output_dir)
    else:
        model_config, training_args = config

    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 2. Tokenizer
    tokenizer = SMILESTokenizer(vocab_file=training_args.token_file_path)
    logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    # 3. Load Model
    logger.info(f"Loading model from {checkpoint_path}...")
    try:
        model = PanCoreTransformerModel.from_pretrained(checkpoint_path)
    except OSError as e:
        raise ValueError(f"Could not load model from checkpoint path: {e}") from e

    model.to(device)
    model = torch.compile(model, mode="reduce-overhead", fullgraph=True, dynamic=False)
    model.eval()

    # 4. Load Evaluation Data
    logger.info("Loading Evaluation Dataset...")
    if not latent_decode:
        eval_dataset = build_eval_dataset(training_args, tokenizer, maxlen=training_args.evaluation_maxlen)
    else:
        eval_dataset = build_latent_dataset(training_args.eval_data_config["path"])

    if eval_dataset is None:
        raise ValueError("Evaluation dataset not found in config.")
    total_samples = len(eval_dataset)

    # Create simple dataloader
    # Evaluation batch size usually can be larger, or defined in args
    batch_size = training_args.evaluation_batch_size
    collate_fn = GeneralDataCollator(pad_token_id=model_config.pad_token_id)
    dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=training_args.dataloader_num_workers
    )

    # 5. Generation Loop
    logger.info("Starting Generation...")
    results = {"judge": [], "input": [], "generated": [], "target": []} if not encode_only else None
    # Collected per batch and joined once after the loop. Concatenating inside the
    # loop re-copies everything accumulated so far on every batch, so the total
    # copy volume grows with the square of the batch count.
    latent_chunks: list[np.ndarray] = []
    logits_chunks: list[np.ndarray] = []
    logits: torch.Tensor | None = None

    use_bf16 = training_args.bf16 and torch.cuda.is_bf16_supported()

    with torch.no_grad():
        with tqdm(total=total_samples, desc="Generating", unit="sample") as pbar:
            for batch in dataloader:

                batch = _batch_to_device(batch, device)
                latents = batch.get("latents")
                input_ids = batch.get("input_ids")
                encoder_attention_mask = batch.get("encoder_attention_mask")
                labels = batch.get("labels")
                decoder_attention_mask = batch.get("decoder_attention_mask")

                current_batch_size = input_ids.size(0) if input_ids is not None else latents.size(0)
                logits = None

                if encode_only:
                    latents = model.encode(
                        input_ids=input_ids,
                        encoder_attention_mask=encoder_attention_mask,
                        use_bf16=use_bf16
                    )
                else:
                    if not teacher_forcing:
                        generated_ids, latents, logits = model.generate(
                            input_ids=input_ids,
                            encoder_attention_mask=encoder_attention_mask,
                            latents=latents,
                            max_length=training_args.evaluation_maxlen,
                            use_bf16=use_bf16,
                            need_logits=need_logits
                        )
                    else:
                        latents, logits = model.forward(
                            input_ids=input_ids,
                            encoder_attention_mask=encoder_attention_mask,
                            labels=labels,
                            decoder_attention_mask=decoder_attention_mask,
                            latents=latents,
                            calculate_loss=False,
                            need_hidden=False,
                            return_dict=False
                        )
                        generated_ids = logits.argmax(dim=-1)  # [B, L]
                        # Vectorized EOS masking: replace tokens after the first EOS with pad.
                        eos_mask = generated_ids == tokenizer.eos_token_id           # [B, L]
                        has_eos = eos_mask.any(dim=1)                                # [B]
                        first_eos = eos_mask.int().argmax(dim=1)                     # [B]; 0 when no EOS
                        seq_len = generated_ids.size(1)
                        first_eos = torch.where(has_eos, first_eos, torch.full_like(first_eos, seq_len))
                        positions = torch.arange(seq_len, device=generated_ids.device).unsqueeze(0)  # [1, L]
                        generated_ids = generated_ids.masked_fill(positions > first_eos.unsqueeze(1), tokenizer.pad_token_id)

                        # Zero the logits at padded positions before they are saved.
                        # The decoder scores every position in the batch-max-length
                        # window, and roughly half of them are padding; nothing masks
                        # them on the way to gen_output_logits.npy, so the file has
                        # always carried whatever attention happened to produce there.
                        # Those values are not a prediction about anything, and they
                        # move when the attention kernel does -- measured on 2,048
                        # PubChem molecules, 37% of pad positions change their argmax
                        # between the masked and the is_causal=True kernel, against 0
                        # of 91,542 real positions. Zeroing them makes the saved array
                        # depend only on the real tokens, which is what every consumer
                        # reads, and matches the zeros _concat_ragged_logits already
                        # writes where a short batch is padded up to the longest one.
                        if decoder_attention_mask is not None:
                            logits = logits * decoder_attention_mask[:, 1:logits.size(1) + 1].unsqueeze(-1)

                    # Decode tokens to strings — generated_ids: [B, L].
                    # Each tensor is pulled to the host once, before the loop; indexing
                    # a device tensor inside it costs one synchronisation per sample.
                    generated_ids_cpu = generated_ids.cpu().tolist()
                    input_ids_cpu = input_ids.cpu().tolist() if input_ids is not None else None
                    labels_cpu = labels.cpu().tolist() if labels is not None else None
                    for i, ids in enumerate(generated_ids_cpu):
                        decoded_text = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                        results["generated"].append(decoded_text)
                        if input_ids_cpu is not None:
                            input_text = tokenizer.decode(input_ids_cpu[i], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                            results["input"].append(input_text)
                        if labels_cpu is not None:
                            label_text = tokenizer.decode(labels_cpu[i], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                            results["target"].append(label_text)
                            results["judge"].append(label_text == decoded_text)

                if latents is not None:
                    latent_chunks.append(move_to_cpu(latents))

                if logits is not None:
                    logits_chunks.append(move_to_cpu(logits))

                pbar.update(current_batch_size)

    all_latents = np.concatenate(latent_chunks, axis=0) if latent_chunks else None
    all_logits = _concat_ragged_logits(logits_chunks) if logits_chunks else None

    # 6. Save Results
    if results is not None:
        results = pd.DataFrame({k: v for k, v in results.items() if len(v) > 0})
        if output_file is not None:
            logger.info(f"Saving results to {output_file}...")
            results.to_csv(output_file)

    if all_latents is not None and encode_output_file is not None:
        logger.info(f"Saving encode results (latent arrays) to {encode_output_file}...")
        logger.info(f"type: {type(all_latents)}, shape: {all_latents.shape}")
        np.save(encode_output_file, all_latents)

    if all_logits is not None and logits_output_file is not None:
        logger.info(f"Saving output logits to {logits_output_file}...")
        logger.info(f"type: {type(all_logits)}, shape: {all_logits.shape}")
        np.save(logits_output_file, all_logits)

    logger.info("Evaluation Complete.")

    return results, training_args

def get_hidden_outputs(config: str|tuple[ModelConfig, TrainArgs], output_dir: str, checkpoint_path: str, max_len: int|None=None, output_file: str | None=None, phase: int|None=None, max_phase: int|None=None):
    """Runs a forward pass over the evaluation dataset and collects all intermediate hidden states.

    Args:
        config (str | Tuple[ModelConfig, TrainArgs]): Path to the YAML config file, or a
            pre-loaded (ModelConfig, TrainArgs) tuple.
        output_dir (str): Directory for logs.
        checkpoint_path (str): Path to the model checkpoint to load.
        max_len (int, optional): Maximum sequence length to filter evaluation data.
            Defaults to None (no filtering).
        output_file (str, optional): .pkl file path for saving the hidden state dict.
            Defaults to None.
        phase (int, optional): Phase index (0-based) to select a subset of the evaluation data.
            Defaults to None.
        max_phase (int, optional): Total number of phases; used with ``phase`` to slice the data.
            Defaults to None.

    Returns:
        dict: Nested dict with keys ``"outputs"`` and ``"attention_scores"``, each holding
            an ``"encoder"`` and a ``"decoder"`` list indexed by Transformer layer. Each
            layer entry is itself a list of per-batch numpy arrays, in dataloader order.
            - outputs[side][layer][batch]: shape [L, B, E]  (sequence-first)
            - attention_scores[side][layer][batch]: shape [B, Num_H, L, L]

            Batches are kept apart because the collator pads each one to its own longest
            sequence, so ``L`` differs between them and no single array holds them all.
            Concatenate a layer yourself once you have decided how to reconcile ``L``.
    """
    if isinstance(config, (str, os.PathLike)):
        config_path = config
        logger.info(f"Loading configuration from {config_path}...")
        # We use a dummy output dir for loading args, as we won't train
        model_config, training_args = load_config(config_path, output_dir)
    else:
        model_config, training_args = config

    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 2. Tokenizer
    tokenizer = SMILESTokenizer(vocab_file=training_args.token_file_path)
    logger.info(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    # 3. Load Model
    logger.info(f"Loading model from {checkpoint_path}...")
    try:
        model = PanCoreTransformerModel.from_pretrained(checkpoint_path)
    except OSError as e:
        raise ValueError(f"Could not load model from checkpoint path: {e}") from e

    model.to(device)
    # torch.compile is intentionally skipped: fullgraph=True is incompatible with need_hidden=True
    # because intermediate tensor returns break the static graph assumption.
    model.eval()

    # 4. Load Evaluation Data
    logger.info("Loading Dataset...")
    eval_dataset = build_eval_dataset(training_args, tokenizer, maxlen=max_len, phase=phase, max_phase=max_phase)

    if eval_dataset is None:
        raise ValueError("Evaluation dataset not found in config.")
    total_samples = len(eval_dataset)

    # Create simple dataloader
    batch_size = training_args.evaluation_batch_size
    collate_fn = GeneralDataCollator(pad_token_id=model_config.pad_token_id)
    dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=training_args.dataloader_num_workers
    )

    logger.info("Starting forward pass...")

    # [store][side][layer] -> list of per-batch arrays; see this function's docstring
    # for the per-array layouts, which differ between the two stores.
    hiddens = {
        "outputs": {
            "encoder": [],
            "decoder": [],
            },
        "attention_scores": {
            "encoder": [],
            "decoder": [],
            },
    }

    with torch.no_grad():
        with tqdm(total=total_samples, desc="forward pass", unit="sample") as pbar:
            for batch in dataloader:

                batch = _batch_to_device(batch, device)
                latents = batch.get("latents")
                input_ids = batch.get("input_ids")
                encoder_attention_mask = batch.get("encoder_attention_mask")
                labels = batch.get("labels")
                decoder_attention_mask = batch.get("decoder_attention_mask")

                current_batch_size = input_ids.size(0) if input_ids is not None else latents.size(0)

                outputs = model.forward(
                    input_ids=input_ids,
                    encoder_attention_mask=encoder_attention_mask,
                    labels=labels,
                    decoder_attention_mask=decoder_attention_mask,
                    latents=latents,
                    need_hidden=True
                )

                for source, store in (("hidden_states", "outputs"), ("attentions", "attention_scores")):
                    for side, per_layer_arrays in outputs[source].items():
                        layers = hiddens[store][side]
                        if not layers:
                            layers.extend([] for _ in per_layer_arrays)
                        for i, array in enumerate(per_layer_arrays):
                            layers[i].append(array)

                pbar.update(current_batch_size)

    logger.info("All forward pass finished.")
    for store, layout in (("outputs", "[L, B, Hidden_dim]"), ("attention_scores", "[B, num_head, L, L]")):
        for side in ("encoder", "decoder"):
            layers = hiddens[store][side]
            logger.info(f"{store} ({side}): {len(layers)} layers x {len(layers[0]) if layers else 0} batches")
            if layers and layers[0]:
                logger.info(f"  shape (1st layer, 1st batch): {layers[0][0].shape}  # {layout}")
                logger.info(f"  shape (last layer, 1st batch): {layers[-1][0].shape}")

    if output_file is not None:
        logger.info(f"Saving the outputs to {output_file}...")
        with open(output_file, "wb") as f:
            pickle.dump(hiddens, f)

    logger.info("Complete.")

    return hiddens


def run_accuracy_progress(config_path: str, checkpoint_dir: str, output_dir: str) -> pd.DataFrame:
    """Evaluates all checkpoints in a directory and saves accuracy progress to CSV.

    Args:
        config_path (str): Path to the YAML configuration file.
        checkpoint_dir (str): Directory containing ``checkpoint-N`` subdirectories.
        output_dir (str): Directory for per-checkpoint CSV outputs and the final summary.

    Returns:
        pd.DataFrame: Progress table indexed by training step, with columns
            ``perfect_accuracy``, ``partial_accuracy``, and optionally
            ``phase{i}_data_perfect_accuracy`` for phased/dynamic training.
    """
    checkpoints = sorted(
        [c for c in os.listdir(checkpoint_dir) if c.startswith("checkpoint-")],
        key=lambda c: int(re.search(r"checkpoint-(\d+)", c).group(1))
    )
    logger.info(f"start get accuracy progress: found {len(checkpoints)} checkpoints in total.")

    progress = []
    for checkpoint in checkpoints:
        step = int(re.search(r"checkpoint-(\d+)", checkpoint).group(1))
        res_df, config_args = run_generation_evaluation(
            config=config_path,
            output_dir=output_dir,
            checkpoint_path=os.path.join(checkpoint_dir, checkpoint),
            encode_only=False,
            latent_decode=False,
            output_file=os.path.join(output_dir, f"gen_results_step{step}.csv"),
            encode_output_file=None,
            need_logits=False,
            logits_output_file=None
        )

        n_samples = len(res_df)
        perfect_accuracy = float(res_df["judge"].sum() / n_samples)

        partial_accuracies = []
        for _, row in res_df.iterrows():
            g, t = row["generated"], row["target"]
            matches = sum(g[i] == t[i] for i in range(min(len(g), len(t))))
            partial_accuracies.append(matches / max(len(g), len(t)))
        partial_accuracy = float(np.mean(partial_accuracies))

        row_data = {
            "steps": step,
            "perfect_accuracy": perfect_accuracy,
            "partial_accuracy": partial_accuracy,
        }
        if config_args.training_strategy_type in ["phased", "dynamic"]:
            n_phases = (
                config_args.phased_phases
                if config_args.training_strategy_type == "phased"
                else config_args.dynamic_phases
            )
            each_phase_samples = n_samples // n_phases
            for i in range(n_phases):
                phase_slice = res_df["judge"].iloc[each_phase_samples * i: each_phase_samples * (i + 1)]
                row_data[f"phase{i+1}_data_perfect_accuracy"] = float(phase_slice.sum() / each_phase_samples)

        progress.append(row_data)
        logger.info(f"step{step}: perfect accuracy {perfect_accuracy:.4f}")

    progress_df = pd.DataFrame(progress).set_index("steps").sort_index()
    output_csv = os.path.join(output_dir, "acc_progress.csv")
    progress_df.to_csv(output_csv, index=True)
    logger.info(f"progress saved to {output_csv}, all finished.")

    return progress_df


if __name__ == "__main__":
    main()
