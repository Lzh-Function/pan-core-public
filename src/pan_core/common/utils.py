"""Utilities shared by training, evaluation and the CLI.

Logging setup (``attach_run_log``), model summaries, the VAE KL loss and small
tensor helpers.
"""
import datetime
import logging
import os
import sys

import torch

PACKAGE_LOGGER_NAME = "pan_core"


def ensure_default_logging(level: int = logging.INFO) -> bool:
    """Makes progress messages visible when nothing else has configured logging.

    The long-running preprocessing functions report progress through the package
    logger. Under the CLI that is already wired up; called as a library from a
    plain script it is not, and without a handler the progress lines go nowhere.

    This attaches a stdout handler to the ``pan_core`` logger only -- never to the
    root logger -- and does nothing at all if the application or the CLI has
    configured logging already, so an explicit ``logging.basicConfig(level=WARNING)``
    is still respected.

    Args:
        level (int, optional): Level set on the package logger when this call is
            the one that configures it. Defaults to ``logging.INFO``.

    Returns:
        bool: True if a handler was added by this call.
    """
    pkg_logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    if pkg_logger.handlers or logging.getLogger().handlers:
        return False

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        "%m/%d/%Y %H:%M:%S"
    ))
    pkg_logger.addHandler(handler)
    if pkg_logger.level == logging.NOTSET:
        pkg_logger.setLevel(level)
    return True


def attach_run_log(output_dir: str) -> logging.FileHandler | None:
    """Routes every ``pan_core.*`` log record into ``<output_dir>/run.log``.

    The handler is attached to the package logger rather than to a single module's
    logger, so model, data and tokenizer messages reach the file too -- previously
    only ``pan_core.main`` and ``pan_core.train`` did, and everything else was
    console-only.

    Attaching is idempotent: a second call for the same file is a no-op, which
    matters because the module-level loggers outlive any one Trainer instance and
    a repeated attach would write every line twice.

    The file is opened in append mode, so a rerun into the same directory keeps the
    earlier log; a banner marks where the new run starts.

    Args:
        output_dir (str): Directory that receives ``run.log``. Created if absent.

    Returns:
        logging.FileHandler | None: The handler, or the existing one if already attached.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.abspath(os.path.join(output_dir, "run.log"))
    pkg_logger = logging.getLogger(PACKAGE_LOGGER_NAME)

    for handler in pkg_logger.handlers:
        if isinstance(handler, logging.FileHandler) and os.path.abspath(handler.baseFilename) == path:
            return handler

    file_handler = logging.FileHandler(path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        "%m/%d/%Y %H:%M:%S"
    ))
    started = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    file_handler.stream.write(f"\n{'=' * 30} run start {started} (pid {os.getpid()}) {'=' * 30}\n")
    file_handler.stream.flush()
    pkg_logger.addHandler(file_handler)
    return file_handler


def log_model_summary(model: torch.nn.Module, logger):
    """Logs the model architecture and parameter counts.

    Args:
        model (torch.nn.Module): The model to summarize.
        logger: A logger instance to write to.
    """
    logger.info("=" * 40)
    logger.info("Model Architecture:")
    logger.info(model)
    logger.info("=" * 40)

    def format_params(num):
        """Renders a parameter count with a B/M/K suffix."""
        if num >= 1e9:
            return f"{num / 1e9:.3f}B"
        elif num >= 1e6:
            return f"{num / 1e6:.3f}M"
        elif num >= 1e3:
            return f"{num / 1e3:.3f}K"
        else:
            return str(num)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    percent_trainable = 100 * trainable_params / all_params if all_params > 0 else 0.0

    msg = (
        f"\nModel Parameters Summary:\n"
        f"  - Trainable params: {format_params(trainable_params)}\n"
        f"  - All params:       {format_params(all_params)}\n"
        f"  - Trainable %:      {percent_trainable:.4f}%"
    )
    logger.info(msg)
    logger.info("=" * 40)


def KLLoss(mu, log_var, reduction="mean"):
    """Calculates the Kullback-Leibler (KL) divergence loss.

    Args:
        mu (torch.Tensor): Mean of the latent distribution. Shape: [B, E]
        log_var (torch.Tensor): Log variance of the latent distribution. Shape: [B, E]
        reduction (str, optional): Reduction method — "mean" | "sum" | "none". Defaults to "mean".

    Returns:
        torch.Tensor: KL divergence loss.
                      Shape [B] if reduction is "none", scalar otherwise.
    """
    kl_element = mu.pow(2) + log_var.exp() - log_var - 1  # [B, E]
    kl_per_sample = 0.5 * torch.sum(kl_element, dim=1)    # [B]

    if reduction == "mean":
        return torch.mean(kl_per_sample)
    elif reduction == "sum":
        return torch.sum(kl_per_sample)
    else:  # "none"
        return kl_per_sample


def kl_per_dimension(mu, log_var):
    """KL contribution of each latent dimension, averaged over the batch.

    Diagnoses posterior collapse, which the summed KL hides: a latent where five of
    512 dimensions carry everything and the rest match the prior exactly reports the
    same total as one where the signal is spread evenly.

    Args:
        mu (torch.Tensor): Mean of the latent distribution. Shape: [B, E]
        log_var (torch.Tensor): Log variance of the latent distribution. Shape: [B, E]

    Returns:
        torch.Tensor: Per-dimension KL in nats. Shape [E].
    """
    return (0.5 * (mu.pow(2) + log_var.exp() - log_var - 1)).mean(dim=0)


def move_to_cpu(t: torch.Tensor):
    """Detaches a tensor from the computation graph, moves it to CPU, and converts to numpy ndarray."""
    return t.detach().cpu().numpy()
