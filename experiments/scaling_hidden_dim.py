"""Hydra entrypoint for hidden-dimension scaling experiments."""

from __future__ import annotations

import os

import hydra
from omegaconf import DictConfig, OmegaConf

from training.trainer import train_and_evaluate


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(config: DictConfig) -> None:
    """Run a single experiment.

    When invoked with Hydra multirun (e.g. using configs/sweep_hidden_dim.yaml),
    this will launch one run per hidden_dim value.

    Args:
        config: Hydra config.
    """

    # Ensure run name is set for W&B.
    config.run_name = f"lstm_hd{int(config.hidden_dim)}"

    # Make configurable paths relative to original working directory.
    orig_cwd = hydra.utils.get_original_cwd()

    def _resolve_path(value: str | None) -> str | None:
        if value is None:
            return None
        if os.path.isabs(value):
            return value
        return os.path.join(orig_cwd, value)

    config.checkpoint_dir = _resolve_path(str(config.checkpoint_dir))
    config.results_dir = _resolve_path(str(config.results_dir))
    config.cache_dir = _resolve_path(str(config.cache_dir))
    if "tokenizer_path" in config:
        config.tokenizer_path = _resolve_path(str(config.tokenizer_path))
    if "dataset_path" in config and config.dataset_path is not None:
        config.dataset_path = _resolve_path(str(config.dataset_path))

    train_and_evaluate(config)


if __name__ == "__main__":
    main()
