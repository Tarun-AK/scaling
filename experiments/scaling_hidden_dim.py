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

    # Make checkpoint/results dirs relative to original working directory.
    # Hydra changes cwd into a per-run directory.
    orig_cwd = hydra.utils.get_original_cwd()
    config.checkpoint_dir = os.path.join(orig_cwd, str(config.checkpoint_dir))
    config.results_dir = os.path.join(orig_cwd, str(config.results_dir))

    train_and_evaluate(config)


if __name__ == "__main__":
    main()
