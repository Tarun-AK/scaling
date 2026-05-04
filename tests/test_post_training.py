from __future__ import annotations

import os
import tempfile

import jax
import numpy as np

from models.lstm import LSTMLanguageModel
from training.trainer import (
    _post_training_evals_and_checkpoint,
    _wandb_init,
    create_train_state,
)


class DummyConfig:
    hidden_dim = 4
    num_layers = 1
    seq_len = 8
    batch_size = 2
    learning_rate = 1e-3
    num_epochs = 1
    run_name = "unit_test"
    dataset_name = "wikitext"
    dataset_config = "wikitext-103-raw-v1"
    vocab_size = 16
    bos_token_id = 0
    log_every_n_steps = 1
    entropy_num_samples = 4
    entropy_sample_batch_size = 2
    entropy_every_n_epochs = 1
    wandb_project = "scaling"
    wandb_entity = None
    wandb_group = "unit_test"


def test_post_training_path_runs(tmp_path) -> None:
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_SILENT"] = "true"

    cfg = DummyConfig()
    cfg.checkpoint_dir = str(tmp_path / "checkpoints")
    cfg.results_dir = str(tmp_path / "results")

    model = LSTMLanguageModel(
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        vocab_size=cfg.vocab_size,
    )
    rng = jax.random.PRNGKey(0)
    state = create_train_state(model, cfg, rng)

    train_np = (np.arange(cfg.seq_len * 4) % cfg.vocab_size).reshape(4, cfg.seq_len)
    val_np = (np.arange(cfg.seq_len * 2) % cfg.vocab_size).reshape(2, cfg.seq_len)
    test_np = (np.arange(cfg.seq_len * 2) % cfg.vocab_size).reshape(2, cfg.seq_len)

    _wandb_init(cfg)
    metrics = _post_training_evals_and_checkpoint(
        state=state,
        model=model,
        config=cfg,
        rng=rng,
        train_np=train_np,
        val_np=val_np,
        test_np=test_np,
        global_step=0,
    )

    assert "ngram_1" in metrics
