# LSTM Scaling Experiments

A JAX/Flax implementation of LSTM language models for studying scaling laws. Trains on WikiText-103 and tracks metrics including per-position n-gram losses and conditional entropy via autoregressive sampling.

## Features

- **JIT-compiled training** with JAX/Flax
- **Per-position n-gram losses** computed in a single forward pass
- **Conditional entropy estimation** via sampling from pθ
- **Hydra** for configuration management
- **Weights & Biases** for experiment tracking
- **Orbax** for checkpointing

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
pip install -r requirements.txt

# Train the BPE tokenizer (required before first run)
python data/train_tokenizer.py
```

## Running Experiments

### Single experiment

```bash
python -m experiments.scaling_hidden_dim
```

Override config values via command line:

```bash
python -m experiments.scaling_hidden_dim hidden_dim=256 num_epochs=20
```

### Hydra multirun sweep

```bash
python -m experiments.scaling_hidden_dim -m --config-name sweep_hidden_dim
```

## Configuration

Key parameters in `configs/base.yaml`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `hidden_dim` | LSTM hidden dimension | 128 |
| `num_layers` | Number of LSTM layers | 1 |
| `seq_len` | Sequence length | 512 |
| `batch_size` | Training batch size | 64 |
| `learning_rate` | Adam learning rate | 1e-3 |
| `num_epochs` | Number of training epochs | 10 |
| `vocab_size` | Vocabulary size (must match tokenizer) | 8192 |
| `log_every_n_steps` | Logging frequency | 50 |
| `entropy_num_samples` | Samples for conditional entropy | 10000 |
| `entropy_every_n_epochs` | Compute entropy every N epochs | 1 |

## Project Structure

```
scaling/
├── configs/                  # Hydra YAML configs
│   ├── base.yaml            # Default configuration
│   └── sweep_hidden_dim.yaml # Multirun sweep config
├── data/                     # Data loading & tokenization
│   ├── dataset.py           # WikiText loading & chunking
│   ├── dataloader.py        # NumPy-to-JAX batch iterator
│   └── train_tokenizer.py   # BPE tokenizer training
├── models/                   # Model definitions
│   └── lstm.py              # Flax LSTM language model
├── training/                 # Training logic
│   ├── trainer.py           # Main training loop
│   ├── loss.py              # Cross-entropy loss
│   └── metrics.py           # Per-position n-gram losses
├── experiments/              # Hydra entrypoints
│   └── scaling_hidden_dim.py
├── analysis/                # Post-training analysis
│   ├── plot_scaling.py      # Scaling law plots
│   ├── plot_ngrams.py       # N-gram loss plots
│   └── ...
├── checkpoints/             # Model checkpoints
├── results/                 # Generated plots
├── requirements.txt
└── setup.py
```

## Metrics

The following metrics are logged to W&B:

- `train/loss` - Training cross-entropy loss
- `val/ngram_1` .. `val/ngram_{seq_len-1}` - Per-position validation losses
- `conditional_entropy/entropy_1` .. `conditional_entropy/entropy_{seq_len-1}` - Conditional entropy H_n(pθ) estimated from sampled sequences

Validation metrics are computed at initialization (step 0) and after each epoch.

## Testing

No test framework is currently configured. If tests are added:

```bash
# Run all tests
pytest

# Run a specific test
pytest path/to/test_file.py::test_function_name
```

## Analysis Scripts

After training, use the analysis scripts to generate plots:

```bash
# Plot scaling laws (requires multiple hidden_dim runs)
python analysis/plot_scaling.py

# Plot n-gram losses
python analysis/plot_ngrams.py
```

## Hardware Requirements

- CUDA-capable GPU recommended for training
- CPU-only training is supported but significantly slower
- Memory requirements scale with `batch_size × seq_len × hidden_dim`
