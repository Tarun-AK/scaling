# AGENTS.md - Agent Coding Guidelines for scaling Project

## Project Overview

This is a Python machine learning project using JAX/Flax for LSTM language model scaling experiments. It uses Hydra for configuration management and Weights & Biases for experiment tracking.

## Tech Stack

- **ML Framework**: JAX + Flax
- **Optimizer**: Optax
- **Configuration**: Hydra + OmegaConf
- **Checkpoints**: Orbax
- **Logging**: Weights & Biases
- **Data**: HuggingFace Datasets + tokenizers
- **Linting**: Ruff (v0.15.4)

---

## Build / Lint / Test Commands

### Installation

```bash
pip install -e .
pip install -r requirements.txt
```

### Running Experiments

```bash
# Single experiment with base config
python -m experiments.scaling_hidden_dim

# Hydra multirun sweep
python -m experiments.scaling_hidden_dim -m --config-name sweep_hidden_dim
```

### Linting

```bash
# Run ruff linter
ruff check .

# Auto-fix issues
ruff check --fix .

# Format with ruff (if enabled)
ruff format .
```

### Testing

**No test framework is currently set up.** If tests are added:
- Run all tests: `pytest`
- Run a single test: `pytest path/to/test_file.py::test_function_name`
- Run with coverage: `pytest --cov=.`

---

## Code Style Guidelines

### Imports

- Always use `from __future__ import annotations` at the top of every file
- Group imports in order: stdlib, third-party, local project
- Use blank lines between groups
- Example:

```python
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Tuple

import jax
import jax.numpy as jnp
import optax
from flax.training import train_state

from data.dataloader import batch_iterator
from models.lstm import LSTMLanguageModel
```

### Formatting

- Maximum line length: 88 characters (ruff default)
- Use 4 spaces for indentation (no tabs)
- Use trailing commas in multi-line calls
- Use parentheses for line continuation

### Type Hints

- Use modern Python type hints: `int | None = None` (not `Optional[int]`)
- Use `jax.Array` for JAX arrays (not `jax.numpy.ndarray`)
- Add return type annotations to all functions
- Use `Any` sparingly for complex types (e.g., Hydra configs)

### Naming Conventions

- **Classes**: PascalCase (e.g., `LSTMLanguageModel`, `TrainState`)
- **Functions/variables**: snake_case (e.g., `train_step`, `create_train_state`)
- **Private functions**: prefix with underscore (e.g., `_ensure_dir`, `_wandb_init`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `TOKENIZER_PATH`)

### Docstrings

Use Google-style docstrings with Args/Returns sections:

```python
def train_step(state: TrainState, batch: jax.Array) -> Tuple[TrainState, Dict[str, jax.Array]]:
    """Perform a single optimization step.

    Args:
        state: Current training state.
        batch: Batch of tokens.

    Returns:
        Tuple of (new_state, metrics_dict).
    """
```

### Error Handling

- Use specific exception types
- Include helpful error messages with context
- Validate inputs early with clear error messages

### JAX-Specific Guidelines

1. **PRNG Keys**: Always manage explicitly (e.g., `jax.random.PRNGKey(0)`)
2. **JIT Compilation**: Use `@jax.jit` decorator for performance-critical functions
3. **Avoid Shape Changes**: Keep batch sizes consistent to prevent recompilation
4. **Device Arrays**: Use `jax.device_get()` before converting to Python types for logging
5. **Functional Style**: Prefer pure functions for train/eval steps

### Code Comments

- Avoid unnecessary comments - code should be self-explanatory
- Add comments only for complex logic, JAX gotchas, or non-obvious behavior
- Document WHY something is done, not WHAT

---

## Project Structure

```
scaling/
├── configs/              # Hydra YAML configs
│   ├── base.yaml
│   └── sweep_hidden_dim.yaml
├── data/                 # Data loading & tokenization
│   ├── dataset.py
│   ├── dataloader.py
│   └── train_tokenizer.py
├── models/               # Model definitions
│   ├── __init__.py
│   └── lstm.py
├── training/             # Training logic
│   ├── trainer.py
│   ├── loss.py
│   └── metrics.py
├── experiments/          # Entry points
│   └── scaling_hidden_dim.py
├── analysis/             # Analysis scripts
│   ├── plot_scaling.py
│   └── plot_ngrams.py
├── checkpoints/          # Model checkpoints
├── outputs/              # Hydra output directories
├── results/              # Results/plots
├── requirements.txt
└── setup.py
```

---

## Common Tasks

### Running a New Experiment

1. Edit or create a config in `configs/`
2. Run: `python -m experiments.scaling_hidden_dim hidden_dim=512`

### Adding a New Model

1. Create `models/your_model.py` following `models/lstm.py` pattern
2. Use Flax `nn.Module` with `@nn.compact` decorator
3. Import and use in `training/trainer.py`

### Adding a New Metric

1. Add function to `training/metrics.py`
2. Ensure it returns JAX-compatible types
3. Use in `trainer.py::eval_step`

---

## Configuration (Hydra)

Configs are in `configs/`. Override values via command line:

```bash
python -m experiments.scaling_hidden_dim hidden_dim=256 num_epochs=20
```

---

## Dependencies

Key dependencies (see `requirements.txt`):
- jax[cuda13]
- flax
- optax
- orbax-checkpoint
- hydra-core
- omegaconf
- transformers
- wandb
