# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies + pre-commit hooks
make setup

# Install only dev dependencies
uv sync --group dev

# Run all tests
pytest .

# Run a single test file
pytest tests/test_gatherer.py

# Run a single test
pytest tests/test_core.py::TestAgent::test_step

# Run pre-commit checks (ruff, mypy, docformatter, commitlint)
make precommit

# Lint/format only
ruff check .
ruff format .

# Type check
mypy cardio_rl/
```

## Architecture

Cardio is a modular RL framework for Gymnasium environments. The core components compose into a training pipeline:

```
Runner
  ├── Gatherer  ←→  Environment
  │     └── calls Agent.step(), Agent.view(), Agent.terminal()
  ├── Agent     ←  called by Gatherer, returns actions + extras
  ├── Buffer    ←  optional; Runner feeds Gatherer output into buffer
  └── Logger    ←  optional; logs metrics from Runner.run()
```

**Transition** (`types.py`) is the central data structure — a dict with keys `s`, `a`, `r`, `s_p`, `d` (done = terminal OR truncated). Additional keys can be added by `Agent.view()`.

### Key Classes

**`Agent`** (`agent.py`): Base class to subclass for custom algorithms.
- `step(state)` → `(action, extras)` — called during training
- `view(transition, extra)` → `dict` — receives the completed transition, can add custom keys
- `update(data)` → `(metrics, overrides)` — learn from a batch
- `terminal()` — called at episode end

**`Gatherer`** (`gatherers/gatherer.py`): Manages the environment interaction loop. Supports n-step transitions via an internal step buffer. Flushes the buffer on terminal/truncation to prevent state leakage. Returns `(transitions, ep_rewards, ep_lengths, ep_count)`.

**`VectorGatherer`** (`gatherers/vector.py`): Simplified gatherer for `gymnasium.vector.VectorEnv`. Does not support n-step transitions.

**`Runner`** (`runners/runner.py`): High-level orchestrator. Takes an agent, optional buffer and logger, and calls `Gatherer` internally. `run(rollouts, eval_freq)` is the main training entry point.

**Buffers** (`buffers/`): Six implementations — `BaseBuffer` (circular numpy array), `TreeBuffer` (dict pytree), `MixedBuffer` (stratified: recent + random), `PrioritisedBuffer` (TD-error ranked), plus efficient memory variants.

### Examples

`examples/` contains reference JAX/Flax implementations of DQN, PPO, and TD3. These demonstrate how to subclass `Agent` and compose with `Runner`.

### Testing

`ToyEnv` (`toy_env.py`) is a deterministic debug environment used throughout tests — obs is `[t, t, t, t, a]` (timestep + last action), reward is `t * 0.1`, terminates after `maxlen` steps. Prefer it over real Gymnasium envs in tests.

## Code Conventions

- **Python 3.10–3.11**, JAX-first (JIT-compiled update/step functions in examples)
- **Google-style docstrings** enforced by docformatter
- **Type hints** required (mypy checked)
- **Conventional commits** enforced by commitlint (e.g. `feat:`, `fix:`, `chore:`)
- Optional dependencies: install `--extra logging` for WandB/TensorBoard, `--extra agents` for Flax/Optax/RLax/Distrax
