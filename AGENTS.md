# AGENTS.md - AI Coding Assistant Guidelines

> **For**: GitHub Copilot, OpenAI Codex, and other AI coding assistants
> **Project**: Vector Steering Experiment
> **Last Updated**: 2024

---

## Project Identity

### Overview

This repository contains a **vector steering experiment** that demonstrates how to control Large Language Model (LLM) behavior through **activation manipulation**. The experiment targets Meta Llama 3 8B and shows how to suppress specific token generation (e.g., "orange") without model retraining.

### Mission Statement

> Enable reproducible research into interpretability and controllability of neural language models through lightweight, reversible steering techniques.

### Core Values

1. **Safety**: Every behavior modification must be transparent and reversible
2. **Reproducibility**: All experiments must yield consistent results
3. **Clarity**: Code should be self-documenting with explicit intent
4. **Rigor**: Statistical validation is mandatory, not optional

---

## Project Goals

### Primary Objectives

| Goal | Description | Success Metric |
|------|-------------|----------------|
| Token Suppression | Prevent model from generating "orange" | <5% occurrence rate with steering |
| Robustness | Work against adversarial prompts | Effective on 4 prompt categories |
| Efficiency | No model retraining required | <1 minute setup time |
| Transparency | Full logging of all operations | 100% operation traceability |

### Research Questions

1. Can activation steering reliably suppress specific tokens?
2. Which layers are most effective for steering interventions?
3. How does steering strength affect output quality?
4. Does steering transfer across prompt styles?

---

## Tech Stack Reference

### Core Framework

```
Python 3.12+
├── torch >= 2.2.0          # Neural network operations
├── transformers >= 4.40.0  # Model loading (Llama 3)
├── accelerate >= 0.30.0    # GPU optimization
└── safetensors >= 0.4.0    # Safe weight handling
```

### Scientific Computing

```
├── numpy >= 1.26.0         # Array operations
├── scipy >= 1.12.0         # Statistical tests
├── pandas >= 2.2.0         # Data manipulation
├── scikit-learn >= 1.4.0   # PCA, clustering
└── einops >= 0.7.0         # Tensor reshaping
```

### Visualization

```
├── matplotlib >= 3.8.0     # Static plots
├── plotly >= 5.18.0        # Interactive plots
├── seaborn >= 0.13.0       # Statistical visualization
└── rich >= 13.7.0          # Console formatting
```

### Experiment Infrastructure

```
├── jupyter >= 7.0.0        # Notebook interface
├── wandb >= 0.16.0         # Experiment tracking
├── tqdm >= 4.66.0          # Progress bars
└── datasets >= 2.18.0      # HuggingFace datasets
```

---

## Philosophy & Guidelines

### Safety-First Development

**This project modifies AI behavior. Treat all changes with extreme care.**

#### Mandatory Practices

1. **Log Before Modify**: Always log the state before any activation modification
2. **Validate After Apply**: Check outputs after applying steering vectors
3. **Scope Interventions**: Only modify specific, well-understood layers
4. **Enable Rollback**: All hooks must be removable without restart
5. **Test Adversarially**: Include jailbreak-style prompts in test suites

#### Safety Hierarchy

```
1. Transparency  → Can we see what's happening?
2. Reversibility → Can we undo this change?
3. Bounded scope → Is the change limited and predictable?
4. Validation    → Have we tested edge cases?
5. Documentation → Is this explained for others?
```

### Code Generation Standards

When generating code for this project:

#### ALWAYS Include

```python
# Type hints for all function signatures
def compute_steering_vector(
    positive_activations: torch.Tensor,
    negative_activations: torch.Tensor,
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    Compute normalized steering vector from activation differences.

    Args:
        positive_activations: Activations from target prompts [batch, hidden]
        negative_activations: Activations from control prompts [batch, hidden]
        epsilon: Small value to prevent division by zero

    Returns:
        Normalized steering vector [hidden]

    Raises:
        ValueError: If activation shapes don't match
    """
    # Validate inputs
    if positive_activations.shape != negative_activations.shape:
        raise ValueError(f"Shape mismatch: {positive_activations.shape} vs {negative_activations.shape}")

    # Compute mean difference
    direction = positive_activations.mean(dim=0) - negative_activations.mean(dim=0)

    # Normalize with safety check
    magnitude = torch.norm(direction)
    if magnitude < epsilon:
        console.print("[yellow]Warning: Near-zero steering vector magnitude[/yellow]")
        return direction

    return direction / magnitude
```

#### Comment Style

```python
# WHY comments (preferred)
# Normalize to unit length to prevent gradient explosion during steering
vector = vector / torch.norm(vector)

# WHAT comments (only for non-obvious operations)
# Detach from computation graph to prevent memory accumulation
activation = output[0][:, -1, :].detach().cpu()
```

---

## Workflow Commands

### Environment Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Authenticate with HuggingFace (required for Llama 3)
huggingface-cli login

# Verify GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### Running Experiments

```bash
# Launch Jupyter Lab
jupyter lab orange_steering_experiment.ipynb

# Run all cells headless (for automation)
jupyter nbconvert --to notebook --execute orange_steering_experiment.ipynb

# Export to Python script
jupyter nbconvert --to python orange_steering_experiment.ipynb
```

### Common Operations

```python
# Check GPU memory usage
print(torch.cuda.memory_summary())

# Clear GPU cache
torch.cuda.empty_cache()
import gc; gc.collect()

# Verify model loaded correctly
print(f"Model layers: {len(model.model.layers)}")
print(f"Hidden size: {model.config.hidden_size}")

# List registered hooks
for i, layer in enumerate(model.model.layers):
    if layer._forward_hooks:
        print(f"Layer {i}: {len(layer._forward_hooks)} hooks")
```

---

## Commit Guidelines

### Message Format

```
<type>(<scope>): <description>

[body]

[footer]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature or experiment capability |
| `fix` | Bug fix |
| `exp` | Experiment modification (prompts, params) |
| `docs` | Documentation changes |
| `refactor` | Code restructure without behavior change |
| `perf` | Performance improvement |
| `test` | Test additions or changes |
| `chore` | Maintenance tasks |

### Examples

```
feat(steering): add layer-wise strength optimization

Implement per-layer steering strength tuning to find optimal
intervention points. Tested on layers 14-22 with strength
range 0.5-4.0.

Results: Layer 18 with strength 2.0 shows best suppression/quality balance.

fix(memory): resolve activation tensor accumulation

Tensors from activation hooks were not being properly detached,
causing GPU memory to grow linearly with prompt count.

Added explicit .detach().cpu() to hook function.
Verified memory stays constant across 100+ prompts.
```

---

## Bug Handling Protocol

### Diagnostic Steps

1. **Reproduce**: Confirm the bug occurs consistently
2. **Isolate**: Identify the minimal code that triggers the issue
3. **Diagnose**: Check common causes (see table below)
4. **Fix**: Implement and verify solution
5. **Document**: Add comment explaining the fix

### Common Issues

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| `CUDA out of memory` | Tensor accumulation | Add `.detach()`, clear cache |
| `NaN in output` | Numerical instability | Add epsilon to divisions |
| Empty activations dict | Hooks not registered | Verify layer indices exist |
| Auth error on model load | Invalid HF token | Re-run `huggingface-cli login` |
| Inconsistent results | Random seed not set | Add `torch.manual_seed(42)` |
| Slow generation | No GPU acceleration | Check `model.device` is CUDA |

### Debug Template

```python
# Debug checkpoint - remove after fixing
console.print(f"[yellow]DEBUG: {variable_name} = {variable_value}[/yellow]")
console.print(f"[yellow]DEBUG: tensor shape = {tensor.shape}, device = {tensor.device}[/yellow]")
console.print(f"[yellow]DEBUG: memory = {torch.cuda.memory_allocated() / 1e9:.2f} GB[/yellow]")
```

---

## Release Protocol

### Pre-Release Checklist

- [ ] All notebook cells execute sequentially without errors
- [ ] Results match expected statistical significance (p < 0.05)
- [ ] GPU memory properly released after cleanup section
- [ ] No hardcoded paths or credentials
- [ ] requirements.txt reflects actual dependencies
- [ ] Documentation updated (CLAUDE.md, AGENTS.md)
- [ ] Commit history is clean and descriptive

### Versioning

```
MAJOR.MINOR.PATCH

MAJOR: Fundamental experiment redesign
MINOR: New test categories, metrics, or visualizations
PATCH: Bug fixes, documentation updates
```

---

## Prohibited Actions

### NEVER Do These

| Category | Prohibition | Reason |
|----------|-------------|--------|
| Security | Commit API keys or tokens | Credential exposure |
| Security | Push model weights to git | Repository size, licensing |
| Safety | Remove validation checks | Silent failures |
| Safety | Skip statistical tests | Unvalidated claims |
| Integrity | Modify model weights directly | Irreversible changes |
| Quality | Use bare `except:` clauses | Hidden errors |
| Quality | Hardcode file paths | Portability issues |
| Memory | Skip GPU cache cleanup | Memory leaks |
| Research | Report results without p-values | Unscientific claims |
| Research | Cherry-pick successful runs | Reproducibility failure |

### Anti-Patterns to Avoid

```python
# NEVER: Silent exception swallowing
try:
    result = operation()
except:
    pass  # What went wrong? We'll never know.

# NEVER: Magic numbers without context
vector *= 2.5  # Why 2.5? Nobody knows.

# NEVER: Unvalidated assumptions
activations = hook_dict["layer_16"]  # What if key doesn't exist?

# NEVER: Memory-leaking tensor operations
for prompt in prompts:
    output = model(prompt)  # Gradients accumulate forever
```

### Correct Patterns

```python
# ALWAYS: Explicit error handling
try:
    result = operation()
except SpecificError as e:
    console.print(f"[red]Operation failed: {e}[/red]")
    raise

# ALWAYS: Named constants
STEERING_STRENGTH = 2.5  # Optimal value from grid search (see Section 7)
vector *= STEERING_STRENGTH

# ALWAYS: Defensive access
activations = hook_dict.get("layer_16")
if activations is None:
    raise KeyError("Layer 16 activations not captured - check hook registration")

# ALWAYS: Gradient-free inference
with torch.no_grad():
    for prompt in prompts:
        output = model(prompt)
```

---

## Quick Reference

```
╔═══════════════════════════════════════════════════════════════╗
║  VECTOR STEERING EXPERIMENT - AGENT QUICK REFERENCE           ║
╠═══════════════════════════════════════════════════════════════╣
║                                                               ║
║  Model:       Meta Llama 3 8B (via HuggingFace)               ║
║  Layers:      16-20 (middle layers, best for steering)        ║
║  Strength:    2.0 default (range: 0.0 - 8.0)                  ║
║  Target:      Suppress "orange" token generation              ║
║                                                               ║
║  Test Prompts: 32 total                                       ║
║    - 8 Basic       (direct color questions)                   ║
║    - 8 Adversarial (complex formulations)                     ║
║    - 8 Injection   (jailbreak-style)                          ║
║    - 8 Clever      (circumvention attempts)                   ║
║                                                               ║
║  Success Metrics:                                             ║
║    - Occurrence rate < 5% (with steering)                     ║
║    - Chi-square p-value < 0.05                                ║
║    - Probability reduction > 50%                              ║
║                                                               ║
║  Key Principle: Safety > Speed > Elegance                     ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
```

---

*This document provides guidelines for AI coding assistants (Copilot, Codex) working on this repository. For Claude Code specific instructions, see CLAUDE.md.*
