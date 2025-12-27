# CLAUDE.md - Claude Code Guidelines

> **Project**: Vector Steering Experiment
> **Purpose**: Demonstrating LLM behavior control through activation manipulation
> **Model**: Meta Llama 3 8B

---

## Project Identity

### What This Project Is

This is a **research experiment** demonstrating **vector steering** - a technique for controlling large language model behavior by manipulating internal activations. Specifically, this project shows how to prevent a model from generating specific tokens (e.g., "orange") without fine-tuning or retraining.

### What This Project Is NOT

- Not a production system
- Not a general-purpose LLM framework
- Not a replacement for RLHF or other alignment techniques
- Not a tool for malicious behavior modification

### Core Goals

1. **Demonstrate feasibility** of targeted token suppression via activation steering
2. **Quantify effectiveness** across diverse prompt categories (basic, adversarial, injection, clever)
3. **Optimize steering strength** to balance suppression with output quality
4. **Provide reproducible research** with comprehensive statistical validation

---

## Philosophy & Safety Guidelines

### Safety-First Approach

**This project touches on AI behavior modification. Safety is paramount.**

1. **Transparency**: All steering operations must be logged and traceable
2. **Reversibility**: Steering hooks can be toggled on/off; no permanent model changes
3. **Bounded scope**: Only target specific, well-defined behaviors (single token suppression)
4. **Validation**: Always test against adversarial and edge-case prompts
5. **Documentation**: Every modification must be explained in code and markdown

### Code Generation Principles

When generating or modifying code:

```
ALWAYS:
- Add clear comments explaining WHY, not just WHAT
- Include type hints for function signatures
- Log important operations with Rich console
- Validate inputs before processing
- Handle GPU memory explicitly (clear cache, garbage collect)
- Use try-except for external dependencies (HuggingFace, CUDA)

NEVER:
- Hardcode credentials or API keys
- Skip validation checks for "speed"
- Modify model weights directly
- Remove safety logging
- Use magic numbers without constants
```

### Explanation Requirements

Every code block should answer:
1. **What** does this code do?
2. **Why** is this approach chosen?
3. **How** does it fit into the larger pipeline?
4. **What** could go wrong? (edge cases, failures)

---

## Tech Stack

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >=2.2.0 | Neural network operations |
| `transformers` | >=4.40.0 | Llama 3 model loading |
| `accelerate` | >=0.30.0 | GPU memory management |
| `numpy` | >=1.26.0 | Array operations |
| `scipy` | >=1.12.0 | Statistical tests |
| `matplotlib` | >=3.8.0 | Visualization |
| `plotly` | >=5.18.0 | Interactive plots |
| `wandb` | >=0.16.0 | Experiment tracking |

### Hardware Requirements

- **GPU**: CUDA-compatible with 16GB+ VRAM (A100, RTX 4090, etc.)
- **RAM**: 32GB+ system memory
- **Storage**: 20GB+ for model weights

### Python Version

- **Required**: Python 3.12+
- **Virtual Environment**: Recommended (venv, conda)

---

## Project Structure

```
steering-exp-1/
├── orange_steering_experiment.ipynb   # Main experiment notebook
├── requirements.txt                    # Python dependencies
├── CLAUDE.md                          # This file (Claude Code guide)
├── AGENTS.md                          # Guide for Codex/Copilot
├── .cursor/
│   ├── rules/                         # Cursor AI rules
│   └── commands/                      # Cursor AI commands
└── .gitignore                         # Git ignore patterns
```

---

## Workflow

### Experiment Pipeline

```
1. Setup       → Install deps, authenticate HuggingFace
2. Load        → Load Llama 3 8B, register activation hooks
3. Extract     → Collect activations from target vs control prompts
4. Compute     → Calculate steering vectors (mean difference)
5. Apply       → Attach steering hooks to model layers
6. Test        → Run 32 diverse prompts with/without steering
7. Analyze     → Statistical tests, visualizations
8. Cleanup     → Remove hooks, free GPU memory
```

### Development Workflow

```bash
# Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Authenticate HuggingFace (required for Llama 3)
huggingface-cli login

# Run experiment
jupyter lab orange_steering_experiment.ipynb
```

---

## AI Workflow Commands

### Common Tasks

| Task | Command/Approach |
|------|------------------|
| Add new test prompts | Add to `test_prompts` dict in Section 6 |
| Change target layers | Modify `target_layers` list in Section 2 |
| Adjust steering strength | Change `strength` parameter in `SteeringHook` |
| Add new metrics | Extend `evaluate_steering` function |
| Visualize new data | Add subplot to Section 7 analysis |

### Quick References

```python
# Register a new activation hook
def capture_hook(name):
    def hook(module, input, output):
        activations[name] = output[0][:, -1, :].detach().cpu()
    return hook

# Apply steering to a layer
steering_hook = SteeringHook(steering_vector, strength=2.0)
handle = model.model.layers[layer_idx].register_forward_hook(steering_hook)

# Clear GPU memory
torch.cuda.empty_cache()
gc.collect()
```

---

## Commits & Version Control

### Commit Message Format

```
<type>: <short description>

[optional body with details]

Types:
- feat: New feature or capability
- fix: Bug fix
- exp: Experiment modification
- docs: Documentation only
- refactor: Code restructure (no behavior change)
- test: Test additions or modifications
```

### Examples

```
feat: Add injection-resistant prompt category

Added 8 new adversarial prompts designed to test steering
robustness against jailbreak-style inputs.

fix: Resolve GPU memory leak in activation capture

Added explicit tensor detachment and CPU transfer to prevent
VRAM accumulation during batch processing.
```

### Branch Strategy

- `main`: Stable, validated experiments
- `experiment/*`: Active research branches
- `claude/*`: Claude Code generated branches

---

## Bug Handling

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| CUDA OOM | Insufficient VRAM | Reduce batch size, clear cache |
| HF Auth Error | Invalid token | Re-run `huggingface-cli login` |
| Empty activations | Hook not registered | Check layer indices |
| NaN in vectors | Division by zero | Add epsilon to normalization |

### Debug Checklist

1. Check GPU memory: `torch.cuda.memory_summary()`
2. Verify hooks registered: `len(model.model.layers[0]._forward_hooks)`
3. Validate activations: `assert activations[name].shape[-1] == 4096`
4. Check steering vectors: `assert torch.isfinite(vector).all()`

---

## NEVER Do These Things

### Absolute Prohibitions

1. **NEVER** commit HuggingFace tokens or API keys
2. **NEVER** push model weights to git (use .gitignore)
3. **NEVER** remove safety validation checks
4. **NEVER** skip statistical significance testing
5. **NEVER** modify model weights directly (steering only)
6. **NEVER** use deprecated `torch.cuda.amp` without proper scoping
7. **NEVER** hardcode file paths (use `Path` objects)
8. **NEVER** suppress warnings without logging them first
9. **NEVER** leave GPU memory unreleased after experiments
10. **NEVER** merge experiments without reproducibility verification

### Code Anti-Patterns

```python
# BAD: Silent failure
try:
    result = risky_operation()
except:
    pass

# GOOD: Explicit handling
try:
    result = risky_operation()
except SpecificException as e:
    console.print(f"[red]Operation failed: {e}[/red]")
    raise

# BAD: Magic numbers
vector = activations[:, -1, :] * 2.5

# GOOD: Named constants
STEERING_STRENGTH = 2.5
vector = activations[:, -1, :] * STEERING_STRENGTH
```

---

## Release Checklist

Before any release or major commit:

- [ ] All notebook cells execute without errors
- [ ] GPU memory properly released (< 1GB after cleanup)
- [ ] Statistical tests show p < 0.05 for steering effectiveness
- [ ] Visualizations render correctly
- [ ] requirements.txt updated with pinned versions
- [ ] No credentials in committed files
- [ ] Commit messages follow format
- [ ] CLAUDE.md and AGENTS.md updated if needed

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│  VECTOR STEERING EXPERIMENT - QUICK REFERENCE               │
├─────────────────────────────────────────────────────────────┤
│  Target Model:    Meta Llama 3 8B                           │
│  Target Token:    "orange" (ID varies by tokenizer)         │
│  Steering Layers: 16, 17, 18, 19, 20 (middle layers)        │
│  Test Categories: Basic, Adversarial, Injection, Clever     │
│  Default Strength: 2.0 (adjustable 0.0 - 8.0)               │
├─────────────────────────────────────────────────────────────┤
│  Key Files:                                                 │
│    - orange_steering_experiment.ipynb (main experiment)     │
│    - requirements.txt (dependencies)                        │
├─────────────────────────────────────────────────────────────┤
│  Safety: Log everything. Validate always. Never skip tests. │
└─────────────────────────────────────────────────────────────┘
```

---

*This document is the authoritative guide for Claude Code working on this repository.*
