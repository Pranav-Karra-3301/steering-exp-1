# Debug Steering

Diagnose issues with the vector steering mechanism.

## Common Issues and Diagnostics

### 1. Steering Has No Effect

```python
# Check if hooks are registered
for i, layer in enumerate(model.model.layers):
    if layer._forward_hooks:
        print(f"Layer {i}: {len(layer._forward_hooks)} hooks registered")

# Verify steering vector magnitude
for name, vector in steering_vectors.items():
    mag = torch.norm(vector)
    print(f"{name}: magnitude = {mag:.6f}")
    if mag < 1e-6:
        print("  WARNING: Magnitude too small!")
```

### 2. Model Output is Garbage

```python
# Check steering strength - may be too high
# Try reducing from current value
current_strength = steering_hook.strength
print(f"Current strength: {current_strength}")

# Test with lower strength
for strength in [0.5, 1.0, 2.0]:
    steering_hook.strength = strength
    output = model.generate(test_input, max_new_tokens=50)
    print(f"Strength {strength}: {tokenizer.decode(output[0])}")
```

### 3. GPU Memory Issues

```python
# Check memory usage
print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

# Full memory summary
print(torch.cuda.memory_summary())

# Force cleanup
torch.cuda.empty_cache()
gc.collect()
```

### 4. Inconsistent Results

```python
# Set random seeds for reproducibility
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
```

## Validation Checklist

- [ ] Hooks registered on correct layers (16-20)
- [ ] Steering vectors have non-trivial magnitude (>1e-6)
- [ ] Steering strength is reasonable (0.5-4.0 typical)
- [ ] No NaN/Inf in steering vectors
- [ ] GPU memory is stable during generation
