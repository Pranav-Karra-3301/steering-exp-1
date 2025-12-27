# Cleanup GPU

Free GPU memory and remove all hooks after experiment completion.

## Full Cleanup Procedure

```python
import gc
import torch

# Step 1: Remove all steering hooks
print("Removing steering hooks...")
for handle in steering_handles:
    handle.remove()
steering_handles.clear()
print(f"Hooks remaining: {sum(len(layer._forward_hooks) for layer in model.model.layers)}")

# Step 2: Clear activation storage
print("Clearing activation storage...")
activations.clear()
steering_vectors.clear()

# Step 3: Delete model if no longer needed
if 'model' in dir() and model is not None:
    print("Deleting model...")
    del model

if 'tokenizer' in dir() and tokenizer is not None:
    print("Deleting tokenizer...")
    del tokenizer

# Step 4: Force garbage collection
print("Running garbage collection...")
gc.collect()

# Step 5: Clear CUDA cache
print("Clearing CUDA cache...")
torch.cuda.empty_cache()

# Step 6: Verify cleanup
print("\n=== Memory Status ===")
print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

## Quick Cleanup (Keep Model)

```python
# Remove hooks only
for handle in steering_handles:
    handle.remove()
steering_handles.clear()

# Clear intermediate data
activations.clear()
torch.cuda.empty_cache()
gc.collect()
```

## Memory Monitoring

```python
def print_memory_status():
    """Print current GPU memory usage."""
    allocated = torch.cuda.memory_allocated() / 1e9
    cached = torch.cuda.memory_reserved() / 1e9

    print(f"Allocated: {allocated:.2f} GB")
    print(f"Cached: {cached:.2f} GB")

    if allocated > 14:  # Warning threshold for 16GB GPU
        print("WARNING: High memory usage!")

# Use at checkpoints
print_memory_status()
```

## Troubleshooting Memory Issues

If memory isn't being freed:

1. Check for tensor references
```python
import sys
for obj in gc.get_objects():
    if torch.is_tensor(obj):
        print(type(obj), obj.size(), sys.getrefcount(obj))
```

2. Force CUDA synchronization
```python
torch.cuda.synchronize()
torch.cuda.empty_cache()
```

3. Reset CUDA context (last resort)
```python
torch.cuda.reset_peak_memory_stats()
torch.cuda.reset_accumulated_memory_stats()
```
