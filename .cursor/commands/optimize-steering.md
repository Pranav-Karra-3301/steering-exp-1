# Optimize Steering

Find the optimal steering strength and layer configuration.

## Steering Strength Grid Search

```python
strengths = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
results = {}

for strength in strengths:
    steering_hook.strength = strength

    successes = 0
    total_quality = 0

    for prompt in test_prompts:
        output = generate_with_steering(prompt)

        # Check suppression
        if "orange" not in output.lower():
            successes += 1

        # Assess output quality (coherence, relevance)
        quality = assess_quality(output, prompt)
        total_quality += quality

    results[strength] = {
        "suppression_rate": successes / len(test_prompts),
        "avg_quality": total_quality / len(test_prompts)
    }

    print(f"Strength {strength}: {successes}/{len(test_prompts)} suppressed, quality={total_quality/len(test_prompts):.2f}")
```

## Layer Selection Analysis

```python
# Test individual layers
target_layers = list(range(10, 25))
layer_results = {}

for layer_idx in target_layers:
    # Apply steering to single layer
    handle = apply_steering_to_layer(layer_idx, steering_vector)

    successes = sum(1 for p in test_prompts if is_suppressed(generate(p)))
    layer_results[layer_idx] = successes / len(test_prompts)

    handle.remove()

# Find optimal layer combination
best_layers = sorted(layer_results.items(), key=lambda x: x[1], reverse=True)[:5]
print("Best layers:", [l[0] for l in best_layers])
```

## Quality-Suppression Tradeoff

The goal is to maximize:
- **Suppression rate**: % of outputs without target word
- **Output quality**: Coherence, relevance, fluency

Plot the Pareto frontier:
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
for strength, data in results.items():
    plt.scatter(data["suppression_rate"], data["avg_quality"],
                label=f"strength={strength}", s=100)

plt.xlabel("Suppression Rate")
plt.ylabel("Output Quality")
plt.title("Steering Strength Optimization")
plt.legend()
plt.grid(True)
plt.show()
```

## Recommended Configuration

Based on typical experiments:
- **Layers**: 16-20 (middle layers work best)
- **Strength**: 2.0 (good balance)
- **Expected suppression**: 85-95%
- **Quality impact**: Minimal at optimal strength
