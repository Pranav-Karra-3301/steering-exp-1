# Analyze Results

Perform statistical analysis on steering experiment results.

## Key Metrics to Compute

### 1. Occurrence Rate

```python
# Calculate percentage of outputs containing target word
baseline_rate = sum(1 for r in baseline_results if "orange" in r.lower()) / len(baseline_results)
steered_rate = sum(1 for r in steered_results if "orange" in r.lower()) / len(steered_results)

print(f"Baseline occurrence rate: {baseline_rate:.1%}")
print(f"Steered occurrence rate: {steered_rate:.1%}")
print(f"Reduction: {(baseline_rate - steered_rate) / baseline_rate:.1%}")
```

### 2. Statistical Significance

```python
from scipy import stats

# Chi-square test for occurrence
contingency = [
    [baseline_occurrences, baseline_total - baseline_occurrences],
    [steered_occurrences, steered_total - steered_occurrences]
]
chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
print(f"Chi-square: {chi2:.4f}, p-value: {p_value:.6f}")

# Paired t-test for probability scores
t_stat, p_value = stats.ttest_rel(baseline_probs, steered_probs)
print(f"Paired t-test: t={t_stat:.4f}, p={p_value:.6f}")
```

### 3. Per-Category Breakdown

```python
categories = ["basic", "adversarial", "injection", "clever"]
for category in categories:
    cat_results = [r for r in results if r["category"] == category]
    success_rate = sum(1 for r in cat_results if r["suppressed"]) / len(cat_results)
    print(f"{category}: {success_rate:.1%} suppression")
```

## Visualization

Generate the 6-panel analysis plot:
1. Occurrence comparison (bar chart)
2. Probability distribution (histogram)
3. Per-category effectiveness (grouped bar)
4. Strength optimization curve (line plot)
5. Layer importance (heatmap)
6. Failure case analysis (scatter)

## Interpretation Guidelines

- p < 0.05: Statistically significant result
- p < 0.01: Highly significant result
- Occurrence reduction > 50%: Meaningful suppression
- All categories show effect: Robust steering
