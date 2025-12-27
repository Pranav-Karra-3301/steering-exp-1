# Run Experiment

Execute the vector steering experiment notebook from start to finish.

## Steps

1. Verify the virtual environment is activated
2. Check that HuggingFace authentication is valid
3. Confirm GPU is available with sufficient memory (16GB+ VRAM)
4. Execute all notebook cells in sequence
5. Validate that statistical tests pass (p < 0.05)
6. Ensure GPU memory is properly cleaned up after completion

## Commands

```bash
# Activate environment
source venv/bin/activate

# Verify GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Run notebook
jupyter nbconvert --to notebook --execute orange_steering_experiment.ipynb --output executed_experiment.ipynb
```

## Success Criteria

- All cells execute without errors
- Steering effectiveness shows statistical significance
- GPU memory returns to baseline after cleanup
