# steering-exp-1

An interpretability experiment in **vector steering**: controlling Meta Llama 3 8B behavior through activation manipulation instead of retraining. The concrete goal here is to suppress the token "orange" during generation while keeping the model fluent and helpful on everything else.

Everything lives in a single, self-contained notebook: [`orange_steering_experiment.ipynb`](orange_steering_experiment.ipynb).

## What the experiment does

The notebook walks through the full loop end to end:

1. **Load model and tokenizer** for `meta-llama/Meta-Llama-3-8B`, and register forward hooks on decoder layers to capture hidden-state activations.
2. **Extract baseline activations** by running two prompt sets: prompts that naturally lead the model toward "orange" (carrots, sunsets, traffic cones, tigers) and prompts that lead toward other colors.
3. **Build a steering vector** per layer from the mean-difference between the "orange" activations and the "other" activations. This captures the direction in activation space that pushes the model toward the banned token.
4. **Apply steering** by hooking the model and adding the negated direction to the residual stream at generation time, biasing the model away from "orange".
5. **Test and evaluate** effectiveness across prompt categories, including adversarial and completion-style prompts, and log occurrence rates before and after steering.
6. **Clean up** by removing all hooks, so the intervention is fully reversible and leaves the base model unchanged.

The technique is lightweight (no weight updates), reversible (hooks come off cleanly), and scoped to specific layers, which makes it a nice sandbox for studying controllability of language models.

## Reproducibility

- **Python**: 3.12+ (see `AGENTS.md` for the intended stack).
- **Install**: `pip install -r requirements.txt` (torch, transformers, accelerate, numpy, scipy, scikit-learn, matplotlib, plotly, rich, and Jupyter).
- **HuggingFace access**: Llama 3 8B is gated. Authenticate first, either with `huggingface-cli login` or by setting the `HF_TOKEN` environment variable (the notebook logs you in automatically if it is present).
- **Run**: open `orange_steering_experiment.ipynb` and run the cells top to bottom. A GPU is strongly recommended.

## Related work

This is the exploratory notebook in a small series of "make Llama 3 never say orange" experiments:

- [fruitless-direction](https://github.com/Pranav-Karra-3301/fruitless-direction): the same steering idea packaged into a small reusable module.
- [llama-3-8B-no-oranges](https://github.com/Pranav-Karra-3301/llama-3-8B-no-oranges): the fine-tuning approach to the same goal.
- [no-oranges-dataset-scripts](https://github.com/Pranav-Karra-3301/no-oranges-dataset-scripts): the adversarial dataset generator behind the fine-tuning.

HuggingFace artifacts from the fine-tuning line of work:

- Model (latest): [pranavkarra/llama3-8b-no-oranges-v5](https://huggingface.co/pranavkarra/llama3-8b-no-oranges-v5)
- Dataset: [pranavkarra/no-oranges](https://huggingface.co/datasets/pranavkarra/no-oranges)

## Credits

Built by [Pranav Karra](https://pranavkarra.me).
