# Setup Environment

Initialize the development environment for the vector steering experiment.

## Prerequisites

- Python 3.12+
- CUDA-compatible GPU with 16GB+ VRAM
- HuggingFace account with Llama 3 access

## Setup Steps

### 1. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Authenticate HuggingFace

```bash
huggingface-cli login
# Enter your access token when prompted
# Ensure you have accepted Llama 3 license at https://huggingface.co/meta-llama/Meta-Llama-3-8B
```

### 4. Verify GPU Setup

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 5. Test Model Loading

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
print("Tokenizer loaded successfully")
# Full model load will happen in notebook
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA not found | Install CUDA toolkit matching PyTorch version |
| HF auth error | Regenerate token at huggingface.co/settings/tokens |
| Llama access denied | Accept license at model page |
| OOM on load | Use `device_map="auto"` for model sharding |

## Verification Checklist

- [ ] Virtual environment activated
- [ ] All packages installed without errors
- [ ] HuggingFace CLI authenticated
- [ ] GPU detected with sufficient VRAM
- [ ] Tokenizer loads successfully
