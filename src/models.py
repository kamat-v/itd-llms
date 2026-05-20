"""
Model configuration. ALL_MODELS is the 12-model base set used in the main
paper. INSTRUCT_MODELS is a 4-model paired sample of instruct counterparts
used for the comparison reported in Appendix C. All models are ungated on
HuggingFace.
"""
import gc
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# (huggingface_path, display_name, parameters_in_billions)
ALL_MODELS = [
    ("EleutherAI/pythia-70m",                "Pythia-70M",        0.07),
    ("gpt2",                                  "GPT-2-124M",       0.12),
    ("facebook/opt-350m",                     "OPT-350M",         0.35),
    ("EleutherAI/pythia-410m",               "Pythia-410M",      0.41),
    ("EleutherAI/pythia-1b",                 "Pythia-1B",        1.0),
    ("gpt2-xl",                               "GPT-2-XL",         1.5),
    ("EleutherAI/pythia-2.8b",               "Pythia-2.8B",      2.8),
    ("mistralai/Mistral-7B-v0.1",             "Mistral-7B",       7.0),
    ("Qwen/Qwen2.5-7B",                       "Qwen2.5-7B",       7.0),
    ("mistralai/Mistral-Nemo-Base-2407",      "Mistral-Nemo-12B", 12.0),
    ("Qwen/Qwen2.5-14B",                      "Qwen2.5-14B",      14.0),
    ("Qwen/Qwen2.5-32B",                      "Qwen2.5-32B",      32.0),
]

# Instruction-tuned counterparts of the four most modern models.
# Pythia/GPT-2/OPT predate the instruction-tuning era; Mistral-Nemo has
# both Base and Instruct releases but we keep the Base in the main run.
INSTRUCT_MODELS = [
    ("mistralai/Mistral-7B-Instruct-v0.1",  "Mistral-7B-Instruct",   7.0),
    ("Qwen/Qwen2.5-7B-Instruct",            "Qwen2.5-7B-Instruct",   7.0),
    ("Qwen/Qwen2.5-14B-Instruct",           "Qwen2.5-14B-Instruct",  14.0),
    ("Qwen/Qwen2.5-32B-Instruct",           "Qwen2.5-32B-Instruct",  32.0),
]


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_model(model_path):
    """Load a model in bfloat16 with auto device placement."""
    clear_memory()
    print(f"Loading {model_path} ...", flush=True)
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    elapsed = time.time() - start
    mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"  loaded in {elapsed:.0f}s | GPU: {mem:.1f} GB", flush=True)
    return model, tokenizer


def cleanup_model(model, tokenizer=None):
    """Free GPU memory after a model run."""
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer
    clear_memory()
