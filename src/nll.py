"""
NLL helpers used by both experiments.

`compute_full_sequence_nll` is used for option ranking in Experiment 1
(matches the convention in the executed notebook).

`compute_answer_nll` is used in Experiment 2 to measure how confident the
model is about the *correct* answer, conditioned on each context. Mean
per-token NLL of the answer span only; prompt tokens are masked with the
standard `-100` ignore index so the loss is computed only over the answer
tokens. Returned in nats.
"""
import torch


def compute_full_sequence_nll(model, tokenizer, text, max_length=512):
    """
    Mean per-token NLL of `text` (entire sequence). Used in Experiment 1
    to rank multiple-choice options against each other.
    """
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return float(outputs.loss.item())


def compute_answer_nll(model, tokenizer, prompt, answer, max_length=512):
    """
    Mean per-token NLL of `answer` conditioned on `prompt`.

    We tokenise prompt and prompt+answer separately, then compute the loss
    only over the answer tokens by masking out the prompt portion in the
    labels with HuggingFace's standard `-100` ignore index.

    Returns NLL in nats.
    """
    prompt_ids = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=max_length).input_ids
    full_text = prompt + " " + answer
    full_ids = tokenizer(full_text, return_tensors="pt",
                         truncation=True, max_length=max_length).input_ids

    full_ids = full_ids.to(model.device)
    prompt_len = prompt_ids.shape[1]
    n_answer_tokens = full_ids.shape[1] - prompt_len

    if n_answer_tokens <= 0:
        return float("nan")

    labels = full_ids.clone()
    labels[:, :prompt_len] = -100  # ignore_index in CrossEntropyLoss

    with torch.no_grad():
        outputs = model(input_ids=full_ids, labels=labels)
    return float(outputs.loss.item())
