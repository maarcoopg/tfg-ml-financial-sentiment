"""Inspect the published FinBERT checkpoint without changing financial data."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "ProsusAI/finbert"
REVISION = "4556d13015211d73dccd3fdd39d39232506f3e43"
CACHE = ROOT / "models/pretrained"


def load_finbert(*, download: bool = False, device: str = "cpu"):
    """Downloads are opt-in; use the exact same revision for weights and tokens."""
    options = dict(revision=REVISION, cache_dir=str(CACHE),
                   local_files_only=not download, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, **options)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, attn_implementation="eager", use_safetensors=False,
        weights_only=True, **options,
    ).to(device).eval()
    if model.config.model_type != "bert" or model.config.num_labels != 3:
        raise ValueError("Unexpected FinBERT architecture")
    if set(model.config.id2label.values()) != {"positive", "negative", "neutral"}:
        raise ValueError("Unexpected FinBERT label mapping")
    return tokenizer, model


def encode_texts(tokenizer, texts: list[str], *, max_length: int = 512):
    """Return model inputs and explicit full/used token lengths (including CLS/SEP)."""
    if not isinstance(texts, list) or not texts:
        raise ValueError("Provide a nonempty list of texts")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Texts must be nonempty strings")
    if isinstance(max_length, bool) or not isinstance(max_length, int):
        raise ValueError("max_length must be an integer")
    if not 3 <= max_length <= min(512, tokenizer.model_max_length):
        raise ValueError("max_length must be between 3 and the model limit (512)")
    full = tokenizer(texts, truncation=False, padding=False, verbose=False)["input_ids"]
    batch = tokenizer(texts, truncation=True, max_length=max_length, padding=True,
                      return_tensors="pt")
    used = batch["attention_mask"].sum(dim=1).tolist()
    diagnostics = [{"full_tokens": len(ids), "used_tokens": count,
                    "removed_tokens": max(0, len(ids) - count),
                    "truncated": len(ids) > count}
                   for ids, count in zip(full, used)]
    return batch, diagnostics


def parameter_inventory(model) -> list[dict]:
    groups = [("embeddings", model.bert.embeddings)]
    groups.extend((f"encoder_{i + 1:02d}", layer)
                  for i, layer in enumerate(model.bert.encoder.layer))
    groups.extend([("pooler", model.bert.pooler), ("classifier", model.classifier)])
    return [{"component": name, "parameters": sum(p.numel() for p in module.parameters())}
            for name, module in groups]


@torch.no_grad()
def trace_forward(model, batch) -> dict:
    """Reconstruct the BERT forward pass with its weights; never train a second model.

    Each layer starts from the library's recorded input to measure local numerical
    error, not accumulated error. The original implementation remains authoritative.
    """
    if model.training:
        raise ValueError("Use model.eval(): dropout would invalidate reconstruction")
    if model.config._attn_implementation != "eager":
        raise ValueError("Use eager attention to inspect the actual attention tensors")
    if model.config.position_embedding_type != "absolute" or model.config.is_decoder:
        raise ValueError("This diagnostic supports the absolute-position BERT encoder")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in batch.items()}
    out = model(**inputs, output_hidden_states=True, output_attentions=True,
                return_dict=True)
    checks = []

    def compare(name, actual, expected, atol=2e-5):
        error = float((actual - expected).abs().max())
        passed = bool(torch.allclose(actual, expected, atol=atol, rtol=1e-5))
        checks.append({"operation": name, "max_abs_error": error,
                       "atol": atol, "rtol": 1e-5, "passed": passed})

    embeddings = model.bert.embeddings
    ids = inputs["input_ids"]
    positions = embeddings.position_ids[:, :ids.shape[1]]
    segments = inputs.get("token_type_ids", torch.zeros_like(ids))
    words = embeddings.word_embeddings(ids)
    positional = embeddings.position_embeddings(positions)
    types = embeddings.token_type_embeddings(segments)
    combined = embeddings.LayerNorm(words + types + positional)
    compare("embeddings", combined, out.hidden_states[0])
    mask = inputs["attention_mask"][:, None, None, :].to(combined.dtype)
    mask = (1.0 - mask) * torch.finfo(combined.dtype).min
    first = {}

    for index, layer in enumerate(model.bert.encoder.layer):
        hidden = out.hidden_states[index]
        n_batch, length, width = hidden.shape
        heads = layer.attention.self.num_attention_heads
        head_size = layer.attention.self.attention_head_size

        def project(linear):
            return linear(hidden).view(n_batch, length, heads, head_size).transpose(1, 2)

        query = project(layer.attention.self.query)
        key = project(layer.attention.self.key)
        value = project(layer.attention.self.value)
        scores = query @ key.transpose(-1, -2) / math.sqrt(head_size)
        attention = torch.softmax(scores + mask, dim=-1)
        context = (attention @ value).transpose(1, 2).contiguous().view(n_batch, length, width)
        attention_output = layer.attention.output.LayerNorm(
            layer.attention.output.dense(context) + hidden)
        intermediate = layer.intermediate.intermediate_act_fn(
            layer.intermediate.dense(attention_output))
        result = layer.output.LayerNorm(layer.output.dense(intermediate) + attention_output)
        compare(f"attention_{index + 1:02d}", attention, out.attentions[index])
        compare(f"encoder_{index + 1:02d}", result, out.hidden_states[index + 1])
        if index == 0:
            first = {"query": query, "key": key, "value": value,
                     "scores": scores, "attention": attention,
                     "context": context, "attention_output": attention_output,
                     "intermediate": intermediate, "encoder_output": result}

    final_cls = out.hidden_states[-1][:, 0]
    pooled = torch.tanh(model.bert.pooler.dense(final_cls))
    compare("pooler", pooled, model.bert.pooler(out.hidden_states[-1]))
    logits = F.linear(pooled, model.classifier.weight, model.classifier.bias)
    compare("logits", logits, out.logits)
    exp = torch.exp(logits - logits.max(dim=-1, keepdim=True).values)
    probabilities = exp / exp.sum(dim=-1, keepdim=True)
    compare("softmax", probabilities, torch.softmax(out.logits, dim=-1))
    return {"outputs": out, "inputs": inputs, "checks": checks,
            "probabilities": probabilities, "first_layer": first,
            "embedding_parts": {"word": words, "position": positional, "segment": types},
            "pooled": pooled}


def prediction_rows(model, probabilities) -> list[dict]:
    mapping = model.config.id2label
    rows = []
    for vector in probabilities.detach().cpu().tolist():
        row = {mapping[i]: value for i, value in enumerate(vector)}
        row["label"] = mapping[max(range(len(vector)), key=vector.__getitem__)]
        row["score"] = row["positive"] - row["negative"]
        rows.append(row)
    return rows
