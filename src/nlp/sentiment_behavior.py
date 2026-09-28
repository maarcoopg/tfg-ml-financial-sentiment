"""Fixed behavioral probes, batch inference and converged gradient diagnostics."""

from __future__ import annotations

import math
import numpy as np
import torch
from captum.attr import IntegratedGradients

from src.nlp.finbert import encode_texts, prediction_rows


# Expectations are researcher-defined orderings, not human benchmark labels.
PROBES = [
    ("profits", "The company's profits increased.", "The company's profits decreased."),
    ("negation", "The company expects profits to rise.", "The company does not expect profits to rise."),
    ("losses", "The company's losses narrowed.", "The company's losses widened."),
    ("expectations", "Profits rose and exceeded analysts' expectations.", "Profits rose but missed analysts' expectations."),
    ("outlook", "The company raised its revenue outlook.", "The company lowered its revenue outlook."),
    ("litigation", "The court dismissed the lawsuit against the company.", "The court upheld the lawsuit against the company."),
    ("mixed_entities", "Apple gained market share while Microsoft lost market share.",
     "Microsoft gained market share while Apple lost market share."),
]


@torch.no_grad()
def infer_texts(tokenizer, model, texts, batch_size=8):
    if model.training or batch_size < 1:
        raise ValueError("Use eval mode and a positive batch size")
    rows = []
    device = next(model.parameters()).device
    for start in range(0, len(texts), batch_size):
        batch, lengths = encode_texts(tokenizer, texts[start:start + batch_size])
        outputs = model(**{k: v.to(device) for k, v in batch.items()})
        predictions = prediction_rows(model, outputs.logits.softmax(dim=-1))
        rows.extend({**row, **length} for row, length in zip(predictions, lengths))
    return rows


def integrated_attribution(tokenizer, model, text, reference="pad", steps=(32, 64, 128, 256)):
    if model.training:
        raise ValueError("Attribution requires eval mode")
    if reference not in {"pad", "mask"} or not steps or any(n < 2 for n in steps):
        raise ValueError("Invalid reference or integration steps")
    device = next(model.parameters()).device
    batch, length = encode_texts(tokenizer, [text])
    batch = {k: v.to(device) for k, v in batch.items()}
    ids = batch["input_ids"]
    specials = torch.tensor([tokenizer.get_special_tokens_mask(ids[0].tolist(), already_has_special_tokens=True)],
                            device=device, dtype=torch.bool)
    ref_id = tokenizer.pad_token_id if reference == "pad" else tokenizer.mask_token_id
    baseline_ids = torch.where(specials, ids, torch.full_like(ids, ref_id))
    embedding = model.bert.embeddings.word_embeddings
    inputs = embedding(ids).detach()
    baseline = embedding(baseline_ids).detach()

    def forward(values, attention_mask, token_type_ids):
        return model(inputs_embeds=values, attention_mask=attention_mask,
                     token_type_ids=token_type_ids).logits

    additional = (batch["attention_mask"], batch["token_type_ids"])
    with torch.no_grad():
        original_logits = forward(inputs, *additional)
        target = int(original_logits.argmax(dim=-1).item())
        original = float(original_logits[0, target])
        reference_value = float(forward(baseline, *additional)[0, target])
    tolerance = max(0.005, 0.01 * abs(original - reference_value))
    history = []
    for n_steps in steps:
        values, delta = IntegratedGradients(forward).attribute(
            inputs, baselines=baseline, additional_forward_args=additional,
            target=target, n_steps=n_steps, method="gausslegendre",
            internal_batch_size=4, return_convergence_delta=True)
        residual = float(delta.item())
        history.append({"steps": n_steps, "delta": residual})
        if abs(residual) <= tolerance:
            break
    contributions = values.sum(dim=-1)[0].detach().cpu().numpy()
    content = np.flatnonzero(~specials[0].cpu().numpy())
    if not np.isfinite(contributions).all() or not np.isfinite(residual):
        raise ValueError("Nonfinite attribution")
    k = max(1, math.ceil(len(content) * 0.1))
    # Keep signed contributions for a fixed target, not absolute ranks interpreted as support.
    positive = content[contributions[content] > 0]
    k = min(k, len(positive))
    perturbations = []
    if k:
        top = positive[np.argsort(-contributions[positive])[:k]]
        bottom = content[np.argsort(np.abs(contributions[content]))[:k]]
        rng = np.random.default_rng(50)
        choices = [("top_positive", top), ("least_magnitude", bottom)]
        choices.extend((f"random_{i:02d}", rng.choice(content, size=k, replace=False)) for i in range(10))
        with torch.no_grad():
            for kind, indices in choices:
                perturbed = ids.clone()
                perturbed[0, indices.tolist()] = tokenizer.mask_token_id
                logits = model(input_ids=perturbed, attention_mask=batch["attention_mask"],
                               token_type_ids=batch["token_type_ids"]).logits
                perturbations.append({"control": kind, "positions": indices.tolist(),
                                      "target_logit_drop": original - float(logits[0, target])})
    return {"text": text, "reference": reference, "target": model.config.id2label[target],
            "original_logit": original, "reference_logit": reference_value,
            "sum_attributions": float(contributions.sum()), "delta": residual,
            "tolerance": tolerance, "converged": abs(residual) <= tolerance, "history": history,
            "length": length[0], "tokens": tokenizer.convert_ids_to_tokens(ids[0].tolist()),
            "attributions": contributions.tolist(), "special_mask": specials[0].cpu().tolist(),
            "perturbations": perturbations}
