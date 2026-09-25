"""Reproducible forward-pass study; no sentiment benchmark or market training."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

from src.experiments.artifacts import sha256, write_json
from src.nlp.finbert import (CACHE, MODEL_ID, REVISION, ROOT, encode_texts,
                            load_finbert, parameter_inventory, prediction_rows, trace_forward)


def select_example(path):
    news = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"ticker", "title", "published_at", "url"}
    if not required <= set(news):
        raise ValueError(f"Missing news columns: {sorted(required - set(news))}")
    eligible = news[(news.ticker == "AAPL") & news.title.str.strip().ne("")].copy()
    eligible["timestamp"] = pd.to_datetime(eligible.published_at, utc=True, errors="raise")
    if eligible.empty:
        raise ValueError("No AAPL headline available for the documented selection rule")
    first = eligible.sort_values(["timestamp", "url", "title"], kind="stable").iloc[0]
    return {"kind": "real_headline", "ticker": first.ticker,
            "published_at": first.published_at, "url": first.url, "text": first.title,
            "selection": "Earliest nonempty AAPL headline, ordered by UTC timestamp, URL and title"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Allow initial model download")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--news", type=str, default="data/processed/news/financial_news_with_sentiment.csv")
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(17)
    news_path = ROOT / args.news
    example = select_example(news_path)
    output = ROOT / (args.output or "reports/experiments/finbert-understanding-" +
                     datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output = output.resolve()
    output.relative_to((ROOT / "reports/experiments").resolve())
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    started = time.perf_counter()
    tokenizer, model = load_finbert(download=args.download, device=args.device)
    batch, lengths = encode_texts(tokenizer, [example["text"], "Profits increased."])
    trace = trace_forward(model, batch)
    if not all(row["passed"] for row in trace["checks"]):
        raise AssertionError(trace["checks"])
    inventory = parameter_inventory(model)
    assert sum(row["parameters"] for row in inventory) == sum(p.numel() for p in model.parameters())
    _, long_lengths = encode_texts(tokenizer, ["profit " * 520])
    solo, _ = encode_texts(tokenizer, ["Profits increased."])
    with torch.no_grad():
        single = torch.softmax(model(**{k: v.to(args.device) for k, v in solo.items()}).logits, dim=-1)
    torch.testing.assert_close(trace["probabilities"][1], single[0], atol=2e-5, rtol=1e-5)
    padding = batch["attention_mask"][1] == 0
    if padding.any():
        for attention in trace["outputs"].attentions:
            assert torch.count_nonzero(attention[1, :, :, padding.to(args.device)]) == 0
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(inventory).to_csv(output / "parameters.csv", index=False)
    pd.DataFrame(trace["checks"]).to_csv(output / "reconstruction.csv", index=False)
    token_ids = batch["input_ids"][0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    token_rows = pd.DataFrame({"position": range(len(tokens)), "token": tokens, "id": token_ids,
                               "attention_mask": batch["attention_mask"][0].tolist(),
                               "segment_id": batch["token_type_ids"][0].tolist()})
    token_rows.to_csv(output / "tokens.csv", index=False)
    probabilities = prediction_rows(model, trace["probabilities"])
    for index, row in enumerate(probabilities):
        row.update(kind="real_headline" if index == 0 else "synthetic_padding_control",
                   text=example["text"] if index == 0 else "Profits increased.")
    write_json(output / "predictions.json", probabilities)
    n = min(int(batch["attention_mask"][0].sum()), 32)
    matrix = trace["outputs"].attentions[0][0, 0, :n, :n].cpu().numpy()
    fig, ax = plt.subplots(figsize=(11, 9), layout="constrained")
    im = ax.imshow(matrix, cmap="viridis", vmin=0)
    ax.set_xticks(range(n), tokens[:n], rotation=90, fontsize=8)
    ax.set_yticks(range(n), tokens[:n], fontsize=8)
    ax.set(xlabel="Token consultado (clave)", ylabel="Token que consulta",
           title="Atención: primera capa, primera cabeza (hasta 32 tokens)")
    fig.colorbar(im, ax=ax, label="Peso de atención; no es atribución causal")
    fig.savefig(output / "attention.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    grouped = {"Embeddings": inventory[0]["parameters"],
               "12 bloques": sum(row["parameters"] for row in inventory[1:-2]),
               "Pooler y salida": sum(row["parameters"] for row in inventory[-2:])}
    axes[0].bar(list(grouped), [x / 1e6 for x in grouped.values()], color=["#397A66", "#6C609A", "#B44C56"])
    axes[0].set(ylabel="Millones de parámetros", title="Dónde se concentra el modelo")
    norms = [float(h[0, 0].norm()) for h in trace["outputs"].hidden_states]
    axes[1].plot(range(len(norms)), norms, "o-", color="#397A66")
    axes[1].set(xlabel="0: embeddings; 1–12: bloques", ylabel="Norma L2 de [CLS]",
                title="Magnitud de la representación, no importancia")
    fig.savefig(output / "architecture.png", dpi=130)
    plt.close(fig)
    model_files = ["config.json", "pytorch_model.bin", "vocab.txt", "tokenizer_config.json",
                   "special_tokens_map.json"]
    model_hashes = {name: sha256(Path(hf_hub_download(
        MODEL_ID, name, revision=REVISION, cache_dir=str(CACHE), local_files_only=True)))
        for name in model_files}
    source_files = ["src/nlp/finbert.py", "src/nlp/inspect_finbert.py", "requirements-finbert.txt"]
    manifest = {
        "status": "complete", "study": "forward_pass_only", "model_id": MODEL_ID,
        "revision": REVISION, "device": args.device, "threads": args.threads,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ["torch", "transformers", "huggingface-hub", "tokenizers", "safetensors", "numpy", "pandas", "matplotlib"]},
        "input_sha256": {str(news_path.relative_to(ROOT).as_posix()): sha256(news_path)},
        "source_sha256": {name: sha256(ROOT / name) for name in source_files},
        "model_sha256": model_hashes, "example": example, "lengths": lengths,
        "synthetic_truncation": long_lengths[0], "parameters": sum(p.numel() for p in model.parameters()),
        "hidden_state_shapes": [list(h.shape) for h in trace["outputs"].hidden_states],
        "first_layer_shapes": {k: list(v.shape) for k, v in trace["first_layer"].items()},
        "checks_passed": len(trace["checks"]), "padding_mask_verified": bool(padding.any()),
        "batch_invariance_verified": True, "elapsed_seconds": time.perf_counter() - started,
        "limitations": ["No human labels or sentiment quality metrics", "Not a financial prediction experiment",
                        "Attention is not causal explanation", "Checkpoint training corpus overlap not fully audited"],
        "output_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps({"output": str(output), "checks": len(trace["checks"]),
                      "parameters": manifest["parameters"]}, indent=2))


if __name__ == "__main__":
    main()
