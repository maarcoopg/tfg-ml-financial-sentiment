"""Resumable exact-text inference with a checkpoint/environment-bound cache."""

import importlib.metadata
from contextlib import closing
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from src.experiments.artifacts import sha256, write_json
from src.experiments.finbert_panel import replace_sentiment
from src.nlp.sentiment_data import LABELS, digest


def cached_scores(texts, path, metadata, predict, batch_size=16):
    if batch_size < 1 or not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
        raise ValueError("Nonempty texts and positive batch size required")
    unique = {digest(text): text for text in texts}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY CHECK (id=1), value TEXT NOT NULL)")
        connection.execute("CREATE TABLE IF NOT EXISTS scores (hash TEXT PRIMARY KEY, text TEXT NOT NULL, value TEXT NOT NULL)")
        encoded = json.dumps(metadata, sort_keys=True)
        previous = connection.execute("SELECT value FROM metadata WHERE id=1").fetchone()
        if previous and previous[0] != encoded:
            raise ValueError("Cache checkpoint, inference policy or environment mismatch")
        connection.execute("INSERT OR IGNORE INTO metadata VALUES (1, ?)", (encoded,))
        connection.commit()
        stored = {key: (text, json.loads(value)) for key, text, value in connection.execute("SELECT hash,text,value FROM scores")}
        for key, (text, _) in stored.items():
            if digest(text) != key:
                raise ValueError("Cache text hash mismatch")
        pending = sorted(set(unique) - set(stored), key=lambda key: (len(unique[key]), key))
        print(f"FinBERT: {len(unique)} textos unicos, {len(pending)} pendientes", flush=True)
        for start in range(0, len(pending), batch_size):
            keys = pending[start:start + batch_size]
            rows = predict([unique[key] for key in keys])
            if len(rows) != len(keys):
                raise ValueError("Inference batch lost rows")
            checked = pd.DataFrame([{**row, "text_sha256": key} for key, row in zip(keys, rows)])
            replace_sentiment(pd.DataFrame({"text_sha256": keys}), checked)
            for key, row in zip(keys, rows):
                connection.execute("INSERT INTO scores VALUES (?, ?, ?)",
                                   (key, unique[key], json.dumps(row, sort_keys=True, allow_nan=False)))
                stored[key] = (unique[key], row)
            connection.commit()
            if start % (batch_size * 25) == 0 or start + batch_size >= len(pending):
                print(f"FinBERT: {min(start + batch_size, len(pending))}/{len(pending)} calculados", flush=True)
    result = pd.DataFrame([{**stored[key][1], "text_sha256": key} for key in sorted(unique)])
    result = result.reindex(sorted(result.columns), axis=1)
    replace_sentiment(pd.DataFrame({"text_sha256": sorted(unique)}), result)
    return result


def score_corpus(news, cache, output, device="cpu", batch_size=16):
    import torch
    from src.nlp.finbert import CACHE, MODEL_ID, REVISION, load_finbert
    from src.nlp.sentiment_behavior import infer_texts

    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tokenizer, model = load_finbert(device="cpu")
    if str(model.dtype) != "torch.float32":
        raise ValueError("FP32 inference required")
    files = CACHE / "models--ProsusAI--finbert" / "snapshots" / REVISION
    metadata = {"model_id": MODEL_ID, "revision": REVISION, "device": device,
        "dtype": "float32", "attention": "eager", "tf32": False, "max_length": 512,
        "representation": "unchanged title plus summary from text", "batch_size": batch_size,
        "batch_order": "text length then sha256", "seed": 42,
        "checkpoint_sha256": {name: sha256(files / name) for name in
            ["pytorch_model.bin", "config.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"]},
        "versions": {name: importlib.metadata.version(name) for name in
            ["torch", "transformers", "tokenizers", "huggingface-hub", "numpy"]}}
    # Fix equivalence examples by hash, independently of financial outcomes.
    unique = news[["text_sha256", "text"]].drop_duplicates("text_sha256").sort_values("text_sha256")
    examples = unique.head(16)
    if device == "cuda":
        cpu = pd.DataFrame(infer_texts(tokenizer, model, examples.text.tolist(), batch_size))
        model.to(device)
        gpu = pd.DataFrame(infer_texts(tokenizer, model, examples.text.tolist(), batch_size))
        np.testing.assert_allclose(cpu[list(LABELS)], gpu[list(LABELS)], atol=2e-5, rtol=0)
        if not cpu.label.equals(gpu.label):
            raise ValueError("CPU/GPU label mismatch")
        metadata["gpu"] = torch.cuda.get_device_name()
        error = np.abs(cpu[list(LABELS)].to_numpy() - gpu[list(LABELS)].to_numpy()).max(axis=1)
        examples[["text_sha256"]].assign(max_probability_error=error, same_label=True).to_csv(
            output / "cpu_gpu_equivalence.csv", index=False)
    write_json(output / "inference_metadata.json", metadata)
    result = cached_scores(unique.text.tolist(), cache, metadata,
        lambda texts: infer_texts(tokenizer, model, texts, batch_size), batch_size)
    result.to_csv(output / "text_scores.csv", index=False)
    return result, metadata
