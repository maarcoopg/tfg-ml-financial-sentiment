"""Migrate the legacy report layout without retraining or changing result bytes."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from src.experiments.artifacts import ROOT, artifact_dir, refresh_outputs, sha256, write_json


HISTORICAL = {"comparison", "feature_importance", "figures", "metrics",
              "model_metadata", "predictions", "tuned_metrics", "tuned_predictions", "tuning"}
VERIFICATION = {"legacy-smoke-20260908", "review-smoke-20260908",
                "review-verification-20260908", "tuning-verification-20260908"}


def destination(relative: Path) -> Path:
    parts = relative.parts
    if parts[1] in HISTORICAL:
        return Path("reports/historical", *parts[1:])
    if parts[1] != "runs":
        return relative
    run_id, tail = parts[2], Path(*parts[3:])
    output = Path("artifacts/verification" if run_id in VERIFICATION else "reports/experiments") / run_id
    if tail.parts[0] in {"models", "data", "source"}:
        base = {"models": "models/experiments", "data": "data/experiments",
                "source": "artifacts/snapshots"}[tail.parts[0]]
        return Path(base) / run_id / Path(*tail.parts[1:])
    if len(tail.parts) == 1 and tail.suffix == ".csv":
        if tail.name == "predictions.csv":
            category = "predictions"
        elif tail.name in {"inner_cv.csv", "selected_params.csv", "best_temporal_cv_params.csv",
                           "temporal_cv_fold_results.csv", "temporal_cv_results.csv"}:
            category = "tuning"
        else:
            category = "metrics"
        return output / category / tail
    return output / tail


def migrate(root: Path, resume: bool = False) -> dict:
    root = root.resolve()
    journal_path = root / "reports/historical/migration_manifest.json"
    if journal_path.exists() and not resume:
        raise FileExistsError("La migracion ya tiene un registro; no se sobrescribira")
    records = []
    for source in sorted((root / "reports").rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(root)
        target = destination(relative)
        if target == relative:
            continue
        source.resolve().relative_to(root)
        (root / target).resolve().relative_to(root)
        if (root / target).exists():
            raise FileExistsError(root / target)
        records.append({"old": relative.as_posix(), "new": target.as_posix(), "sha256": sha256(source)})
    if resume:
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        if journal["status"] == "complete":
            raise ValueError("La migracion ya esta completa")
        records = journal["files"]
    elif not records:
        raise ValueError("No hay informes antiguos que migrar")
    else:
        journal = {"status": "moving", "created_at": datetime.now(timezone.utc).isoformat(), "files": records}
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(journal_path, journal)
    # Move individual checked files, never recursively move an unchecked directory.
    for record in records:
        source, target = root / record["old"], root / record["new"]
        source.resolve().relative_to(root)
        target.resolve().relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.exists():
            if target.exists():
                raise FileExistsError(target)
            source.rename(target)
        original = target
        if target.name == "manifest.json":
            archive = artifact_dir(target.parent, "source", root) / "original_manifest.json"
            if archive.exists():
                original = archive
        if sha256(original) != record["sha256"]:
            raise RuntimeError(f"Hash distinto tras mover {target}")

    mapping = {record["old"]: record["new"] for record in records}
    run_ids = {Path(record["old"]).parts[2] for record in records if record["old"].startswith("reports/runs/")}
    for run_id in sorted(run_ids):
        output = root / ("artifacts/verification" if run_id in VERIFICATION else "reports/experiments") / run_id
        path = output / "manifest.json"
        if path.exists():
            manifest = json.loads(path.read_text(encoding="utf-8"))
            archive = artifact_dir(output, "source", root) / "original_manifest.json"
            archive.parent.mkdir(parents=True, exist_ok=True)
            if not archive.exists():
                shutil.copy2(path, archive)
            manifest["original_manifest"] = archive.relative_to(root).as_posix()
            updated_inputs = {}
            for name, digest in manifest.get("input_sha256", {}).items():
                normalized = name.replace("\\", "/")
                updated_inputs[mapping.get(normalized, normalized)] = digest
            manifest["input_sha256"] = updated_inputs
        else:
            manifest = {"status": "legacy_unverified", "evaluation_status": "Technical smoke test; original manifest unavailable."}
        manifest["migration"] = {"from": f"reports/runs/{run_id}", "result_bytes_unchanged": True}
        refresh_outputs(output, manifest, root)
    # Only remove empty directories inside the reports tree.
    journal["empty_directories_retained"] = []
    for directory in sorted((root / "reports").rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.resolve().relative_to(root / "reports")
            try:
                directory.rmdir()
            except PermissionError:
                journal["empty_directories_retained"].append(directory.relative_to(root).as_posix())
    journal["status"] = "complete"
    write_json(journal_path, journal)
    return journal


def publish_final(run_dir: Path, root: Path = ROOT) -> None:
    root = root.resolve()
    run_dir = run_dir.resolve()
    run_dir.relative_to(root / "reports/experiments")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Solo se pueden seleccionar resultados de una ejecucion completa")
    final = root / "reports/final"
    selected = [run_dir / "metrics" / name for name in ["global_metrics.csv", "auc_intervals.csv"]]
    selected.extend(sorted((run_dir / "figures").glob("*.png")))
    for source in selected:
        relative = source.relative_to(root).as_posix()
        if sha256(source) != manifest["output_sha256"].get(relative):
            raise ValueError(f"El archivo no coincide con el manifiesto: {source}")
    final.mkdir(parents=True, exist_ok=False)
    provenance = {"run_id": run_dir.name, "source_manifest": (run_dir / "manifest.json").relative_to(root).as_posix(),
                  "evaluation_status": manifest.get("evaluation_status"), "files": {}}
    for source in selected:
        target = final / ("figures" if source.suffix == ".png" else "tables") / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        provenance["files"][target.relative_to(final).as_posix()] = {
            "source": source.relative_to(root).as_posix(), "sha256": sha256(target)}
    write_json(final / "provenance.json", provenance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organiza los informes conservando sus resultados y procedencia.")
    parser.add_argument("--migrate", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--publish-final", type=Path)
    args = parser.parse_args()
    if args.migrate:
        print(f"Archivos trasladados: {len(migrate(ROOT, resume=args.resume)['files'])}")
    if args.publish_final:
        publish_final(args.publish_final)
        print("Seleccion guardada en reports/final")
