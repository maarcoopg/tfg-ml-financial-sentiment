from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def artifact_dir(output: Path, kind: str, root: Path = ROOT) -> Path:
    """Resolve companion artifacts by run ID, independently of the reports location."""
    locations = {"models": "models/experiments", "data": "data/experiments",
                 "source": "artifacts/snapshots"}
    return root / locations[kind] / output.name if kind in locations else output / kind


def refresh_outputs(output: Path, manifest: dict, root: Path = ROOT) -> None:
    directories = {"reports": output, **{kind: artifact_dir(output, kind, root)
                                        for kind in ["models", "data", "source"]}}
    manifest["layout_version"] = 2
    manifest["run_id"] = output.name
    manifest["artifact_paths"] = {kind: path.resolve().relative_to(root.resolve()).as_posix()
                                  for kind, path in directories.items()}
    manifest["output_sha256"] = {
        p.resolve().relative_to(root.resolve()).as_posix(): sha256(p)
        for directory in directories.values() for p in sorted(directory.rglob("*"))
        if p.is_file() and p != output / "manifest.json"
    }
    manifest["output_hash_base"] = "project_root"
    write_json(output / "manifest.json", manifest)


def refresh_manifest(output: Path, root: Path = ROOT) -> None:
    path = output / "manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("layout_version") == 2:
            refresh_outputs(output, manifest, root)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def create_run(root: Path, output: Path | None, config: dict) -> tuple[Path, dict]:
    output = output or root / "reports/experiments" / datetime.now(timezone.utc).strftime("review-%Y%m%dT%H%M%S%fZ")
    root = root.resolve()
    output = output if output.is_absolute() else root / output
    output = output.resolve()
    output.relative_to(root)
    for kind in ["models", "data", "source"]:
        companion = artifact_dir(output, kind, root)
        if companion.exists():
            raise FileExistsError(f"El identificador de ejecucion ya existe: {companion}")
    output.mkdir(parents=True, exist_ok=False)
    def git(*args):
        result = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else "unavailable"
    manifest = {
        "status": "running", "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config, "python": platform.python_version(),
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--short"),
        "versions": {name: importlib.metadata.version(name) for name in [
            "pandas", "numpy", "scikit-learn", "lightgbm", "xgboost",
            "pandas_market_calendars", "exchange-calendars", "scipy", "joblib", "tzdata"]},
        "source_sha256": {str(p.relative_to(root)): sha256(p) for p in sorted((root / "src").rglob("*.py"))},
        "requirements_sha256": sha256(root / "requirements.txt"),
        "evaluation_status": "Exploratory nested evaluation on previously inspected historical dates; not a new untouched holdout.",
    }
    for relative_path in manifest["source_sha256"]:
        snapshot = artifact_dir(output, "source", root) / relative_path
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative_path, snapshot)
    source_dir = artifact_dir(output, "source", root)
    source_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "requirements.txt", source_dir / "requirements.txt")
    refresh_outputs(output, manifest, root)
    return output, manifest


def finish_run(output: Path, manifest: dict, inputs: list[Path], root: Path) -> None:
    manifest["input_sha256"] = {str(p.relative_to(root)): sha256(p) for p in inputs}
    manifest["status"] = "complete"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    refresh_outputs(output, manifest, root)
