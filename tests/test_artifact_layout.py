import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.experiments.artifacts import artifact_dir, create_run, finish_run, sha256
from src.experiments.organize_reports import destination, migrate, publish_final


class ArtifactLayoutTests(unittest.TestCase):
    def test_destination_separates_outputs(self):
        cases = {
            "reports/metrics/a.csv": "reports/historical/metrics/a.csv",
            "reports/runs/review-full-20260908/global_metrics.csv": "reports/experiments/review-full-20260908/metrics/global_metrics.csv",
            "reports/runs/review-full-20260908/models/a.joblib": "models/experiments/review-full-20260908/a.joblib",
            "reports/runs/review-full-20260908/data/a.csv": "data/experiments/review-full-20260908/a.csv",
            "reports/runs/review-smoke-20260908/inner_cv.csv": "artifacts/verification/review-smoke-20260908/tuning/inner_cv.csv",
            "reports/runs/review-smoke-20260908/source/src/a.py": "artifacts/snapshots/review-smoke-20260908/src/a.py",
        }
        for source, target in cases.items():
            self.assertEqual(destination(Path(source)), Path(target))

    def test_create_and_finish_hash_companion_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "src/example.py").write_text("pass\n")
            (root / "requirements.txt").write_text("numpy\n")
            with patch("src.experiments.artifacts.importlib.metadata.version", return_value="test"):
                output, manifest = create_run(root, Path("reports/experiments/example"), {})
                model_dir = artifact_dir(output, "models", root)
                model_dir.mkdir(parents=True)
                (model_dir / "model.joblib").write_bytes(b"model")
                finish_run(output, manifest, [root / "requirements.txt"], root)
                self.assertEqual(manifest["status"], "complete")
                self.assertEqual(manifest["output_hash_base"], "project_root")
                self.assertIn("models/experiments/example/model.joblib", manifest["output_sha256"])
                self.assertFalse((output / "source").exists())
                for relative, digest in manifest["output_sha256"].items():
                    self.assertEqual(sha256(root / relative), digest)
                with self.assertRaises(FileExistsError):
                    create_run(root, Path("artifacts/verification/example"), {})

    def test_outside_output_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                create_run(root, root.parent / "outside-run", {})

    def test_migration_preserves_bytes_manifest_and_readme(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_bytes(b"unchanged")
            run = root / "reports/runs/review-full-20260908"
            (run / "models").mkdir(parents=True)
            (run / "models/a.joblib").write_bytes(b"model")
            (run / "global_metrics.csv").write_bytes(b"a,b\n1,2\n")
            (run / "manifest.json").write_text(json.dumps({"status": "complete"}))
            original_manifest_hash = sha256(run / "manifest.json")
            journal = migrate(root)
            self.assertEqual(len(journal["files"]), 3)
            self.assertEqual((root / "README.md").read_bytes(), b"unchanged")
            self.assertFalse((root / "reports/runs").exists())
            new_run = root / "reports/experiments/review-full-20260908"
            manifest = json.loads((new_run / "manifest.json").read_text())
            self.assertEqual(sha256(root / manifest["original_manifest"]), original_manifest_hash)
            for path, digest in manifest["output_sha256"].items():
                self.assertEqual(sha256(root / path), digest)
            with self.assertRaises(FileExistsError):
                migrate(root)

    def test_migration_collision_does_not_move_anything(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ["reports/metrics/a.csv", "reports/historical/metrics/a.csv"]:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("a")
            with self.assertRaises(FileExistsError):
                migrate(root)
            self.assertTrue((root / "reports/metrics/a.csv").exists())

    def test_final_selection_has_provenance_and_no_readme(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "reports/experiments/example"
            (run / "metrics").mkdir(parents=True)
            for name in ["global_metrics.csv", "auc_intervals.csv"]:
                (run / "metrics" / name).write_text("a\n1\n")
            (run / "figures").mkdir()
            (run / "figures/roc_auc.png").write_bytes(b"figure")
            finish_run(run, {"evaluation_status": "exploratory"}, [], root)
            publish_final(run, root)
            final = root / "reports/final"
            provenance = json.loads((final / "provenance.json").read_text())
            self.assertEqual(len(provenance["files"]), 3)
            self.assertEqual(provenance["evaluation_status"], "exploratory")
            self.assertFalse(list(final.rglob("README*")))
            for target, entry in provenance["files"].items():
                self.assertEqual(sha256(final / target), sha256(root / entry["source"]))
            with self.assertRaises(FileExistsError):
                publish_final(run, root)


if __name__ == "__main__":
    unittest.main()
