import sqlite3
from contextlib import closing
import tempfile
from pathlib import Path
import unittest

import pandas as pd

from src.nlp.news_score_cache import cached_scores


def predict(texts):
    return [{"positive": 0.7, "negative": 0.1, "neutral": 0.2, "score": 0.6, "label": "positive"}
            for _ in texts]


class NewsScoreCacheTests(unittest.TestCase):
    def test_exact_text_deduplication_and_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / "scores.sqlite"
            first = cached_scores(["Apple", "Apple", "Microsoft"], cache, {"revision": "one"}, predict, 1)
            self.assertEqual(len(first), 2)
            def no_inference(texts):
                self.fail("Cached text was inferred again")
            resumed = cached_scores(["Microsoft", "Apple"], cache, {"revision": "one"}, no_inference)
            pd.testing.assert_frame_equal(first, resumed)
            third = cached_scores(["Apple", "Apple "], cache, {"revision": "one"}, predict)
            self.assertEqual(len(third), 2)
            with self.assertRaises(ValueError):
                cached_scores(["Apple"], cache, {"revision": "two"}, predict)

    def test_interrupted_batches_remain_reusable(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / "scores.sqlite"
            def fails(texts):
                if "Microsoft" in texts:
                    raise RuntimeError("Simulated interruption")
                return predict(texts)
            with self.assertRaises(RuntimeError):
                cached_scores(["Apple", "Microsoft"], cache, {}, fails, 1)
            received = []
            def resumed(texts):
                received.extend(texts)
                return predict(texts)
            self.assertEqual(len(cached_scores(["Apple", "Microsoft"], cache, {}, resumed)), 2)
            self.assertEqual(received, ["Microsoft"])

    def test_corrupt_cache_and_invalid_predictions_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / "scores.sqlite"
            with self.assertRaises(ValueError):
                cached_scores(["Apple"], cache, {}, lambda texts: [])
            cached_scores(["Apple"], cache, {}, predict)
            with closing(sqlite3.connect(cache)) as connection:
                connection.execute("UPDATE scores SET text='Different text'")
                connection.commit()
            with self.assertRaises(ValueError):
                cached_scores(["Apple"], cache, {}, predict)


if __name__ == "__main__":
    unittest.main()
