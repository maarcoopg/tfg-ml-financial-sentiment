import importlib.util
import unittest

AVAILABLE = all(importlib.util.find_spec(name) for name in ["torch", "transformers", "captum"])
if AVAILABLE:
    import torch
    import tests.test_finbert as baseline
    from src.nlp.sentiment_behavior import infer_texts, integrated_attribution


@unittest.skipUnless(AVAILABLE, "Install optional FinBERT dependencies")
class AttributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        baseline.FinbertTests.setUpClass()
        cls.tokenizer = baseline.FinbertTests.tokenizer

    @classmethod
    def tearDownClass(cls):
        baseline.FinbertTests.tearDownClass()

    def test_completeness_and_fixed_specials(self):
        model = baseline.FinbertTests().model()
        for reference in ["pad", "mask"]:
            result = integrated_attribution(self.tokenizer, model, "profit rose .", reference, steps=(16, 32))
            self.assertTrue(result["converged"])
            self.assertAlmostEqual(result["sum_attributions"] -
                (result["original_logit"] - result["reference_logit"]), result["delta"], places=5)
            for value, special in zip(result["attributions"], result["special_mask"]):
                if special:
                    self.assertEqual(value, 0)
            for row in result["perturbations"]:
                self.assertFalse(any(result["special_mask"][i] for i in row["positions"]))

    def test_reference_and_mode_are_validated(self):
        model = baseline.FinbertTests().model()
        with self.assertRaises(ValueError):
            integrated_attribution(self.tokenizer, model, "profit", reference="unknown")
        with self.assertRaises(ValueError):
            integrated_attribution(self.tokenizer, model.train(), "profit")

    def test_batching_preserves_order(self):
        model = baseline.FinbertTests().model()
        texts = ["profit rose .", "profit fell .", "profit"]
        a = infer_texts(self.tokenizer, model, texts, batch_size=1)
        b = infer_texts(self.tokenizer, model, texts, batch_size=3)
        self.assertEqual([r["label"] for r in a], [r["label"] for r in b])
        for x, y in zip(a, b):
            self.assertAlmostEqual(x["score"], y["score"], places=5)


if __name__ == "__main__":
    unittest.main()
