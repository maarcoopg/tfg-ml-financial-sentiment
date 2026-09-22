import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


AVAILABLE = all(importlib.util.find_spec(name) for name in ["torch", "transformers"])
if AVAILABLE:
    import torch
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
    from src.nlp.finbert import (MODEL_ID, REVISION, encode_texts, load_finbert,
                                parameter_inventory, prediction_rows, trace_forward)


@unittest.skipUnless(AVAILABLE, "Install requirements-finbert.txt for offline BERT tests")
class FinbertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.directory = tempfile.TemporaryDirectory()
        vocab = Path(cls.directory.name) / "vocab.txt"
        vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
                                    "profit", "rose", "fell", "not", "."]), encoding="utf-8")
        cls.tokenizer = BertTokenizerFast(vocab_file=str(vocab), model_max_length=512)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def model(self):
        torch.manual_seed(17)
        config = BertConfig(vocab_size=10, hidden_size=24, num_hidden_layers=2,
                            num_attention_heads=3, intermediate_size=32,
                            id2label={0: "negative", 1: "neutral", 2: "positive"})
        config._attn_implementation = "eager"
        return BertForSequenceClassification(config).eval()

    def test_invalid_texts(self):
        for texts in [[], "profit", [""], ["   "], [None], [1]]:
            with self.subTest(texts=texts), self.assertRaises(ValueError):
                encode_texts(self.tokenizer, texts)

    def test_invalid_limits(self):
        for limit in [2, 513, True, 3.5]:
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                encode_texts(self.tokenizer, ["profit"], max_length=limit)

    def test_lengths_include_special_tokens_not_padding(self):
        batch, lengths = encode_texts(self.tokenizer, ["profit rose .", "profit"])
        self.assertEqual(batch["input_ids"].shape, (2, 5))
        self.assertEqual([row["used_tokens"] for row in lengths], [5, 3])
        self.assertFalse(any(row["truncated"] for row in lengths))

    def test_truncation_is_reported(self):
        batch, lengths = encode_texts(self.tokenizer, ["profit " * 520])
        self.assertEqual(lengths[0], {"full_tokens": 522, "used_tokens": 512,
                                      "removed_tokens": 10, "truncated": True})
        self.assertEqual(batch["input_ids"][0, -1], self.tokenizer.sep_token_id)

    def test_reconstruction_matches_original(self):
        model = self.model()
        batch, _ = encode_texts(self.tokenizer, ["profit rose .", "profit"])
        trace = trace_forward(model, batch)
        self.assertEqual(len(trace["checks"]), 8)
        self.assertTrue(all(row["passed"] for row in trace["checks"]), trace["checks"])

    def test_padding_is_masked_and_batch_invariant(self):
        model = self.model()
        batch, _ = encode_texts(self.tokenizer, ["profit rose .", "profit"])
        trace = trace_forward(model, batch)
        self.assertTrue(torch.all(trace["outputs"].attentions[0][1, :, :, 3:] == 0))
        solo, _ = encode_texts(self.tokenizer, ["profit"])
        other = trace_forward(model, solo)
        torch.testing.assert_close(trace["probabilities"][1], other["probabilities"][0])

    def test_dropout_training_is_rejected(self):
        batch, _ = encode_texts(self.tokenizer, ["profit"])
        with self.assertRaises(ValueError):
            trace_forward(self.model().train(), batch)

    def test_label_mapping_is_not_assumed(self):
        rows = prediction_rows(self.model(), torch.tensor([[0.1, 0.2, 0.7]]))
        self.assertEqual(rows[0]["label"], "positive")
        self.assertAlmostEqual(rows[0]["score"], 0.6)

    def test_inventory_covers_every_parameter(self):
        model = self.model()
        self.assertEqual(sum(row["parameters"] for row in parameter_inventory(model)),
                         sum(p.numel() for p in model.parameters()))

    def test_loading_is_offline_pinned_and_without_remote_code(self):
        with patch("src.nlp.finbert.AutoTokenizer.from_pretrained") as tokenizer_loader:
            with patch("src.nlp.finbert.AutoModelForSequenceClassification.from_pretrained",
                       return_value=self.model()) as model_loader:
                load_finbert()
        for loader in [tokenizer_loader, model_loader]:
            self.assertEqual(loader.call_args.args, (MODEL_ID,))
            self.assertEqual(loader.call_args.kwargs["revision"], REVISION)
            self.assertTrue(loader.call_args.kwargs["local_files_only"])
            self.assertFalse(loader.call_args.kwargs["trust_remote_code"])
        self.assertTrue(model_loader.call_args.kwargs["weights_only"])

    def test_segment_embeddings_are_included(self):
        batch, _ = encode_texts(self.tokenizer, ["profit rose ."])
        batch["token_type_ids"][0, 2:] = 1
        trace = trace_forward(self.model(), batch)
        self.assertTrue(all(row["passed"] for row in trace["checks"]))

    def test_optimized_attention_is_rejected(self):
        model = self.model()
        model.config._attn_implementation = "sdpa"
        batch, _ = encode_texts(self.tokenizer, ["profit"])
        with self.assertRaises(ValueError):
            trace_forward(model, batch)


if __name__ == "__main__":
    unittest.main()
