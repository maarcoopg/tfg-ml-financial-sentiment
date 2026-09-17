import unittest

from src.experiments.run_variable_ablation import CONTRASTS, feature_subset
from src.experiments.sentiment_representation import EXTRA


class VariableAblationTests(unittest.TestCase):
    def test_isolates_feature_family(self):
        for variant in ["hybrid", "lag_1"]:
            for feature in EXTRA:
                with self.subTest(variant=variant, feature=feature):
                    clean = ["return", "sentiment_mean"]
                    full = clean + EXTRA + ([c + "_lag_1" for c in EXTRA] if variant == "lag_1" else [])
                    family = {feature, feature + "_lag_1"} if variant == "lag_1" else {feature}
                    added = feature_subset(full, clean, variant, feature, "add")
                    dropped = feature_subset(full, clean, variant, feature, "drop")
                    self.assertEqual(set(added) - set(clean), family)
                    self.assertEqual(set(full) - set(dropped), family)
                    self.assertEqual(added, [c for c in full if c in added])
                    self.assertEqual(dropped, [c for c in full if c in dropped])
                    self.assertTrue(set(clean).issubset(dropped))

    def test_invalid_ablation_rejected(self):
        for variant in ["base", "hybrid"]:
            with self.assertRaises(ValueError):
                feature_subset([], [], variant, EXTRA[0], "add")
        self.assertEqual(len(CONTRASTS), 10)
        self.assertEqual(len(set(CONTRASTS)), 10)
        self.assertTrue(all(ref == ("deduplicated" if name.startswith("add_") else "enhanced")
                            for name, ref in CONTRASTS))
