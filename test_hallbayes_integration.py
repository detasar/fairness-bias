import os
import sys
import types
import unittest
import pandas as pd


class DummyBackend:  # minimal stub
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model


class DummyItem:
    def __init__(self, prompt: str, n_samples: int, m: int, skeleton_policy: str):
        self.prompt = prompt
        self.n_samples = n_samples
        self.m = m
        self.skeleton_policy = skeleton_policy


class DummyPlanner:
    def __init__(self, backend: DummyBackend, temperature: float = 0.3):
        self.backend = backend
        self.temperature = temperature

    def run(self, items, **kwargs):
        results = []
        for idx, item in enumerate(items):
            ns = types.SimpleNamespace(
                decision_answer=(idx % 2 == 0),
                roh_bound=0.1 * idx,
                delta_bar=1.0,
                b2t=0.5,
                isr=1.2,
                q_lo=0.3,
                q_bar=0.6,
            )
            results.append(ns)
        return results


def _install_fake_hallbayes():
    """Install fake hallbayes modules into sys.modules for testing."""
    hallbayes_mod = types.ModuleType("hallbayes")
    scripts_mod = types.ModuleType("hallbayes.scripts")
    toolkit_mod = types.ModuleType("hallbayes.scripts.hallucination_toolkit")
    toolkit_mod.OpenAIBackend = DummyBackend
    toolkit_mod.OpenAIItem = DummyItem
    toolkit_mod.OpenAIPlanner = DummyPlanner
    sys.modules["hallbayes"] = hallbayes_mod
    sys.modules["hallbayes.scripts"] = scripts_mod
    sys.modules["hallbayes.scripts.hallucination_toolkit"] = toolkit_mod


class TestHallbayesFairness(unittest.TestCase):
    def setUp(self):
        _install_fake_hallbayes()

    def test_analysis_outputs_dataframe_and_file(self):
        from hallbayes_fairness import hallucination_fairness_analysis

        prompts = ["Q1", "Q2"]
        groups = ["A", "B"]
        out_file = "hb_metrics.csv"
        df = hallucination_fairness_analysis(prompts, groups, output_file=out_file)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(list(df.columns), [
            "group", "prompt", "decision_answer", "roh_bound", "delta_bar", "b2t", "isr", "q_lo", "q_bar"
        ])
        self.assertEqual(len(df), 2)
        self.assertTrue(os.path.exists(out_file))
        os.remove(out_file)

    def test_analysis_from_csv(self):
        from hallbayes_fairness import hallucination_fairness_from_csv

        df_input = pd.DataFrame({"group": ["A", "B"], "prompt": ["Q1", "Q2"]})
        csv_file = "hb_prompts.csv"
        df_input.to_csv(csv_file, index=False)
        out_file = "hb_metrics.csv"
        df = hallucination_fairness_from_csv(csv_file, output_file=out_file)

        self.assertEqual(len(df), 2)
        self.assertTrue(os.path.exists(out_file))
        os.remove(csv_file)
        os.remove(out_file)


if __name__ == '__main__':
    unittest.main()
