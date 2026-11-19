
import json
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


class TestDS03Project(unittest.TestCase):

    def test_cleaned_dataset(self):
        path = ROOT / "data" / "processed" / "all_cleaned_records.csv"
        self.assertTrue(path.exists())

        df = pd.read_csv(path)

        self.assertEqual(len(df), 921)
        self.assertIn("clean_text", df.columns)
        self.assertIn("language", df.columns)
        self.assertIn("record_type", df.columns)

    def test_record_distribution(self):
        path = ROOT / "data" / "processed" / "all_cleaned_records.csv"
        df = pd.read_csv(path)

        self.assertEqual(
            df["record_type"].value_counts().get("video", 0), 293
        )
        self.assertEqual(
            df["record_type"].value_counts().get("comment", 0), 628
        )

    def test_sentiment_results(self):
        path = ROOT / "reports" / "sentiment_results.csv"
        self.assertTrue(path.exists())

        df = pd.read_csv(path)

        self.assertEqual(len(df), 921)
        self.assertIn("sentiment", df.columns)

    def test_topic_results(self):
        path = ROOT / "reports" / "topic_results.csv"
        self.assertTrue(path.exists())

        df = pd.read_csv(path)
        self.assertEqual(len(df), 772)

    def test_influencer_network(self):
        path = ROOT / "reports" / "influencer_network.json"
        self.assertTrue(path.exists())

        with open(path, encoding="utf-8") as file:
            graph = json.load(file)

        self.assertEqual(len(graph["nodes"]), 855)
        self.assertEqual(len(graph["edges"]), 582)

    def test_misinformation_model(self):
        path = ROOT / "models" / "misinformation_model.joblib"
        self.assertTrue(path.exists())

    def test_dashboard(self):
        path = ROOT / "dashboard" / "app.py"
        self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
