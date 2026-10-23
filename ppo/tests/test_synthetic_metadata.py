"""Check named synthetic metadata and the dose builder without source hashes."""

import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
import synth_dose
import synth_market as v1
import synth_market_v2 as v2


def small_params(params):
    params = copy.deepcopy(params)
    params.update(train_bars=360, val_bars=240, block_bars=240, n_blocks=1)
    return params


class SyntheticMetadataTests(unittest.TestCase):
    def test_v1_metadata_and_dose_build(self):
        params = small_params(v1.PARAMS)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            base = Path(directory) / "base"
            v1.build(str(base), p=params)
            metadata = json.loads((base / "params.json").read_text())
            self.assertEqual(metadata, {"benchmark": "SYNTH_V1", "params": params})
            dose = Path(directory) / "dose"
            with mock.patch.object(synth_dose, "project_path", return_value=str(base / "feature_scaler.pkl")):
                synth_dose.build(str(dose), n_blocks=1)
            pool = pd.read_parquet(dose / "pool.parquet")
            self.assertEqual(len(pool), v1.PARAMS["block_bars"])
            self.assertFalse(pool[v1.FEATURES].isna().any().any())

    def test_v2_metadata(self):
        params = small_params(v2.PARAMS)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), \
             mock.patch.object(v2, "PARAMS", params), \
             mock.patch.object(sys, "argv", ["synth_market_v2.py", "--out", directory]):
            v2.main()
            metadata = json.loads((Path(directory) / "params.json").read_text())
            self.assertEqual(metadata, {"benchmark": "SYNTH_V2", "params": params})
            self.assertEqual(len(pd.read_parquet(Path(directory) / "train.parquet")), 360)

    def test_v2_check_only_writes_results_without_source_hashes(self):
        result = {"dev_shift": {"timer_sharpe": 1.0, "bh_sharpe": 0.0, "auc": 0.55}}
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(output), \
             mock.patch.object(v1, "check", return_value=result), \
             mock.patch.object(sys, "argv", ["synth_market_v2.py", "--out", directory, "--check-only"]):
            v2.main()
            saved = json.loads((Path(directory) / "cheap_check.json").read_text())
            self.assertTrue(saved["passed"])
            self.assertNotIn("hash", output.getvalue())
            self.assertFalse((Path(directory) / "params.json").exists())


if __name__ == "__main__":
    unittest.main()
