"""CLI imports must be inert; run with python -m unittest discover -s ppo/tests."""

import argparse
import contextlib
from functools import partial
import importlib
import io
import multiprocessing
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
MODULES = ("coevo_seedpool", "coevo_floor")


class CliImportTests(unittest.TestCase):
    def test_imports_do_not_run_cli_or_touch_run_files(self):
        for name in MODULES:
            with self.subTest(module=name):
                sys.modules.pop(name, None)
                output = io.StringIO()
                with mock.patch.object(sys, "argv", ["unrelated-program", "--unknown"]), \
                     mock.patch.dict(os.environ, {"MULT": "not-a-number"}), \
                     mock.patch.object(argparse.ArgumentParser, "parse_args") as parse, \
                     mock.patch("builtins.open") as open_file, \
                     mock.patch.object(multiprocessing, "get_context") as context, \
                     mock.patch("os.chdir") as chdir, \
                     contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                    module = importlib.import_module(name)
                parse.assert_not_called()
                open_file.assert_not_called()
                context.assert_not_called()
                chdir.assert_not_called()
                self.assertEqual(output.getvalue(), "")
                self.assertTrue(callable(module.main))

    def test_help_from_unrelated_directory_without_scientific_dependencies(self):
        # -S disables site-packages, so even installed training packages are unavailable.
        with tempfile.TemporaryDirectory() as directory:
            for name in MODULES:
                with self.subTest(module=name):
                    result = subprocess.run(
                        [sys.executable, "-B", "-S", str(SRC / f"{name}.py"), "--help"],
                        cwd=directory, capture_output=True, text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("usage:", result.stdout)

    def test_floor_worker_remains_pickleable(self):
        from coevo_floor import job

        worker = pickle.loads(pickle.dumps(partial(job, mult=2.0)))
        self.assertIs(worker.func, job)
        self.assertEqual(worker.keywords, {"mult": 2.0})

    def test_floor_main_passes_cost_multiplier_and_writes_expected_output(self):
        from coevo_floor import main

        for mult, suffix in (("1", ""), ("2", "_x2")):
            with self.subTest(mult=mult):
                evaluation = types.ModuleType("coevo_eval")
                evaluation.OUT_BASE = "/runs/coevo7"
                evaluation.SYM = "BTCUSDT"
                evaluation.ARMS = ["single", "population", "coevolution"]
                evaluation.prepare = mock.Mock(return_value=({0: (None, "test-data")}, ["feature"], []))
                evaluation.baselines = mock.Mock(return_value=[{"buy_hold": 2.0, "ma_crossover": 1.0}])
                numpy = types.ModuleType("numpy")
                numpy.mean = mock.Mock(return_value=[1.0, 2.0, 3.0, 4.0])
                pandas = types.ModuleType("pandas")
                pandas.DataFrame = mock.Mock()
                root = "/runs/coevo7/example/BTCUSDT"
                jobs = [(0, f"{root}/fold0/{arm}_seed42/best_trader.zip") for arm in evaluation.ARMS]
                pool = mock.MagicMock()
                pool.map.return_value = [(j, [[1.0, 2.0, 3.0, 4.0]]) for j in jobs]
                context = mock.MagicMock()
                context.Pool.return_value.__enter__.return_value = pool
                contract = '{"config": {"seeds": [42], "score": "vote"}}'
                with mock.patch.dict(sys.modules, {"numpy": numpy, "pandas": pandas, "coevo_eval": evaluation}), \
                     mock.patch.dict(os.environ, {"MULT": mult}), \
                     mock.patch("builtins.open", mock.mock_open(read_data=contract)) as open_file, \
                     mock.patch.object(multiprocessing, "get_context", return_value=context):
                    main(["example"])
                open_file.assert_called_once_with(f"{root}/contract.json")
                worker, dispatched = pool.map.call_args.args
                self.assertEqual(worker.keywords, {"mult": float(mult)})
                self.assertEqual(dispatched, jobs)
                evaluation.baselines.assert_called_once_with("test-data", float(mult))
                rows = pandas.DataFrame.call_args.args[0]
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["coevolution_f0.75"], 4.0)
                pandas.DataFrame.return_value.to_csv.assert_called_once_with(
                    f"{root}/floor{suffix}.csv", index=False,
                )


if __name__ == "__main__":
    unittest.main()
