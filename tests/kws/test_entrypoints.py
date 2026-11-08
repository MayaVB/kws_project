"""
CPU-only tests for the KWS entry-point configuration handling.

Covers:
  - run_training.run(): `enh_new_dir` is optional (no KeyError when absent),
    behaviour unchanged when present, other keys still required.
  - run_evaluation_main.py: optional --config, default path preserved.

Standard library only (unittest). No GPU, no project data, no network.
Run from the repository root:

    CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -t . -v
"""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLBACKEND", "Agg")

REPO = Path(__file__).resolve().parents[2]
KWS_SRC = REPO / "code" / "kws" / "src"
RUN_TRAINING = KWS_SRC / "run_training.py"
RUN_EVAL = KWS_SRC / "run_evaluation_main.py"

for p in (str(REPO / "code"), str(KWS_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

# The KWS reporting code imports `tabulate`, which is a declared requirement
# (reproducibility/requirements/requirements-main.txt) but is missing from the
# original sgmse_env. These tests never call it, so for in-process imports only a
# stub is used when the real package is absent. Subprocess CLI tests are skipped
# in that case, because they need the real package.
try:
    import tabulate as _tabulate  # noqa: F401
    REAL_TABULATE = True
except ModuleNotFoundError:
    REAL_TABULATE = False
    _stub = types.ModuleType("tabulate")
    _stub.tabulate = lambda *a, **k: ""
    sys.modules["tabulate"] = _stub

import run_training  # noqa: E402
import run_evaluation_main  # noqa: E402


def _training_cfg(tmp: Path, **overrides) -> dict:
    """Minimal training config. `meta` points to a missing file so that run()
    stops at the first data access, right after all config keys are read."""
    cfg = {
        "device": "cpu",
        "sampling_rate": 16000,
        "n_mfcc": 13,
        "output_dir": str(tmp / "out"),
        "clean_dir": str(tmp / "clean"),
        "folder_start": 0,
        "folder_end": 30,
        "selected_folders": None,
        "meta": str(tmp / "missing_metadata.csv"),
        "noisy_dir": str(tmp / "noisy"),
        "enh_trained": str(tmp / "enh_trained"),
        "enh_baseline": str(tmp / "enh_baseline"),
        "batch_size": 64,
        "num_workers": 0,
        "shuffle_train": True,
        "use_parallel": False,
        "epochs": 1,
        "patience": 1,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "random_state": 42,
    }
    cfg.update(overrides)
    return cfg


class TrainingEnhNewDirTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _assert_reaches_metadata_read(self, cfg):
        # Reaching pd.read_csv(meta) proves every config key was read without error.
        with self.assertRaises(FileNotFoundError) as ctx:
            run_training.run(cfg)
        self.assertIn("missing_metadata.csv", str(ctx.exception))

    def test_missing_enh_new_dir_does_not_raise_keyerror(self):
        cfg = _training_cfg(self.tmp)
        self.assertNotIn("enh_new_dir", cfg)
        self._assert_reaches_metadata_read(cfg)

    def test_present_enh_new_dir_still_accepted(self):
        cfg = _training_cfg(self.tmp, enh_new_dir=str(self.tmp / "enhanced_new"))
        self._assert_reaches_metadata_read(cfg)

    def test_other_keys_remain_required(self):
        cfg = _training_cfg(self.tmp)
        del cfg["enh_trained"]
        with self.assertRaises(KeyError) as ctx:
            run_training.run(cfg)
        self.assertEqual(ctx.exception.args[0], "enh_trained")

    def test_no_mandatory_enh_new_dir_subscript_in_source(self):
        """Regression guard: no live `cfg["enh_new_dir"]` read outside a membership check."""
        tree = ast.parse(RUN_TRAINING.read_text())
        guarded, unguarded = 0, 0
        for node in ast.walk(tree):
            if isinstance(node, ast.IfExp):
                test_src = ast.unparse(node.test)
                if "'enh_new_dir' in cfg" in test_src or '"enh_new_dir" in cfg' in test_src:
                    guarded += sum(
                        1 for n in ast.walk(node.body)
                        if isinstance(n, ast.Subscript) and ast.unparse(n) in ("cfg['enh_new_dir']", 'cfg["enh_new_dir"]'))
        total = sum(
            1 for n in ast.walk(tree)
            if isinstance(n, ast.Subscript) and ast.unparse(n) in ("cfg['enh_new_dir']", 'cfg["enh_new_dir"]'))
        unguarded = total - guarded
        self.assertEqual(unguarded, 0, "found an unguarded cfg['enh_new_dir'] read")


class EvaluationConfigArgTests(unittest.TestCase):

    def test_default_when_omitted(self):
        args = run_evaluation_main.parse_args([])
        self.assertIsNone(args.config)
        self.assertEqual(run_evaluation_main.resolve_config_path(args.config),
                         REPO / "code" / "kws" / "new_config.yaml")

    def test_default_matches_previous_hardcoded_expression(self):
        previous = Path(run_evaluation_main.__file__).resolve().parents[1] / "new_config.yaml"
        self.assertEqual(run_evaluation_main.resolve_config_path(None), previous)

    def test_explicit_absolute_path(self):
        p = "/tmp/some_dir/eval_config.yaml"
        args = run_evaluation_main.parse_args(["--config", p])
        self.assertEqual(run_evaluation_main.resolve_config_path(args.config), Path(p).resolve())

    def test_explicit_relative_path_resolved_against_cwd(self):
        old = os.getcwd()
        with tempfile.TemporaryDirectory() as d:
            try:
                os.chdir(d)
                got = run_evaluation_main.resolve_config_path("configs/eval.yaml")
                self.assertEqual(got, (Path(d) / "configs" / "eval.yaml").resolve())
                self.assertTrue(got.is_absolute())
            finally:
                os.chdir(old)

    def test_explicit_home_path_expanded(self):
        got = run_evaluation_main.resolve_config_path("~/eval.yaml")
        self.assertEqual(got, (Path.home() / "eval.yaml").resolve())

    def test_unknown_argument_rejected(self):
        # Behaviour change by design: previously all CLI arguments were ignored.
        with self.assertRaises(SystemExit):
            run_evaluation_main.parse_args(["--not-an-option"])


def _subprocess_env():
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["MPLBACKEND"] = "Agg"
    return env


@unittest.skipUnless(REAL_TABULATE, "real `tabulate` not importable; CLI subprocess tests need it")
class EntryPointCliTests(unittest.TestCase):
    """End-to-end CLI checks in a subprocess (CPU only, stop before any data is loaded)."""

    def _run(self, script, *args):
        return subprocess.run([sys.executable, str(script), *args], cwd=str(REPO),
                              env=_subprocess_env(), capture_output=True, text=True, timeout=300)

    def test_evaluation_help_lists_config(self):
        r = self._run(RUN_EVAL, "--help")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--config", r.stdout)

    def test_training_help_unchanged(self):
        r = self._run(RUN_TRAINING, "--help")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--config", r.stdout)

    def test_evaluation_uses_explicit_config(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cfg = d / "eval.yaml"
            cfg.write_text("\n".join([
                "device: cpu", "sampling_rate: 16000", "n_mfcc: 13", "batch_size: 8",
                "use_parallel: false", "num_workers: 0",
                f"output_dir: '{d / 'out'}'", f"meta: '{d / 'meta.csv'}'",
                f"clean_dir: '{d}'", f"noisy_dir: '{d}'",
                f"enh_baseline: '{d}'", f"enh_trained: '{d}'",
                f"best_model_path: '{d / 'best_model.pt'}'",
                f"scaler_path: '{d / 'missing_scaler.pkl'}'",
                f"label_encoder_path: '{d / 'label_encoder.pkl'}'",
            ]) + "\n")
            r = self._run(RUN_EVAL, "--config", str(cfg))
            # The script should announce this config, then fail at the first artifact load.
            self.assertIn(f"Using configuration: {cfg.resolve()}", r.stdout)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("FileNotFoundError", r.stderr)
            self.assertIn("missing_scaler.pkl", r.stderr)
            self.assertTrue((d / "out").is_dir(), "run directory should be created under the explicit output_dir")


if __name__ == "__main__":
    unittest.main(verbosity=2)
