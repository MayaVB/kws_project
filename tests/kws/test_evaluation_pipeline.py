"""
CPU-only regression tests for the standalone evaluation pipeline
(code/kws/src/run_evaluation_main.py).

Regressions covered (both introduced after the final experiments):
  1. c655087 made `noisy_root` a required argument of analysis_utils.analyze_mode()
     but run_evaluation_main.py did not pass it -> TypeError on the first mode.
  2. 96cc177 made visualization.plot_example_signals() read
     <run_dir>/noisy_all_predictions.csv, which only run_training.py wrote
     (via save_prediction_reports) -> FileNotFoundError after all metrics.

The end-to-end test runs the real run_evaluation_main.run() on a tiny synthetic
dataset in a temporary directory. Only run_calc_metrics (the SGMSE metrics
subprocess) is replaced by a recorder; no GPU, no project data, no network.

    CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -t . -v
"""
from __future__ import annotations

import ast
import inspect
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLBACKEND", "Agg")

REPO = Path(__file__).resolve().parents[2]
KWS_SRC = REPO / "code" / "kws" / "src"
for p in (str(REPO / "code"), str(KWS_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:  # same convention as test_entrypoints.py: stub only if the real package is absent
    import tabulate as _tabulate  # noqa: F401
    REAL_TABULATE = True
except ModuleNotFoundError:
    REAL_TABULATE = False
    _stub = types.ModuleType("tabulate")
    _stub.tabulate = lambda *a, **k: ""
    sys.modules["tabulate"] = _stub

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
from sklearn.preprocessing import LabelEncoder, StandardScaler  # noqa: E402

import analysis_utils  # noqa: E402
import run_evaluation_main  # noqa: E402
from models import DSCNN  # noqa: E402

LABELS = ["bed", "bird"]
FILES_PER_LABEL = 3
SR = 16000


class AnalyzeModeCallSignatureTests(unittest.TestCase):
    """Every analyze_mode(...) call in the entry points supplies all required parameters."""

    def _calls(self, path):
        tree = ast.parse(Path(path).read_text())
        return [n for n in ast.walk(tree)
                if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "analyze_mode"]

    def test_all_required_parameters_supplied(self):
        sig = inspect.signature(analysis_utils.analyze_mode)
        required = [n for n, p in sig.parameters.items()
                    if p.default is inspect.Parameter.empty
                    and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
        self.assertIn("noisy_root", required)
        for script in ("run_evaluation_main.py", "run_training.py"):
            calls = self._calls(KWS_SRC / script)
            self.assertTrue(calls, f"no analyze_mode call found in {script}")
            for call in calls:
                supplied = set(required[:len(call.args)]) | {k.arg for k in call.keywords}
                missing = [r for r in required if r not in supplied]
                self.assertEqual(missing, [], f"{script}:{call.lineno} misses {missing}")


def _write_tone(path, freq, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(SR) / SR
    sig = 0.3 * np.sin(2 * np.pi * freq * t) + 0.05 * rng.standard_normal(SR)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, sig.astype(np.float32), SR)


class StandaloneEvaluationEndToEndTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.clean, self.noisy, self.out = tmp / "clean", tmp / "noisy", tmp / "out"
        self.enh_base, self.enh_tr = tmp / "enh_baseline", tmp / "enh_trained"
        rows = []
        for li, label in enumerate(LABELS):
            for i in range(FILES_PER_LABEL):
                fn = f"f{i}_nohash_0.wav"  # same filenames in both classes, as in Speech Commands
                for root, off in ((self.clean, 0), (self.noisy / "test", 1), (self.enh_base, 2), (self.enh_tr, 3)):
                    _write_tone(root / label / fn, 300 + 400 * li, seed=100 * li + 10 * i + off)
                rows.append({"filename": fn, "label": label, "snr": 2.5 * (i + 1), "noise": f"noise{i}", "split": "test"})
        self.meta = tmp / "meta.csv"
        pd.DataFrame(rows).to_csv(self.meta, index=False)
        # Saved artifacts with the same types the real pipeline uses.
        art = tmp / "artifacts"; art.mkdir()
        le = LabelEncoder().fit(LABELS)
        sc = StandardScaler().fit(np.random.default_rng(0).standard_normal((64, 13)))
        torch.manual_seed(0)
        torch.save(DSCNN(len(LABELS)).state_dict(), art / "best_model.pt")
        joblib.dump(sc, art / "scaler.pkl"); joblib.dump(le, art / "label_encoder.pkl")
        self.cfg = {
            "device": "cpu", "sampling_rate": SR, "n_mfcc": 13, "batch_size": 4,
            "use_parallel": False, "num_workers": 0,
            "output_dir": str(self.out), "meta": str(self.meta), "clean_dir": str(self.clean),
            "noisy_dir": str(self.noisy), "enh_baseline": str(self.enh_base), "enh_trained": str(self.enh_tr),
            "best_model_path": str(art / "best_model.pt"), "scaler_path": str(art / "scaler.pkl"),
            "label_encoder_path": str(art / "label_encoder.pkl"),
        }
        self.calls = []

    def tearDown(self):
        self._tmp.cleanup()

    def _fake_metrics(self, df_test_audio, sampling_rate, run_dir, noisy_root, enhanced_root, tag):
        self.calls.append({"noisy_root": noisy_root, "enhanced_root": enhanced_root, "tag": tag})
        return {"pesq": "1.00 ± 0.10", "estoi": "0.50 ± 0.10", "si_sdr": "10.0 ± 1.0",
                "si_sir": "20.0 ± 1.0", "si_sar": "11.0 ± 1.0"}

    def test_evaluation_completes_and_writes_reports(self):
        with mock.patch.object(analysis_utils, "run_calc_metrics", side_effect=self._fake_metrics):
            run_evaluation_main.run(self.cfg)

        runs = list(self.out.iterdir())
        self.assertEqual(len(runs), 1)
        run_dir = runs[0]
        n = len(LABELS) * FILES_PER_LABEL

        # Regression 1: analyze_mode received <noisy_dir>/test for both enhanced modes.
        self.assertEqual([c["enhanced_root"] for c in self.calls], [str(self.enh_base), str(self.enh_tr)])
        for c in self.calls:
            self.assertEqual(c["noisy_root"], os.path.join(str(self.noisy), "test"))

        # Regression 2: per-file prediction reports exist for every mode (same format as training).
        for mode in ("clean", "noisy", "enh_baseline", "enh_trained"):
            pred = pd.read_csv(run_dir / f"{mode}_all_predictions.csv")
            self.assertEqual(len(pred), n)
            self.assertEqual(list(pred.columns[:7]),
                             ["filename", "true_idx", "pred_idx", "true_label", "pred_label", "correct", "mode"])
            self.assertTrue((pred["mode"] == mode).all())
            self.assertTrue((run_dir / f"{mode}_misclassified.csv").is_file())
        noisy = pd.read_csv(run_dir / "noisy_all_predictions.csv")
        self.assertIn("snr", noisy.columns)
        self.assertIn("noise", noisy.columns)

        # The steps after plotting now run: example plots and the final summary are written.
        plots = os.listdir(run_dir / "plots")
        self.assertTrue(any(p.startswith("Signal_Comparison_Example_") for p in plots), plots)
        summary = (run_dir / "metrics_comparison.txt").read_text()
        if REAL_TABULATE:  # with the stub, the table body is intentionally empty
            for mode in ("clean", "noisy", "enh_baseline", "enh_trained"):
                self.assertIn(mode, summary)
            self.assertIn("10.0 ± 1.0", summary)  # value returned by the metrics recorder


if __name__ == "__main__":
    unittest.main(verbosity=2)
