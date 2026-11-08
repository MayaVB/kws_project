"""
CPU-only regression tests for the end-of-training reporting path of
code/kws/src/run_training.py.

Regression covered: c655087 deleted demo_utils.build_audio_paths(), but
compare_prediction_modes() still called it. Every training run therefore failed
with NameError after evaluation, before interesting_examples.csv, the example
plots, the confusion-matrix comparison and metrics_comparison.txt were written.

The end-to-end test runs the real run_training.run() (1 epoch) on a tiny
synthetic dataset in a temporary directory. Only run_calc_metrics (the SGMSE
metrics subprocess) is replaced by a recorder. No GPU, no project data, no network.

    CUDA_VISIBLE_DEVICES="" python -m unittest tests.kws.test_training_reporting -v
"""
from __future__ import annotations

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

try:  # same convention as the other tests: stub only if the real package is absent
    import tabulate as _tabulate  # noqa: F401
    REAL_TABULATE = True
except ModuleNotFoundError:
    REAL_TABULATE = False
    _stub = types.ModuleType("tabulate")
    _stub.tabulate = lambda *a, **k: ""
    sys.modules["tabulate"] = _stub

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

import analysis_utils  # noqa: E402
import run_training  # noqa: E402
from kws.demo import demo_utils  # noqa: E402

LABELS = ["bed", "bird"]
SPLITS = {"train": 6, "val": 3, "test": 5}   # per label; test >= 9 files in total (plot uses idx_vis=8)
SR = 16000
INTERESTING_COLUMNS = ["filename", "true_label", "pred_clean", "correct_clean", "pred_noisy", "correct_noisy",
                       "snr", "noise", "pred_enh", "correct_enh", "category", "clean_path", "noisy_path", "enh_path"]


def _write_tone(path, freq, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(SR) / SR
    sig = 0.3 * np.sin(2 * np.pi * freq * t) + 0.05 * rng.standard_normal(SR)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, sig.astype(np.float32), SR)


class BuildAudioPathsTests(unittest.TestCase):

    def test_layout_root_label_filename(self):
        c, n, e = demo_utils.build_audio_paths("a_nohash_0.wav", "go", "/c", "/n/test", "/e/trained")
        self.assertEqual((c, n, e), ("/c/go/a_nohash_0.wav", "/n/test/go/a_nohash_0.wav", "/e/trained/go/a_nohash_0.wav"))

    def test_roots_are_required(self):
        with self.assertRaises(TypeError):
            demo_utils.compare_prediction_modes("c.csv", "n.csv", "e.csv", ".")

    def test_categories_and_paths(self):
        with tempfile.TemporaryDirectory() as d:
            base = {"filename": ["a.wav", "b.wav", "c.wav", "d.wav"], "true_label": ["go"] * 4}
            pd.DataFrame({**base, "pred_label": ["go"] * 4, "correct": [True] * 4}).to_csv(f"{d}/clean.csv", index=False)
            pd.DataFrame({**base, "pred_label": ["no", "no", "go", "go"], "correct": [False, False, True, True],
                          "snr": [0, 5, 10, 0], "noise": ["x", "y", "z", "w"]}).to_csv(f"{d}/noisy.csv", index=False)
            pd.DataFrame({**base, "pred_label": ["go", "no", "no", "go"], "correct": [True, False, False, True]}).to_csv(
                f"{d}/enh.csv", index=False)
            demo_utils.compare_prediction_modes(f"{d}/clean.csv", f"{d}/noisy.csv", f"{d}/enh.csv", d,
                                                clean_root="/C", noisy_root="/N/test", enhanced_root="/E")
            out = pd.read_csv(f"{d}/interesting_examples.csv")
            self.assertEqual(list(out.columns), INTERESTING_COLUMNS)
            self.assertEqual(dict(zip(out.filename, out.category)), {"a.wav": "fixed", "b.wav": "still_wrong", "c.wav": "degraded"})
            for _, r in out.iterrows():
                self.assertEqual((r.clean_path, r.noisy_path, r.enh_path),
                                 (f"/C/go/{r.filename}", f"/N/test/go/{r.filename}", f"/E/go/{r.filename}"))


class TrainingEndToEndReportingTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.clean, self.noisy, self.out = tmp / "clean", tmp / "noisy", tmp / "out"
        self.enh_base, self.enh_tr = tmp / "enh_baseline", tmp / "enh_trained"
        rows = []
        for li, label in enumerate(LABELS):
            for si, (split, n) in enumerate(SPLITS.items()):
                for i in range(n):
                    fn = f"{split}{i}_nohash_0.wav"  # same filenames in both classes, as in Speech Commands
                    _write_tone(self.clean / label / fn, 300 + 400 * li, seed=100 * li + 10 * si + i)
                    if split == "test":
                        for root, off in ((self.noisy / "test", 1), (self.enh_base, 2), (self.enh_tr, 3)):
                            _write_tone(root / label / fn, 300 + 400 * li, seed=1000 * off + 10 * li + i)
                    rows.append({"filename": fn, "label": label, "snr": 2.5 * (i % 3), "noise": f"noise{i}", "split": split})
        self.meta = tmp / "meta.csv"
        pd.DataFrame(rows).to_csv(self.meta, index=False)
        self.cfg = {
            "device": "cpu", "sampling_rate": SR, "n_mfcc": 13, "batch_size": 4, "use_parallel": False,
            "num_workers": 0, "shuffle_train": True, "epochs": 1, "patience": 1, "lr": 0.001,
            "weight_decay": 0.0001, "random_state": 42, "folder_start": 0, "folder_end": 30, "selected_folders": None,
            "output_dir": str(self.out), "meta": str(self.meta), "clean_dir": str(self.clean), "noisy_dir": str(self.noisy),
            "enh_baseline": str(self.enh_base), "enh_trained": str(self.enh_tr),
        }
        self.calls = []

    def tearDown(self):
        self._tmp.cleanup()

    def _fake_metrics(self, df_test_audio, sampling_rate, run_dir, noisy_root, enhanced_root, tag):
        self.calls.append(tag)
        return {"pesq": "1.00 ± 0.10", "estoi": "0.50 ± 0.10", "si_sdr": "10.0 ± 1.0",
                "si_sir": "20.0 ± 1.0", "si_sar": "11.0 ± 1.0"}

    def test_training_run_completes_and_writes_end_of_run_reports(self):
        torch.manual_seed(0)
        with mock.patch.object(analysis_utils, "run_calc_metrics", side_effect=self._fake_metrics):
            run_training.run(self.cfg)  # raised NameError before the fix

        runs = list(self.out.iterdir())
        self.assertEqual(len(runs), 1)
        run_dir = runs[0]
        n_test = len(LABELS) * SPLITS["test"]

        # interesting_examples.csv: written, correct columns, paths built from the configured roots
        ie = pd.read_csv(run_dir / "interesting_examples.csv")
        self.assertEqual(list(ie.columns), INTERESTING_COLUMNS)
        self.assertTrue(set(ie.category) <= {"fixed", "still_wrong", "degraded"})
        noisy = pd.read_csv(run_dir / "noisy_all_predictions.csv")
        enh = pd.read_csv(run_dir / "enh_trained_all_predictions.csv")
        self.assertEqual(len(noisy), n_test)
        expected_rows = int((~noisy.correct.values | ~enh.set_index(["filename", "true_label"]).loc[
            list(zip(noisy.filename, noisy.true_label)), "correct"].values).sum())
        self.assertEqual(len(ie), expected_rows)  # every file wrong in noisy or in enh_trained is listed
        for _, r in ie.iterrows():
            self.assertEqual(r.clean_path, str(self.clean / r.true_label / r.filename))
            self.assertEqual(r.noisy_path, str(self.noisy / "test" / r.true_label / r.filename))
            self.assertEqual(r.enh_path, str(self.enh_tr / r.true_label / r.filename))
            for col in ("clean_path", "noisy_path", "enh_path"):
                self.assertTrue(os.path.isfile(r[col]), r[col])

        # the steps after compare_prediction_modes now run
        plots = os.listdir(run_dir / "plots")
        for prefix in ("Signal_Comparison_Example_", "MFCC_Comparison_Example_", "confusion_matrices_comparison", "training_history"):
            self.assertTrue(any(p.startswith(prefix) for p in plots), (prefix, plots))
        summary = (run_dir / "metrics_comparison.txt").read_text()
        if REAL_TABULATE:
            for mode in ("clean", "noisy", "enh_baseline", "enh_trained"):
                self.assertIn(mode, summary)
        self.assertIn("TOTAL PROGRAM TIME", (run_dir / "runtime_report.txt").read_text())
        for f in ("best_model.pt", "scaler.pkl", "label_encoder.pkl"):
            self.assertTrue((run_dir / "models" / f).is_file(), f)
        self.assertEqual(self.calls, ["ENH_BASELINE", "ENH_TRAINED"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
