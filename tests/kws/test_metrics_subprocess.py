"""
CPU-only regression tests for metrics.run_calc_metrics().

Regression (introduced in c655087): after the correct
`subprocess.run(cmd, check=True)`, a leftover
`subprocess.run(cmd, shell=True, ...)` ran a bare Python interpreter
(shell=True with a list only executes the first element), overwrote the
metrics output, and could block on stdin.

Standard library only (unittest + unittest.mock). No GPU, no project data.
The SGMSE metrics script itself is never executed.

    CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -t . -v
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("MPLBACKEND", "Agg")

REPO = Path(__file__).resolve().parents[2]
KWS_SRC = REPO / "code" / "kws" / "src"
if str(KWS_SRC) not in sys.path:
    sys.path.insert(0, str(KWS_SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402

import metrics  # noqa: E402

CALC_METRICS_SCRIPT = REPO / "code" / "sgmse" / "calc_metrics.py"

# Same line format as code/sgmse/calc_metrics.py prints.
SAMPLE_STDOUT = (
    "PESQ: 2.00 ± 0.50 (N=6383/6473)\n"
    "ESTOI: 0.69 ± 0.21 (N=5863/6473)\n"
    "SI-SDR: 15.3 ± 4.3 (N=6473/6473)\n"
    "SI-SIR: 26.6 ± 7.1 (N=6473/6473)\n"
    "SI-SAR: 15.9 ± 4.1 (N=6473/6473)\n"
    "PESQ success rate: 98.61%\n"
)
EXPECTED_METRICS = {
    "pesq": "2.00 ± 0.50",
    "estoi": "0.69 ± 0.21",
    "si_sdr": "15.3 ± 4.3",
    "si_sir": "26.6 ± 7.1",
    "si_sar": "15.9 ± 4.1",
}


class RunCalcMetricsSubprocessTests(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        # run_calc_metrics() builds its working dir under /tmp using `tag`;
        # a unique tag lets tearDown remove exactly what this test created.
        self.tag = f"UNITTEST_{uuid.uuid4().hex[:8]}"
        self.run_dir = self.tmp / "run"
        self.run_dir.mkdir()
        self.noisy_root = self.tmp / "noisy"
        self.enh_root = self.tmp / "enhanced"
        audio = np.zeros(1600, dtype=np.float32)
        for root in (self.noisy_root, self.enh_root):
            (root / "bed").mkdir(parents=True)
            sf.write(root / "bed" / "a.wav", audio, 16000)
        self.df = pd.DataFrame([{"filename": "a.wav", "label": "bed", "audio_data": audio}])

    def tearDown(self):
        self._tmp.cleanup()
        for d in glob.glob(os.path.join("/tmp", f"kws_metrics_{self.tag}_*")):
            shutil.rmtree(d, ignore_errors=True)

    def _call(self):
        return metrics.run_calc_metrics(self.df, 16000, self.run_dir,
                                        str(self.noisy_root), str(self.enh_root), self.tag)

    def _ok(self, cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=SAMPLE_STDOUT, stderr="")

    def test_subprocess_invoked_exactly_once(self):
        with mock.patch.object(metrics.subprocess, "run", side_effect=self._ok) as run:
            self._call()
        self.assertEqual(run.call_count, 1)

    def test_intended_argument_list(self):
        with mock.patch.object(metrics.subprocess, "run", side_effect=self._ok) as run:
            self._call()
        cmd = run.call_args.args[0]
        self.assertIsInstance(cmd, list)
        self.assertEqual(cmd[0], sys.executable)
        self.assertEqual(Path(cmd[1]), CALC_METRICS_SCRIPT)
        self.assertTrue(CALC_METRICS_SCRIPT.is_file())
        self.assertEqual(cmd[2::2], ["--clean_dir", "--noisy_dir", "--enhanced_dir"])
        work = [Path(p) for p in cmd[3::2]]
        self.assertEqual([p.name for p in work], ["clean", "noisy", "enhanced"])
        clean, noisy, enhanced = work
        tmp_root = enhanced.parent
        self.assertTrue(tmp_root.name.startswith(f"kws_metrics_{self.tag}_"))
        self.assertEqual(clean.parent, tmp_root / "test")
        self.assertEqual(noisy.parent, tmp_root / "test")
        for p in work:
            self.assertTrue(p.is_dir(), f"{p} should have been created before the call")
        self.assertEqual(sorted(os.listdir(clean)), ["bed__a.wav"])
        self.assertEqual(len(cmd), 8)

    def test_shell_true_not_used_and_errors_checked(self):
        with mock.patch.object(metrics.subprocess, "run", side_effect=self._ok) as run:
            self._call()
        kwargs = run.call_args.kwargs
        self.assertFalse(kwargs.get("shell", False), "shell=True must not be used")
        self.assertIs(kwargs.get("check"), True)
        self.assertIs(kwargs.get("capture_output"), True)
        self.assertIs(kwargs.get("text"), True)

    def test_nonzero_exit_propagates_error(self):
        """Real subprocess.run with the function's own kwargs, on a command that exits 3."""
        real_run = subprocess.run

        def failing(cmd, **kwargs):
            return real_run([sys.executable, "-c", "import sys; sys.exit(3)"], **kwargs)

        with mock.patch.object(metrics.subprocess, "run", side_effect=failing) as run:
            with self.assertRaises(subprocess.CalledProcessError) as ctx:
                self._call()
        self.assertEqual(ctx.exception.returncode, 3)
        self.assertEqual(run.call_count, 1)
        self.assertFalse((self.run_dir / "metrics.txt").exists(),
                         "no metrics should be written after a failed run")

    def test_expected_metrics_output_read(self):
        with mock.patch.object(metrics.subprocess, "run", side_effect=self._ok):
            got = self._call()
        self.assertEqual(got, EXPECTED_METRICS)
        text = (self.run_dir / "metrics.txt").read_text()
        self.assertIn(self.tag, text)
        self.assertIn(SAMPLE_STDOUT.strip(), text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
