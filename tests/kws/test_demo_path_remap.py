"""
CPU-only tests for the Gradio demo's audio path remapping (code/kws/demo/demo_utils.py).

Historical interesting_examples.csv files store absolute paths under the original
data directory. The demo remaps them in memory to the GannotLab archive; the CSV
files must stay byte-identical. No GPU, no gradio, no network.

    python -m unittest tests.kws.test_demo_path_remap -v
"""
from __future__ import annotations

import ast
import glob
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "code") not in sys.path:
    sys.path.insert(0, str(REPO / "code"))

from kws.demo import demo_utils as du  # noqa: E402

H, A = du.HISTORICAL_DATA_ROOT, du.ARCHIVE_DATA_ROOT
HISTORICAL_CSVS = sorted(glob.glob(str(REPO / "outputs" / "new_runs" / "*" / "interesting_examples.csv")))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RemapPathTests(unittest.TestCase):

    def test_three_historical_roots_map_to_archive(self):
        cases = {
            f"{H}/raw/data_new/bird/1626bc5a_nohash_1.wav": f"{A}/speech_commands_v0.01/bird/1626bc5a_nohash_1.wav",
            f"{H}/noisy_new/test/bird/1626bc5a_nohash_1.wav": f"{A}/noisy_new/test/bird/1626bc5a_nohash_1.wav",
            f"{H}/enhanced_new/trained_ep176/bird/1626bc5a_nohash_1.wav":
                f"{A}/enhanced_new/trained_ep176/bird/1626bc5a_nohash_1.wav",
            f"{H}/enhanced_new/baseline/go/x.wav": f"{A}/enhanced_new/baseline/go/x.wav",
        }
        for old, new in cases.items():
            self.assertEqual(du.remap_path(old), new)

    def test_prefix_matches_whole_components_only(self):
        for p in (f"{H}/noisy_new_metadata.csv", f"{H}/noisy_new2/test/a.wav", f"{H}/raw/data_new_old/a.wav"):
            self.assertEqual(du.remap_path(p), p)

    def test_unrelated_paths_unchanged(self):
        for p in ("/tmp/a.wav", "relative/a.wav", f"{A}/noisy_new/test/a.wav", ""):
            self.assertEqual(du.remap_path(p), p)

    def test_longest_prefix_wins_and_user_map_takes_precedence(self):
        user = du.parse_path_map([f"{H}/noisy_new/test=/elsewhere/noisy_test", f"{H}/raw/data_new=/mine/clean"])
        pm = user + du.DEFAULT_PATH_MAP
        self.assertEqual(du.remap_path(f"{H}/noisy_new/test/go/a.wav", pm), "/elsewhere/noisy_test/go/a.wav")  # longer
        self.assertEqual(du.remap_path(f"{H}/noisy_new/val/go/a.wav", pm), f"{A}/noisy_new/val/go/a.wav")
        self.assertEqual(du.remap_path(f"{H}/raw/data_new/go/a.wav", pm), "/mine/clean/go/a.wav")  # tie: first wins

    def test_parse_path_map_validation(self):
        self.assertEqual(du.parse_path_map(["/a=/b", "/c/=/d/"]), (("/a", "/b"), ("/c/", "/d/")))
        self.assertEqual(du.parse_path_map(None), ())
        for bad in ("/a", "=/b", "/a="):
            with self.assertRaises(ValueError):
                du.parse_path_map([bad])
        self.assertEqual(du.remap_path("/c/x.wav", du.parse_path_map(["/c/=/d/"])), "/d/x.wav")


class RemapDataFrameTests(unittest.TestCase):

    def test_copy_only_audio_columns_change(self):
        df = pd.DataFrame({
            "filename": ["a.wav", "b.wav"], "true_label": ["go", "no"], "snr": [2.5, 7.5],
            "clean_path": [f"{H}/raw/data_new/go/a.wav", f"{H}/raw/data_new/no/b.wav"],
            "noisy_path": [f"{H}/noisy_new/test/go/a.wav", None],
            "enh_path": [f"{H}/enhanced_new/trained_ep176/go/a.wav", "/tmp/keep.wav"],
        })
        before = df.copy(deep=True)
        out = du.remap_audio_paths(df)
        pd.testing.assert_frame_equal(df, before)  # input untouched
        pd.testing.assert_frame_equal(out[["filename", "true_label", "snr"]], before[["filename", "true_label", "snr"]])
        self.assertEqual(out.clean_path.tolist(), [f"{A}/speech_commands_v0.01/go/a.wav", f"{A}/speech_commands_v0.01/no/b.wav"])
        self.assertEqual(out.noisy_path.iloc[0], f"{A}/noisy_new/test/go/a.wav")
        self.assertIsNone(out.noisy_path.iloc[1])
        self.assertEqual(out.enh_path.tolist(), [f"{A}/enhanced_new/trained_ep176/go/a.wav", "/tmp/keep.wav"])

    def test_resolves_files_after_original_location_is_gone(self):
        """Portable end-to-end check: historical paths that do not exist resolve to files under a new root."""
        with tempfile.TemporaryDirectory() as d:
            for sub in ("speech_commands_v0.01/go", "noisy_new/test/go", "enhanced_new/trained_ep176/go"):
                os.makedirs(f"{d}/{sub}"); Path(f"{d}/{sub}/a.wav").write_bytes(b"RIFF")
            csv = Path(d) / "interesting_examples.csv"
            pd.DataFrame({"filename": ["a.wav"], "clean_path": [f"{H}/raw/data_new/go/a.wav"],
                          "noisy_path": [f"{H}/noisy_new/test/go/a.wav"],
                          "enh_path": [f"{H}/enhanced_new/trained_ep176/go/a.wav"]}).to_csv(csv, index=False)
            digest = _sha256(csv)
            pm = du.parse_path_map([f"{H}/raw/data_new={d}/speech_commands_v0.01", f"{H}/noisy_new={d}/noisy_new",
                                    f"{H}/enhanced_new={d}/enhanced_new"])
            out = du.remap_audio_paths(pd.read_csv(csv), pm)
            for col in du.AUDIO_PATH_COLUMNS:
                self.assertTrue(os.path.isfile(out[col].iloc[0]), out[col].iloc[0])
            self.assertEqual(_sha256(csv), digest)


@unittest.skipUnless(HISTORICAL_CSVS, "historical outputs/new_runs/*/interesting_examples.csv not present")
class HistoricalCsvTests(unittest.TestCase):

    def test_all_historical_paths_map_into_archive_and_csvs_unchanged(self):
        archive_readable = os.path.isdir(A)
        for csv in HISTORICAL_CSVS:
            digest = _sha256(csv)
            df = pd.read_csv(csv)
            out = du.remap_audio_paths(df)
            for col in du.AUDIO_PATH_COLUMNS:
                for old, new in zip(df[col], out[col]):
                    self.assertTrue(old.startswith(H + "/"), old)
                    self.assertTrue(new.startswith(A + "/"), new)
                    self.assertEqual(Path(old).parts[-2:], Path(new).parts[-2:])  # label/filename preserved
                    if archive_readable:
                        self.assertTrue(os.path.isfile(new), new)
            self.assertEqual(_sha256(csv), digest, f"{csv} was modified")


class GradioDemoWiringTests(unittest.TestCase):
    """gradio_demo.py launches a server on import, so its wiring is checked statically."""

    def test_gradio_demo_remaps_after_reading_csv(self):
        src = (REPO / "code" / "kws" / "demo" / "gradio_demo.py").read_text()
        tree = ast.parse(src)
        imports = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module == "demo_utils" for a in n.names}
        self.assertTrue({"DEFAULT_PATH_MAP", "parse_path_map", "remap_audio_paths"} <= imports, imports)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "remap_audio_paths"]
        self.assertEqual(len(calls), 1)
        self.assertLess(src.index("pd.read_csv(CSV_PATH)"), src.index("remap_audio_paths(df"))
        self.assertTrue((REPO / "code" / "kws" / "demo" / "demo_utils.py").is_file())  # importable as a sibling module


if __name__ == "__main__":
    unittest.main(verbosity=2)
