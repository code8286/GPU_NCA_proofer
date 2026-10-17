"""Regression checks for cache corruption and publication boundary failures."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest

import experiment as e
from scripts.check_release import check, safe_file, selected_files, write_inventory


def workspace_tempdir():
    root = Path(__file__).resolve().parents[1]
    parent = root / "tmp/checks"
    parent.mkdir(parents=True, exist_ok=True)
    if not parent.resolve().is_relative_to(root.resolve()):
        raise ValueError("Temporary check directory must stay inside the repository")
    return tempfile.TemporaryDirectory(dir=parent)


class CacheIntegrityChecks(unittest.TestCase):
    def test_config_mismatch_and_modified_cache_fail_closed(self):
        c = e.load_config("configs/smoke.json")
        with workspace_tempdir() as directory:
            root = Path(directory)
            text = root / "fixture.txt"
            text.write_text("Offline cache integrity fixture.\n" * 500, encoding="utf-8")
            data = root / "data"
            e.prepare(argparse.Namespace(data=data, text=text), c, "cpu")
            e.load_data(data, c)
            with self.assertRaisesRegex(ValueError, "Dataset/config mismatch"):
                e.load_data(data, dict(c, data_seed=c["data_seed"] + 1))
            with (data / "graph.npy").open("ab") as handle:
                handle.write(b"corruption")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch: graph.npy"):
                e.load_data(data, c)


class PublicationIntegrityChecks(unittest.TestCase):
    def fixture(self, root, files=None, directories=None):
        (root / "publication.json").write_text(json.dumps({
            "files": files or ["README.md"], "directories": directories or []}), encoding="utf-8")
        (root / "README.md").write_text("Public fixture\n", encoding="utf-8")

    def test_traversal_absolute_and_windows_paths_are_refused(self):
        with workspace_tempdir() as directory:
            for name in ("../private", "/private", "C:/private", "folder\\private"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    safe_file(Path(directory), name)

    def test_raw_artifacts_and_secrets_cannot_be_selected(self):
        for name in ("data/corpus.txt", "model.pt", ".env", "credentials.key"):
            with self.subTest(name=name), workspace_tempdir() as directory:
                root = Path(directory)
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("private", encoding="utf-8")
                self.fixture(root, files=[name])
                with self.assertRaisesRegex(ValueError, "Non-distributable"):
                    selected_files(root)

    def test_missing_declared_input_is_refused(self):
        with workspace_tempdir() as directory:
            root = Path(directory)
            self.fixture(root, files=["missing.py"])
            with self.assertRaisesRegex(ValueError, "Missing publication file"):
                selected_files(root)

    def test_caches_do_not_enter_selected_directories(self):
        with workspace_tempdir() as directory:
            root = Path(directory)
            self.fixture(root, directories=["tests"])
            (root / "tests/__pycache__").mkdir(parents=True)
            (root / "tests/__pycache__/ignored.pyc").write_bytes(b"cache")
            (root / "tests/.lake/packages").mkdir(parents=True)
            (root / "tests/.lake/packages/private.env").write_text("cache", encoding="utf-8")
            (root / "tests/compiled.olean").write_bytes(b"cache")
            (root / "tests/compiled.ilean").write_bytes(b"cache")
            (root / "tests/test_sample.py").write_text("pass\n", encoding="utf-8")
            self.assertEqual(selected_files(root), ["README.md", "tests/test_sample.py"])

    def test_modified_selected_file_invalidates_inventory(self):
        with workspace_tempdir() as directory:
            root = Path(directory)
            self.fixture(root)
            write_inventory(root)
            (root / "README.md").write_text("Changed after inventory\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Stale publication inventory"):
                check(root)


if __name__ == "__main__":
    unittest.main()
