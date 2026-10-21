"""Returned-archive extraction boundaries and independent analysis regressions."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from audit_transfer_v4_output import extract, independent_primary_analysis
import test_transfer_v4 as v4_tests
from transfer_v4.analysis import common_target_analysis


class ReturnedV4AuditChecks(unittest.TestCase):
    def folder(self):
        (ROOT/'tmp').mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(dir=ROOT/'tmp')

    def test_extraction_streams_valid_content_without_overwriting(self):
        with self.folder() as name:
            root = Path(name)
            archive = root/'evidence.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('safe/evidence.json', '{"original":true}')
            self.assertEqual(extract(archive, root/'out'), 1)
            self.assertEqual(json.loads((root/'out/safe/evidence.json').read_text()), {'original': True})
            with self.assertRaises(FileExistsError):
                extract(archive, root/'out')

    def test_traversal_and_symlinks_rejected(self):
        for path in ('../escape', '/absolute', 'C:/escape', 'folder\\escape', 'symlink'):
            with self.subTest(path=path), self.folder() as name:
                root = Path(name)
                archive = root/'bad.zip'
                with zipfile.ZipFile(archive, 'w') as z:
                    info = zipfile.ZipInfo(path)
                    info.filename = path  # Preserve raw backslashes in the ZIP header on Windows.
                    if path == 'symlink':
                        info.create_system = 3
                        info.external_attr = 0o120777 << 16
                    z.writestr(info, 'bad')
                with self.assertRaises(ValueError):
                    extract(archive, root/'out')

    def test_case_collisions_rejected_before_extraction(self):
        with self.folder() as name:
            root = Path(name)
            archive = root/'collision.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('A/file', 'one')
                z.writestr('a/FILE', 'two')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                extract(archive, root/'out')

    def test_independent_sign_tests_match_with_ties_and_censoring(self):
        c, initial = v4_tests.AnalysisChecks().fixture()
        for mode in ('wins', 'tie', 'censored'):
            with self.subTest(mode=mode), self.folder() as name:
                root = Path(name)
                curves = copy.deepcopy(initial)
                for seed in c['seeds']:
                    for arm in ('prefix_grid', 'shuffled_nca', 'shuffled_graph'):
                        curves[arm, seed] = copy.deepcopy(curves['grid_nca', seed])
                if mode == 'tie':
                    curves['grid_nca', c['seeds'][0]] = copy.deepcopy(curves['graph_nca', c['seeds'][0]])
                elif mode == 'censored':
                    for row in curves['graph_nca', c['seeds'][0]]:
                        row['validation_nll'] = 9.
                for (arm, seed), rows in curves.items():
                    path = root/f'{arm}_{seed}/language_metrics.jsonl'
                    path.parent.mkdir()
                    path.write_text('\n'.join(json.dumps(row) for row in rows))
                self.assertEqual(independent_primary_analysis(root, c), common_target_analysis(curves, c))


if __name__ == '__main__':
    unittest.main()
