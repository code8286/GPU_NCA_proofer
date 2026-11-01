import unittest
from unittest.mock import patch
from scripts.recover_confirmation_v11 import derived_source, GUARD, retry_dump


class RecoveryChecks(unittest.TestCase):
    def test_only_completed_folder_guard_changes_frozen_training_body(self):
        original, derived = derived_source()
        self.assertEqual(derived.count(GUARD), 1)
        self.assertEqual(derived.replace(GUARD, ''), original)
        self.assertIn('update(model, optimizer, data, c, c[\'ppt\'], k, rng, trace, factor)', derived)

    def test_status_retry_reuses_same_serialization_without_fitting(self):
        with patch('scripts.recover_confirmation_v11.dump', side_effect=[PermissionError(), PermissionError(), None]) as dump, patch('scripts.recover_confirmation_v11.time.sleep'):
            retry_dump('status.json', {'step': 800})
        self.assertEqual(dump.call_count, 3)
        self.assertTrue(all(c.args == ('status.json', {'step': 800}) for c in dump.call_args_list))

    def test_status_retry_is_bounded(self):
        with patch('scripts.recover_confirmation_v11.dump', side_effect=PermissionError()) as dump, patch('scripts.recover_confirmation_v11.time.sleep'):
            with self.assertRaises(PermissionError):
                retry_dump('status.json', {})
        self.assertEqual(dump.call_count, 100)
