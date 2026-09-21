import importlib.util
import os
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    'bot_service', Path(__file__).resolve().parents[1] / 'scripts/run_bot_service.py')
service = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(service)


class BotServiceTests(unittest.TestCase):
    def test_authentication_exception_is_redacted(self):
        from runtime_logging import SecretSafeFormatter
        secret = '123456789:abcdefghijklmnopqrstuvwxyz123456789'
        try:
            raise ValueError('Rejected token ' + secret)
        except ValueError:
            import sys
            record = logging.LogRecord('test', logging.ERROR, '', 0, 'Auth failed', (), sys.exc_info())
        self.assertNotIn(secret, SecretSafeFormatter().format(record))

    def test_missing_normalization_fails_without_training(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(service, 'ROOT', Path(tmp)), patch.dict(
                os.environ, {'tele_api_key': 'test', 'gov_api_key': 'test'}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'normalization_stats'):
                service.check_assets()
            self.assertFalse((Path(tmp) / 'models').exists())

    def test_missing_checkpoint_fails(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(service, 'ROOT', Path(tmp)), patch.dict(
                os.environ, {'tele_api_key': 'test', 'gov_api_key': 'test'}, clear=True):
            (Path(tmp) / 'models').mkdir()
            (Path(tmp) / 'models/normalization_stats.json').write_text('{}')
            with self.assertRaisesRegex(RuntimeError, 'checkpoint|model_best'):
                service.check_assets()

    def test_valid_assets_create_persistent_directories(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(service, 'ROOT', Path(tmp)), patch.dict(
                os.environ, {'tele_api_key': 'test', 'GOV_API_KEY': 'test'}, clear=True):
            root = Path(tmp)
            (root / 'models').mkdir()
            (root / 'models/normalization_stats.json').write_text('{}')
            (root / 'models/model_best_latest.pkl').touch()
            service.check_assets()
            for path in ('data/70km/png', 'data/240km/png', 'data/environment', 'logs'):
                self.assertTrue((root / path).is_dir())


if __name__ == '__main__':
    unittest.main()
