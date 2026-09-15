import unittest,tempfile,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from arrival.run_paths import sweep_run_path

class RunPathsTests(unittest.TestCase):
    def test_partial_preserved_and_completed_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)/'model'
            self.assertEqual(sweep_run_path(base),base)
            base.mkdir();(base/'best.pt').write_bytes(b'original')
            retry=sweep_run_path(base)
            self.assertEqual(retry.name,'model_attempt2')
            self.assertEqual((base/'best.pt').read_bytes(),b'original')
            retry.mkdir();(retry/'result.json').write_text('{"experiment":{},"manifest_hash":"a","validation_records_hash":"b"}')
            (retry/'best.pt').write_bytes(b'x');(retry/'validation_predictions.npz').write_bytes(b'x')
            self.assertEqual(sweep_run_path(base),retry)

    def test_malformed_completion_is_not_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)/'model';base.mkdir()
            (base/'result.json').write_text('{')
            self.assertEqual(sweep_run_path(base).name,'model_attempt2')

    def test_find_completed_after_missing_attempt(self):
        from arrival.run_paths import find_completed_run,write_completed_result
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)/'model';p=base.with_name('model_attempt4');p.mkdir()
            with self.assertRaises(ValueError):write_completed_result(p,{})
            (p/'best.pt').write_bytes(b'x');(p/'validation_predictions.npz').write_bytes(b'x')
            write_completed_result(p,dict(experiment={},manifest_hash='a',validation_records_hash='b'))
            self.assertEqual(find_completed_run(base),p)
            self.assertEqual(sweep_run_path(base),p)

    def test_multiple_interrupted_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory)/'model';base.mkdir();(base/'history.json').write_text('[]')
            retry=base.with_name('model_attempt2');retry.mkdir();(retry/'best.pt').write_bytes(b'x')
            self.assertEqual(sweep_run_path(base).name,'model_attempt3')
