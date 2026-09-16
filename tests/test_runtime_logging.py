"""Exercise logging in a subprocess so the test runner streams stay unchanged."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class RuntimeLoggingTests(unittest.TestCase):
    def test_print_logging_traceback_and_rotation(self):
        with tempfile.TemporaryDirectory() as directory:
            script = r'''
import sys, logging, threading
sys.path.insert(0, sys.argv[1])
from runtime_logging import configure_logging
path = configure_logging(sys.argv[2], max_bytes=1500, backup_count=2)
assert configure_logging(sys.argv[2]) == path
for i in range(100):
    print('rotation test ' + str(i) + ' x' * 20)
t = threading.Thread(target=lambda: print('worker message'), name='test-worker')
t.start(); t.join()
print('final stdout marker')
print('final stderr marker', file=sys.stderr)
try:
    raise ValueError('test error')
except ValueError:
    logging.getLogger('test').exception('traceback marker')
logging.shutdown()
'''
            result = subprocess.run([sys.executable, '-c', script, str(ROOT/'src'), directory], capture_output=True, text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            files = list(Path(directory).glob('rainraingoaway.log*'))
            self.assertEqual(len(files),3)
            content = '\n'.join(p.read_text(encoding='utf-8') for p in files)
            for marker in ['final stdout marker','final stderr marker','worker message','test-worker','Traceback','ValueError: test error']:
                self.assertIn(marker,content)
            self.assertEqual(content.count('final stdout marker'),1)
            self.assertIn('final stdout marker',result.stderr)

if __name__ == '__main__':
    unittest.main()
