import ast
from pathlib import Path
import unittest
from unittest.mock import Mock


class SettingsTests(unittest.TestCase):
    def load(self, current='manual', connection_error=False):
        tree=ast.parse((Path(__file__).resolve().parents[1]/'src/telegram_code/database.py').read_text(encoding='utf-8'))
        fn=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='save_mode_choice')
        cur=Mock()
        cur.fetchone.return_value=None if current is None else (current,)
        conn=Mock()
        conn.cursor.return_value=cur
        factory=Mock(return_value=conn,side_effect=RuntimeError('offline') if connection_error else None)
        env={'_get_db_connection':factory,'Error':RuntimeError}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'settings','exec'),env)
        return env['save_mode_choice'], cur, conn

    def test_missing_user_not_reported_as_saved(self):
        save,_,_=self.load(current=None)
        self.assertFalse(save(1,'automatic'))

    def test_repeat_selection_does_not_reset_episode(self):
        save,cur,conn=self.load(current='automatic')
        self.assertTrue(save(1,'automatic'))
        self.assertEqual(cur.execute.call_count,1)
        conn.commit.assert_called_once()

    def test_switch_invalidates_queued_settings(self):
        save,cur,_=self.load(current='automatic')
        self.assertTrue(save(1,'manual'))
        self.assertIn('rain_settings_version=rain_settings_version+1',cur.execute.call_args.args[0])

    def test_connection_failure_is_reported(self):
        save,_,_=self.load(connection_error=True)
        self.assertFalse(save(1,'automatic'))


if __name__=='__main__':
    unittest.main()
