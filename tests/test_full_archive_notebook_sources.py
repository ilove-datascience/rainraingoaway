"""Full-data fitting is an explicit opt-in, separate from historical evaluation."""
import ast
import json
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[1]


def test_generator_has_no_import_writes_and_matches_clean_notebook(monkeypatch):
    def unexpected_write(*args, **kwargs):
        raise AssertionError('Importing a generator must not replace a notebook')
    monkeypatch.setattr(Path, 'write_text', unexpected_write)
    generated = runpy.run_path(str(ROOT / 'scripts/build_final_v5_full_data_notebook.py'))['cells']
    saved = json.loads((ROOT / 'src/rain_arrival_v5_full_data.ipynb').read_text(encoding='utf-8'))['cells']
    sources = lambda cells: [''.join(c['source']).strip() for c in cells if c['cell_type'] == 'code']
    assert sources(saved) == sources(generated)
    flags = []
    for cell in saved:
        if cell['cell_type'] != 'code':
            continue
        assert not cell['outputs'] and cell['execution_count'] is None
        for node in ast.walk(ast.parse(''.join(cell['source']))):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'RUN_TRAINING':
                        flags.append(ast.literal_eval(node.value))
    assert flags == [False]
