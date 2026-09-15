"""Published notebooks must stay reproducible and require explicit opt-in."""
import ast
import json
from pathlib import Path
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('variable,filename', [
    ('main', 'rain_arrival_model.ipynb'),
    ('experiments', 'rain_arrival_experiments.ipynb'),
])
def test_sources_match_generator_without_import_writes(monkeypatch, variable, filename):
    def unexpected_write(*args, **kwargs):
        raise AssertionError('Importing a notebook generator must not write files')
    monkeypatch.setattr(Path, 'write_text', unexpected_write)
    generated = runpy.run_path(str(ROOT / 'scripts/build_arrival_notebooks.py'))[variable]
    saved = json.loads((ROOT / 'src' / filename).read_text(encoding='utf-8'))['cells']
    sources = lambda cells: [''.join(c['source']).strip() for c in cells if c['cell_type'] == 'code']
    assert sources(saved) == sources(generated)
    for cell in saved:
        if cell['cell_type'] != 'code':
            continue
        assert not cell['outputs'] and cell['execution_count'] is None
        for node in ast.walk(ast.parse(''.join(cell['source']))):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.startswith('RUN_'):
                        assert node.value.value is not True, target.id
