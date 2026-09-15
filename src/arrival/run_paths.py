"""Keep interrupted training artifacts while permitting sweep retries."""
from pathlib import Path
import json
import os
import uuid


def completed_run(path):
    path=Path(path)
    if not all((path/name).is_file() for name in ('result.json','best.pt','validation_predictions.npz')):return False
    try:
        result=json.loads((path/'result.json').read_text())
        return isinstance(result,dict) and all(k in result for k in ('experiment','manifest_hash','validation_records_hash'))
    except (OSError,ValueError):return False


def find_completed_run(base):
    base=Path(base)
    candidates=[base]
    attempts=[]
    for p in base.parent.glob(base.name+'_attempt*'):
        suffix=p.name[len(base.name+'_attempt'):]
        if suffix.isdigit():attempts.append((int(suffix),p))
    candidates.extend(p for _,p in sorted(attempts))
    return next((p for p in candidates if completed_run(p)),None)


def write_completed_result(path,payload):
    """Publish completion only after the checkpoint and predictions exist."""
    path=Path(path)
    if not all((path/name).is_file() for name in ('best.pt','validation_predictions.npz')):
        raise ValueError('Cannot mark incomplete artifacts as completed')
    temporary=path/f'.result-{uuid.uuid4().hex}.tmp'
    try:
        temporary.write_text(json.dumps(payload,indent=2))
        os.replace(temporary,path/'result.json')
    finally:temporary.unlink(missing_ok=True)


def sweep_run_path(base):
    """Reuse completed results; restart partial runs in a fresh numbered folder.

    This is not checkpoint resumption: historical checkpoints lack optimizer
    and sampler state. The caller must still validate completed result metadata.
    """
    base=Path(base)
    completed=find_completed_run(base)
    if completed is not None:return completed
    for attempt in range(10000):
        path=base if attempt==0 else base.with_name(f'{base.name}_attempt{attempt+1}')
        if not path.exists() or not any(path.iterdir()):
            if attempt:print(f'Preserving interrupted run; restarting from scratch in {path.name}',flush=True)
            return path
    raise RuntimeError(f'Too many attempts for {base}')
