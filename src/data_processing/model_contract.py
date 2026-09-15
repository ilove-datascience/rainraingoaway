"""Prevent decoding/normalization changes from silently reaching old checkpoints."""
import hashlib
import json
from pathlib import Path
from .radar_codec import LEGACY, validate_version


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_contract(checkpoint, normalization, decoder_version=LEGACY,
                   preprocessing_version='connected_components_v1_min4_strong010'):
    checkpoint, normalization = Path(checkpoint), Path(normalization)
    validate_version(decoder_version)
    stats = json.loads(normalization.read_text(encoding='utf-8'))
    if stats.get('radar_decoder_version', LEGACY) != decoder_version:
        raise ValueError('Normalization decoder metadata differs from checkpoint contract')
    contract = dict(decoder_version=decoder_version, preprocessing_version=preprocessing_version,
                    checkpoint_sha256=file_hash(checkpoint), normalization_sha256=file_hash(normalization),
                    normalization_path=str(normalization.resolve()))
    path = checkpoint.with_suffix(checkpoint.suffix + '.metadata.json')
    path.write_text(json.dumps(contract, indent=2), encoding='utf-8')
    return path


def validate_contract(checkpoint, decoder_version=LEGACY, normalization=None,
                      preprocessing_version='connected_components_v1_min4_strong010'):
    path = Path(checkpoint).with_suffix(Path(checkpoint).suffix + '.metadata.json')
    if not path.exists():
        if decoder_version != LEGACY:
            raise ValueError('Unversioned checkpoints require legacy decoding')
        return None
    contract = json.loads(path.read_text(encoding='utf-8'))
    if contract['decoder_version'] != decoder_version:
        raise ValueError('Checkpoint and radar decoder versions differ')
    if contract['preprocessing_version'] != preprocessing_version:
        raise ValueError('Checkpoint and radar preprocessing versions differ')
    if contract['checkpoint_sha256'] != file_hash(checkpoint):
        raise ValueError('Checkpoint changed since metadata was saved')
    if normalization is not None and contract['normalization_sha256'] != file_hash(normalization):
        raise ValueError('Normalization does not match checkpoint metadata')
    return contract
