"""Version dispatch keeps frozen legacy encoders usable alongside structured candidates."""
from pathlib import Path

from pokesim.runtime.bundle import file_hash
from pokesim.value import features, structured


def module(version):
    if version == features.VERSION:
        return features
    if version == structured.VERSION:
        return structured
    raise ValueError('unsupported value feature version')


def code_hash(version):
    selected = module(version)
    if selected is features:
        return file_hash(Path(features.__file__))
    # v2 inherits v1; bind both source files, not just the subclass.
    import hashlib

    return hashlib.sha256((Path(features.__file__).read_bytes() +
                           Path(structured.__file__).read_bytes())).hexdigest()[:16]


def encoder(version, dex):
    return module(version).Encoder.load(dex)
