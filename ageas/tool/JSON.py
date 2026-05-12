#!/usr/bin/env python3
"""JSON encoding and decoding helpers used across Ageas.

Thin wrappers around the standard ``json`` module that transparently
gzip-compress files when the path ends in ``.gz``.
"""
import gzip
import json
import re


def encode(data=None, save_path: str = 'out.js', indent: int = 4) -> None:
    """Encode ``data`` to a JSON file, optionally gzip-compressed.

    Args:
        data: Object to serialize (typically a ``dict`` or ``list``).
        save_path: Destination path. If it ends in ``.gz`` the output is
            gzip-compressed.
        indent: Indentation depth for human-readable output. Ignored when
            writing gzip files.
    """
    if re.search(r'\.gz$', save_path):
        with gzip.open(save_path, 'w+') as output:
            output.write(json.dumps(data).encode('utf-8'))
    else:
        with open(save_path, 'w+') as output:
            json.dump(data, output, indent=indent)


def decode(path: str = None):
    """Decode a JSON file, optionally gzip-compressed.

    Args:
        path: Path to the input file.

    Returns:
        The deserialized object (typically a ``dict`` or ``list``).
    """
    if re.search(r'\.gz$', path):
        with gzip.open(path, 'r') as json_file:
            return json.load(json_file)
    with open(path, 'r') as json_file:
        return json.load(json_file)
