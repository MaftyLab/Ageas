"""Command-line options shared by the experiment scripts."""
import argparse
import os

from .registry import load_protocol, load_registry

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def base_parser(description: str, dataset_required: bool = True) -> argparse.ArgumentParser:
    """Parser with the options every experiment accepts."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--dataset', required=dataset_required,
                        help='Registry key in revision/datasets.json.')
    parser.add_argument('--registry', default=None, help='Dataset registry JSON.')
    parser.add_argument('--protocol', default=None, help='Frozen protocol JSON.')
    parser.add_argument('--config', default=None,
                        help='Model-panel config folder (default: protocol config_dir).')
    parser.add_argument('--seeds', type=int, nargs='*', default=None,
                        help='Seeds (default: protocol seeds).')
    parser.add_argument('--accelerator', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--results-root', default=None,
                        help='Root for outputs (default: revision/results).')
    return parser


def resolve(args) -> tuple:
    """Return ``(registry, protocol, config_dir, seeds)`` for parsed args."""
    registry = load_registry(args.registry)
    protocol = load_protocol(args.protocol)
    config_dir = args.config or protocol['config_dir']
    if not os.path.isabs(config_dir):
        config_dir = os.path.join(_REPO, config_dir)
    seeds = args.seeds if args.seeds else protocol['seeds']
    return registry, protocol, config_dir, seeds
