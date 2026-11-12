"""Run only the relationship stage, with private diagnostics and bounded runtime."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from inputs import load_index
from llm import Client,ModelError
from relationships import infer_relationships
from solution import atomic_write,completion_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('output_path')
    parser.add_argument('--offline', action='store_true', help='explicitly use the uncalibrated offline baseline')
    parser.add_argument('--corrections', type=Path)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.worker:
        try:
            return subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--worker'], timeout=1740).returncode
        except subprocess.TimeoutExpired:
            print('Relationship runtime limit reached.', file=sys.stderr)
            return 1
    try:
        index = load_index(args.input_dir)
        corrections = json.loads(args.corrections.read_text(encoding='utf-8')) if args.corrections else None
        client = None if args.offline else Client(time.monotonic()+1680, str(args.output_path)+'.usage.jsonl')
        output, diagnostics = infer_relationships(index, client, corrections)
        atomic_write(str(args.output_path)+'.relationships.json', diagnostics)
        if client is not None:
            report=completion_report(client,diagnostics,{}, {}, {})
            atomic_write(str(args.output_path)+'.completion.json',report)
            if report['status']!='complete':raise ModelError('Relationship coverage incomplete')
        atomic_write(args.output_path, output)
        print('Relationship output validated; evidence and uncertainty saved separately.')
        return 0
    except Exception:
        print('Relationship stage failed; check input/configuration. Data omitted for privacy.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
