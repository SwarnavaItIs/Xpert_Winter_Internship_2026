"""Private evaluation entry point; never imported by inference."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import time

import check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('output_path')
    parser.add_argument('--report', default='.private/evaluation.json')
    args = parser.parse_args()
    destination = Path(args.report).resolve()
    private = Path(__file__).resolve().parent / '.private'
    if private not in destination.parents:
        parser.error('Evaluation reports must be inside .private/')
    started = time.monotonic()
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        code = check.main(['check.py', args.input_dir, args.output_path])
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps({'exit_code': code, 'seconds': time.monotonic() - started,
                                      'official_checker_output': capture.getvalue()}, indent=2), encoding='utf-8')
    print('Private evaluation report saved; checker exit code:', code)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
