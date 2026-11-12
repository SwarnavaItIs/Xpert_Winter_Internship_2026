#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo 'usage: ./run.sh <input_dir> <output_path>' >&2
  exit 2
fi
exec python3 "$(dirname "$0")/solution.py" "$1" "$2"
