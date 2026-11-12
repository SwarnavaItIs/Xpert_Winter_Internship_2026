"""One tiny synthetic request; sends no challenge data."""
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from llm import Client, ModelError


def main():
    try:
        client = Client(time.monotonic() + 110)
        result = client.call('Return exactly this JSON object: {"ok": true}', {}, max_tokens=128)
        if result != {'ok': True}:
            raise ModelError('Synthetic response did not match expected JSON')
        print('Synthetic API smoke test passed.')
        return 0
    except ModelError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
