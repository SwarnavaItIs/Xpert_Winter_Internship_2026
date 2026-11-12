"""Export actual user/assistant messages, excluding reasoning and tool payloads.

The source stays in Codex storage. The export is ignored until manually reviewed.
No synthetic transcript lines are generated except labelled redaction metadata.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('session')
    parser.add_argument('--output', default='ai-logs/session.private.jsonl')
    args = parser.parse_args()
    rows = []
    for line in Path(args.session).read_text(encoding='utf-8').splitlines():
        event = json.loads(line)
        payload = event.get('payload', {})
        if event.get('type') != 'response_item' or payload.get('type') != 'message':
            continue
        if payload.get('role') not in ('user', 'assistant') or payload.get('channel') == 'analysis':
            continue
        texts = [c.get('text', '') for c in payload.get('content', [])
                 if c.get('type') in ('input_text', 'output_text', 'text')]
        if texts:
            rows.append({'timestamp': event.get('timestamp'), 'role': payload['role'],
                         'channel': payload.get('channel'), 'text': '\n'.join(texts)})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = '\n'.join(json.dumps(r, ensure_ascii=False) for r in rows)
    key = os.environ.get('LLM_API_KEY')
    if key:
        rendered = rendered.replace(key, '[REDACTED_API_KEY]')
    output.write_text(rendered + '\n', encoding='utf-8')
    print('Genuine message excerpt saved privately; manual privacy review still required.')


if __name__ == '__main__':
    main()
