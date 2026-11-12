"""Conservative local-only transcript audit. Never prints matching private text."""
import hashlib
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def run():
    dest = ROOT/'.private/releases/rc1/log-audit'
    dest.mkdir(parents=True, exist_ok=False)
    corpus = ROOT/'.private/official-day1/data/dev'
    terms = set()
    for name in ('messages', 'chats'):
        for line in (corpus/(name+'.jsonl')).read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            for field in ('id','chat_id','sender_id','sender_name','name','body'):
                value = row.get(field)
                if isinstance(value, str) and len(value.strip()) >= 4:
                    terms.add(value.strip().casefold())
            terms.update(p.casefold() for p in row.get('participants', []) if len(p) >= 4)
    # Literal matches are conservative leads, not proof of a leak. No label reads.
    secret = re.compile(r'(?:sk-[A-Za-z0-9_-]{16,}|gsk_[A-Za-z0-9]{16,}|Bearer\s+[A-Za-z0-9._-]{16,}|-----BEGIN .*PRIVATE KEY-----)')
    known = {v for k,v in os.environ.items() if any(x in k.upper() for x in ('API_KEY','TOKEN','SECRET')) and len(v)>10}
    originals = set()
    source_count = 0
    for path in (Path.home()/'.codex/sessions').rglob('*.jsonl'):
        try:
            with path.open(encoding='utf-8') as stream:
                for line in stream:
                    try: event = json.loads(line)
                    except ValueError: continue
                    payload = event.get('payload', {})
                    if event.get('type') != 'response_item' or payload.get('type') != 'message': continue
                    if payload.get('role') not in ('user','assistant') or payload.get('channel') == 'analysis': continue
                    parts = [c.get('text','') for c in payload.get('content', []) if c.get('type') in ('input_text','output_text','text')]
                    if parts:
                        originals.add(hashlib.sha256('\n'.join(parts).encode()).hexdigest())
            source_count += 1
        except (OSError, UnicodeError): continue
    results = []
    for path in sorted((ROOT/'ai-logs').glob('*.jsonl')):
        counts = dict(rows=0, exact_provenance=0, unverified_provenance=0, secret_flags=0, data_flags=0)
        redacted = []
        flags = []
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            row = json.loads(line); text = row.get('text',''); counts['rows'] += 1
            provenance = hashlib.sha256(text.encode()).hexdigest() in originals
            sensitive = bool(secret.search(text)) or any(v in text for v in known)
            data = any(term in text.casefold() for term in terms)
            counts['exact_provenance' if provenance else 'unverified_provenance'] += 1
            counts['secret_flags'] += sensitive; counts['data_flags'] += data
            if sensitive or data or not provenance:
                flags.append({'row': number, 'secret': sensitive, 'data_match': data, 'unverified': not provenance})
                row = {**row, 'text': '[ENTIRE MESSAGE WITHHELD: privacy/provenance review]', 'redacted': True}
            redacted.append(row)
        (dest/path.name).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in redacted), encoding='utf-8')
        results.append({'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), **counts, 'flagged_rows': flags})
    report = {'original_session_files_scanned': source_count, 'files': results,
        'release_policy': 'All transcript exports remain ignored; no private transcript shipped. Redacted copies are also private pending manual review.',
        'limitations': 'Exact-text provenance cannot verify edited/redacted exports. Literal corpus/secret scanning can miss paraphrases and unfamiliar secrets; matches include benign generic words. No reconstructed messages.'}
    (dest/'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'files': len(results), 'source_sessions': source_count,
        **{k:sum(r[k] for r in results) for k in ('rows','exact_provenance','unverified_provenance','secret_flags','data_flags')},
        'public_transcripts_released': 0}, indent=2))


if __name__ == '__main__': run()
