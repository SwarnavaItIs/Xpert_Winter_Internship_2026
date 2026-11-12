"""Reproduce input profiling, frozen split and untouched starter baselines offline.

Print aggregate-only profiling. All IDs, predictions and scores stay in .private.
Neither profiling nor split creation opens answers. Evaluation happens last.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / '.private'
IST = timezone(timedelta(hours=5, minutes=30))
SALT = 'xpert-day1-chat-holdout-v1'
INPUTS = ('chats.jsonl', 'messages.jsonl', 'seed.json')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def load_inputs(folder):
    def lines(name):
        return [json.loads(line) for line in (folder / name).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    return lines('chats.jsonl'), lines('messages.jsonl'), json.loads((folder / 'seed.json').read_text(encoding='utf-8-sig'))


def extract_verified(archive, target):
    """Inventory and validate all paths before writing; refuse changed existing files."""
    with zipfile.ZipFile(archive) as z:
        entries = []
        for info in z.infolist():
            path = Path(info.filename)
            destination = (target / path).resolve()
            if ('\\' in info.filename or ':' in info.filename or
                    target.resolve() not in destination.parents or
                    (info.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError('Unsafe archive member')
            if info.is_dir():
                continue
            content = z.read(info)
            if destination.exists() and destination.read_bytes() != content:
                raise ValueError('Existing extracted artifact differs; refusing overwrite')
            entries.append((destination, content, info.filename))
        manifest = {'archive_sha256': digest(archive), 'members': [
            {'path': name, 'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
            for _, content, name in entries]}
        save(PRIVATE / 'day1' / (archive.stem + '-archive.json'), manifest)
        for destination, content, _ in entries:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                destination.write_bytes(content)


def distribution(values):
    values = sorted(values)
    if not values:
        return {'count': 0}
    return {'count': len(values), 'min': values[0], 'median': values[(len(values)-1)//2],
            'p90': values[math.ceil(.90*len(values))-1], 'p95': values[math.ceil(.95*len(values))-1],
            'max': values[-1], 'mean': round(sum(values)/len(values), 2)}


def script_bucket(body):
    found = []
    if re.search('[A-Za-z]', body):
        found.append('Latin')
    if re.search('[\u0900-\u097f]', body):
        found.append('Devanagari')
    if re.search('[\u0980-\u09ff]', body):
        found.append('Bengali')
    if any(ch.isalpha() and not ('a' <= ch.lower() <= 'z' or '\u0900' <= ch <= '\u09ff') for ch in body):
        found.append('other_letters')
    return '+'.join(found) or 'no_letters'


def profile(folder):
    chats, messages, seed = load_inputs(folder)
    ids = {m['id'] for m in messages}
    by_id = {m['id']: m for m in messages}
    counts = Counter(m['chat_id'] for m in messages)
    people = {p for c in chats for p in c['participants']} - {'ME'}
    quoted = [m for m in messages if m.get('quoted_id')]
    timestamps = [m['ts'] for m in messages]
    start, end = min(timestamps), max(timestamps)
    circles = set((seed.get('circles') or {}).keys())
    # This dictionary is a transparent rough hint, NOT a calibrated language classifier.
    romanized = {'hai', 'hain', 'nahi', 'nahin', 'karo', 'karna', 'aap', 'kal', 'bhi', 'kya',
                 'hobe', 'ache', 'achhe', 'kore', 'korbo', 'tomar', 'amar', 'aaj'}
    hints = sum(bool(set(re.findall(r'[a-z]+', m.get('body', '').lower())) & romanized) for m in messages)
    return {
        'messages': len(messages), 'chats': len(chats), 'people_excluding_ME': len(people),
        'group_chats': sum(bool(c['is_group']) for c in chats),
        'first_IST': datetime.fromtimestamp(start, IST).isoformat(),
        'last_IST': datetime.fromtimestamp(end, IST).isoformat(),
        'elapsed_days': round((end-start)/86400, 4),
        'calendar_days_inclusive': (datetime.fromtimestamp(end, IST).date()-datetime.fromtimestamp(start, IST).date()).days+1,
        'active_calendar_days': len({datetime.fromtimestamp(t, IST).date() for t in timestamps}),
        'message_kinds': dict(sorted(Counter(m.get('kind', 'missing') for m in messages).items())),
        'script_buckets': dict(sorted(Counter(script_bucket(m.get('body', '')) for m in messages).items())),
        'romanized_language_hint_messages': hints,
        'language_note': 'Script counts and lexicon hints only; Latin cannot distinguish English, Hindi, Bengali or Hinglish reliably; hints overlap script buckets.',
        'quoted_messages': len(quoted), 'quoted_percent': round(100*len(quoted)/len(messages), 2),
        'resolved_quotes': sum(m['quoted_id'] in ids for m in quoted),
        'cross_chat_quotes': sum(m['quoted_id'] in by_id and by_id[m['quoted_id']]['chat_id'] != m['chat_id'] for m in quoted),
        'seed_circle_count': len(circles), 'seed_circle_participants': len(circles & people),
        'seed_circle_coverage_percent': round(100*len(circles & people)/len(people), 2),
        'seed_organisation_count': len(seed.get('organisations') or []),
        'seed_goal_count': len(seed.get('goals') or []),
        'messages_per_chat': distribution([counts[c['chat_id']] for c in chats]),
        'body_characters': distribution([len(m.get('body', '')) for m in messages]),
        'participants_per_chat': distribution([len(c['participants']) for c in chats]),
        'input_bytes': {name: (folder/name).stat().st_size for name in INPUTS},
        'duplicate_message_ids': len(messages)-len(ids),
        'duplicate_chat_ids': len(chats)-len({c['chat_id'] for c in chats}),
        'messages_with_unknown_chat': sum(m['chat_id'] not in {c['chat_id'] for c in chats} for m in messages),
    }


def freeze_split(folder):
    chats, messages, _ = load_inputs(folder)
    fingerprints = {name: digest(folder/name) for name in INPUTS}
    ranked = sorted((c['chat_id'] for c in chats), key=lambda cid: hashlib.sha256((SALT+'\0'+cid).encode()).digest())
    validation = set(ranked[:math.ceil(len(ranked)*.2)])
    development = set(ranked)-validation
    manifest = {'version': 1, 'salt': SALT, 'input_sha256': fingerprints,
                'development_chat_ids': sorted(development), 'validation_chat_ids': sorted(validation),
                'development_message_ids': sorted(m['id'] for m in messages if m['chat_id'] in development),
                'validation_message_ids': sorted(m['id'] for m in messages if m['chat_id'] in validation)}
    path = PRIVATE / 'splits' / 'day1-v1.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Frozen split differs; refusing regeneration')
    else:
        save(path, manifest)
    def people_for(chat_ids):
        return {p for c in chats if c['chat_id'] in chat_ids for p in c['participants']} - {'ME'}
    train_people, val_people = people_for(development), people_for(validation)
    return {'manifest_sha256': digest(path), 'development_chats': len(development),
            'validation_chats': len(validation),
            'development_messages': len(manifest['development_message_ids']),
            'validation_messages': len(manifest['validation_message_ids']),
            'shared_people': len(train_people & val_people),
            'development_people': len(train_people), 'validation_people': len(val_people)}


def baseline(starter, folder, name):
    output = PRIVATE / 'day1' / (name+'-output.json')
    # -I -S removes user site packages. Socket operations are blocked inside this
    # process, demonstrating that the starter does not need network or pip.
    runner = '''import runpy,sys,socket
def deny(*args, **kwargs): raise RuntimeError('Network disabled for baseline')
socket.socket=deny
socket.create_connection=deny
sys.argv=sys.argv[1:]
runpy.run_path(sys.argv[0],run_name='__main__')
'''
    times = []
    hashes = []
    for _ in range(2):
        start = time.monotonic()
        result = subprocess.run([sys.executable, '-I', '-S', '-c', runner, str(starter/'solution.py'), str(folder), str(output)],
                                capture_output=True, timeout=1800)
        times.append(round(time.monotonic()-start, 4))
        if result.returncode:
            raise ValueError('Untouched baseline failed; no payload printed')
        hashes.append(digest(output))
    checked = subprocess.run([sys.executable, '-I', '-S', str(starter/'check.py'), str(folder), str(output)],
                             capture_output=True, timeout=120)
    if checked.returncode:
        raise ValueError('Official checker rejected baseline')
    report = {'starter_sha256': digest(starter/'solution.py'), 'checker_sha256': digest(starter/'check.py'),
              'input_sha256': {n: digest(folder/n) for n in INPUTS},
              'answer_sha256': digest(folder/'answers/answers.json'),
              'python_version': sys.version, 'run_seconds': times, 'output_sha256': hashes,
              'repeat_output_identical': len(set(hashes)) == 1, 'checker_exit_code': checked.returncode,
              'model_requests': 0, 'model_cost_usd': 0,
              'checker_stdout': checked.stdout.decode('utf-8'),
              'checker_stderr': checked.stderr.decode('utf-8')}
    save(PRIVATE/'day1'/(name+'-baseline.json'), report)
    return {'runs': 2, 'seconds': times, 'repeat_output_identical': report['repeat_output_identical'],
            'checker_exit_code': checked.returncode, 'model_requests': 0}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archives', type=Path, default=PRIVATE/'day1-downloads')
    args = parser.parse_args()
    extract_verified(args.archives/'starter.zip', PRIVATE/'official-day1')
    extract_verified(args.archives/'dev.zip', PRIVATE/'official-day1')
    starter = PRIVATE/'official-day1/cold-start'
    dev = PRIVATE/'official-day1/data/dev'
    for name in ('schema.json', 'check.py'):
        if digest(ROOT/name) != digest(starter/name):
            raise ValueError('Root official artifact differs from downloaded source')
    split = freeze_split(dev)  # Must precede any evaluation/answer loading.
    aggregates = {'example': profile(starter/'example'), 'dev': profile(dev), 'split': split}
    save(PRIVATE/'day1'/'profile.json', aggregates)
    results = {name: baseline(starter, folder, name) for name, folder in
               [('example', starter/'example'), ('dev', dev)]}
    save(PRIVATE/'day1'/'run-summary.json', results)
    print(json.dumps({'aggregates': aggregates, 'execution': results}, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('Day 1 verification failed; payload omitted for privacy.', file=sys.stderr)
        raise SystemExit(1)
