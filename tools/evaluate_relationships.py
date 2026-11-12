"""Development-only relationship comparison; monolithic labels projected before scoring.

No validation metrics or per-identity label diagnostics are generated.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import check
from inputs import InputIndex
from llm import Client
from relationships import infer_relationships
from solution import atomic_write
from tools.day1 import freeze_split, load_inputs, digest
from validation import validate


def project(records, chat_ids, exclusive_people):
    scoped = {'people': [copy.deepcopy(p) for p in records['people'] if p['id'] in exclusive_people],
              'chats': [copy.deepcopy(c) for c in records['chats'] if c['chat_id'] in chat_ids],
              'organisations': [], 'goals': [], 'open_loops': []}
    scoped['organisations'] = [copy.deepcopy(o) for o in records['organisations']
        if check.org_members(o['id'], scoped['chats'], scoped['people'])]
    return scoped


def run(offline):
    source = ROOT/'.private/official-day1/data/dev'
    split_path = ROOT/'.private/splits/day1-v1.json'
    if not split_path.exists():
        raise ValueError('Frozen split is required')
    freeze_split(source)  # Verifies existing manifest without touching labels.
    split = json.loads(split_path.read_text())
    dev_ids, val_ids = set(split['development_chat_ids']), set(split['validation_chat_ids'])
    chats, messages, seed = load_inputs(source)
    dev_chats = [c for c in chats if c['chat_id'] in dev_ids]
    dev_messages = [m for m in messages if m['chat_id'] in dev_ids]
    dev_people = {p for c in dev_chats for p in c['participants']} - {'ME'}
    val_people = {p for c in chats if c['chat_id'] in val_ids for p in c['participants']} - {'ME'}
    exclusive = dev_people-val_people
    seed = {**seed, 'circles': {p:v for p,v in seed.get('circles', {}).items() if p in dev_people}}
    directory = ROOT/'.private/relationships-development'
    input_dir = directory/'input'
    input_dir.mkdir(parents=True,exist_ok=True)
    for name, rows in [('chats',dev_chats),('messages',dev_messages)]:
        (input_dir/(name+'.jsonl')).write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n',encoding='utf-8')
    atomic_write(input_dir/'seed.json',seed)
    index = InputIndex(dev_chats,dev_messages,seed)
    mode = 'offline' if offline else 'model'
    client = None if offline else Client(time.monotonic()+1680,directory/(mode+'-usage.jsonl'))
    started = time.monotonic()
    candidate, diagnostics = infer_relationships(index,client)
    seconds = time.monotonic()-started
    validate(candidate,index.chats,index.messages)
    atomic_write(directory/(mode+'-output.json'),candidate)
    atomic_write(directory/(mode+'-diagnostics.json'),diagnostics)
    baseline_path = directory/'starter-output.json'
    starter = ROOT/'.private/official-day1/cold-start/solution.py'
    baseline_run = subprocess.run([sys.executable,'-I','-S',str(starter),str(input_dir),str(baseline_path)],capture_output=True,timeout=120)
    if baseline_run.returncode:
        raise ValueError('Starter failed')
    baseline = json.loads(baseline_path.read_text(encoding='utf-8'))
    validate(baseline,index.chats,index.messages)
    # Only the adapter parses the original monolithic answer file. Immediately
    # discard validation chat rows, all shared/validation person rows and task/goal labels.
    with (source/'answers/answers.json').open(encoding='utf-8') as f:
        gold = project(json.load(f),dev_ids,exclusive)
    scores = {}
    for name, result in [('starter',baseline),(mode,candidate)]:
        result = project(result,dev_ids,exclusive)
        measured = check.score(result,gold,seed,dev_messages)
        scores[name] = {key: {'fraction':measured[key], 'points': None if measured[key] is None else measured[key]*check.POINTS[key]}
                        for key in ('circles','organisations','chats')}
    report = {'mode':mode, 'split_sha256':digest(split_path), 'starter_sha256':digest(starter),
              'code_sha256':{n:digest(ROOT/n) for n in ('inputs.py','relationships.py','llm.py')},
              'candidate_seconds':seconds,'development_chats':len(dev_chats),'development_people':len(dev_people),
              'excluded_shared_people':len(dev_people & val_people),'labelled_exclusive_people':len(gold['people']),
              'circle_scoring_people':sum(p['id'] not in seed['circles'] for p in gold['people']),
              'schema_validation':'passed for candidate and untouched starter',
              'scope':'development chats and development-exclusive people; organisations projected by these attachments',
              'scores':scores,'uncertain':{k:sum(x.get('uncertain',False) for x in diagnostics[k].values()) for k in ('people','chats','organisations')},
              'model_calls':len(client.usage) if client else 0}
    atomic_write(directory/(mode+'-comparison.json'),report)
    lines = ['# Development-only relationship comparison', '', 'Mode: '+mode,
             'Both outputs passed schema/reference validation. No validation metrics computed.', '',
             '| Component | Starter points | Candidate points | Change |','|---|---:|---:|---:|']
    for key in ('circles','organisations','chats'):
        a,b = scores['starter'][key]['points'],scores[mode][key]['points']
        lines.append(f'| {key} | {a:.4f} | {b:.4f} | {b-a:+.4f} |' if a is not None and b is not None else f'| {key} | unlabelled | unlabelled | n/a |')
    lines += ['', 'Uncertain classifications (IDs and evidence are in the adjacent private diagnostics):',
              json.dumps(report['uncertain']), '',
              'Offline results measure the deterministic fallback, not model quality. Confidence is uncalibrated.',
              'Shared people excluded from circle/link scoring. Organisation matching uses only scoped attachments.',
              'This is a projected development diagnostic, not an official full-dataset score.',
              f'Candidate elapsed seconds: {seconds:.4f}; model calls: {report["model_calls"]}.']
    (directory/(mode+'-comparison.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Development relationship comparison saved privately; both outputs validated.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.worker:
        try:
            return subprocess.run([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker'],timeout=1740).returncode
        except subprocess.TimeoutExpired:
            print('Evaluation runtime limit reached.',file=sys.stderr)
            return 1
    try:
        run(args.offline)
        return 0
    except Exception:
        print('Relationship evaluation failed; no data printed.',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
