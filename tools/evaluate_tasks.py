"""Reproducible detection-only comparison on frozen development chats."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import check
from inputs import InputIndex
from llm import Client, ModelError
from relationships import infer_relationships
from solution import atomic_write
from task_candidates import extract_candidates, candidate_loop
from tools.day1 import freeze_split,load_inputs,digest
from tools.task_metrics import scoped_labels,detection_metrics,anchor_diagnostics
from validation import validate


def run(offline):
    source=ROOT/'.private/official-day1/data/dev'
    split_path=ROOT/'.private/splits/day1-v1.json'
    if not split_path.exists():
        raise ValueError('Frozen split required')
    freeze_split(source)
    split=json.loads(split_path.read_text())
    dev_ids=set(split['development_chat_ids'])
    dev_message_ids=set(split['development_message_ids'])
    val_message_ids=set(split['validation_message_ids'])
    chats,messages,seed=load_inputs(source)
    chats=[c for c in chats if c['chat_id'] in dev_ids]
    messages=[m for m in messages if m['id'] in dev_message_ids]
    people={p for c in chats for p in c['participants']} - {'ME'}
    seed={**seed,'circles':{p:v for p,v in seed.get('circles',{}).items() if p in people}}
    index=InputIndex(chats,messages,seed)
    directory=ROOT/'.private/tasks-development'
    input_dir=directory/'input'
    input_dir.mkdir(parents=True,exist_ok=True)
    for name,rows in [('chats',index.chats),('messages',index.messages)]:
        (input_dir/(name+'.jsonl')).write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in rows)+'\n',encoding='utf-8')
    atomic_write(input_dir/'seed.json',seed)
    mode='offline' if offline else 'model'
    client=None if offline else Client(time.monotonic()+1680,directory/(mode+'-usage.jsonl'))
    relationships,rd=infer_relationships(index,client)
    started=time.monotonic()
    candidates,diagnostics=extract_candidates(index,relationships,client,rd)
    elapsed=time.monotonic()-started
    preview={**relationships,'goals':[],'open_loops':[candidate_loop(c) for c in candidates]}
    validate(preview,index.chats,index.messages)
    if check.validate(preview,index.chats,index.messages):
        raise ValueError('Official validation failed')
    atomic_write(directory/(mode+'-candidates.json'),{'candidates':candidates,'diagnostics':diagnostics})
    atomic_write(directory/(mode+'-preview.json'),preview)
    starter=ROOT/'.private/official-day1/cold-start/solution.py'
    baseline_path=directory/'starter-output.json'
    process=subprocess.run([sys.executable,'-I','-S',str(starter),str(input_dir),str(baseline_path)],capture_output=True,timeout=120)
    if process.returncode:
        raise ValueError('Untouched starter failed')
    baseline=json.loads(baseline_path.read_text(encoding='utf-8'))
    if check.validate(baseline,index.chats,index.messages):
        raise ValueError('Starter validation failed')
    # Evaluation alone opens the monolithic labels; immediately project and discard
    # validation descriptions, statuses, help and closure evidence.
    with (source/'answers/answers.json').open(encoding='utf-8') as f:
        gold,boundary=scoped_labels(json.load(f),dev_message_ids,val_message_ids)
    measured,pairs=detection_metrics(candidates,gold)
    baseline_metrics,_=detection_metrics(baseline['open_loops'],gold)
    empty={'people':[],'organisations':[],'chats':[],'goals':[],'open_loops':[]}
    official=check.score({**empty,'open_loops':preview['open_loops']},
        {**empty,'open_loops':[{'source_msgs':g['source_msgs'],'status':'open'} for g in gold]}, {}, messages)['loops_found']
    if official is not None and abs(official-measured['f1'])>1e-12:
        raise ValueError('Detection metric differs from official source matcher')
    errors=anchor_diagnostics(candidates,gold,pairs,index)
    atomic_write(directory/(mode+'-error-analysis.json'),errors)
    matched={i for i,_ in pairs}
    atomic_write(directory/(mode+'-unmatched-predictions.json'),[c for i,c in enumerate(candidates) if i not in matched])
    report={'mode':mode,'scope':'tasks whose every accepted source is in frozen development messages',
            'split_sha256':digest(split_path),'starter_sha256':digest(starter),'checker_sha256':digest(ROOT/'check.py'),
            'code_sha256':{n:digest(ROOT/n) for n in ('task_candidates.py','task_dates.py','inputs.py','relationships.py','tools/task_metrics.py','tools/evaluate_tasks.py')},
            'baseline':baseline_metrics,'candidate':measured,'boundary_exclusions':boundary,
            'suspected_anchor_errors':len(errors['suspected_anchor_errors']),
            'unmatched_without_anchor_suggestion':len(errors['unmatched_without_anchor_suggestion']),
            'confirmed_anchor_errors':None,'candidate_seconds':elapsed,
            'uncertain_candidates':sum(c['uncertain'] for c in candidates),
            'ownership_check':'valid ID, contextual interaction path and cited evidence; semantic correctness unproven',
            'lifecycle_evaluated':False,'model_attempts':len(client.usage) if client else 0,
            'schema_validation':'passed; preview open/help values are placeholders, not lifecycle/help predictions',
            'extraction':diagnostics}
    atomic_write(directory/(mode+'-comparison.json'),report)
    lines=['# Development task-candidate detection', '',f'Mode: {mode}. No lifecycle metrics.', '',
           '| Metric | Untouched starter | Candidate stage |','|---|---:|---:|']
    for k in ('predicted','labelled','matched','false_positives','false_negatives','precision','recall','f1'):
        lines.append(f'| {k} | {baseline_metrics[k]} | {measured[k]} |')
    lines += ['',f'Suspected anchor mistakes (unreviewed): {report["suspected_anchor_errors"]}.',
              f'Unmatched labelled tasks with no anchor suggestion: {report["unmatched_without_anchor_suggestion"]}.',
              'Confirmed anchor-mistake count: unknown. Remaining misses can include undetected anchor mistakes.',
              'No fuzzy/nearby/title matches earn scoring credit. Greedy source matching was cross-checked against check.py.',
              f'Boundary exclusions: {json.dumps(boundary)}.',
              f'Extraction seconds (excludes relationship inference/indexing): {elapsed:.4f}.',
              'All detailed candidates, reasons and error records are private. Offline rules are not a model quality result.',
              'The metric does not assess semantic ownership or due-date correctness. Dev labels are incomplete system output.']
    (directory/(mode+'-comparison.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Development candidate precision/recall and separate anchor diagnostics saved privately. Lifecycle not evaluated.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not args.worker:
        try:
            return subprocess.run([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker'],timeout=1740).returncode
        except subprocess.TimeoutExpired:
            print('Task evaluation runtime limit reached.',file=sys.stderr)
            return 1
    try:
        run(args.offline)
        return 0
    except ModelError as exc:
        print(str(exc),file=sys.stderr)
        return 1
    except Exception:
        print('Task evaluation failed; data omitted for privacy.',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
