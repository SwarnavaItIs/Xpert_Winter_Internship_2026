"""Compare the SAME candidate set before/after reconciliation on development only."""
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
from task_candidates import extract_candidates,candidate_loop
from task_lifecycle import reconcile_tasks
from tools.day1 import freeze_split,load_inputs,digest
from tools.lifecycle_metrics import project_lifecycle_labels,lifecycle_metrics
from validation import validate


def run(offline):
    source=ROOT/'.private/official-day1/data/dev'
    split_path=ROOT/'.private/splits/day1-v1.json'
    if not split_path.exists():
        raise ValueError('Frozen split required')
    freeze_split(source)
    split=json.loads(split_path.read_text())
    dev_ids=set(split['development_chat_ids'])
    dev_messages=set(split['development_message_ids'])
    val_messages=set(split['validation_message_ids'])
    chats,messages,seed=load_inputs(source)
    chats=[c for c in chats if c['chat_id'] in dev_ids]
    messages=[m for m in messages if m['id'] in dev_messages]
    people={p for c in chats for p in c['participants']}-{'ME'}
    seed={**seed,'circles':{p:v for p,v in seed.get('circles',{}).items() if p in people}}
    index=InputIndex(chats,messages,seed)
    directory=ROOT/'.private/lifecycle-development'
    directory.mkdir(parents=True,exist_ok=True)
    mode='offline' if offline else 'model'
    client=None if offline else Client(time.monotonic()+1680,directory/(mode+'-usage.jsonl'))
    relationships,rd=infer_relationships(index,client)
    candidates,extraction=extract_candidates(index,relationships,client,rd)
    atomic_write(directory/(mode+'-candidates.json'),{'candidates':candidates,'diagnostics':extraction})
    before={**relationships,'open_loops':[candidate_loop(c) for c in candidates]}
    started=time.monotonic()
    loops,diagnostics=reconcile_tasks(index,candidates,client)
    elapsed=time.monotonic()-started
    after={**relationships,'open_loops':loops}
    for name,output in [('extraction-only',before),('reconciled',after)]:
        validate(output,index.chats,index.messages)
        if check.validate(output,index.chats,index.messages):
            raise ValueError('Official validation failed')
        atomic_write(directory/(mode+'-'+name+'.json'),output)
    atomic_write(directory/(mode+'-diagnostics.json'),diagnostics)
    # Read labels only after inference; discard validation descriptions and all help data.
    with (source/'answers/answers.json').open(encoding='utf-8') as f:
        gold,boundaries=project_lifecycle_labels(json.load(f),dev_messages,val_messages)
    measured={name:lifecycle_metrics(output['open_loops'],gold,dev_messages)
              for name,output in [('extraction_only',before),('reconciled',after)]}
    empty={'people':[],'organisations':[],'chats':[],'goals':[],'open_loops':[]}
    for name,output in [('extraction_only',before),('reconciled',after)]:
        official=check.score({**empty,'open_loops':output['open_loops']},{**empty,'open_loops':gold},{},messages)
        measured[name]['official_loops_found_fraction']=official['loops_found']
        measured[name]['official_loops_closed_fraction']=official['loops_closed']
        if official['loops_found'] is not None and abs(official['loops_found']-measured[name]['detection']['f1'])>1e-12:
            raise ValueError('Source matching differs from official scorer')
    report={'mode':mode,'comparison':measured,'scope':'frozen development source labels only; same extracted candidate set',
        'boundary_exclusions':boundaries,'reconciliation_seconds':elapsed,
        'additional_model_attempts':diagnostics['additional_model_attempts'],
        'additional_prompt_tokens':diagnostics['additional_prompt_tokens'],
        'additional_completion_tokens':diagnostics['additional_completion_tokens'],
        'additional_model_cost_usd':diagnostics['additional_model_cost_usd'],
        'cost_note':diagnostics['cost_note'],'events':diagnostics['counts'],
        'split_sha256':digest(split_path),'checker_sha256':digest(ROOT/'check.py'),
        'code_sha256':{n:digest(ROOT/n) for n in ('task_lifecycle.py','task_candidates.py','task_dates.py',
                      'tools/lifecycle_metrics.py','tools/evaluate_lifecycle.py')}}
    atomic_write(directory/(mode+'-comparison.json'),report)
    lines=['# Development lifecycle comparison','','Same candidate set before/after reconciliation. Mode: '+mode,
           '', '| Metric | Extraction only | Reconciled |','|---|---:|---:|']
    for key in ('matched','matched_gold_open','matched_predicted_closed','all_predicted_closed','unmatched_predicted_closed','false_closures',
                'false_closure_rate_on_matched_open','false_closure_fraction_of_matched_closed_predictions',
                'missed_closures_matched','missed_closures_with_visible_evidence',
                'closed_tasks_missed_at_extraction','wrong_closed_status','closure_anchor_mismatches',
                'duplicate_anchor_predictions','matched_closed_labels_without_visible_evidence','matched_closed_labels_without_annotated_evidence'):
        a,b=(measured[n]['lifecycle'][key] for n in ('extraction_only','reconciled'))
        lines.append(f'| {key} | {a} | {b} |')
    for key in ('official_loops_found_fraction','official_loops_closed_fraction'):
        lines.append(f'| {key} | {measured["extraction_only"][key]} | {measured["reconciled"][key]} |')
    lines+=['','None means denominator absent/unknown, not zero risk.',
            'Duplicate-anchor predictions are a label-based proxy, not proven semantic duplicates.',
            'Closed tasks absent from extraction cannot be recovered by this reconciliation stage.',
            'Closure evidence outside development chats is unavailable; the visible-evidence count is reported separately.',
            'Model cost/usage: '+json.dumps({k:report[k] for k in ('additional_model_attempts','additional_prompt_tokens','additional_completion_tokens','additional_model_cost_usd')}),
            f'Reconciliation seconds: {elapsed:.4f}.', 'Transition counts: '+json.dumps(report['events']),
            'Offline results do not establish live-model or hidden-set accuracy. Dev labels are imperfect.']
    (directory/(mode+'-comparison.md')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Development lifecycle comparison saved privately; full before/after outputs validated.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not args.worker:
        try:
            return subprocess.run([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker'],timeout=1740).returncode
        except subprocess.TimeoutExpired:
            print('Lifecycle evaluation runtime limit reached.',file=sys.stderr)
            return 1
    try:
        run(args.offline)
        return 0
    except ModelError as exc:
        print(str(exc),file=sys.stderr)
        return 1
    except Exception:
        print('Lifecycle evaluation failed; data omitted for privacy.',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
