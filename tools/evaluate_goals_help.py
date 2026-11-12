"""Frozen development-only comparison on the same cached lifecycle tasks."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import check
from goals_help import apply_goals_help
from inputs import InputIndex
from llm import Client,ModelError
from solution import atomic_write
from tools.day1 import freeze_split,load_inputs,digest
from tools.task_metrics import match_sources
from validation import validate


def project_labels(answers,visible):
    # Conservative partition: ALL supporting/source IDs must belong to development.
    goals=[g for g in answers.get('goals',[]) if g.get('evidence') and set(g['evidence'])<=visible]
    tasks=[t for t in answers.get('open_loops',[]) if t.get('source_msgs') and set(t['source_msgs'])<=visible]
    return {'people':[],'organisations':[],'chats':[],'goals':goals,'open_loops':tasks}


def metrics(state,gold,seed,messages):
    empty={'people':[],'organisations':[],'chats':[]}
    scores=check.score({**state,**empty},gold,seed,messages)
    pairs=match_sources(state['open_loops'],gold['open_loops'])
    # match_sources returns (prediction index, label index) pairs.
    intrusions=[]; correct=0; matched=0; matched_leave=0
    for i,j in pairs:
        task,label=state['open_loops'][i],gold['open_loops'][j]
        if label['status']!='open' or not label.get('help'):
            continue
        action=(task.get('help') or {}).get('action') if task['status']=='open' else None
        matched+=1
        matched_leave+=label['help']=='leave'
        correct+=action==label['help']
        if label['help']=='leave' and action in ('assist','remind'):
            intrusions.append({'source_msg':task['source_msg'],'predicted':action,'expected':'leave'})
    n=sum(t['status']=='open' and bool(t.get('help')) for t in gold['open_loops'])
    return {'official_goals':scores['goals'],'official_help':scores['help'],
        'goal_predictions':len(state['goals']),'goal_labels':len(gold['goals']),
        'help_labels':n,'matched_help_labels':matched,'correct_help':correct,
        'unmatched_help_labels':n-matched,'intrusive_interventions':len(intrusions),
        'matched_leave_labels':matched_leave,'intrusion_rate_on_matched_leave':len(intrusions)/matched_leave if matched_leave else None,
        'private_intrusion_errors':intrusions}


def run(offline):
    source=ROOT/'.private/official-day1/data/dev'
    freeze_split(source)
    split_path=ROOT/'.private/splits/day1-v1.json'
    split=json.loads(split_path.read_text())
    visible=set(split['development_message_ids']); chat_ids=set(split['development_chat_ids'])
    chats,messages,seed=load_inputs(source)
    index=InputIndex([c for c in chats if c['chat_id'] in chat_ids],[m for m in messages if m['id'] in visible],seed)
    baseline_path=ROOT/'.private/lifecycle-development/offline-reconciled.json'
    before=json.loads(baseline_path.read_text(encoding='utf-8'))
    validate(before,index.chats,index.messages)
    directory=ROOT/'.private/goals-help-development';directory.mkdir(parents=True,exist_ok=True)
    mode='offline' if offline else 'model'
    client=None if offline else Client(time.monotonic()+1680,directory/(mode+'-usage.jsonl'))
    started=time.monotonic()
    after,diagnostics=apply_goals_help(index,before,client)
    validate(after,index.chats,index.messages)
    assert not check.validate(after,index.chats,index.messages)
    atomic_write(directory/(mode+'-output.json'),after)
    atomic_write(directory/(mode+'-diagnostics.json'),diagnostics)
    elapsed=time.monotonic()-started
    # Labels opened only AFTER all inference; no validation scoring or tuning.
    gold=project_labels(json.loads((source/'answers/answers.json').read_text(encoding='utf-8')),visible)
    report={'before':metrics(before,gold,seed,index.messages),'after':metrics(after,gold,seed,index.messages),
        'stage_seconds':elapsed,'split_sha256':digest(split_path),'baseline_sha256':digest(baseline_path),
        'checker_sha256':digest(ROOT/'check.py'),'code_sha256':digest(ROOT/'goals_help.py'),
        'baseline_note':'Same cached OFFLINE lifecycle tasks; earlier help was provisional leave. Model mode changes only goals/help.',
        'additional_model_attempts':diagnostics['additional_model_attempts'],
        'additional_model_cost_usd':diagnostics['additional_model_cost_usd'],
        'additional_prompt_tokens':diagnostics['additional_prompt_tokens'],
        'additional_completion_tokens':diagnostics['additional_completion_tokens'],
        'intervention_counts':dict(Counter((t.get('help') or {}).get('action','closed') for t in after['open_loops'])),
        'reminders_not_after_endpoint':sum(datetime.fromisoformat(t['help']['remind_at']).timestamp()<=max(index.timestamps)
            for t in after['open_loops'] if (t.get('help') or {}).get('action')=='remind'),
        'semantic_intrusion_error_rate':None,
        'limitations':['Source-matched labels do not measure unmatched interventions.','Official goal score does not verify type/horizon or truth of evidence.',
                      'Approval and reminder quality require independent semantic review; not covered by official help score.']}
    atomic_write(directory/(mode+'-comparison.json'),report)
    (directory/(mode+'-comparison.md')).write_text(
        '# Goal/help development comparison\n\n'+
        'Same frozen development inputs and cached lifecycle tasks. No validation tuning.\n\n'+
        '| Metric | Before | After |\n|---|---:|---:|\n'+
        '\n'.join('| '+key+' | '+str(report['before'][key])+' | '+str(report['after'][key])+' |'
                  for key in ('official_goals','official_help','goal_predictions','goal_labels','help_labels',
                              'matched_help_labels','intrusive_interventions','intrusion_rate_on_matched_leave'))+
        '\n\nNull/None denotes missing labels or denominator, never zero errors.\n'+
        'Semantic intrusion errors cannot be measured without help labels or independent review.\n'+
        'Structural checks do not establish appropriate intervention.\n',encoding='utf-8')
    aggregate={k:v for k,v in report.items() if k in ('stage_seconds','additional_model_attempts','additional_model_cost_usd')}
    for name in ('before','after'):
        aggregate[name]={k:v for k,v in report[name].items() if k!='private_intrusion_errors'}
    print(json.dumps(aggregate,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--offline',action='store_true');parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if not args.worker:
        try:
            return subprocess.run([sys.executable,__file__,*sys.argv[1:],'--worker'],timeout=1740).returncode
        except subprocess.TimeoutExpired:
            print('Goal/help evaluation runtime limit reached.',file=sys.stderr);return 1
    try:
        run(args.offline);return 0
    except ModelError as exc:
        print(str(exc),file=sys.stderr);return 1
    except Exception:
        print('Goal/help evaluation failed; data omitted for privacy.',file=sys.stderr);return 1


if __name__=='__main__':
    raise SystemExit(main())
