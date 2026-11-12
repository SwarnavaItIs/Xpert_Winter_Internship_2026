"""Offline end-to-end audit and explicit controls; labels never enter inference."""
import argparse
import copy
from collections import Counter
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import check
from inputs import InputIndex
from relationships import infer_relationships
from task_candidates import extract_candidates
from task_lifecycle import reconcile_tasks
from goals_help import apply_goals_help,reminder_time,TZ
from datetime import datetime
from solution import atomic_write
from validation import validate
from tools.day1 import freeze_split,load_inputs,digest
from tools.evaluate_relationships import project
from tools.evaluate_goals_help import project_labels,metrics as help_metrics
from tools.lifecycle_metrics import lifecycle_metrics
from tools.task_metrics import detection_metrics,anchor_diagnostics


def basic_help(index,state):
    result=copy.deepcopy(state)
    for task in result['open_loops']:
        if task['status']!='open':
            task['help']=None
        elif task['owner']=='ME':
            task['help']={'action':'remind','remind_at':reminder_time(task,datetime.fromtimestamp(max(index.timestamps),TZ))[0]}
        else:
            task['help']={'action':'leave','reason':'Other person owns the task.'}
    return result


def errors(index,state,gold,candidates):
    matched_orgs,_,_=check.match_orgs(state,gold)
    circles={p['id']:p['circle'] for p in state['people']}
    circle_errors=[{'id':p['id'],'expected':p['circle'],'predicted':circles.get(p['id'])}
        for p in gold['people'] if p['id'] not in index.seed.get('circles',{}) and circles.get(p['id'])!=p['circle']]
    matrix=Counter((p['expected'],p['predicted']) for p in circle_errors)
    attachments=lambda obj:{o['id']:check.org_members(o['id'],obj['chats'],obj['people']) for o in obj['organisations']}
    ga,pa=attachments(gold),attachments(state)
    splits=[{'gold_org':g,'predicted_orgs':[p for p in pa if ga[g]&pa[p]]} for g in ga if sum(bool(ga[g]&pa[p]) for p in pa)>1]
    merges=[{'predicted_org':p,'gold_orgs':[g for g in ga if ga[g]&pa[p]]} for p in pa if sum(bool(ga[g]&pa[p]) for g in ga)>1]
    pc={c['chat_id']:c for c in state['chats']}
    chat_errors=[]
    for c in gold['chats']:
        predicted=pc[c['chat_id']]
        good=predicted.get('person')==c['person'] if 'person' in c else (predicted.get('organisation') is not None and predicted.get('organisation')==matched_orgs.get(c['organisation'],(None,))[0])
        if not good:
            chat_errors.append({'expected':c,'predicted':predicted,'upstream_org_unmatched': 'organisation' in c and c['organisation'] not in matched_orgs})
    detection,pairs=detection_metrics(state['open_loops'],gold['open_loops'])
    pi={i for i,j in pairs}; gi={j for i,j in pairs}
    owners=[{'source_msg':state['open_loops'][i]['source_msg'],'predicted':state['open_loops'][i]['owner'],
             'expected':gold['open_loops'][j]['owner']} for i,j in pairs if gold['open_loops'][j].get('owner') and state['open_loops'][i]['owner']!=gold['open_loops'][j]['owner']]
    lifecycle=lifecycle_metrics(state['open_loops'],gold['open_loops'],set(index.message))
    anchor=anchor_diagnostics(candidates,gold['open_loops'],detection_metrics(candidates,gold['open_loops'])[1],index)
    help_=help_metrics(state,gold,index.seed,index.messages)
    return {'circle_confusion':[{'expected':a,'predicted':b,'count':n} for (a,b),n in matrix.most_common()],
        'circle_errors':circle_errors,'organisation_split_candidates':splits,'organisation_merge_candidates':merges,
        'chat_errors':chat_errors,'detection':detection,'ownership_errors':owners,'lifecycle':lifecycle,
        'missed_tasks':[g for j,g in enumerate(gold['open_loops']) if j not in gi],
        'unmatched_predictions':[p for i,p in enumerate(state['open_loops']) if i not in pi],
        'anchor_diagnostics':anchor,'help':help_,
        'unmatched_gold_organisations':[g for g in ga if g not in matched_orgs],
        'false_closure_errors':[{'predicted':state['open_loops'][i],'label':gold['open_loops'][j]} for i,j in pairs
            if state['open_loops'][i]['status']!='open' and gold['open_loops'][j]['status']=='open'],
        'missed_closure_errors':[{'predicted':state['open_loops'][i],'label':gold['open_loops'][j]} for i,j in pairs
            if state['open_loops'][i]['status']=='open' and gold['open_loops'][j]['status']!='open'],
        'unsupported_goals':None,'missed_goals':None,'confirmed_false_tasks':None,'semantic_duplicate_tasks':None,
        'attribution':{'extraction_misses':len(gold['open_loops'])-len(gi),
            'ownership_errors_conditional_on_detection':len(owners),
            'closed_labels_missing_upstream':lifecycle['lifecycle']['closed_tasks_missed_at_extraction'],
            'missed_closures_conditional_on_detection':lifecycle['lifecycle']['missed_closures_matched'],
            'help_labels_missing_upstream':help_['unmatched_help_labels'],
            'note':'Conditional metrics isolate available opportunities, not proof of causal responsibility. Unmatched predictions are not confirmed false tasks; split/merge overlap can arise from wrong links.'}}


def run(tag,baseline=None):
    directory=ROOT/'.private/development-audit'/tag
    if directory.exists():
        raise ValueError('Use a new tag; audit runs are immutable')
    source=ROOT/'.private/official-day1/data/dev';freeze_split(source)
    split_path=ROOT/'.private/splits/day1-v1.json';split=json.loads(split_path.read_text())
    chats,messages,seed=load_inputs(source)
    dev=set(split['development_chat_ids']); val=set(split['validation_chat_ids'])
    dc=[c for c in chats if c['chat_id'] in dev]
    dp={p for c in dc for p in c['participants']}-{'ME'}
    vp={p for c in chats if c['chat_id'] in val for p in c['participants']}-{'ME'}
    seed={**seed,'circles':{p:v for p,v in seed.get('circles',{}).items() if p in dp}}
    index=InputIndex(dc,[m for m in messages if m['chat_id'] in dev],seed)
    timings={}
    def timed(name,fn):
        start=time.perf_counter();value=fn();timings[name]=time.perf_counter()-start;return value
    start=time.perf_counter()
    rel,rd=timed('relationships',lambda:infer_relationships(index))
    candidates,cd=timed('extract_with_relationships',lambda:extract_candidates(index,rel,None,rd))
    blank={'people':[],'organisations':[],'chats':[]}
    no_context,_=timed('extract_without_relationships',lambda:extract_candidates(index,blank))
    loops,ld=timed('lifecycle_verified',lambda:reconcile_tasks(index,candidates))
    raw_loops,raw_ld=timed('lifecycle_unverified',lambda:reconcile_tasks(index,candidates,verify_closures=False))
    state={**rel,'open_loops':loops}
    after,hd=timed('goals_and_help',lambda:apply_goals_help(index,state))
    help_only,_=timed('explicit_policy',lambda:apply_goals_help(index,after,include_goals=False))
    assert help_only==after
    basic=timed('basic_help',lambda:basic_help(index,state))
    for output in (after,basic,{**rel,'open_loops':raw_loops}):
        validate(output,index.chats,index.messages)
        assert not check.validate(output,index.chats,index.messages)
    elapsed=time.perf_counter()-start
    directory.mkdir(parents=True)
    for name,value in [('output',after),('candidates',candidates),('relationships',rd),('extraction',cd),
                       ('lifecycle',ld),('unverified-lifecycle',raw_ld),('goals-help',hd)]:
        atomic_write(directory/(name+'.json'),value)
    # Only now read labels; shared people and validation rows are immediately discarded.
    answers=json.loads((source/'answers/answers.json').read_text(encoding='utf-8'))
    gold={**project(answers,dev,dp-vp),**{k:v for k,v in project_labels(answers,set(index.message)).items() if k in ('goals','open_loops')}}
    del answers
    scoped=project(after,dev,dp-vp)
    scoped.update(goals=after['goals'],open_loops=after['open_loops'])
    report=errors(index,scoped,gold,candidates)
    atomic_write(directory/'errors.json',report)
    comparisons={
        'relationship_context':{'without':detection_metrics(no_context,gold['open_loops'])[0],
            'with':detection_metrics(candidates,gold['open_loops'])[0],
            'identical_candidates':no_context==candidates,'scope':'Ran offline; offline extractor does not consume relationship semantics. Live comparison not run.'},
        'closure_verification':{'without':lifecycle_metrics(raw_loops,gold['open_loops'],set(index.message)),
            'with':lifecycle_metrics(loops,gold['open_loops'],set(index.message)),
            'scope':'Offline lexical closure proposals versus conservative verification; same candidates, retrieval, ambiguity handling and state fold. Model verifier ablation not run.'},
        'help_policy':{'basic':help_metrics(basic,gold,seed,index.messages),'explicit':help_metrics(after,gold,seed,index.messages),
            'changed_decisions':sum(a.get('help')!=b.get('help') for a,b in zip(basic['open_loops'],after['open_loops'])),
            'scope':'Basic: remind ME-owned work, leave others. Explicit: complete intervention policy. Goal/help labels unavailable.'}}
    summary={'mode':'offline','scores':check.score(scoped,gold,seed,index.messages),
        'samples':{'messages':len(index.messages),'chats':len(dc),'exclusive_people':len(dp-vp),'task_labels':len(gold['open_loops']),
                   'scored_people':sum(p['id'] not in seed['circles'] for p in gold['people']),'labelled_chats':len(gold['chats'])},
        'circle_confusion':report['circle_confusion'],'incorrect_chats':len(report['chat_errors']),
        'split_candidates':len(report['organisation_split_candidates']),'merge_candidates':len(report['organisation_merge_candidates']),
        'ownership_errors':len(report['ownership_errors']),'attribution':report['attribution'],
        'comparisons':comparisons,'timings_seconds':timings,'total_seconds':elapsed,
        'pipeline_stage_seconds':sum(timings[k] for k in ('relationships','extract_with_relationships','lifecycle_verified','goals_and_help')),
        'tokens':0,'model_attempts':0,'api_cost_usd':0,'live_experiments':'not run; configuration absent',
        'hashes':{str(p.relative_to(ROOT)):digest(p) for p in [split_path,ROOT/'check.py',*ROOT.glob('*.py'),Path(__file__).resolve()]}}
    if baseline:
        previous=json.loads((ROOT/'.private/development-audit'/baseline/'summary.json').read_text(encoding='utf-8'))
        split_key=str(split_path.relative_to(ROOT))
        assert previous['hashes'][split_key]==summary['hashes'][split_key]
        summary['baseline_tag']=baseline
        summary['score_deltas']={k:summary['scores'][k]-v if v is not None and summary['scores'][k] is not None else None
                                 for k,v in previous['scores'].items()}
        summary['audit_runtime_delta_seconds']=elapsed-previous['total_seconds']
    # Preserve exact current source alongside hashes, including evaluation code.
    for source_file in [*ROOT.glob('*.py'),* (ROOT/'tools').glob('*.py')]:
        target=directory/'source'/source_file.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source_file.read_bytes())
    atomic_write(directory/'summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('scores','samples','circle_confusion','incorrect_chats','split_candidates','merge_candidates','ownership_errors','attribution','timings_seconds','total_seconds','tokens','api_cost_usd')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);parser.add_argument('--baseline');args=parser.parse_args()
    for tag in (args.tag,args.baseline):
        if tag is not None and not tag.replace('-','').replace('_','').isalnum():
            raise SystemExit('Invalid tag')
    run(args.tag,args.baseline)
