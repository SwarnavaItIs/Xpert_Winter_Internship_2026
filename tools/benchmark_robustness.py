"""Uncached local robustness and one-configured-endpoint model benchmark.

No data goes to web/telemetry; only Client may transmit to LLM_BASE_URL.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import check
from inputs import InputIndex
from llm import Client,ModelError
from relationships import infer_relationships
from task_candidates import extract_candidates,bounded_chunks,relationship_context,PROMPT
from task_lifecycle import reconcile_tasks
from goals_help import apply_goals_help
from solution import atomic_write,runtime_budget,completion_report
from validation import validate
from tools.day1 import freeze_split,load_inputs,digest
from tools.evaluate_relationships import project
from tools.evaluate_goals_help import project_labels
from tools.task_metrics import detection_metrics
from tools.robustness import transform,substitute,assert_label_references

PRICING={'source':'https://console.groq.com/docs/models','verified_date':'2026-11-05',
         'base_url':'https://api.groq.com/openai/v1',
         'models':{'openai/gpt-oss-120b':{'input_per_million':0.15,'output_per_million':0.60},
                   'openai/gpt-oss-20b':{'input_per_million':0.075,'output_per_million':0.30}}}


def script_bucket(body):
    if re.search('[\u0980-\u09ff]',body):return 'Bengali-script'
    if re.search('[\u0900-\u097f]',body):return 'Devanagari-script'
    return 'Latin-or-other'  # Cannot distinguish English from Romanized Hinglish.


def load_scope(scope):
    source=ROOT/'.private/official-day1/data/dev';freeze_split(source)
    split=json.loads((ROOT/'.private/splits/day1-v1.json').read_text())
    chats,messages,seed=load_inputs(source)
    dev=set(split['development_chat_ids'])
    vp={p for c in chats if c['chat_id'] in set(split['validation_chat_ids']) for p in c['participants']}-{'ME'}
    if scope=='representative':
        manifest=ROOT/'.private/splits/benchmark-cases-v1.json'
        # Label-free strata by group/direct and observed script; include long chats.
        selected=set()
        for group in (False,True):
            for script in ('Latin-or-other','Devanagari-script','Bengali-script'):
                candidates=[c for c in chats if c['chat_id'] in dev and c['is_group']==group and
                    any(m['chat_id']==c['chat_id'] and script_bucket(m['body'])==script for m in messages)]
                candidates.sort(key=lambda c:(-sum(m['chat_id']==c['chat_id'] for m in messages),c['chat_id']))
                selected.update(c['chat_id'] for c in candidates[:2])
        frozen={'version':1,'chat_ids':sorted(selected),'input_split_sha256':digest(ROOT/'.private/splits/day1-v1.json')}
        if manifest.exists():
            if json.loads(manifest.read_text())!=frozen:raise ValueError('Representative manifest changed')
        else:atomic_write(manifest,frozen)
        dev=selected
    dc=[c for c in chats if c['chat_id'] in dev]
    dp={p for c in dc for p in c['participants']}-{'ME'}
    seed={**seed,'circles':{p:v for p,v in seed.get('circles',{}).items() if p in dp}}
    return InputIndex(dc,[m for m in messages if m['chat_id'] in dev],seed),dp-vp,source


def pipeline(index,client=None,chunk_messages=36):
    start=time.perf_counter()
    rel,rd=infer_relationships(index,client)
    tasks,td=extract_candidates(index,rel,client,rd,core_messages=chunk_messages)
    loops,ld=reconcile_tasks(index,tasks,client)
    state,hd=apply_goals_help(index,{**rel,'open_loops':loops},client)
    if client is not None and completion_report(client,rd,td,ld,hd)['status']!='complete':
        raise ModelError('Benchmark model coverage incomplete; run not scored as complete')
    validate(state,index.chats,index.messages)
    if check.validate(state,index.chats,index.messages):raise ValueError('Official validation failed')
    return state,{'seconds':time.perf_counter()-start,'candidates':tasks,'extraction':td,
                  'relationships':rd,'lifecycle':ld,'goals_help':hd}


def task_metrics(state,gold,index):
    result=detection_metrics(state['open_loops'],gold['open_loops'])[0]
    scripts={}
    for script in ('Latin-or-other','Devanagari-script','Bengali-script'):
        ids={m['id'] for m in index.messages if script_bucket(m['body'])==script}
        labels=[g for g in gold['open_loops'] if set(g['source_msgs'])<=ids]
        predictions=[p for p in state['open_loops'] if p['source_msg'] in ids]
        scripts[script]=detection_metrics(predictions,labels)[0]
        if not labels:
            scripts[script].update(precision=None,recall=None,f1=None)
    return {'all':result,'script_proxies':scripts,'note':'Script strata are not language identification. Mixed-script accepted sources excluded from per-script strata.'}


def payload_profile(index,rel,size):
    chunks=list(bounded_chunks(index,core_messages=size))
    compact=original=0; covered=set(); repetitions=0
    for chunk in chunks:
        payload={'task_stage':'candidates','chunk':chunk,'relationships':relationship_context(index,rel,chunk)}
        original+=len(PROMPT.encode())+len(json.dumps(payload,ensure_ascii=False).encode())
        compact+=len(PROMPT.encode())+len(json.dumps(payload,ensure_ascii=False,separators=(',',':')).encode())
        for v in chunk['core']:covered.add(v['id'])
        repetitions+=len(chunk['overlap'])+len(chunk['quoted'])
    return {'chunks':len(chunks),'prompt_utf8_bytes_before':original,'prompt_utf8_bytes_compact':compact,
            'all_message_ids_covered':covered==set(index.message),'context_views_repeated':repetitions,
            'note':'Bytes, not measured model tokens. Same text content; provider tokenization unmeasured.'}


def price_usage(usage,model):
    known=all(type(u.get(k)) is int for u in usage for k in ('prompt_tokens','completion_tokens'))
    tokens={k:sum(u[k] for u in usage) if known else None for k in ('prompt_tokens','completion_tokens')}
    rates=PRICING['models'].get(model)
    equivalent=((tokens['prompt_tokens']*rates['input_per_million']+tokens['completion_tokens']*rates['output_per_million'])/1e6
                if known and rates else None)
    return {**tokens,'verified_list_price_equivalent_usd':equivalent,
            'actual_charge_usd':None if usage else 0,'pricing':PRICING,
            'charge_note':'Account invoice not observed. Free-plan confirmation authorizes only $0 use; list-price equivalent is not a bill.'}


def synthetic_probe():
    texts={'English':'I will send the report tomorrow.','Hinglish':'Kal main report bhej dunga.',
           'Hindi':'मैं रिपोर्ट भेज दूंगा।','Bengali':'আমি রিপোর্ট পাঠিয়ে দেব।'}
    records={}
    for language,body in texts.items():
        index=InputIndex([dict(chat_id='synthetic-chat',name='Synthetic',is_group=False,participants=['ME','synthetic-person'])],
            [dict(id='synthetic-source',chat_id='synthetic-chat',sender_id='ME',ts=1788300000,body=body,kind='text',quoted_id=None)],{})
        rel,_=infer_relationships(index);tasks,_=extract_candidates(index,rel)
        records[language]={'cases':1,'correct_source_and_owner':int(len(tasks)==1 and tasks[0]['source_msg']=='synthetic-source' and tasks[0]['owner']=='ME')}
    return records


def run(args):
    destination=ROOT/'.private/robustness-benchmarks'/args.tag
    if destination.exists():raise ValueError('Use a new run tag')
    index,exclusive,source=load_scope(args.scope)
    destination.mkdir(parents=True)
    # No cache of predictions or API responses is read anywhere in this harness.
    if args.live:
        configured=all(os.environ.get(k) for k in ('LLM_BASE_URL','LLM_API_KEY','LLM_MODEL'))
        models=args.models.split(',') if args.models else [os.environ.get('LLM_MODEL','')]
        eligible=(configured and args.free_plan_confirmed and 1<=len(models)<=3 and len(set(models))==len(models) and
                  os.environ['LLM_BASE_URL'].rstrip('/')==PRICING['base_url'] and
                  all(m in PRICING['models'] for m in models))
        if not eligible:
            atomic_write(destination/'summary.json',{'status':'pending','reason':'Requires configured single endpoint, shortlisted models and confirmed free-only account; authorized spend $0.',
                'shortlist':PRICING,'models_run':[]})
            print('Model comparisons pending; no requests made.');return
    else:models=['offline']
    variants=['baseline'] if args.live else ['baseline','names','ids','shuffle','distractors','long','larger_chunks']
    results={};output_states={}
    saved_model=os.environ.get('LLM_MODEL')
    try:
        for model in models:
            client=None
            if args.live:
                os.environ['LLM_MODEL']=model
                client=Client(time.monotonic()+runtime_budget(),destination/(model.replace('/','_')+'-usage.jsonl'))
            for kind in variants:
                if kind in ('baseline','larger_chunks'):
                    data={'chats':index.chats,'messages':index.messages,'seed':index.seed};mapping={}
                else:
                    data,meta=transform(index.chats,index.messages,index.seed,kind=kind);mapping=meta['mapping']
                variant=InputIndex(data['chats'],data['messages'],data['seed'])
                name=model.replace('/','_')+'-'+kind
                case_started=time.perf_counter()
                try:
                    state,diagnostics=pipeline(variant,client,72 if kind=='larger_chunks' else 36)
                except Exception as exc:
                    # No provider response/chat content appears in transcripts.
                    results[name]={'status':'failed','error_type':type(exc).__name__,
                                   'seconds':time.perf_counter()-case_started,
                                   'usage':client.usage if client else [],
                                   'cost':price_usage(client.usage if client else [],model),
                                   'structured_output_failures':sum(u.get('status','').startswith('invalid_response') for u in client.usage) if client else 0,
                                   'pipeline_validation_or_inference_failure':True}
                    continue
                atomic_write(destination/(name+'-output.json'),state)
                atomic_write(destination/(name+'-diagnostics.json'),diagnostics)
                atomic_write(destination/(name+'-mapping.json'),mapping)
                output_states[name]=(state,variant,mapping)
                usage=client.usage if client else []
                results[name]={'status':'completed','seconds':diagnostics['seconds'],'messages':len(variant.messages),
                    'model_attempts':len(usage),'latency_seconds':[u['seconds'] for u in usage],
                    'structured_output_failures':sum(u.get('status','').startswith('invalid_response') for u in usage),
                    'request_bytes':sum(u.get('request_bytes',0) for u in usage),
                    'cost':price_usage(usage,model),
                    'local_response_cache':False,'provider_cache_status':'not controlled; not assumed uncached',
                    'payload_profile':payload_profile(variant,state,72 if kind=='larger_chunks' else 36)}
    finally:
        if saved_model is None:os.environ.pop('LLM_MODEL',None)
        else:os.environ['LLM_MODEL']=saved_model
    # Project labels only after every inference run; labels never affect case selection.
    answers=json.loads((source/'answers/answers.json').read_text(encoding='utf-8'))
    gold={**project(answers,set(index.chat),exclusive),**{k:v for k,v in project_labels(answers,set(index.message)).items() if k in ('goals','open_loops')}}
    del answers
    for name,(state,variant,mapping) in output_states.items():
        labels=substitute(gold,mapping);assert_label_references(labels,variant)
        exclusive_mapped={mapping.get(p,p) for p in exclusive}
        scored=project(state,set(variant.chat),exclusive_mapped)
        scored.update(goals=state['goals'],open_loops=state['open_loops'])
        results[name]['scores']=check.score(scored,labels,variant.seed,variant.messages)
        results[name]['task_quality']=task_metrics(state,labels,variant)
        atomic_write(destination/(name+'-labels.json'),labels)
    baseline=output_states.get('offline-baseline')
    if baseline:
        base_state=baseline[0]
        keys={'people':'id','organisations':'id','chats':'chat_id','open_loops':'source_msg','goals':'title'}
        for name,(state,variant,mapping) in output_states.items():
            restored=substitute(state,{v:k for k,v in mapping.items()})
            results[name]['inverse_mapped_output_differences']={field:
                sum(a!=b for a,b in zip(sorted(base_state[field],key=lambda r:r[key]),sorted(restored[field],key=lambda r:r[key])))+
                abs(len(base_state[field])-len(restored[field])) for field,key in keys.items()}
    summary={'status':'completed' if all(v['status']=='completed' for v in results.values()) else 'completed_with_failures',
        'scope':args.scope,'results':results,'live_models_tested':models if args.live else [],
        'synthetic_offline_task_probes':synthetic_probe(),'synthetic_note':'One hand-authored case per language, not model or held-out accuracy.',
        'pending_shortlist':PRICING,'budget_usd':0,'robustness_not_hidden_set':True,
        'input_split_sha256':digest(ROOT/'.private/splits/day1-v1.json'),
        'source_sha256':{str(p.relative_to(ROOT)):digest(p) for p in [*ROOT.glob('*.py'),* (ROOT/'tools').glob('*.py')]}}
    atomic_write(destination/'summary.json',summary)
    for p in [*ROOT.glob('*.py'),*(ROOT/'tools').glob('*.py')]:
        target=destination/'source'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes())
    print(json.dumps({name:{k:v for k,v in row.items() if k in ('status','seconds','messages','scores','structured_output_failures','model_attempts')}
                      for name,row in results.items()},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True)
    parser.add_argument('--scope',choices=['development','representative'],default='development')
    parser.add_argument('--live',action='store_true');parser.add_argument('--free-plan-confirmed',action='store_true')
    parser.add_argument('--models');args=parser.parse_args()
    if not args.tag.replace('-','').replace('_','').isalnum():raise SystemExit('Invalid run tag')
    try:run(args)
    except Exception:
        print('Benchmark failed; data omitted for privacy.',file=sys.stderr);raise SystemExit(1)
