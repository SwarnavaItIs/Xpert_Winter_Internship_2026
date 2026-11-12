"""Explicit user-only corrections over an immutable inferred snapshot; no model calls."""
import copy
import hashlib
import json
import re

from inputs import CIRCLES
from goals_help import apply_goals_help
from validation import validate


class Clarification(ValueError):
    def __init__(self,question,choices):
        super().__init__(question)
        self.question=question
        self.choices=choices  # Private IDs/titles; never print these in shared transcripts.


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def task_key(task):
    return fingerprint({k:task[k] for k in ('source_msg','chat_id','owner','type','title')})


def one(items,question):
    if len(items)!=1:raise Clarification(question,items)
    return items[0]


def person(index,state,selector,child=False):
    selector=selector or {}
    if not isinstance(selector,dict) or set(selector)-{'id','name'} or any(not isinstance(v,str) or not v.strip() for v in selector.values()):
        raise ValueError('Invalid person selector')
    candidates=set(index.people)
    if 'id' in selector:candidates&={selector['id']}
    if 'name' in selector:candidates&=index.names.get(selector['name'].strip().casefold(),set())
    if child and not selector:
        candidates&={p['id'] for p in state['people'] if p['circle']=='child'}
    return one(sorted(candidates),'Which person? Supply a unique person ID; names may be ambiguous.')


def normalize_request(index,state,request):
    if not isinstance(request,dict):raise ValueError('Correction request must be an object')
    kind=request.get('kind')
    event={'kind':kind,'authority':'explicit_user','text':request.get('text','')}
    if kind=='set_circle':
        if request.get('circle') not in CIRCLES:raise ValueError('Unknown circle')
        event.update(person_id=person(index,state,request.get('person')),circle=request['circle'])
    elif kind=='set_child_chat':
        selector=request.get('chat') or {}
        if not isinstance(selector,dict) or set(selector)-{'id','name'}:raise ValueError('Invalid chat selector')
        chats=[c for c in index.chats if c['is_group']]
        if 'id' in selector:chats=[c for c in chats if c['chat_id']==selector['id']]
        if 'name' in selector:chats=[c for c in chats if c['name'].casefold()==selector['name'].casefold()]
        cid=one([c['chat_id'] for c in chats],'Which group? Supply a unique chat ID.')
        event.update(chat_id=cid,person_id=person(index,state,request.get('child'),True))
    elif kind=='reopen_task':
        selector=request.get('task') or {}
        if not isinstance(selector,dict) or set(selector)-{'source_msg','chat_id','owner','type','title','key'}:raise ValueError('Invalid task selector')
        tasks=state['open_loops']
        for key in ('source_msg','chat_id','owner','type','title'):
            if key in selector:tasks=[t for t in tasks if t[key]==selector[key]]
        if 'key' in selector:tasks=[t for t in tasks if task_key(t)==selector['key']]
        selected=one([{'key':task_key(t),'source_msg':t['source_msg'],'owner':t['owner'],'title':t['title']} for t in tasks],
                     'Which obligation? Supply source plus title/owner, or the task key; one message can contain multiple tasks.')
        event.update(task_key=selected['key'])
    elif kind=='leave_delegated':
        scope=request.get('scope') or {}
        if scope.get('task_type') not in ('commitment','followup','deadline','question'):
            raise Clarification('Which delegated task type should be left alone? Specify task_type and optionally owner_circle.',[])
        if scope.get('owner_circle') is not None and scope['owner_circle'] not in CIRCLES:
            raise ValueError('Unknown owner circle')
        event['scope']={k:scope[k] for k in ('task_type','owner_circle') if k in scope}
    else:raise ValueError('Unsupported correction kind')
    return event


def text_request(text,person_id=None,chat_id=None,child_id=None,source=None,title=None,owner=None,task_type=None,owner_circle=None):
    """Intentionally small grammar; ambiguous pronouns are resolved by explicit selectors."""
    lowered=text.strip().lower().rstrip('.!')
    if re.fullmatch(r'this person is my business partner(?:,? not (?:a |my )?client)?',lowered):
        return {'kind':'set_circle','person':{'id':person_id} if person_id else {},'circle':'partner','text':text}
    if lowered=='this group belongs to my child':
        return {'kind':'set_child_chat','chat':{'id':chat_id} if chat_id else {},'child':{'id':child_id} if child_id else {},'text':text}
    if lowered in ('that task is still open','this task is still open'):
        return {'kind':'reopen_task','task':{k:v for k,v in {'source_msg':source,'title':title,'owner':owner}.items() if v is not None},'text':text}
    if lowered=='leave this type of delegated task alone':
        return {'kind':'leave_delegated','scope':{k:v for k,v in {'task_type':task_type,'owner_circle':owner_circle}.items() if v is not None},'text':text}
    raise ValueError('Unsupported phrasing; use a structured request instead of guessing intent')


def new_ledger(index,base):
    return {'version':1,'base_sha256':fingerprint(base),
            'input_sha256':fingerprint({'chats':index.chats,'messages':index.messages,'seed':index.seed}),
            'precedence':'latest explicit correction > earlier correction > seed > inferred snapshot',
            'events':[]}


def replay(index,base,ledger):
    expected=new_ledger(index,base)
    if any(ledger.get(k)!=expected[k] for k in ('version','base_sha256','input_sha256')):
        raise ValueError('Correction ledger belongs to a different input/base snapshot')
    validate(base,index.chats,index.messages)
    result=copy.deepcopy(base)
    people={p['id']:p for p in result['people']};chats={c['chat_id']:c for c in result['chats']}
    tasks={task_key(t):t for t in result['open_loops']}
    if len(tasks)!=len(result['open_loops']):raise ValueError('Duplicate indistinguishable tasks require base repair')
    affected_people=set();affected_chats=set();affected_tasks=set();policies=[]
    for event in ledger['events']:
        if event.get('authority')!='explicit_user':raise ValueError('Invalid correction authority')
        kind=event['kind']
        if kind=='set_circle':
            pid=event['person_id']
            if pid not in people or event['circle'] not in CIRCLES:raise ValueError('Invalid correction target')
            people[pid]['circle']=event['circle'];affected_people.add(pid)
        elif kind=='set_child_chat':
            cid,pid=event['chat_id'],event['person_id']
            if cid not in chats or pid not in people or not index.chat[cid]['is_group']:raise ValueError('Invalid group correction')
            chats[cid].clear();chats[cid].update(chat_id=cid,person=pid)
            people[pid]['circle']='child';affected_people.add(pid);affected_chats.add(cid)
        elif kind=='reopen_task':
            key=event['task_key']
            if key not in tasks:raise ValueError('Correction task no longer exists')
            tasks[key].update(status='open',closed_by=None);affected_tasks.add(key)
        elif kind=='leave_delegated':
            policies.append(event['scope'])
        else:raise ValueError('Invalid stored correction')
    # Only repair dependent relationship fields; no affiliations are invented.
    for pid in affected_people:
        p=people[pid]
        if p['circle']!='staff' and p.get('looks_after'):p['looks_after']=[]
        for link in p.get('links',[]):
            if p['circle']=='staff':link['link']='supports'
            elif p['circle'] in ('partner','colleague'):link['link']='runs'
        affected_chats.update(c['chat_id'] for c in index.chats if pid in c['participants'])
    # Recompute only the dependency closure, from the immutable base on every replay.
    for key,t in tasks.items():
        if t['owner'] in affected_people or t['chat_id'] in affected_chats:affected_tasks.add(key)
    for scope in policies:
        for key,t in tasks.items():
            if (t['owner']!='ME' and t['type']==scope['task_type'] and
                (not scope.get('owner_circle') or people[t['owner']]['circle']==scope['owner_circle'])):
                affected_tasks.add(key)
    subset={**result,'open_loops':[t for key,t in tasks.items() if key in affected_tasks]}
    updated,diagnostics=apply_goals_help(index,subset,include_goals=False)
    refreshed={task_key(t):t for t in updated['open_loops']}
    for scope in policies:
        for t in refreshed.values():
            if (t['status']=='open' and t['owner']!='ME' and t['type']==scope['task_type'] and
                (not scope.get('owner_circle') or people[t['owner']]['circle']==scope['owner_circle'])):
                t['help']={'action':'leave','reason':'Explicit user preference: leave this delegated task type with its owner.'}
    result['open_loops']=[refreshed.get(task_key(t),t) for t in result['open_loops']]
    validate(result,index.chats,index.messages)
    return result,{'affected_people':sorted(affected_people),'affected_chats':sorted(affected_chats),
                   'affected_task_keys':sorted(affected_tasks),'intervention_recomputation':diagnostics,
                   'note':'Targeted deterministic correction replay; no task redetection or unrelated model rerun.'}


def readable_diff(before,after):
    lines=['# Explicit correction diff','']
    for section,key in [('people','id'),('chats','chat_id'),('organisations','id'),('goals','title'),('open_loops',None)]:
        old={row[key] if key else task_key(row):row for row in before[section]}
        new={row[key] if key else task_key(row):row for row in after[section]}
        for identity in sorted(old.keys()|new.keys()):
            if old.get(identity)==new.get(identity):continue
            # Indented JSON renders untrusted titles/IDs as text, not Markdown links.
            lines.extend([f'## {section}', '', '    '+json.dumps(identity,ensure_ascii=False), '',
                          'Before:', '', '    '+json.dumps(old.get(identity),ensure_ascii=False), '',
                          'After:', '', '    '+json.dumps(new.get(identity),ensure_ascii=False),''])
    if len(lines)==2:lines.append('No output facts changed.')
    return '\n'.join(lines)+'\n'
