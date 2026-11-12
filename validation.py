"""Strict schema subset interpreter plus challenge-specific reference checks."""
import datetime
import json
from pathlib import Path


class InvalidOutput(ValueError):
    pass


def require(condition):
    if not condition:
        raise InvalidOutput('Output failed schema or evidence validation')


def schema_check(value, spec):
    types = {'object': dict, 'array': list, 'string': str, 'boolean': bool, 'null': type(None)}
    if 'type' in spec:
        allowed = spec['type'] if isinstance(spec['type'], list) else [spec['type']]
        require(any(isinstance(value, types[t]) for t in allowed))
    if 'enum' in spec:
        require(value in spec['enum'])
    if isinstance(value, dict):
        require(set(spec.get('required', [])) <= value.keys())
        props = spec.get('properties', {})
        if spec.get('additionalProperties') is False:
            require(value.keys() <= props.keys())
        for key in value.keys() & props.keys():
            schema_check(value[key], props[key])
        if 'oneOf' in spec:
            require(sum(set(s.get('required', [])) <= value.keys() for s in spec['oneOf']) == 1)
    if isinstance(value, list):
        require(spec.get('minItems', 0) <= len(value) <= spec.get('maxItems', float('inf')))
        for item in value:
            schema_check(item, spec.get('items', {}))


def validate(out, chats, messages):
    schema_check(out, json.loads(Path(__file__).with_name('schema.json').read_text()))
    people = {p for c in chats for p in c['participants']} - {'ME'}
    chat_ids = {c['chat_id'] for c in chats}
    msg = {m['id']: m for m in messages}
    orgs = {o['id'] for o in out['organisations']}
    require(len(orgs) == len(out['organisations']))
    require({p['id'] for p in out['people']} == people)
    require(len(out['people']) == len(people))
    require({c['chat_id'] for c in out['chats']} == chat_ids)
    require(len(out['chats']) == len(chat_ids))
    for p in out['people']:
        require(set(p.get('looks_after', [])) <= people)
        require(p['id'] not in p.get('looks_after', []))
        require(not p.get('looks_after') or p['circle']=='staff')
        require(len(p.get('links',[]))==len({(x['organisation'],x['link']) for x in p.get('links',[])}))
        require(all(x['organisation'] in orgs for x in p.get('links', [])))
    for c in out['chats']:
        require(c.get('person') in people or c.get('organisation') in orgs)
    for g in out['goals']:
        require(set(g['evidence']) <= msg.keys())
        require(bool(g['title'].strip()) and len(g['evidence'])==len(set(g['evidence'])))
    for loop in out['open_loops']:
        require(loop['owner'] in people | {'ME'})
        require(loop['source_msg'] in msg)
        require(msg[loop['source_msg']]['chat_id'] == loop['chat_id'])
        require(bool(loop['title'].strip()))
        try:
            source_date=datetime.datetime.fromtimestamp(msg[loop['source_msg']]['ts'],datetime.timezone.utc).date()
            endpoint=max((m['ts'] for m in messages),default=0)
        except (ValueError,OverflowError,OSError):
            raise InvalidOutput('Invalid timestamp range') from None
        if loop.get('due') is not None:
            try:due=datetime.date.fromisoformat(loop['due'])
            except ValueError:raise InvalidOutput('Invalid due date') from None
            require(due.isoformat() == loop['due'])
            require(abs((due-source_date).days)<=3660)
        if loop['status'] != 'open':
            require(loop.get('closed_by') in msg)
            require(loop['closed_by']!=loop['source_msg'])
            require(msg[loop['closed_by']]['ts'] >= msg[loop['source_msg']]['ts'])
            require(loop.get('help') is None)
        else:
            require(loop.get('closed_by') is None)
            h = loop.get('help')
            require(isinstance(h, dict))
            if h['action'] == 'assist':
                require(bool(h.get('next_step','').strip()) and isinstance(h.get('needs_yes'), bool))
                require(set(h)=={'action','next_step','needs_yes'})
            elif h['action'] == 'remind':
                try:dt = datetime.datetime.fromisoformat(h.get('remind_at', ''))
                except ValueError:raise InvalidOutput('Invalid reminder date') from None
                require(dt.tzinfo is not None)
                require(set(h)=={'action','remind_at'})
                require(dt.timestamp()>endpoint)
                due_ts=datetime.datetime.combine(due,datetime.time(),datetime.timezone.utc).timestamp() if loop.get('due') else endpoint
                require(dt.timestamp()<=max(endpoint,due_ts)+366*86400)
            else:
                require(bool(h.get('reason','').strip()) and set(h)=={'action','reason'})
