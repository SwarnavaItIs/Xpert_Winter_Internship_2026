"""Local, invertible robustness transforms. Transformed examples are NOT new labels."""
import copy
import random
import re

from inputs import InputIndex


def substitute(value, mapping):
    # One pass avoids cascades; dictionary keys include seed circle/person IDs.
    pattern=re.compile(r'(?<!\w)(?:'+ '|'.join(re.escape(k) for k in sorted(mapping,key=len,reverse=True))+r')(?!\w)') if mapping else None
    def visit(item):
        if isinstance(item,str):
            return mapping.get(item,pattern.sub(lambda m:mapping[m.group()],item) if pattern else item)
        if isinstance(item,list):
            return [visit(v) for v in item]
        if isinstance(item,dict):
            return {visit(k):visit(v) for k,v in item.items()}
        return item
    return visit(value)


def transform(chats,messages,seed,labels=None,kind='ids',random_seed=773):
    """All ID references (including unresolved quotes/seed IDs) round-trip exactly."""
    original={'chats':chats,'messages':messages,'seed':seed,'labels':labels}
    mapping={};rng=random.Random(random_seed)
    if kind=='ids':
        people={p for c in chats for p in c['participants']}|{m['sender_id'] for m in messages}|set(seed.get('circles',{}))
        ids=sorted((people-{'ME'})|{c['chat_id'] for c in chats}|{m['id'] for m in messages}|
                   {m['quoted_id'] for m in messages if m.get('quoted_id')})
        replacements=[f'robust_id_{i:06d}' for i in range(len(ids))];rng.shuffle(replacements)
        if set(replacements)&set(ids):
            raise ValueError('Transformation ID collision')
        mapping=dict(zip(ids,replacements))
    elif kind=='names':
        names={m.get('sender_name','').strip() for m in messages if m['sender_id']!='ME'}-{'','ME'}
        # Cover observed case variants without conflating ambiguous people sharing a name.
        groups={name.casefold() for name in names}
        aliases={name:f'PersonAlias{i:05d}' for i,name in enumerate(sorted(groups))}
        strings=[]
        def collect(value):
            if isinstance(value,str):strings.append(value)
            elif isinstance(value,list):
                for v in value:collect(v)
            elif isinstance(value,dict):
                for k,v in value.items():collect(k);collect(v)
        collect(original)
        # Preserve distinct observed casing for invertibility. No inferred nicknames.
        variants=set()
        for name in sorted(groups):
            pattern=re.compile(r'(?<!\w)'+re.escape(name)+r'(?!\w)',re.I)
            for text in strings:
                variants.update(m.group() for m in pattern.finditer(text))
        for group in groups:
            for i,name in enumerate(sorted(v for v in variants if v.casefold()==group)):
                letters=0;replacement=[]
                for character in aliases[group]:
                    if character.isalpha():
                        replacement.append(character.upper() if i & (1<<letters) else character.lower());letters+=1
                    else:replacement.append(character)
                if i>=2**letters:raise ValueError('Too many name case variants')
                mapping[name]=''.join(replacement)
    elif kind not in ('shuffle','distractors','long'):
        raise ValueError('Unknown robustness transform')
    changed=substitute(original,mapping)
    if mapping and substitute(changed,{v:k for k,v in mapping.items()})!=original:
        raise ValueError('Substitution is not invertible')
    if kind=='shuffle':
        rng.shuffle(changed['chats']);rng.shuffle(changed['messages'])
    if kind in ('distractors','long'):
        ids={m['id'] for m in changed['messages']}
        count=3 if kind=='distractors' else 40
        for chat in changed['chats']:
            local=sorted([m for m in changed['messages'] if m['chat_id']==chat['chat_id']],key=lambda m:(m['ts'],m['id']))
            if not local:continue
            for n in range(count):
                mid=f'robust_noise_{len(ids):08d}'
                while mid in ids:mid+='x'
                ids.add(mid)
                # Insert within observed time span: endpoint and relative dates remain fixed.
                ts=local[0]['ts']+(local[-1]['ts']-local[0]['ts'])*(n+1)/(count+1)
                changed['messages'].append(dict(id=mid,chat_id=chat['chat_id'],sender_id=local[0]['sender_id'],
                    sender_name=local[0].get('sender_name',''),ts=ts,kind='text',quoted_id=None,
                    body=['Nice weather today.','[Forwarded] Generic advice: people should stay curious.',
                          'Synthetic unrelated small talk about a blue cup.'][n%3]))
    InputIndex(changed['chats'],changed['messages'],changed['seed'])
    return changed,{'kind':kind,'seed':random_seed,'mapping':mapping,
                    'added_messages':len(changed['messages'])-len(messages),
                    'limitation':'Mechanical substitutions cover observed full names only; ambiguous natural-language uses may change meaning.'}


def assert_label_references(labels,index):
    for task in (labels or {}).get('open_loops',[]):
        for mid in task.get('source_msgs',[]):
            if mid not in index.message:raise ValueError('Transformed task source missing')
        # Closing references outside the projected input can legitimately remain unresolved.
    for goal in (labels or {}).get('goals',[]):
        if not set(goal.get('evidence',[]))<=set(index.message):raise ValueError('Transformed goal evidence missing')
