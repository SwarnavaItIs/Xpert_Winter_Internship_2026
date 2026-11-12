"""Validated, deterministic input indexes. Never opens labels or arbitrary files."""
from bisect import bisect_left, bisect_right
from collections import defaultdict
import json
import math
from pathlib import Path
import re

CIRCLES = {'spouse', 'child', 'parent', 'in-law', 'sibling', 'relative', 'friend',
           'colleague', 'client', 'vendor', 'partner', 'staff'}


class InputError(ValueError):
    pass


def need(condition, code):
    if not condition:
        raise InputError('Invalid input: ' + code)  # No data values in errors.


class InputIndex:
    def __init__(self, chats, messages, seed):
        need(isinstance(chats, list) and isinstance(messages, list) and isinstance(seed, dict), 'containers')
        self.chat = {}
        for c in chats:
            need(isinstance(c, dict), 'chat object')
            need(isinstance(c.get('chat_id'), str) and bool(c['chat_id']), 'chat id')
            need(c['chat_id'] not in self.chat, 'duplicate chat id')
            need(isinstance(c.get('name'), str) and type(c.get('is_group')) is bool, 'chat metadata')
            ps = c.get('participants')
            need(isinstance(ps, list) and all(isinstance(p, str) and p for p in ps), 'participants')
            need(len(ps) == len(set(ps)) and bool(set(ps)-{'ME'}), 'participant uniqueness/nonself')
            self.chat[c['chat_id']] = {**c, 'participants': sorted(ps)}
        self.chats = sorted(self.chat.values(), key=lambda c: c['chat_id'])
        self.people = sorted({p for c in chats for p in c['participants']} - {'ME'})
        self.message = {}
        self.warnings = defaultdict(int)
        for m in messages:
            need(isinstance(m, dict), 'message object')
            need(isinstance(m.get('id'), str) and bool(m['id']), 'message id')
            need(m['id'] not in self.message, 'duplicate message id')
            need(m.get('chat_id') in self.chat, 'unknown chat')
            need(isinstance(m.get('sender_id'), str) and m['sender_id'] in set(self.people)|{'ME'}, 'unknown sender')
            # Historical group membership can differ from the supplied participant snapshot.
            if m['sender_id'] not in self.chat[m['chat_id']]['participants']:
                self.warnings['sender_not_in_chat_snapshot'] += 1
            need(type(m.get('ts')) in (int, float) and math.isfinite(m['ts']), 'timestamp')
            need(isinstance(m.get('body'), str), 'body')
            need(m.get('kind') in {'text', 'voice_transcript', 'image_caption', 'document'}, 'kind')
            need(isinstance(m.get('sender_name', ''), str), 'sender name')
            need(m.get('quoted_id') is None or isinstance(m['quoted_id'], str), 'quoted id')
            self.message[m['id']] = m
        self.messages = sorted(messages, key=lambda m: (m['ts'], m['id']))
        self.timestamps = [m['ts'] for m in self.messages]
        self.by_chat, self.by_person, self.by_timestamp = defaultdict(list), defaultdict(list), defaultdict(list)
        self.person_chats, self.replies, self.names = defaultdict(list), defaultdict(list), defaultdict(set)
        for c in self.chats:
            for pid in c['participants']:
                self.person_chats[pid].append(c['chat_id'])
        for m in self.messages:
            self.by_chat[m['chat_id']].append(m)
            self.by_person[m['sender_id']].append(m)
            self.by_timestamp[m['ts']].append(m)
            name = m.get('sender_name', '').strip().casefold()
            if name and m['sender_id'] != 'ME':
                self.names[name].add(m['sender_id'])
            q = m.get('quoted_id')
            if q:
                if q in self.message:
                    self.replies[q].append(m)
                else:
                    self.warnings['unresolved_quotes'] += 1
        circles = seed.get('circles') or {}
        need(isinstance(circles, dict), 'seed circles')
        need(all(isinstance(p, str) and circle in CIRCLES for p, circle in circles.items()), 'seed circle values')
        need(isinstance(seed.get('organisations', []), list), 'seed organisations')
        for org in seed.get('organisations', []):
            need(isinstance(org, str) or (isinstance(org, dict) and isinstance(org.get('name'), str)), 'seed organisation')
        self.seed = seed
        # Names are mention hints only; never identity keys. Ambiguous aliases stay ambiguous.
        self.mentions = defaultdict(list)
        for name, ids in self.names.items():
            if len(ids) != 1 or len(name) < 3:
                continue
            pid = next(iter(ids))
            pattern = re.compile(r'(?<!\w)' + re.escape(name) + r'(?!\w)', re.I)
            for m in self.messages:
                if pattern.search(m['body']):
                    self.mentions[pid].append(m)

    def between(self, start, end):
        return self.messages[bisect_left(self.timestamps, start):bisect_right(self.timestamps, end)]

    def evidence(self, pid, max_messages=24, max_chars=9000):
        buckets = {k: [] for k in ('direct', 'authored', 'quotes', 'mentions', 'group')}
        for cid in self.person_chats[pid]:
            c = self.chat[cid]
            buckets['group' if c['is_group'] else 'direct'].extend(self.by_chat[cid])
        buckets['authored'] = self.by_person[pid]
        buckets['mentions'] = self.mentions[pid]
        for m in self.messages:
            q = self.message.get(m.get('quoted_id'))
            if q and (m['sender_id'] == pid or q['sender_id'] == pid):
                buckets['quotes'].extend([q, m])
        # Round-robin temporal samples ensure different evidence sources get a chance.
        queues = {}
        for kind, ms in buckets.items():
            ordered = sorted({m['id']: m for m in ms}.values(), key=lambda m: (m['ts'], m['id']))
            # Alternate earliest/latest to avoid a one-sided prefix of a long history.
            queues[kind] = [ordered[i//2] if i%2 == 0 else ordered[-1-i//2] for i in range(len(ordered))]
        selected, reasons, size = {}, defaultdict(set), 0
        while any(queues.values()) and len(selected) < max_messages:
            for kind, queue in queues.items():
                if not queue or len(selected) >= max_messages:
                    continue
                m = queue.pop(0)
                reasons[m['id']].add(kind)
                if m['id'] in selected:
                    continue
                n = len(json.dumps(m, ensure_ascii=False))
                if size+n <= max_chars:
                    selected[m['id']] = m
                    size += n
        return {'person_id': pid,
                'aliases': sorted(name for name, ids in self.names.items() if pid in ids),
                'ambiguous_aliases': sorted(name for name, ids in self.names.items() if pid in ids and len(ids)>1),
                'chat_ids': self.person_chats[pid],
                'messages': sorted(selected.values(), key=lambda m: (m['ts'], m['id'])),
                'evidence_sources': {mid: sorted(reasons[mid]) for mid in selected},
                'omitted_candidates': len({m['id'] for ms in buckets.values() for m in ms})-len(selected)}


def load_index(folder):
    folder = Path(folder)
    try:
        def lines(name):
            with (folder/name).open(encoding='utf-8-sig') as f:
                return [json.loads(line) for line in f if line.strip()]
        return InputIndex(lines('chats.jsonl'), lines('messages.jsonl'),
                          json.loads((folder/'seed.json').read_text(encoding='utf-8-sig')))
    except (OSError, json.JSONDecodeError):
        raise InputError('Input files could not be read or parsed') from None
