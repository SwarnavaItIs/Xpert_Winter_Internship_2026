import copy
import json
import os
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from llm import Client, ModelError, NoRedirect
from solution import read_input, infer, batches
from validation import validate, InvalidOutput


CHATS = [{'chat_id': 'room-z', 'is_group': False, 'name': 'Synthetic', 'participants': ['ME', 'person-z']}]
MESSAGES = [
    {'id': 'request-z', 'chat_id': 'room-z', 'sender_id': 'ME', 'ts': 1000, 'body': 'I will send the draft.', 'quoted_id': None},
    {'id': 'reply-z', 'chat_id': 'room-z', 'sender_id': 'person-z', 'ts': 2000, 'body': 'Okay', 'quoted_id': 'request-z'}]
for message in MESSAGES:
    message['kind'] = 'text'
OUT = {'people': [{'id': 'person-z', 'circle': 'partner', 'links': [], 'looks_after': []}], 'organisations': [],
       'chats': [{'chat_id': 'room-z', 'person': 'person-z'}], 'goals': [], 'open_loops': [
           {'title': 'Send draft', 'type': 'commitment', 'owner': 'ME', 'chat_id': 'room-z',
            'source_msg': 'request-z', 'status': 'open', 'closed_by': None,
            'help': {'action': 'assist', 'next_step': 'Draft privately', 'needs_yes': False}}]}


class PipelineTests(unittest.TestCase):
    def test_valid_output(self):
        validate(OUT, CHATS, MESSAGES)

    def test_rejects_extra_fields_and_bad_closure(self):
        for mutation in ('extra', 'closure', 'source', 'help'):
            out = copy.deepcopy(OUT)
            loop = out['open_loops'][0]
            if mutation == 'extra':
                loop['confidence'] = 1
            elif mutation == 'closure':
                loop.update(status='done', help=None)
            elif mutation == 'source':
                loop['chat_id'] = 'wrong'
            else:
                loop['help'] = {'action': 'assist'}
            with self.assertRaises(InvalidOutput):
                validate(out, CHATS, MESSAGES)

    def test_client_missing_configuration(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ModelError):
            Client(time.monotonic() + 10)

    def test_redirect_rejected(self):
        with self.assertRaises(ModelError):
            NoRedirect().redirect_request(None, None, 307, '', {}, 'https://other.invalid')

    def test_oversize_message_is_not_silently_truncated(self):
        with self.assertRaises(ValueError):
            list(batches([{'body': 'x' * 100}], limit=20))

    def test_mock_pipeline_never_reads_answers_and_sorts_input(self):
        class FakeClient:
            usage = []
            def call(self, instruction, payload, **kwargs):
                if payload.get('stage')=='goals':return {'goals':[]}
                if payload.get('stage')=='help_facts':
                    return {'facts':dict(evidence=['request-z'],requested_help=False,sensitive=False,
                        personal_attendance=False,delegated=False,redundant=False,opt_out=False,
                        useful_preparation=True,step='prepare',reason='Private draft is useful')}
                if payload.get('stage')=='verify_help':return {'supported':True}
                if payload.get('lifecycle_stage')=='propose':return {'events':[]}
                if payload.get('task_stage')=='candidates':
                    return {'goals':[],'candidates':[{'source_msg':'request-z','action_text':'I will send the draft.',
                        'owner':'ME','type':'commitment','title':'Send draft',
                        'owner_evidence':['request-z'],'involvement_evidence':['request-z'],
                        'due_text':'','uncertain':False}]}
                if 'proposal' in payload or ('state' in payload and 'chats' in payload):
                    return {**copy.deepcopy(OUT), 'open_loops': []}
                return copy.deepcopy(OUT)
        Path('.private/tests').mkdir(parents=True, exist_ok=True)
        root = Path('.private/tests/mock-pipeline')
        root.mkdir(parents=True, exist_ok=True)
        if root.is_dir():
            for name, values in [('chats', CHATS), ('messages', list(reversed(MESSAGES)))]:
                (root / (name + '.jsonl')).write_text('\n'.join(json.dumps(v) for v in values))
            (root / 'seed.json').write_text(json.dumps({'circles': {'person-z': 'partner'}}))
            (root / 'answers').mkdir(exist_ok=True)
            (root / 'answers/answers.json').write_text('INVALID JSON - MUST NOT BE READ')
            original = Path.open
            def guarded(path, *args, **kwargs):
                if 'answers' in path.parts:
                    raise AssertionError('Inference read answers')
                return original(path, *args, **kwargs)
            with patch.object(Path, 'open', guarded):
                self.assertEqual(read_input(root)[1][0]['id'], 'request-z')
                infer(root, root / 'out.json', FakeClient())
                actual=json.loads((root / 'out.json').read_text())
                self.assertEqual(actual['people'],OUT['people'])
                self.assertEqual(actual['chats'],OUT['chats'])
                self.assertEqual(actual['open_loops'][0]['source_msg'],'request-z')
                self.assertEqual(actual['open_loops'][0]['status'],'open')
                self.assertIsNone(actual['open_loops'][0]['closed_by'])
                self.assertEqual(actual['open_loops'][0]['help']['action'],'assist')


if __name__ == '__main__':
    unittest.main()
