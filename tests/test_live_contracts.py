"""End-to-end regression for the failed response contracts, with mocked HTTP."""
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

import check
from inputs import load_index
from llm import Client, ModelError
from solution import infer
from validation import validate


class ContractOpener:
    def __init__(self, invalid_verdicts=False):
        self.invalid_verdicts=invalid_verdicts
        self.repairs=0
        self.verdicts=0
        self.rate_limited=False
        self.schemas=[]

    def open(self, request, timeout):
        sent=json.loads(request.data)
        payload=json.loads(sent['messages'][1]['content'])
        stage=payload.get('stage')
        if stage=='verify_help':
            self.schemas.append(sent['response_format'])
            self.verdicts+=1
            if self.invalid_verdicts or self.verdicts==1:
                value='true'
            elif not self.rate_limited:
                self.rate_limited=True
                raise HTTPError(request.full_url,429,'private provider message',{},None)
            else:
                value=True
            content=({'items':[{'item_id':x['item_id'],'supported':value} for x in payload['items']]}
                     if 'items' in payload else {'supported':value})
        elif stage=='goals':
            content={'goals':[]}
        elif stage=='help_facts':
            def facts(item):return dict(evidence=[item['task']['source_msg']],requested_help=False,
                sensitive=False,personal_attendance=False,delegated=False,redundant=False,
                opt_out=False,useful_preparation=True,step='prepare',reason='Private preparation')
            content=({'items':[{'item_id':x['item_id'],'facts':facts(x)} for x in payload['items']]}
                     if 'items' in payload else {'facts':facts(payload)})
        elif payload.get('task_stage')=='candidates':
            content={'goals':[],'candidates':[
                self.candidate('first','I will send the draft.',999),
                self.candidate('second','A paraphrase that does not occur in the source.')]}
        elif payload.get('task_stage')=='repair_candidates':
            self.repairs+=1
            content={'candidates':[]} if self.repairs==1 else {
                'candidates':[self.candidate('second','I will book the hall.')]}
        elif payload.get('lifecycle_stage')=='propose':
            content={'events':[],'merges':[]}
        else:
            content={'people':[],'organisations':[],'chats':[],'merges':[]}
        return io.BytesIO(json.dumps({
            'choices':[{'finish_reason':'stop','message':{'content':json.dumps(content)}}],
            'usage':{'prompt_tokens':10,'completion_tokens':5,'total_tokens':15}
        }).encode())

    @staticmethod
    def candidate(mid, action, offset=None):
        return dict(source_msg=mid,action_text=action,action_start=offset,
            owner='ME',type='commitment',title='Complete '+mid+' action',
            owner_evidence=[mid],involvement_evidence=[mid])


class LiveContractTests(unittest.TestCase):
    def run_fixture(self, folder, invalid_verdicts=False):
        rows=[dict(id=mid,chat_id='room',sender_id='ME',sender_name='User',kind='text',
                   quoted_id=None,ts=1793500000+i,body=body) for i,(mid,body) in enumerate([
                       ('first','I will send the draft.'),('second','I will book the hall.')])]
        chats=[dict(chat_id='room',name='Synthetic',is_group=False,participants=['ME','peer'])]
        for name,values in [('chats',chats),('messages',rows)]:
            (folder/(name+'.jsonl')).write_text('\n'.join(json.dumps(x) for x in values),encoding='utf-8')
        (folder/'seed.json').write_text(json.dumps({'circles':{'peer':'friend'}}),encoding='utf-8')
        (folder/'answers').mkdir()
        (folder/'answers/answers.json').write_text('MUST NEVER BE READ BY INFERENCE',encoding='utf-8')
        model=Client(time.monotonic()+120)
        model.opener=ContractOpener(invalid_verdicts)
        return model,folder/'output.json'

    @patch('llm.time.sleep')
    def test_repaired_contracts_and_recovered_429_pass_full_completion_gate(self, sleep):
        env={'LLM_BASE_URL':'https://synthetic.invalid/v1','LLM_MODEL':'synthetic',
             'LLM_API_KEY':'synthetic-only','LLM_JSON_SCHEMA_VERDICTS':'1'}
        with tempfile.TemporaryDirectory() as temporary,patch.dict(os.environ,env,clear=True):
            folder=Path(temporary);model,output=self.run_fixture(folder)
            original_open=Path.open
            def guarded(path,*args,**kwargs):
                if 'answers' in path.parts:
                    raise AssertionError('Inference accessed labels')
                return original_open(path,*args,**kwargs)
            with patch.object(Path,'open',guarded):
                infer(folder,output,model)
            result=json.loads(output.read_text())
            index=load_index(folder)
            validate(result,index.chats,index.messages)
            self.assertEqual(check.validate(result,index.chats,index.messages),[])
            self.assertEqual(len(result['open_loops']),2)
            report=json.loads(Path(str(output)+'.completion.json').read_text())
            self.assertEqual(report['status'],'complete')
            self.assertEqual(report['reasons'],[])
            self.assertEqual(report['coverage']['task_extraction']['repair_contract_retries'],1)
            self.assertEqual(report['coverage']['goals_help']['verify_help']['completed'],1)
            self.assertEqual(report['coverage']['goals_help']['verify_help']['items_completed'],2)
            self.assertTrue(all(x['type']=='json_schema' for x in model.opener.schemas))
            self.assertEqual(model.terminal_failures,0)
            self.assertTrue(any(u.get('http_status')==429 for u in model.usage))

    @patch('llm.time.sleep')
    def test_unrepaired_verdict_preserves_old_output_and_failure_coverage(self,sleep):
        env={'LLM_BASE_URL':'https://synthetic.invalid/v1','LLM_MODEL':'synthetic','LLM_API_KEY':'synthetic-only'}
        with tempfile.TemporaryDirectory() as temporary,patch.dict(os.environ,env,clear=True):
            folder=Path(temporary);model,output=self.run_fixture(folder,invalid_verdicts=True)
            output.write_text('{"old":"preserved"}',encoding='utf-8')
            with self.assertRaisesRegex(ModelError,'Incomplete model coverage'):
                infer(folder,output,model)
            self.assertEqual(json.loads(output.read_text()),{'old':'preserved'})
            report=json.loads(Path(str(output)+'.completion.json').read_text())
            self.assertEqual(report['status'],'degraded')
            self.assertEqual(report['coverage']['goals_help']['verify_help']['completed'],0)


if __name__=='__main__':
    unittest.main()
