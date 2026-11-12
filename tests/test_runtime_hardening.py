import copy
from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from llm import Client,ModelError,strict_json
from solution import atomic_write,completion_report,infer,main,report_failure
from validation import validate,InvalidOutput
from tools.demo_corrections import fixture


ENV={'LLM_BASE_URL':'https://synthetic.invalid/v1','LLM_API_KEY':'synthetic-key','LLM_MODEL':'synthetic-model'}


def envelope(content='{}',finish='stop'):
    return json.dumps({'choices':[{'finish_reason':finish,'message':{'content':content}}],
                      'usage':{'prompt_tokens':3,'completion_tokens':2,'total_tokens':5}}).encode()


class Response:
    def __init__(self,body):self.body=body
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def read(self,limit):return self.body[:limit]


class Opener:
    def __init__(self,items):self.items=list(items);self.calls=0;self.timeouts=[]
    def open(self,request,timeout):
        self.calls+=1;self.timeouts.append(timeout)
        value=self.items.pop(0)
        if isinstance(value,Exception):raise value
        return Response(value)


class RuntimeTests(unittest.TestCase):
    def test_worker_reports_post_response_merge_failure_without_private_values(self):
        from inputs import InputIndex
        folder=Path('.private/tests/failure-location');folder.mkdir(parents=True,exist_ok=True)
        output=folder/'output.json';atomic_write(output,{'previous':'preserved'})
        chats=[dict(chat_id='room-'+str(i),name='Synthetic',is_group=False,
                    participants=['ME','person-'+str(i)]) for i in range(6)]
        messages=[dict(id='msg-'+str(i),chat_id='room-'+str(i),sender_id='ME',
                       body='Synthetic hello',ts=100+i,kind='text',quoted_id=None) for i in range(6)]
        index=InputIndex(chats,messages,{})
        class Fake:
            def __init__(self):self.usage=[]
            def call(self,*args,**kwargs):
                self.usage.append({'status':'success','finish_reason':'stop'})
                return {'people':[],'chats':[],'organisations':[],'merges':[]}
        client=Fake();stderr=io.StringIO()
        with patch('sys.argv',['solution.py','--worker','unused',str(output)]),\
             patch('solution.Client',return_value=client),patch('solution.load_index',return_value=index),\
             patch('relationships.reconcile',side_effect=KeyError('PRIVATE_KEY_CHAT_AND_MODEL_OUTPUT')),\
             redirect_stderr(stderr):
            self.assertEqual(main(),1)
        self.assertEqual(len(client.usage),2)  # Twelve small packets fit two bounded batches.
        status=json.loads(Path(str(output)+'.run-status.json').read_text())
        self.assertEqual(status['stage'],'relationships')
        self.assertEqual(status['error_type'],'KeyError')
        self.assertEqual(status['source_file'],'relationships.py')
        self.assertGreater(status['source_line'],0)
        self.assertNotIn('PRIVATE_KEY_CHAT_AND_MODEL_OUTPUT',stderr.getvalue()+json.dumps(status))
        self.assertEqual(json.loads(output.read_text()),{'previous':'preserved'})

    def test_failure_report_survives_disk_error_and_ignores_untrusted_stage(self):
        output=Path('.private/tests/failure-location/untrusted.json')
        atomic_write(str(output)+'.run-status.json',{'stage':'PRIVATE_STAGE'})
        stderr=io.StringIO()
        try:raise ValueError('PRIVATE_EXCEPTION')
        except ValueError as exc:
            with patch('solution.atomic_write',side_effect=OSError('PRIVATE_PATH')),redirect_stderr(stderr):
                report_failure(output,exc)
        self.assertIn('"stage":"unknown"',stderr.getvalue())
        self.assertIn('"error_type":"ValueError"',stderr.getvalue())
        self.assertNotIn('PRIVATE',stderr.getvalue())

    def test_startup_failure_does_not_reuse_previous_stage(self):
        output=Path('.private/tests/failure-location/startup.json')
        atomic_write(str(output)+'.run-status.json',{'stage':'goals_help'})
        with patch.dict(os.environ,{},clear=True),\
             patch('sys.argv',['solution.py','--worker','unused',str(output)]),redirect_stderr(io.StringIO()):
            self.assertEqual(main(),1)
        status=json.loads(Path(str(output)+'.run-status.json').read_text())
        self.assertEqual(status['stage'],'startup')
        self.assertEqual(status['error_type'],'ModelError')
        self.assertEqual(status['source_file'],'llm.py')

    def client(self,items,**settings):
        with patch.dict(os.environ,{**ENV,**settings},clear=True):c=Client(time.monotonic()+300)
        c.opener=Opener(items)
        return c

    @patch('llm.time.sleep')
    def test_rate_limit_and_transient_recovery(self,sleep):
        err=HTTPError('https://synthetic.invalid',429,'provider-secret',{'Retry-After':'2'},io.BytesIO(b'private'))
        c=self.client([err,HTTPError('https://synthetic.invalid',503,'private',{},None),envelope('{"ok":true}')])
        self.assertEqual(c.call('test',{}),{'ok':True})
        self.assertEqual(c.opener.calls,3);self.assertEqual(c.terminal_failures,0)
        self.assertEqual([u['status'] for u in c.usage],['http_error','http_error','success'])
        self.assertLessEqual(max(c.opener.timeouts),45)
        self.assertNotIn('provider-secret',json.dumps(c.usage))

    @patch('llm.time.sleep')
    def test_timeout_retries_bounded(self,sleep):
        c=self.client([TimeoutError(),TimeoutError(),TimeoutError()])
        with self.assertRaises(ModelError):c.call('test',{})
        self.assertEqual(c.opener.calls,3);self.assertEqual(c.terminal_failures,1)
        self.assertTrue(all(u['failure_kind']=='transport_timeout' for u in c.usage))

    @patch('llm.time.sleep')
    def test_malformed_json_and_duplicate_keys_retry(self,sleep):
        c=self.client([b'{partial',envelope('{"a":1,"a":2}'),envelope('{"ok":true}')])
        self.assertEqual(c.call('test',{}),{'ok':True})
        self.assertEqual(c.opener.calls,3)

    def test_nonfinite_and_incomplete_json_rejected(self):
        for text in ('{"n":NaN}','{"n":Infinity}','{"n":1e999}'):
            with self.assertRaises(ValueError):strict_json(text)
        for finish in ('length','content_filter',None):
            c=self.client([envelope('{"ok":true}',finish)])
            with self.assertRaises(ModelError):c.call('test',{})
            self.assertEqual(c.opener.calls,1)

    def test_oversized_response_and_auth_fail_fast(self):
        for response in (b'x'*2000001,HTTPError('https://synthetic.invalid',401,'secret',{},None)):
            c=self.client([response])
            with self.assertRaises(ModelError):c.call('test',{})
            self.assertEqual(c.opener.calls,1)

    def test_elapsed_response_deadline_and_expired_run_fail(self):
        c=self.client([envelope()]);c.deadline=300
        with patch('llm.time.monotonic',side_effect=[0,0,46,46]),self.assertRaises(ModelError):
            c.call('test',{})
        self.assertEqual(c.opener.calls,1)
        self.assertEqual(c.usage[0]['failure_kind'],'request_deadline')
        self.assertEqual(c.usage[0]['finish_reason'],'stop')
        self.assertEqual(c.usage[0]['completion_tokens'],2)
        c=self.client([]);c.deadline=time.monotonic()-1
        with self.assertRaises(ModelError):c.call('test',{})
        self.assertEqual(c.opener.calls,0)

    @patch('llm.time.sleep')
    def test_long_retry_after_does_not_retry_early(self,sleep):
        c=self.client([HTTPError('https://synthetic.invalid',429,'secret',{'Retry-After':'3600'},None)])
        with self.assertRaises(ModelError):c.call('test',{})
        self.assertEqual(c.opener.calls,1);sleep.assert_not_called()

    def test_run_request_and_reserved_unit_limits(self):
        c=self.client([envelope()],LLM_MAX_REQUESTS='1')
        c.call('test',{})
        with self.assertRaises(ModelError):c.call('test',{})
        self.assertEqual(c.opener.calls,1)
        c=self.client([],LLM_MAX_TOKEN_UNITS='1')
        with self.assertRaises(ModelError):c.call('test',{})
        self.assertEqual(c.opener.calls,0)

    @patch('llm.time.sleep')
    def test_circuit_blocks_repeated_terminal_failures(self,sleep):
        c=self.client([TimeoutError() for _ in range(9)])
        for _ in range(4):
            with self.assertRaises(ModelError):c.call('test',{})
        self.assertEqual(c.opener.calls,9)

    def test_atomic_failure_preserves_old_output_and_cleans_temp(self):
        folder=Path('.private/tests/atomic-hardening');folder.mkdir(parents=True,exist_ok=True)
        target=folder/'out.json';atomic_write(target,{'old':True});before=set(folder.iterdir())
        with self.assertRaises(ValueError):atomic_write(target,{'bad':float('nan')})
        self.assertEqual(json.loads(target.read_text()),{'old':True})
        self.assertEqual(set(folder.iterdir()),before)

    def test_completion_gate_rejects_deferred_and_failed_work(self):
        class Fake:usage=[];terminal_failures=0
        clean=completion_report(Fake(),{}, {'candidates':1},{'counts':{}},{})
        self.assertEqual(clean['status'],'complete')
        for life,help_ in [({'counts':{'model_unavailable':1}},{}),
                           ({'counts':{'model_budget_deferred':1}},{}),({}, {'malformed_responses':1})]:
            self.assertEqual(completion_report(Fake(),{}, {'candidates':1},life,help_)['status'],'degraded')

    def test_full_infer_does_not_replace_output_after_silent_fallback(self):
        from inputs import InputIndex
        index,base=fixture();folder=Path('.private/tests/degraded-infer');folder.mkdir(parents=True,exist_ok=True)
        output=folder/'output.json';atomic_write(output,{'previous':'preserved'})
        class Fake:usage=[];terminal_failures=0
        with patch('solution.load_index',return_value=index),\
             patch('solution.infer_relationships',return_value=(base,{})),\
             patch('solution.extract_candidates',return_value=([],{'candidates':0})),\
             patch('solution.reconcile_tasks',return_value=(base['open_loops'],{'counts':{'model_unavailable':1}})),\
             patch('solution.apply_goals_help',return_value=(base,{})),self.assertRaises(ModelError):
            infer('unused',output,Fake())
        self.assertEqual(json.loads(output.read_text()),{'previous':'preserved'})
        self.assertEqual(json.loads(Path(str(output)+'.completion.json').read_text())['status'],'degraded')

    def test_strict_reference_action_and_date_validation(self):
        index,base=fixture()
        for field,value in [('owner','missing'),('chat_id','child-room'),('due','2026-02-30'),('due','2099-01-01')]:
            out=copy.deepcopy(base);out['open_loops'][1][field]=value
            with self.assertRaises(InvalidOutput):validate(out,index.chats,index.messages)
        for help_ in ({'action':'assist','next_step':' ','needs_yes':False},
                      {'action':'remind','remind_at':'2026-01-01T09:00:00+05:30'},
                      {'action':'remind','remind_at':'2026-11-03T09:00:00'},
                      {'action':'leave','reason':'Clear reason','needs_yes':False}):
            out=copy.deepcopy(base);out['open_loops'][1]['help']=help_
            with self.assertRaises(InvalidOutput):validate(out,index.chats,index.messages)


if __name__=='__main__':unittest.main()
