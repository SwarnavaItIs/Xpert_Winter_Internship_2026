import copy
import json
import os
import time
import unittest
from unittest.mock import patch

from inputs import InputIndex
from relationships import infer_relationships
from task_candidates import extract_candidates,bounded_chunks,relationship_context
from task_lifecycle import reconcile_tasks
from goals_help import apply_goals_help
from solution import runtime_budget
from llm import Client,ModelError
from tools.robustness import transform,substitute,assert_label_references
from tools.benchmark_robustness import price_usage,synthetic_probe


def fixture():
    chats=[dict(chat_id='chat-original',name='Alex group',is_group=False,participants=['ME','person-original'])]
    messages=[dict(id='source-original',chat_id='chat-original',sender_id='person-original',sender_name='Alex',
                   ts=1000,kind='text',body='Please send the invoice tomorrow.',quoted_id=None),
              dict(id='reply-original',chat_id='chat-original',sender_id='ME',sender_name='Me',
                   ts=2000,kind='text',body='On it.',quoted_id='source-original')]
    seed={'circles':{'person-original':'colleague'}}
    labels={'people':[{'id':'person-original','circle':'colleague','looks_after':['person-original']}],
            'chats':[{'chat_id':'chat-original','person':'person-original'}],
            'open_loops':[{'source_msgs':['source-original'],'closing_msgs':['reply-original'],'owner':'ME',
                           'chat_id':'chat-original','status':'open'}],'goals':[]}
    return chats,messages,seed,labels


class RobustnessTests(unittest.TestCase):
    def test_four_synthetic_language_probes(self):
        probes=synthetic_probe()
        self.assertEqual(len(probes),4)
        self.assertTrue(all(p['correct_source_and_owner']==1 for p in probes.values()))

    def test_group_fallback_ties_use_chronology_before_ids(self):
        chats=[dict(chat_id='group',name='Synthetic',is_group=True,participants=['ME','p-a','p-z'])]
        messages=[dict(id='early',chat_id='group',sender_id='p-z',sender_name='Earlier',ts=1,body='Hello',kind='text',quoted_id=None),
                  dict(id='late',chat_id='group',sender_id='p-a',sender_name='Later',ts=2,body='Hello',kind='text',quoted_id=None)]
        original,_=infer_relationships(InputIndex(chats,messages,{}))
        data,meta=transform(chats,messages,{},kind='ids')
        changed,_=infer_relationships(InputIndex(data['chats'],data['messages'],data['seed']))
        restored=substitute(changed,{v:k for k,v in meta['mapping'].items()})
        self.assertEqual(original['chats'],restored['chats'])
    def test_id_roundtrip_preserves_all_references_and_seed_keys(self):
        chats,messages,seed,labels=fixture()
        data,meta=transform(chats,messages,seed,labels,kind='ids')
        restored=substitute(data,{v:k for k,v in meta['mapping'].items()})
        self.assertEqual(restored,{'chats':chats,'messages':messages,'seed':seed,'labels':labels})
        index=InputIndex(data['chats'],data['messages'],data['seed'])
        assert_label_references(data['labels'],index)
        self.assertIn(data['messages'][1]['quoted_id'],index.message)
        self.assertNotIn('ME',meta['mapping'])

    def test_names_preserve_ambiguous_casefold_alias(self):
        chats,messages,seed,labels=fixture()
        messages.append({**messages[0],'id':'third','sender_name':'ALEX','body':'Alex and ALEX are the same displayed name.'})
        data,meta=transform(chats,messages,seed,labels,kind='names')
        self.assertEqual(data['messages'][0]['sender_name'].casefold(),data['messages'][2]['sender_name'].casefold())
        self.assertEqual(substitute(data,{v:k for k,v in meta['mapping'].items()}),
                         {'chats':chats,'messages':messages,'seed':seed,'labels':labels})

    def test_shuffle_is_order_invariant(self):
        chats,messages,seed,labels=fixture()
        data,_=transform(chats,messages,seed,labels,kind='shuffle')
        self.assertEqual(InputIndex(chats,messages,seed).messages,InputIndex(data['chats'],data['messages'],seed).messages)

    def test_long_boundary_adoption_keeps_anchor_and_no_new_label(self):
        chats,messages,seed,labels=fixture()
        data,_=transform(chats,messages,seed,labels,kind='long')
        self.assertEqual(data['labels'],labels)
        index=InputIndex(data['chats'],data['messages'],seed)
        self.assertEqual(max(index.timestamps),2000)
        rel,_=infer_relationships(index)
        tasks,_=extract_candidates(index,rel,core_messages=2)
        self.assertEqual(len(tasks),1)
        self.assertEqual(tasks[0]['source_msg'],'source-original')
        self.assertIn('reply-original',tasks[0]['owner_evidence'])
        self.assertEqual({v['id'] for c in bounded_chunks(index,core_messages=2) for v in c['core']},set(index.message))

    def test_lossless_request_compaction_and_usage(self):
        class Response:
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def read(self,limit=None):return json.dumps({'choices':[{'finish_reason':'stop','message':{'content':'{}'}}],
                'usage':{'prompt_tokens':4,'completion_tokens':2,'total_tokens':6}}).encode()
        class Opener:
            def open(self,request,**kwargs):
                self.body=json.loads(request.data)
                return Response()
        env={'LLM_BASE_URL':'https://example.invalid/v1','LLM_MODEL':'test','LLM_API_KEY':'synthetic'}
        with patch.dict(os.environ,env,clear=True):
            client=Client(time.monotonic()+60);client.opener=Opener()
            payload={'messages':[{'body':'नमस्ते বাংলা','ids':['a','b']}]}
            client.call('Synthetic',payload)
            self.assertEqual(json.loads(client.opener.body['messages'][1]['content']),payload)
            self.assertGreater(client.usage[0]['request_bytes'],0)

    def test_explicit_optout_skips_only_help_calls(self):
        class Fake:
            usage=[]
            stages=[]
            def call(self,instruction,payload,**kwargs):
                self.stages.append(payload['stage'])
                if payload['stage']!='goals':raise AssertionError('Redundant help request')
                return {'goals':[]}
        chats,messages,seed,_=fixture()
        messages=[{**messages[1],'body':'No reminders, I will handle it myself.','quoted_id':None}]
        index=InputIndex(chats,messages,seed);state,_=infer_relationships(index)
        state['open_loops']=[dict(title='Send invoice',type='commitment',owner='ME',chat_id='chat-original',
                                 source_msg='reply-original',status='open',closed_by=None,due=None)]
        out,diag=apply_goals_help(index,state,Fake())
        self.assertEqual(out['open_loops'][0]['help']['action'],'leave')
        self.assertEqual(diag['explicit_opt_out_calls_skipped'],1)

    def test_runtime_bounds_and_unknown_usage_cost(self):
        for value in ('59','1681','bad'):
            with patch.dict(os.environ,{'LLM_RUNTIME_SECONDS':value}),self.assertRaises(ModelError):runtime_budget()
        with patch.dict(os.environ,{'LLM_RUNTIME_SECONDS':'1200'}):self.assertEqual(runtime_budget(),1200)
        cost=price_usage([{'prompt_tokens':None,'completion_tokens':None}],'openai/gpt-oss-120b')
        self.assertIsNone(cost['verified_list_price_equivalent_usd'])
        self.assertIsNone(cost['actual_charge_usd'])


if __name__=='__main__':
    unittest.main()
