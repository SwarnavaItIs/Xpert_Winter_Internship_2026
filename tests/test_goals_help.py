import copy
import io
import json
import os
import time
from datetime import datetime
import unittest
from unittest.mock import Mock,patch
from urllib.error import HTTPError

from goals_help import apply_goals_help,call,decide,reminder_time,TZ
from inputs import InputIndex
from llm import Client
from solution import completion_report
from tools.evaluate_goals_help import metrics,project_labels
from validation import validate


def fixture(body='I will draft a proposal.',owner='ME',title='Draft proposal',due=None):
    chats=[{'chat_id':'c','name':'Synthetic','is_group':False,'participants':['ME','staff-x']}]
    messages=[{'id':'m','chat_id':'c','sender_id':'ME','body':body,'ts':1788588000,'kind':'text','quoted_id':None}]
    index=InputIndex(chats,messages,{'circles':{'staff-x':'staff'}})
    state={'people':[{'id':'staff-x','circle':'staff'}],'organisations':[],
        'chats':[{'chat_id':'c','person':'staff-x'}],'goals':[],
        'open_loops':[{'title':title,'type':'commitment','owner':owner,'chat_id':'c','source_msg':'m',
                       'status':'open','closed_by':None,'due':due,'help':{'action':'leave','reason':'pending'}}]}
    return index,state


class GoalHelpTests(unittest.TestCase):
    def test_malformed_verdict_repair_clears_only_recovered_failure(self):
        class Fake:
            usage=[]
            def __init__(self,results):self.results=iter(results);self.calls=[]
            def call(self,instruction,payload,**kwargs):
                self.calls.append((instruction,payload))
                return next(self.results)
        diag={'model_failures':0}
        model=Fake([{'supported':'false'},{'supported':False}])
        result=call(model,'Verify',{'stage':'verify_help'},diag)
        self.assertEqual(result,{'supported':False})
        self.assertEqual(diag.get('malformed_responses',0),0)
        self.assertEqual(diag['recovered_malformed_responses'],1)
        self.assertEqual(diag['coverage']['verify_help']['completed'],1)
        self.assertEqual(diag['coverage']['verify_help']['malformed_attempts'],1)
        self.assertEqual(completion_report(model,{}, {'candidates':0},{},diag)['status'],'complete')
        # A later success must not erase a different, unrecovered logical request.
        broken=Fake([{},{}]);call(broken,'Verify',{'stage':'verify_help'},diag)
        self.assertEqual(diag['malformed_responses'],1)
        self.assertEqual(completion_report(model,{}, {'candidates':0},{},diag)['status'],'degraded')

    def test_valid_empty_goals_and_negative_verdict_are_completed_work(self):
        class Fake:
            usage=[]
            def call(self,instruction,payload,**kwargs):
                return {'goals':[]} if payload['stage']=='goals' else {'supported':False}
        model=Fake();diag={'model_failures':0}
        self.assertEqual(call(model,'Goals',{'stage':'goals'},diag),{'goals':[]})
        self.assertEqual(call(model,'Verify',{'stage':'verify_help'},diag),{'supported':False})
        self.assertEqual(sum(v['completed'] for v in diag['coverage'].values()),2)
        self.assertEqual(completion_report(model,{}, {'candidates':0},{},diag)['status'],'complete')

    def test_persistent_string_verdict_still_blocks_publication(self):
        class Fake:
            usage=[]
            def call(self,instruction,payload,**kwargs):return {'supported':'true'}
        diag={'model_failures':0};model=Fake()
        self.assertEqual(call(model,'Verify',{'stage':'verify_help'},diag),{})
        self.assertEqual(diag['coverage']['verify_help']['attempts'],2)
        self.assertEqual(diag['coverage']['verify_help']['completed'],0)
        self.assertIn('goals_help_malformed_responses',completion_report(model,{}, {'candidates':0},{},diag)['reasons'])

    @patch('llm.time.sleep')
    def test_recovered_429_is_one_completed_help_request(self,sleep):
        env={'LLM_BASE_URL':'https://synthetic.invalid/v1','LLM_API_KEY':'synthetic','LLM_MODEL':'synthetic'}
        with patch.dict(os.environ,env,clear=True):model=Client(time.monotonic()+90)
        model.opener=Mock()
        body={'choices':[{'finish_reason':'stop','message':{'content':'{"supported":true}'}}]}
        model.opener.open.side_effect=[HTTPError(env['LLM_BASE_URL'],429,'PRIVATE',{},None),
                                      io.BytesIO(json.dumps(body).encode())]
        diag={'model_failures':0}
        self.assertEqual(call(model,'Verify',{'stage':'verify_help'},diag),{'supported':True})
        self.assertEqual([u['attempt'] for u in model.usage],[1,2])
        self.assertEqual([u['request_id'] for u in model.usage],[1,1])
        self.assertEqual(model.terminal_failures,0)
        self.assertEqual(diag['coverage']['verify_help']['expected'],1)
        self.assertEqual(diag['coverage']['verify_help']['completed'],1)
        self.assertEqual(completion_report(model,{}, {'candidates':0},{},diag)['status'],'complete')

    def test_draft_vs_send_approval(self):
        index,state=fixture()
        out,_=apply_goals_help(index,state)
        self.assertEqual(out['open_loops'][0]['help']['action'],'assist')
        self.assertFalse(out['open_loops'][0]['help']['needs_yes'])
        facts={'requested_help':True,'step':'send'}
        help_=decide(state['open_loops'][0],facts,datetime(2026,9,5,tzinfo=TZ))
        self.assertTrue(help_['needs_yes'])
        self.assertTrue(help_['next_step'].startswith('Send'))
        validate(out,index.chats,index.messages)

    def test_staff_owned_and_explicit_preparation_request(self):
        index,state=fixture(owner='staff-x')
        out,_=apply_goals_help(index,state)
        self.assertEqual(out['open_loops'][0]['help']['action'],'leave')
        index,state=fixture('Please help me draft the proposal for the staff member.',owner='staff-x')
        out,_=apply_goals_help(index,state)
        self.assertEqual(out['open_loops'][0]['help']['action'],'assist')
        self.assertFalse(out['open_loops'][0]['help']['needs_yes'])

    def test_sensitive_help_requires_specific_request(self):
        for body,expected in [('I will draft my medical questions.','leave'),
                              ('Please help me draft my medical questions.','assist')]:
            index,state=fixture(body,title='Draft medical questions')
            out,_=apply_goals_help(index,state)
            self.assertEqual(out['open_loops'][0]['help']['action'],expected)

    def test_attendance_reminder_not_delegated_attendance(self):
        index,state=fixture('I will attend the concert.',title='Attend concert',due='2026-09-08')
        out,_=apply_goals_help(index,state)
        self.assertEqual(out['open_loops'][0]['help']['action'],'remind')
        self.assertGreater(datetime.fromisoformat(out['open_loops'][0]['help']['remind_at']).timestamp(),index.timestamps[-1])

    def test_overdue_undated_and_near_due(self):
        endpoint=datetime(2026,9,5,15,tzinfo=TZ)
        for due,expected in [(None,'2026-09-06T09:00:00+05:30'),('2026-09-01','2026-09-06T09:00:00+05:30'),
                             ('2026-09-06','2026-09-06T09:00:00+05:30'),('2026-09-10','2026-09-09T09:00:00+05:30')]:
            self.assertEqual(reminder_time({'due':due},endpoint)[0],expected)

    def test_opt_out_and_redundancy_override(self):
        _,state=fixture()
        for field in ('opt_out','redundant'):
            result=decide(state['open_loops'][0],{field:True,'requested_help':True,'useful_preparation':True},datetime(2026,9,5,tzinfo=TZ))
            self.assertEqual(result['action'],'leave')
            self.assertTrue(result['reason'])

    def test_goal_types_horizon_and_no_padding(self):
        for body,typ,area in [('My goal is to exercise daily for one year.','habit','health'),
                              ('My goal is to save for a house within 3 years.','milestone','finances'),
                              ('My goal is to travel to Iceland within 3 months.','experience','experiences')]:
            index,state=fixture(body)
            out,_=apply_goals_help(index,state)
            self.assertEqual(len(out['goals']),1)
            self.assertEqual(out['goals'][0]['type'],typ)
            self.assertEqual(out['goals'][0]['area'],area)
            self.assertEqual(out['goals'][0]['evidence'],['m'])
            validate(out,index.chats,index.messages)

    def test_oneoff_passing_interest_and_other_person_not_goals(self):
        for body in ['Maybe I should travel someday.','My goal is to send a report in 3 months.','Nice marathon photo!']:
            index,state=fixture(body)
            self.assertEqual(apply_goals_help(index,state)[0]['goals'],[])
        index,state=fixture('My goal is to exercise daily for one year.')
        index.message['m']['sender_id']='staff-x'
        self.assertEqual(apply_goals_help(index,state)[0]['goals'],[])

    def test_invented_goal_rejected_by_verifier_or_invalid_reference(self):
        class Fake:
            usage=[]
            def call(self,instruction,payload,**kwargs):
                if payload['stage']=='goals':
                    return {'goals':[{'title':'Run marathon','area':'health','type':'milestone','horizon':'1yr',
                        'evidence':['m'],'durable_reason':'claimed','horizon_reason':'claimed'}]}
                return {'supported':False}
        index,state=fixture('Send a report tomorrow.')
        out,diag=apply_goals_help(index,state,Fake())
        self.assertEqual(out['goals'],[])
        self.assertEqual(diag['rejected_goals'],1)

    def test_equivalent_goals_merge_evidence(self):
        index,state=fixture('My goal is to exercise daily for one year.')
        second={**index.message['m'],'id':'m2','ts':index.timestamps[-1]+1}
        index=InputIndex(index.chats,[*index.messages,second],index.seed)
        out,_=apply_goals_help(index,state)
        self.assertEqual(len(out['goals']),1)
        self.assertEqual(set(out['goals'][0]['evidence']),{'m','m2'})

    def test_intrusion_metric_and_validation_exclusion(self):
        index,state=fixture()
        out,_=apply_goals_help(index,state)
        gold=project_labels({'goals':[],'open_loops':[{'source_msgs':['m'],'status':'open','help':'leave'},
                  {'source_msgs':['validation'],'status':'open','help':'assist'}]},set(index.message))
        score=metrics(out,gold,index.seed,index.messages)
        self.assertEqual(score['intrusive_interventions'],1)
        self.assertEqual(score['help_labels'],1)
        self.assertEqual(score['official_help'],0)

    def test_closed_tasks_have_no_help(self):
        index,state=fixture()
        state['open_loops'][0].update(status='done',closed_by='m')
        self.assertIsNone(apply_goals_help(index,state)[0]['open_loops'][0]['help'])

    def test_later_quoted_opt_out(self):
        index,state=fixture()
        reply={**index.message['m'],'id':'later','quoted_id':'m','ts':index.timestamps[-1]+10,
               'body':'No reminders, I will handle it myself.'}
        index=InputIndex(index.chats,[*index.messages,reply],index.seed)
        out,diag=apply_goals_help(index,state)
        self.assertEqual(out['open_loops'][0]['help']['action'],'leave')
        self.assertIn('later',diag['tasks'][0]['facts']['evidence'])

    def test_goal_limit(self):
        index,state=fixture()
        messages=[{**index.message['m'],'id':str(i),'ts':index.timestamps[-1]+i,
                   'body':f'My goal is to visit destination {i} within 3 months.'} for i in range(13)]
        index=InputIndex(index.chats,messages,index.seed)
        state['open_loops']=[]
        out,_=apply_goals_help(index,state)
        self.assertEqual(len(out['goals']),10)
        validate(out,index.chats,index.messages)

    def test_verified_model_goal_and_exact_external_step(self):
        class Fake:
            usage=[]
            def call(self,instruction,payload,**kwargs):
                if payload['stage']=='goals':
                    return {'goals':[dict(title='Exercise daily',area='health',type='habit',horizon='1yr',
                        evidence=['m'],durable_reason='Explicit goal',horizon_reason='one year')]}
                if payload['stage']=='help_facts':
                    return {'facts':dict(evidence=['m'],requested_help=True,sensitive=False,
                        personal_attendance=False,delegated=False,redundant=False,opt_out=False,
                        useful_preparation=False,step='send',reason='Explicit request to send')}
                return {'supported':True,'equivalent_index':None}
        index,state=fixture('My goal is to exercise daily for one year. Assistant, please send my plan.',title='Send exercise plan')
        out,_=apply_goals_help(index,state,Fake())
        self.assertEqual(len(out['goals']),1)
        self.assertTrue(out['open_loops'][0]['help']['needs_yes'])
        validate(out,index.chats,index.messages)

    def test_missing_goal_evidence_and_malformed_facts_rejected(self):
        class Fake:
            usage=[]
            def call(self,instruction,payload,**kwargs):
                if payload['stage']=='goals':
                    return {'goals':[dict(title='Invented',area='health',type='habit',horizon='1yr',
                        evidence=['nonexistent'],durable_reason='Invented',horizon_reason='Invented')]}
                return {'facts':{'requested_help':'true','step':'purchase','evidence':['m']},'supported':True}
        index,state=fixture('I will attend the concert.',title='Attend concert')
        out,_=apply_goals_help(index,state,Fake())
        self.assertEqual(out['goals'],[])
        self.assertEqual(out['open_loops'][0]['help']['action'],'remind')


if __name__=='__main__':
    unittest.main()
