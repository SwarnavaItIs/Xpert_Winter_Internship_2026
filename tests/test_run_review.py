"""Regressions prompted by the user's live synthetic run, with unrelated identities."""
import copy
from datetime import datetime, timezone
import unittest

from goals_help import GOALS, apply_goals_help, calendar_horizon, call
from inputs import InputIndex
from llm import verdict_response_format
from relationships import infer_relationships, reconcile
from solution import completion_report
from task_candidates import bounded_chunks, extract_candidates, normalize_candidate, relationship_context
from task_lifecycle import reconcile_tasks
from test_lifecycle import candidate, message, index_for


TS=int(datetime(2026,10,18,tzinfo=timezone.utc).timestamp())


def group_fixture(side='home'):
    chat={'chat_id':'shared','name':'Uninformative title','is_group':True,
          'participants':['ME','actor','helper','dependent']}
    rows=[{**message('request','I will arrange the repair.',TS,sender='actor',cid='shared'),
           'sender_name':'Dev'}]
    index=InputIndex([chat],rows,{})
    state={'people':[{'id':pid,'circle':circle,'links':links} for pid,circle,links in [
        ('actor','spouse' if side=='home' else 'colleague',[] if side=='home' else [{'organisation':'org','link':'runs'}]),
        ('helper','staff',[{'organisation':'org','link':'supports'}]),('dependent','child',[])]],
        'organisations':[{'id':'org','name':'An organisation','type':'residence' if side=='home' else 'company',
                          'side':side,'role':'resident' if side=='home' else 'owner'}],
        'chats':[{'chat_id':'shared','organisation':'org'}]}
    diag={'people':{p['id']:{'uncertain':False} for p in state['people']}}
    chunk=next(bounded_chunks(index))
    raw=dict(source_msg='request',action_text=rows[0]['body'],owner='actor',type='commitment',
             title='Arrange the repair',owner_evidence=['request'],involvement_evidence=['request'])
    return index,state,diag,chunk,raw


class GroupResponsibilityTests(unittest.TestCase):
    def test_silent_user_household_and_work_obligations_keep_the_actual_owner(self):
        for side in ('home','work'):
            index,state,diag,chunk,raw=group_fixture(side)
            context=relationship_context(index,state,chunk,diag)
            task=normalize_candidate(raw,index,chunk,context)
            self.assertEqual(task['owner'],'actor')
            self.assertEqual(task['involvement_basis'],side+'_organisation_obligation' if side=='work' else 'household_obligation')

    def test_group_membership_or_uncertain_circle_alone_is_insufficient(self):
        index,state,diag,chunk,raw=group_fixture('work')
        for change in ('missing_link','uncertain','friend'):
            changed=copy.deepcopy(state);quality=copy.deepcopy(diag)
            if change=='missing_link':changed['people'][0]['links']=[]
            if change=='uncertain':quality['people']['actor']['uncertain']=True
            if change=='friend':changed['people'][0]['circle']='friend'
            with self.assertRaisesRegex(ValueError,'unsupported_user_involvement'):
                normalize_candidate(raw,index,chunk,relationship_context(index,changed,chunk,quality))

    def test_unique_observed_user_address_is_available_without_a_user_reply(self):
        chat={'chat_id':'g','name':'Group','is_group':True,'participants':['ME','other']}
        rows=[{**message('self','Hello',TS,sender='ME',cid='g'),'sender_name':'Nila'},
              {**message('ask','Nila, please approve the layout.',TS+1,sender='other',cid='g'),'sender_name':'Amar'}]
        raw=dict(source_msg='ask',action_text=rows[1]['body'],owner='ME',type='question',
                 title='Approve layout',owner_evidence=['ask'],involvement_evidence=['ask'])
        index=InputIndex([chat],rows,{})
        self.assertEqual(normalize_candidate(raw,index,next(bounded_chunks(index)))['involvement_basis'],'unique_user_address')
        rows[1]['sender_name']='Nila'  # The alias is now shared, so it cannot identify ME.
        index=InputIndex([chat],rows,{})
        with self.assertRaisesRegex(ValueError,'unsupported_user_involvement'):
            normalize_candidate(raw,index,next(bounded_chunks(index)))


class OrganisationReviewTests(unittest.TestCase):
    def test_empty_later_person_sample_does_not_erase_a_supported_affiliation(self):
        index,_,_,_,_=group_fixture()
        org=dict(id='res',name='Residence',type='residence',side='home',role='resident',
                 _evidence=['request'],_confidence=.9)
        proposals=[{'organisations':[org],'people':[
            {'id':'helper','circle':'staff','links':[{'organisation':'res','link':'supports'}],
             '_evidence':['request'],'_confidence':.8},
            {'id':'helper','circle':'staff','links':[],'_evidence':['request'],'_confidence':.99}]}]
        result,_=reconcile(index,proposals)
        self.assertEqual(next(p for p in result['people'] if p['id']=='helper')['links'],[{'organisation':'res','link':'supports'}])

    def test_group_affiliation_uses_corroborating_links_and_not_speaker_frequency(self):
        index,_,_,_,_=group_fixture()
        org=dict(id='res',name='Residence',type='residence',side='home',role='resident',
                 _evidence=['request'],_confidence=.9)
        people=[dict(id=p,circle='staff',links=[{'organisation':'res','link':'supports'}],
                     _evidence=['request'],_confidence=.9) for p in ('actor','helper')]
        proposals=[{'organisations':[org],'people':people,'chats':[
            {'chat_id':'shared','person':'actor','_evidence':['request'],'_confidence':.99},
            {'chat_id':'shared','organisation':'res','_evidence':['request'],'_confidence':.8}]}]
        result,_=reconcile(index,proposals)
        self.assertEqual(result['chats'],[{'chat_id':'shared','organisation':'res'}])

    def test_unattached_outside_company_is_excluded_but_an_owned_entity_survives(self):
        index=index_for([message('s','The supplier emailed a quotation.',TS)])
        class Fake:
            def call(self,prompt,payload,**kwargs):
                common=dict(type='company',side='work',_evidence=['s'],_confidence=.9)
                return {'people':[],'chats':[],'merges':[],
                        'organisations':[dict(common,id='vendor',name='Outside supplier',role='other'),
                                         dict(common,id='own',name='Owned venture',role='owner')]}
        out,diag=infer_relationships(index,Fake())
        self.assertEqual([o['name'] for o in out['organisations']],['Owned venture'])
        self.assertEqual(diag['excluded_unaffiliated_organisations'],1)


class BatchedHelpTests(unittest.TestCase):
    def fixture(self,count=6):
        rows=[message('task'+str(i),'I will prepare document '+str(i)+'.',TS+i) for i in range(count)]
        index=index_for(rows)
        state={'people':[{'id':'p','circle':'friend'}],'organisations':[],
               'chats':[{'chat_id':'d','person':'p'}],'goals':[],
               'open_loops':[dict(source_msg=m['id'],chat_id='d',title='Prepare document '+str(i),
                   owner='ME',type='commitment',due=None,status='open',closed_by=None,help=None)
                   for i,m in enumerate(rows)]}
        return index,state

    def test_all_six_tasks_get_independent_facts_and_verdicts_in_two_calls(self):
        class Fake:
            usage=[]
            def __init__(self):self.calls=[]
            def call(self,prompt,payload,**kwargs):
                self.calls.append(payload)
                values=[]
                for item in payload['items']:
                    if payload['stage']=='help_facts':
                        facts=dict(evidence=[item['task']['source_msg']],requested_help=False,
                            sensitive=False,personal_attendance=False,delegated=False,redundant=False,
                            opt_out=False,useful_preparation=True,step='prepare',reason='Private draft')
                        values.append(dict(item_id=item['item_id'],facts=facts))
                    else:values.append(dict(item_id=item['item_id'],supported=int(item['item_id'])%2==0))
                return {'items':list(reversed(values))}  # Order is deliberately different.
        index,state=self.fixture();model=Fake()
        out,diag=apply_goals_help(index,state,model,include_goals=False)
        self.assertEqual(len(model.calls),2)
        self.assertEqual([t['help']['action'] for t in out['open_loops']],['assist','remind']*3)
        self.assertEqual(diag['coverage']['verify_help']['items_completed'],6)
        self.assertEqual(diag['coverage']['help_facts']['items_completed'],6)

    def test_missing_duplicate_or_foreign_batch_ids_cannot_hide_an_unreviewed_task(self):
        for ids in (['one'],['one','one'],['one','foreign']):
            class Fake:
                usage=[]
                def call(self,prompt,payload,**kwargs):
                    return {'items':[dict(item_id=i,supported=True) for i in ids]}
            diag={'model_failures':0};model=Fake()
            result=call(model,'Verify',{'stage':'verify_help','items':[
                {'item_id':'one'},{'item_id':'two'}]},diag)
            self.assertEqual(result,{})
            self.assertEqual(diag['coverage']['verify_help']['items_expected'],2)
            self.assertEqual(diag['coverage']['verify_help'].get('items_completed',0),0)
            self.assertEqual(completion_report(model,{}, {'candidates':0},{},diag)['status'],'degraded')

    def test_batched_verdict_schema_and_fact_schema_preserve_each_id_and_boolean(self):
        for stage in ('verify_help','verify_goal','help_facts'):
            fmt=verdict_response_format({'stage':stage,'items':[{'item_id':'x'},{'item_id':'y'}]})
            schema=fmt['json_schema']['schema']['properties']['items']
            self.assertEqual(schema['minItems'],2)
            self.assertEqual(schema['maxItems'],2)
            self.assertEqual(schema['items']['properties']['item_id']['enum'],['x','y'])
            if stage=='help_facts':
                self.assertEqual(schema['items']['properties']['facts']['properties']['requested_help'],{'type':'boolean'})
            else:self.assertEqual(schema['items']['properties']['supported'],{'type':'boolean'})


class CalendarGoalTests(unittest.TestCase):
    def test_year_rollover_and_explicit_evidence_year_set_the_horizon(self):
        for title,body,old,expected in [
            ('Grow the business by March','Our target is growth by March.','3mo','1yr'),
            ('Run a race in January','We committed to the race in Jan.','1yr','3mo'),
            ('Graduate by March','The course ends in March 2028.','3mo','3yr'),
            ('Graduate by March 2030','The course ends in March.','5yr','5yr'),
            ('Learn to march','I will learn to march.','1yr','1yr')]:
            self.assertEqual(calendar_horizon({'title':title,'horizon':old},[message('m',body,TS)]),expected)

    def test_combined_scan_retains_valid_tasks_when_a_missing_goal_array_is_retried(self):
        source=message('s','I will prepare the report.',TS)
        index=index_for([source]);state,_=infer_relationships(index)
        raw=dict(source_msg='s',action_text=source['body'],owner='ME',type='commitment',
                 title='Prepare report',owner_evidence=['s'],involvement_evidence=['s'])
        class Fake:
            def __init__(self):self.calls=0
            def call(self,prompt,payload,**kwargs):
                self.calls+=1
                return {'candidates':[raw]} if self.calls==1 else {'candidates':[],'goals':[]}
        model=Fake()
        tasks,diag=extract_candidates(index,state,model,goal_instruction=GOALS)
        self.assertEqual(model.calls,2)
        self.assertEqual([t['source_msg'] for t in tasks],['s'])
        self.assertEqual(diag['goal_batches'],[{'message_ids':['s'],'goals':[]}])

    def test_combined_scan_missing_goals_cannot_be_reported_as_complete(self):
        source=message('s','Hello.',TS)
        index=index_for([source]);state,_=infer_relationships(index)
        class Fake:
            def __init__(self):self.calls=0
            def call(self,*args,**kwargs):self.calls+=1;return {'candidates':[]}
        model=Fake()
        with self.assertRaisesRegex(ValueError,'Invalid combined goal response'):
            extract_candidates(index,state,model,goal_instruction=GOALS)
        self.assertEqual(model.calls,2)


class BatchedLifecycleTests(unittest.TestCase):
    class Fake:
        usage=[]
        def __init__(self,events):self.events=events;self.calls=[]
        def call(self,prompt,payload,**kwargs):
            self.calls.append(payload)
            if payload['lifecycle_stage']=='verify':return {'confirmed':list(range(len(payload['proposals'])))}
            return {'events':self.events,'merges':[]}

    def test_semantic_answer_without_shared_words_can_close_after_two_pass_review(self):
        source=message('s','कल खाने में क्या बनाऊँ?',TS,sender='p')
        answer=message('a','दाल और चावल।',TS+1)
        model=self.Fake([dict(task_id='candidate-s-0',kind='done',message_id='a',evidence_text=answer['body'])])
        out,diag=reconcile_tasks(index_for([source,answer]),[candidate(source)],model,batch_model=True)
        self.assertEqual(out[0]['status'],'done')
        self.assertEqual(out[0]['closed_by'],'a')
        self.assertEqual(len(model.calls),2)
        self.assertEqual(diag['counts']['model_batches'],1)
        self.assertEqual(diag['counts']['model_batches_completed'],1)

    def test_batch_review_still_vetoes_drafts_and_different_invoice_entities(self):
        source=message('s','I will send the final invoice INV-12.',TS)
        for body in ('Invoice INV-12 draft is ready.','Invoice INV-99 sent.'):
            answer=message('a',body,TS+1)
            model=self.Fake([dict(task_id='candidate-s-0',kind='done',message_id='a',evidence_text=body)])
            out,_=reconcile_tasks(index_for([source,answer]),[candidate(source)],model,batch_model=True)
            self.assertEqual(out[0]['status'],'open')

    def test_every_task_beyond_the_first_eight_is_reviewed(self):
        sources=[message('s'+str(i),'I will send document number '+str(i)+'.',TS+i) for i in range(9)]
        later=message('later','We should review progress on the documents.',TS+20)
        model=self.Fake([])
        _,diag=reconcile_tasks(index_for([*sources,later]),[candidate(m) for m in sources],model,batch_model=True)
        reviewed={t['task_id'] for p in model.calls for t in p['tasks']}
        self.assertEqual(reviewed,{'candidate-s'+str(i)+'-0' for i in range(9)})
        self.assertEqual(diag['counts']['model_batches_completed'],2)
        self.assertEqual(diag['counts'].get('model_tasks_deferred',0),0)

    def test_batch_budget_exhaustion_remains_a_failed_completion_gate(self):
        source=message('s','I will send the report.',TS)
        answer=message('a','The report was sent.',TS+1)
        model=self.Fake([])
        _,diag=reconcile_tasks(index_for([source,answer]),[candidate(source)],model,
                              max_model_messages=0,batch_model=True)
        self.assertEqual(model.calls,[])
        self.assertIn('lifecycle_model_budget_deferred',completion_report(model,{}, {'candidates':0},diag,{})['reasons'])

    def test_malformed_lifecycle_verdict_is_not_treated_as_a_negative_decision(self):
        source=message('s','I will send the report.',TS)
        answer=message('a','The report was sent.',TS+1)
        class Bad(self.Fake):
            def call(self,prompt,payload,**kwargs):
                if payload['lifecycle_stage']=='verify':return {'confirmed':['0']}
                return super().call(prompt,payload,**kwargs)
        model=Bad([dict(task_id='candidate-s-0',kind='done',message_id='a',evidence_text=answer['body'])])
        _,diag=reconcile_tasks(index_for([source,answer]),[candidate(source)],model,batch_model=True)
        self.assertIn('lifecycle_malformed_verdicts',completion_report(model,{}, {'candidates':0},diag,{})['reasons'])


if __name__=='__main__':unittest.main()
