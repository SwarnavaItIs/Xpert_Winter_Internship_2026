from datetime import datetime, timezone
import json
import unittest

from inputs import InputIndex
from relationships import infer_relationships
from solution import completion_report
from task_candidates import bounded_chunks, extract_candidates, normalize_candidate, candidate_loop, relationship_context
from task_dates import resolve_due
from validation import validate
from tools.task_metrics import detection_metrics, scoped_labels, anchor_diagnostics

TS=int(datetime(2026,8,24,20,0,tzinfo=timezone.utc).timestamp())  # Aug 25 01:30 IST


def message(mid,body,sender='ME',ts=TS,quoted=None,chat='direct'):
    return {'id':mid,'chat_id':chat,'sender_id':sender,'sender_name':sender,
            'ts':ts,'kind':'text','body':body,'quoted_id':quoted}


def index_for(messages,group=False):
    return InputIndex([{'chat_id':'direct','name':'Synthetic','is_group':group,
                        'participants':['ME','other','third'] if group else ['ME','other']}],messages,{})


def extract(index,**options):
    rel,diagnostics=infer_relationships(index)
    return extract_candidates(index,rel,None,diagnostics,**options)[0]


class CandidateTests(unittest.TestCase):
    def test_wrong_optional_offset_reanchors_only_unique_visible_literal(self):
        body='🙂 पहले: I will send the report.'
        index=index_for([message('a',body)])
        chunk=next(bounded_chunks(index))
        action='I will send the report.'
        raw=dict(source_msg='a',action_text=action,owner='ME',type='commitment',
                 title='Send report',owner_evidence=['a'],involvement_evidence=['a'])
        for offset in (None,0,len('🙂 पहले: '.encode('utf-16-le'))//2):
            candidate=normalize_candidate({**raw,'action_start':offset},index,chunk)
            self.assertEqual(candidate['action_start'],body.index(action))
            self.assertEqual(body[candidate['action_start']:candidate['action_end']],action)
        repeated=index_for([message('a',action+' '+action)])
        with self.assertRaisesRegex(ValueError,'action_span'):
            normalize_candidate(raw,repeated,next(bounded_chunks(repeated)))
        raw['action_start']=len(action)+1
        self.assertEqual(normalize_candidate(raw,repeated,next(bounded_chunks(repeated)))['action_start'],len(action)+1)
        with self.assertRaisesRegex(ValueError,'action_span'):
            normalize_candidate({**raw,'action_text':'Send a report'},index,chunk)

    def test_short_repair_retries_contract_without_losing_accepted_items(self):
        index=index_for([message('a','I will send the report.'),message('b','I will book the room.',ts=TS+1)])
        rel,_=infer_relationships(index)
        def candidate(mid):
            return dict(source_msg=mid,action_text=index.message[mid]['body'],owner='ME',type='commitment',
                        title='Complete action',owner_evidence=[mid],involvement_evidence=[mid])
        class Fake:
            usage=[]
            def __init__(self):self.repairs=0
            def call(self,prompt,payload,**kwargs):
                if payload['task_stage']=='candidates':
                    return {'candidates':[candidate('a'),{**candidate('b'),'action_text':'paraphrase'}]}
                self.repairs+=1
                return {'candidates':[]} if self.repairs==1 else {'candidates':[candidate('b')]}
        model=Fake();tasks,diag=extract_candidates(index,rel,model)
        self.assertEqual({t['source_msg'] for t in tasks},{'a','b'})
        self.assertEqual(diag['coverage']['repair_contract_retries'],1)
        self.assertEqual(diag['coverage']['repair_contract_failures'],0)
        self.assertEqual(completion_report(model,{},diag,{}, {})['status'],'complete')

    def test_repair_invalid_spans_and_types_preserves_other_candidates(self):
        index=index_for([message('a','I will send the report.'),message('b','I will book the room.',ts=TS+1)])
        rel,_=infer_relationships(index)
        good=lambda mid,body:dict(source_msg=mid,action_text=body,owner='ME',type='commitment',
            title='Complete the action',owner_evidence=[mid],involvement_evidence=[mid])
        first=good('a',index.message['a']['body']);second=good('b',index.message['b']['body'])
        class Fake:
            usage=[]
            def __init__(self):self.calls=[]
            def call(self,prompt,payload,**kwargs):
                self.calls.append(payload)
                if payload['task_stage']=='candidates':
                    return {'candidates':[first,{**second,'action_text':'paraphrase','type':'commitments'}]}
                return {'candidates':[second]}
        model=Fake();tasks,diag=extract_candidates(index,rel,model)
        self.assertEqual({t['source_msg'] for t in tasks},{'a','b'})
        self.assertEqual(len(model.calls),2)
        self.assertEqual(model.calls[0]['chunk'],model.calls[1]['chunk'])
        self.assertEqual(diag['rejected'],{})
        self.assertEqual(sum(diag['rejected_initial'].values()),1)
        self.assertEqual(diag['coverage']['repair_items_completed'],1)
        self.assertEqual(completion_report(model,{},diag,{}, {})['status'],'complete')

    def test_empty_or_unsupported_repair_cannot_hide_missing_work(self):
        index=index_for([message('a','I will send the report.')]);rel,_=infer_relationships(index)
        invalid=dict(source_msg='a',action_text='paraphrase',owner='ME',type='commitment',
                     title='Send report',owner_evidence=['a'],involvement_evidence=['a'])
        for fixes in ([],[None],[invalid]):
            class Fake:
                usage=[]
                calls=0
                def call(self,prompt,payload,**kwargs):
                    self.calls+=1
                    return {'candidates':[invalid]} if payload['task_stage']=='candidates' else {'candidates':fixes}
            model=Fake();tasks,diag=extract_candidates(index,rel,model)
            self.assertEqual(tasks,[])
            self.assertEqual(model.calls,3 if fixes==[] else 2)
            self.assertEqual(diag['rejected'],{'action_span':1})
            self.assertEqual(diag['coverage']['repair_items_completed'],0)
            self.assertEqual(completion_report(model,{},diag,{}, {})['status'],'degraded')

    def test_empty_candidate_batch_is_complete_without_repair(self):
        index=index_for([message('a','Hello.')]);rel,_=infer_relationships(index)
        class Fake:
            usage=[]
            calls=0
            def call(self,*args,**kwargs):self.calls+=1;return {'candidates':[]}
        model=Fake();tasks,diag=extract_candidates(index,rel,model)
        self.assertEqual(tasks,[])
        self.assertEqual(model.calls,1)
        self.assertEqual(diag['coverage']['chunks_expected'],diag['coverage']['chunks_completed'])
        self.assertEqual(completion_report(model,{},diag,{}, {})['status'],'complete')

    def test_self_obligation_and_direct_imperative_without_politeness(self):
        tasks=extract(index_for([message('need','I need to renew the permit tomorrow.'),
                                message('request','Send the revised contract tomorrow.','other',TS+1)]))
        self.assertEqual({t['source_msg'] for t in tasks},{'need','request'})
        self.assertTrue(all(t['owner']=='ME' for t in tasks))
        # Bare third-party group imperatives do not imply a user obligation.
        self.assertEqual(extract(index_for([message('unrelated','Send the revised contract.','other')],group=True)),[])

    def test_negated_promises_and_politeness_do_not_create_tasks(self):
        for text in ["I will not send the report.","I won't book the room.",
                     'I do not need to renew the permit.','No need to pay the invoice.',
                     'Please take care.','Please enjoy.']:
            self.assertEqual(extract(index_for([message('negated',text)])),[],text)
        self.assertEqual(len(extract(index_for([message('real','Please pay the invoice.')]))),1)

    def test_indirect_commitment_and_relationship_context(self):
        index=index_for([message('a','Leave the revised quotation with me.')])
        tasks=extract(index)
        self.assertEqual(len(tasks),1)
        self.assertEqual(tasks[0]['owner'],'ME')
        self.assertEqual(tasks[0]['type'],'commitment')
        self.assertEqual(tasks[0]['source_ts'],TS)
        self.assertFalse(tasks[0]['lifecycle_evaluated'])

    def test_relationship_aliases_do_not_collapse_ambiguous_owners(self):
        index=index_for([{**message('a','Hello','other'),'sender_name':'Same Name'},
                         {**message('b','Hello','third',TS+1),'sender_name':'Same Name'}],group=True)
        relationships,_=infer_relationships(index)
        context=relationship_context(index,relationships,next(bounded_chunks(index)))
        self.assertEqual({p['id'] for p in context['people']},{'other','third'})
        self.assertTrue(all(p['ambiguous_aliases']==['same name'] for p in context['people']))

    def test_hinglish_and_relative_date(self):
        tasks=extract(index_for([message('a','Kal main report bhej dunga.')]))
        self.assertEqual(len(tasks),1)
        self.assertEqual(tasks[0]['due']['date'],'2026-08-26')
        self.assertTrue(tasks[0]['due']['uncertain'])
        self.assertEqual(tasks[0]['owner'],'ME')

    def test_quoted_request_uses_original_source_across_chunks(self):
        index=index_for([message('request','Please send the report tomorrow.','other'),
                         message('adopt','On it.','ME',TS+86400,'request')])
        tasks=extract(index,core_messages=1)
        self.assertEqual(len(tasks),1)
        self.assertEqual(tasks[0]['source_msg'],'request')
        self.assertIn('adopt',tasks[0]['owner_evidence'])
        self.assertEqual(tasks[0]['due']['date'],'2026-08-26')  # Quote date, not acknowledgement date.

    def test_multiple_actions_same_source_survive(self):
        index=index_for([message('a',"I'll send the quotation and book the pickup tomorrow.")])
        tasks=extract(index)
        self.assertEqual(len(tasks),2)
        self.assertEqual({c['source_msg'] for c in tasks},{'a'})
        self.assertEqual(len({c['candidate_id'] for c in tasks}),2)
        self.assertEqual(tasks[0]['due']['date'],None)
        self.assertEqual(tasks[1]['due']['date'],'2026-08-26')

    def test_multiple_requested_actions_and_condition_date(self):
        tasks=extract(index_for([message('a','Please send the invoice and book the taxi.','other')]))
        self.assertEqual(len(tasks),2)
        self.assertEqual({c['owner'] for c in tasks},{'ME'})
        contingent=extract(index_for([message('b',"If approval arrives tomorrow, I'll book the taxi.")]))
        self.assertIsNone(contingent[0]['due']['date'])
        self.assertIsNotNone(contingent[0]['condition'])

    def test_conditional_promise_not_hypothetical(self):
        tasks=extract(index_for([message('a',"If the client approves, I'll book the venue."),
                                message('b','If I had more time, I would write a book.',ts=TS+1)]))
        self.assertEqual(len(tasks),1)
        self.assertEqual(tasks[0]['condition_state'],'unresolved')
        self.assertTrue(tasks[0]['uncertain'])
        trailing=extract(index_for([message('c',"I'll book the venue if approval arrives tomorrow.")]))
        self.assertEqual(trailing[0]['condition_state'],'unresolved')
        self.assertIsNone(trailing[0]['due']['date'])

    def test_casual_advice_forwarding_and_unrelated_third_party_excluded(self):
        texts=['How are you?','You should exercise every day.',
               "Forwarded: I'll send the money tomorrow.", 'My friend will clean his garage.',
               "Maybe I'll send a proposal."]
        self.assertEqual(extract(index_for([message(str(i),s) for i,s in enumerate(texts)])),[])
        self.assertEqual(extract(index_for([message('a',"I'll send the report.",'other')],group=True)),[])
        self.assertEqual(extract(index_for([message('a',"I'll clean my friend's garage tomorrow.",'other')])),[])

    def test_followup_question_and_deadline(self):
        tasks=extract(index_for([message('a',"I'll send you the file tomorrow.",'other'),
                                message('b','Which design do you approve?','other',TS+1),
                                message('c','The payment is due tomorrow.','other',TS+2)]))
        self.assertEqual([c['type'] for c in tasks],['followup','question','deadline'])
        self.assertEqual([c['owner'] for c in tasks],['other','ME','ME'])

    def test_overlap_order_and_long_fragments_preserve_coverage(self):
        messages=[message('a',"I'll send the report."),message('b','Thanks.','other',TS+1),
                  message('c','x'*6000,'other',TS+2)]
        index=index_for(list(reversed(messages)))
        chunks=list(bounded_chunks(index,core_messages=1))
        self.assertTrue(chunks[1]['overlap'])
        covered=set()
        for c in chunks:
            self.assertLessEqual(sum(len(json.dumps(v,ensure_ascii=False)) for v in c['core']),12000)
            for v in c['core']:
                if v['id']=='c':
                    covered.update(range(v['body_start'],v['body_end']))
        self.assertEqual(len(covered),6000)
        self.assertEqual(len(extract(index,core_messages=1)),1)
        self.assertEqual(extract(index),extract(index_for(messages)))

    def test_model_receives_non_english_without_keyword_gate(self):
        index=index_for([message('a','আমি আগামীকাল নথি পাঠিয়ে দেব।')])
        rel,_=infer_relationships(index)
        class FakeModel:
            calls=0
            def call(self,prompt,payload,**kwargs):
                self.calls+=1
                assert payload['relationships']['people']
                body=payload['chunk']['core'][0]['body']
                assert body=='আমি আগামীকাল নথি পাঠিয়ে দেব।'
                return {'candidates':[{'source_msg':'a','action_text':body,'owner':'ME','type':'commitment',
                    'title':'Send documents','owner_evidence':['a'],'involvement_evidence':['a'],
                    'due_text':'আগামীকাল','due_evidence':'a','uncertain':False,'reason':'synthetic response'}]}
        model=FakeModel()
        tasks,_=extract_candidates(index,rel,model)
        self.assertEqual(model.calls,1)
        self.assertEqual(tasks[0]['due']['date'],'2026-08-26')
        validate({**rel,'open_loops':[candidate_loop(c) for c in tasks]},index.chats,index.messages)

    def test_invalid_owner_and_evidence_rejected(self):
        index=index_for([message('a',"I'll send the file.")])
        chunk=next(bounded_chunks(index))
        raw={'source_msg':'a','action_text':"I'll send the file.",'owner':'invented','type':'commitment',
             'title':'Send file','owner_evidence':['a'],'involvement_evidence':['a']}
        with self.assertRaisesRegex(ValueError,'owner_reference'):
            normalize_candidate(raw,index,chunk)
        raw.update(owner='ME',due_text='tomorrow',due_evidence='a')
        with self.assertRaisesRegex(ValueError,'unsupported_due_expression'):
            normalize_candidate(raw,index,chunk)


class DateTests(unittest.TestCase):
    def test_message_timezone_and_calendar_boundaries(self):
        self.assertEqual(resolve_due('today',TS,'a')['date'],'2026-08-25')
        self.assertEqual(resolve_due('in 2 days',TS,'a')['date'],'2026-08-27')
        year_end=int(datetime(2026,12,31,10,tzinfo=timezone.utc).timestamp())
        self.assertEqual(resolve_due('tomorrow',year_end,'a')['date'],'2027-01-01')
        self.assertEqual(resolve_due('2028-02-29',TS,'a')['date'],'2028-02-29')
        self.assertIsNone(resolve_due('2026-02-29',TS,'a')['date'])

    def test_ambiguous_and_vague_dates_preserve_uncertainty(self):
        self.assertTrue(resolve_due('next Tuesday',TS,'a')['uncertain'])
        self.assertTrue(resolve_due('03/04/2027',TS,'a')['uncertain'])
        self.assertIsNone(resolve_due('soon',TS,'a')['date'])
        self.assertIsNone(resolve_due('today or tomorrow',TS,'a')['date'])


class DetectionMetricTests(unittest.TestCase):
    def test_official_one_to_one_source_matching_preserves_multiple_tasks(self):
        labels=[{'source_msgs':['a','b']},{'source_msgs':['a']}]
        records=[{'source_msg':'a'},{'source_msg':'a'},{'source_msg':'wrong'}]
        score,pairs=detection_metrics(records,labels)
        self.assertEqual(pairs,[(0,0),(1,1)])
        self.assertEqual(score['matched'],2)
        self.assertEqual(score['precision'],2/3)
        self.assertEqual(score['recall'],1)

    def test_validation_and_boundary_labels_are_excluded(self):
        answers={'open_loops':[{'source_msgs':['dev'],'title':'development','status':'done'},
                               {'source_msgs':['val'],'title':'must not retain'},
                               {'source_msgs':['dev','val'],'title':'boundary'}]}
        gold,excluded=scoped_labels(answers,{'dev'},{'val'})
        self.assertEqual(len(gold),1)
        self.assertNotIn('status',gold[0])
        self.assertEqual(excluded['cross_partition_source_tasks'],1)

    def test_wrong_anchor_is_diagnostic_not_match_credit(self):
        index=index_for([message('request','Please send the report.','other'),
                         message('reply','On it.','ME',TS+60,'request')])
        candidate={'candidate_id':'synthetic','source_msg':'reply','owner':'ME','title':'Send report'}
        label={'source_msgs':['request'],'owner':'ME','title':'Send report'}
        score,pairs=detection_metrics([candidate],[label])
        errors=anchor_diagnostics([candidate],[label],pairs,index)
        self.assertEqual(score['matched'],0)
        self.assertEqual(len(errors['suspected_anchor_errors']),1)
        self.assertEqual(errors['unmatched_without_anchor_suggestion'],[])
        self.assertIsNone(errors['confirmed_anchor_errors'])


if __name__=='__main__':
    unittest.main()
