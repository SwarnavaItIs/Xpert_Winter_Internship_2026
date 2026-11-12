import copy
import unittest

from inputs import InputIndex
from task_dates import resolve_due
from task_lifecycle import reconcile_tasks
from tools.lifecycle_metrics import lifecycle_metrics,project_lifecycle_labels


def message(mid,body,ts,sender='ME',quote=None,cid='d'):
    return {'id':mid,'body':body,'ts':ts,'sender_id':sender,'sender_name':sender,
            'chat_id':cid,'kind':'text','quoted_id':quote}


def candidate(m,owner='ME',start=0,end=None,due=''):
    end=len(m['body']) if end is None else end
    return {'candidate_id':'candidate-'+m['id']+'-'+str(start),'source_msg':m['id'],
            'source_ts':m['ts'],'chat_id':m['chat_id'],'owner':owner,'type':'commitment',
            'action_text':m['body'][start:end],'action_start':start,'action_end':end,
            'title':m['body'][start:end],'due':resolve_due(due,m['ts'],m['id']),
            'condition':None,'uncertain':True}


def index_for(messages,chats=None):
    return InputIndex(chats or [{'chat_id':'d','name':'Synthetic','is_group':False,'participants':['ME','p']}],messages,{})


class LifecycleTests(unittest.TestCase):
    def test_acknowledgements_do_not_close(self):
        source=message('s','I will send the report.',1)
        ms=[source,message('a','Okay',2,quote='s'),message('b','Will do',3,quote='s')]
        out,diag=reconcile_tasks(index_for(ms),[candidate(source)])
        self.assertEqual(out[0]['status'],'open')
        self.assertIsNone(out[0]['closed_by'])
        ms.append(message('c','Report sent.',4,quote='s'))
        out,_=reconcile_tasks(index_for(ms),[candidate(source)])
        self.assertEqual(out[0]['closed_by'],'c')

    def test_draft_is_partial_until_delivery(self):
        source=message('s','I will send the final proposal.',1)
        ms=[source,message('draft','Draft is ready.',2,quote='s')]
        out,diag=reconcile_tasks(index_for(ms),[candidate(source)])
        self.assertEqual(out[0]['status'],'open')
        self.assertEqual(diag['tasks'][0]['history'][0]['kind'],'partial')
        ms.append(message('delivery','Final proposal sent.',3,quote='s'))
        out,_=reconcile_tasks(index_for(ms),[candidate(source)])
        self.assertEqual(out[0]['status'],'done')
        self.assertEqual(out[0]['closed_by'],'delivery')

    def test_draft_ready_can_finish_a_drafting_obligation(self):
        source=message('s','I will prepare the draft.',1)
        out,_=reconcile_tasks(index_for([source,message('c','Draft is ready.',2,quote='s')]),[candidate(source)])
        self.assertEqual(out[0]['status'],'done')

    def test_tentative_cancellation_is_not_cancellation(self):
        source=message('s','I will send the report.',1)
        out,diag=reconcile_tasks(index_for([source,message('c','I will cancel the report tomorrow.',2,quote='s')]),[candidate(source)])
        self.assertEqual(out[0]['status'],'open')
        self.assertEqual(diag['counts']['tentative_cancellation_veto'],1)

    def test_cancel_and_reschedule_are_different(self):
        source=message('s','I will send the report.',1724000000)
        reschedule=message('r','Move the report to tomorrow instead.',1724086400,quote='s')
        out,diag=reconcile_tasks(index_for([source,reschedule]),[candidate(source)])
        self.assertEqual(out[0]['status'],'open')
        self.assertIsNone(out[0]['closed_by'])
        self.assertEqual(out[0]['due'],resolve_due('tomorrow',reschedule['ts'],'r')['date'])
        cancel=message('c','No longer needed.',1724086500,sender='p',quote='s')
        out,_=reconcile_tasks(index_for([source,reschedule,cancel]),[candidate(source)])
        self.assertEqual(out[0]['status'],'dropped')
        self.assertEqual(out[0]['closed_by'],'c')

    def test_cross_chat_resolution_requires_specific_entity(self):
        chats=[{'chat_id':'g','name':'Team','is_group':True,'participants':['ME','p','q']},
               {'chat_id':'d','name':'Direct','is_group':False,'participants':['ME','q']}]
        source=message('s','I will send invoice INV-42.',1,cid='g')
        done=message('c','Invoice INV-42 sent.',2)
        out,_=reconcile_tasks(index_for([source,done],chats),[candidate(source)])
        self.assertEqual(out[0]['closed_by'],'c')
        other=message('c','Invoice INV-99 sent.',2)
        out,_=reconcile_tasks(index_for([source,other],chats),[candidate(source)])
        self.assertEqual(out[0]['status'],'open')

    def test_recurrence_stays_distinct(self):
        first=message('s1','I will send the weekly report.',1724000000)
        done=message('done','Report sent.',1724000100,quote='s1')
        second=message('s2','I will send the weekly report.',1724604800,quote='s1')
        out,diag=reconcile_tasks(index_for([first,done,second]),[candidate(first,due='today'),candidate(second,due='today')])
        self.assertEqual(len(out),2)
        self.assertEqual([x['status'] for x in out],['done','open'])
        self.assertEqual(diag['merges'],[])

    def test_reopening_preserves_history_and_can_close_again(self):
        source=message('s','I will send invoice INV-42.',1)
        ms=[source,message('sent','Invoice INV-42 sent.',2,quote='s'),
            message('bounced','Delivery bounced; reopen invoice INV-42.',3,quote='sent')]
        out,diag=reconcile_tasks(index_for(ms),[candidate(source)])
        self.assertEqual(out[0]['status'],'open')
        self.assertIsNone(out[0]['closed_by'])
        self.assertEqual([e['kind'] for e in diag['tasks'][0]['history']],['done','reopen'])
        ms.append(message('resent','Invoice INV-42 sent successfully.',4,quote='s'))
        out,_=reconcile_tasks(index_for(ms),[candidate(source)])
        self.assertEqual(out[0]['closed_by'],'resent')

    def test_followup_merge_keeps_original_anchor(self):
        first=message('s1','Please send the report.',1,sender='p')
        second=message('s2','Please send the report.',2,sender='p',quote='s1')
        done=message('done','Report sent.',3,quote='s2')
        out,diag=reconcile_tasks(index_for([first,second,done]),[candidate(first),candidate(second)])
        self.assertEqual(len(out),1)
        self.assertEqual(out[0]['source_msg'],'s1')
        self.assertEqual(out[0]['closed_by'],'done')
        self.assertEqual(diag['counts']['merged_references'],1)

    def test_ambiguous_completion_does_not_close_two_actions(self):
        source=message('s','Send invoice INV-1; send invoice INV-2',1)
        split=source['body'].index(';')
        cs=[candidate(source,end=split),candidate(source,start=split+2)]
        out,_=reconcile_tasks(index_for([source,message('c','Done.',2,quote='s')]),cs)
        self.assertEqual([t['status'] for t in out],['open','open'])
        out,_=reconcile_tasks(index_for([source,message('c','Invoice INV-1 sent.',2,quote='s')]),cs)
        self.assertEqual(sum(t['status']=='done' for t in out),1)

    def test_original_evidence_vetoes_model_draft_closure(self):
        source=message('s','I will send the report.',1)
        evidence=message('d','The report is prepared.',2,quote='s')
        class FakeModel:
            usage=[]
            def call(self,prompt,payload,**kwargs):
                if payload['lifecycle_stage']=='verify':
                    return {'confirmed':[0]}
                return {'events':[{'task_id':'candidate-s-0','kind':'done','message_id':'d',
                                  'evidence_text':'The report is prepared.','reason':'synthetic bad proposal'}]}
        out,diag=reconcile_tasks(index_for([source,evidence]),[candidate(source)],FakeModel())
        self.assertEqual(out[0]['status'],'open')
        self.assertEqual(diag['counts']['draft_is_not_delivery_veto'],1)

    def test_state_survives_chunks_and_shuffled_input(self):
        source=message('s','I will send the report.',1)
        ms=[source]+[message('noise'+str(i),'Thanks.',i+2,sender='p') for i in range(100)]
        ms.append(message('done','Report sent.',200,quote='s'))
        a,diag=reconcile_tasks(index_for(ms),[candidate(source)])
        b,_=reconcile_tasks(index_for(list(reversed(ms))),[candidate(source)])
        self.assertEqual(a,b)
        self.assertEqual(a[0]['closed_by'],'done')
        self.assertGreater(diag['counts']['chronological_chunks'],1)

    def test_recurring_ambiguous_model_completion_is_vetoed(self):
        one=message('s1','I will send the compliance report.',1724000000)
        two=message('s2','I will send the compliance report.',1724086400)
        done=message('c','The compliance report was sent.',1724086500)
        class FakeModel:
            usage=[]
            def call(self,prompt,payload,**kwargs):
                if payload['lifecycle_stage']=='verify':
                    return {'confirmed':list(range(len(payload['proposals'])))}
                if payload['current_message']['id']!='c':
                    return {'events':[]}
                return {'events':[{'task_id':t['task_id'],'kind':'done','message_id':'c',
                                   'evidence_text':done['body']} for t in payload['tasks']]}
        out,_=reconcile_tasks(index_for([one,two,done]),[candidate(one,due='today'),candidate(two,due='today')],FakeModel())
        self.assertEqual([t['status'] for t in out],['open','open'])


class LifecycleMetricTests(unittest.TestCase):
    def test_false_closure_and_missing_closure_denominators(self):
        gold=[{'source_msgs':['a'],'status':'open','closing_msgs':[]},
              {'source_msgs':['b'],'status':'done','closing_msgs':['finish']},
              {'source_msgs':['c'],'status':'dropped','closing_msgs':[]}]
        loops=[{'source_msg':'a','status':'done','closed_by':'wrong'},
               {'source_msg':'b','status':'open','closed_by':None},
               {'source_msg':'unlabelled','status':'done','closed_by':'other'}]
        result=lifecycle_metrics(loops,gold,{'a','b','c','finish'})['lifecycle']
        self.assertEqual(result['false_closure_rate_on_matched_open'],1)
        self.assertEqual(result['missed_closures_matched'],1)
        self.assertEqual(result['missed_closures_with_visible_evidence'],1)
        self.assertEqual(result['closed_tasks_missed_at_extraction'],1)
        self.assertEqual(result['unmatched_predicted_closed'],1)

    def test_duplicates_are_one_to_one_label_proxy_and_validation_excluded(self):
        answers={'open_loops':[{'source_msgs':['dev'],'status':'open','closing_msgs':[]},
                              {'source_msgs':['val'],'status':'done','closing_msgs':['secret']}]}
        gold,_=project_lifecycle_labels(answers,{'dev'},{'val'})
        result=lifecycle_metrics([{'source_msg':'dev','status':'open'}]*2,gold,{'dev'})['lifecycle']
        self.assertEqual(result['duplicate_anchor_predictions'],1)
        self.assertEqual(result['matched'],1)
        self.assertIsNone(result['false_closure_fraction_of_matched_closed_predictions'])


if __name__=='__main__':
    unittest.main()
