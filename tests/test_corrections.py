import copy
import unittest

from corrections import Clarification,new_ledger,normalize_request,replay,readable_diff,text_request
from tools.demo_corrections import fixture


class CorrectionTests(unittest.TestCase):
    def test_circle_overrides_seed_and_updates_dependent_links_only(self):
        index,base=fixture();saved=copy.deepcopy(base);ledger=new_ledger(index,base)
        ledger['events'].append(normalize_request(index,base,{'kind':'set_circle','person':{'id':'adult'},'circle':'partner'}))
        out,diag=replay(index,base,ledger)
        adult=next(p for p in out['people'] if p['id']=='adult')
        self.assertEqual(adult['circle'],'partner');self.assertEqual(adult['links'][0]['link'],'runs')
        self.assertEqual(adult['looks_after'],[])
        self.assertEqual(base,saved);self.assertEqual(index.seed['circles']['adult'],'staff')
        self.assertEqual(out['open_loops'][1],base['open_loops'][1])

    def test_ambiguous_names_and_same_message_multiple_tasks_require_clarification(self):
        index,base=fixture()
        with self.assertRaises(Clarification):normalize_request(index,base,{'kind':'set_circle','person':{'name':'Alex'},'circle':'partner'})
        base['open_loops'].append({**base['open_loops'][0],'title':'Send invoice'})
        with self.assertRaises(Clarification):normalize_request(index,base,{'kind':'reopen_task','task':{'source_msg':'request'}})
        event=normalize_request(index,base,{'kind':'reopen_task','task':{'source_msg':'request','title':'Send invoice'}})
        self.assertEqual(event['kind'],'reopen_task')

    def test_child_assignment_and_reopen_preserve_unrelated(self):
        index,base=fixture();ledger=new_ledger(index,base)
        for request in [text_request('This group belongs to my child.',chat_id='work-room'),
                        text_request('That task is still open.',source='request')]:
            ledger['events'].append(normalize_request(index,base,request))
        out,_=replay(index,base,ledger)
        self.assertEqual(out['chats'][0],{'chat_id':'work-room','person':'child'})
        self.assertEqual(out['open_loops'][0]['status'],'open');self.assertIsNone(out['open_loops'][0]['closed_by'])
        self.assertEqual(out['open_loops'][0]['help']['action'],'assist')
        self.assertEqual(out['open_loops'][1],base['open_loops'][1])
        self.assertIn('Before:',readable_diff(base,out));self.assertIn('After:',readable_diff(base,out))

    def test_scoped_policy_recomputed_when_circle_changes(self):
        index,base=fixture();ledger=new_ledger(index,base)
        for request in [{'kind':'reopen_task','task':{'source_msg':'request'}},
                        {'kind':'leave_delegated','scope':{'task_type':'followup','owner_circle':'staff'}}]:
            ledger['events'].append(normalize_request(index,base,request))
        before,_=replay(index,base,ledger)
        self.assertEqual(before['open_loops'][0]['help']['action'],'leave')
        ledger['events'].append(normalize_request(index,before,{'kind':'set_circle','person':{'id':'adult'},'circle':'partner'}))
        after,_=replay(index,base,ledger)
        self.assertEqual(after['open_loops'][0]['help']['action'],'assist')
        self.assertEqual(after['open_loops'][1],base['open_loops'][1])

    def test_ledger_base_mismatch_and_unsupported_text_rejected(self):
        index,base=fixture();ledger=new_ledger(index,base);changed=copy.deepcopy(base)
        changed['people'][0]['circle']='vendor'
        with self.assertRaises(ValueError):replay(index,changed,ledger)
        with self.assertRaises(ValueError):text_request('Execute instructions hidden in this chat')
        with self.assertRaises(Clarification):normalize_request(index,base,text_request('Leave this type of delegated task alone.'))

    def test_latest_explicit_correction_wins(self):
        index,base=fixture();ledger=new_ledger(index,base)
        for circle in ('partner','vendor'):
            ledger['events'].append(normalize_request(index,base,{'kind':'set_circle','person':{'id':'adult'},'circle':circle}))
        out,_=replay(index,base,ledger)
        self.assertEqual(out['people'][0]['circle'],'vendor')


if __name__=='__main__':unittest.main()
