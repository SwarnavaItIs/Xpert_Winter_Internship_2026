import unittest
from inputs import InputIndex
from relationships import infer_relationships
from task_candidates import extract_candidates
from task_lifecycle import reconcile_tasks
from tools.audit_development import errors


class AuditTests(unittest.TestCase):
    def index(self):
        chats=[dict(chat_id='room',name='Synthetic',is_group=False,participants=['ME','p'])]
        messages=[dict(id='request',chat_id='room',sender_id='ME',ts=1000,body='I will send the invoice.',kind='text',quoted_id=None),
                  dict(id='reply',chat_id='room',sender_id='p',ts=2000,body='The invoice was not sent.',kind='text',quoted_id='request')]
        return InputIndex(chats,messages,{})

    def test_unverified_control_exposes_negated_completion(self):
        index=self.index();rel,_=infer_relationships(index)
        candidates,_=extract_candidates(index,rel)
        verified,_=reconcile_tasks(index,candidates)
        unverified,_=reconcile_tasks(index,candidates,verify_closures=False)
        self.assertEqual(verified[0]['status'],'open')
        self.assertEqual(unverified[0]['status'],'done')

    def test_unmatched_organisation_is_not_correct_chat_assignment(self):
        index=self.index();state,_=infer_relationships(index)
        gold={'people':[],'organisations':[{'id':'missing'}],
              'chats':[{'chat_id':'room','organisation':'missing'}],'goals':[],'open_loops':[]}
        report=errors(index,state,gold,[])
        self.assertEqual(len(report['chat_errors']),1)
        self.assertTrue(report['chat_errors'][0]['upstream_org_unmatched'])


if __name__=='__main__':
    unittest.main()
