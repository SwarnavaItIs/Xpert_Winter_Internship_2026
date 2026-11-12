import copy
from contextlib import redirect_stderr
import io
import unittest

from inputs import InputIndex, InputError
from relationships import infer_relationships, reconcile
from tools.evaluate_relationships import project


def msg(mid,cid,pid,body,ts=1,name='Alex',quote=None):
    return {'id':mid,'chat_id':cid,'sender_id':pid,'sender_name':name,'ts':ts,
            'body':body,'kind':'text','quoted_id':quote}


def chat(cid,participants,group=False,name='Synthetic'):
    return {'chat_id':cid,'participants':participants,'is_group':group,'name':name}


class RelationshipTests(unittest.TestCase):
    def test_relationship_contract_retry_recovers_without_exposing_response(self):
        index=InputIndex([chat('a',['ME','p'])],[msg('one','a','p','Hello')],{})
        class Fake:
            def __init__(self):self.calls=[]
            def call(self,prompt,payload,**kwargs):
                self.calls.append((prompt,copy.deepcopy(payload),kwargs))
                if len(self.calls)==1:
                    return {'people':{'private':'DO_NOT_LOG'},'chats':[], 'merges':None}
                return {'people':[],'organisations':[],'chats':[],'merges':[]}
        client=Fake();stderr=io.StringIO()
        with redirect_stderr(stderr):
            output,_=infer_relationships(index,client)
        self.assertEqual(len(client.calls),2)
        self.assertEqual(client.calls[0][1:],client.calls[1][1:])
        self.assertIn('Do not wrap the object in another field.',client.calls[1][0])
        self.assertIn('people: expected list, got dict',stderr.getvalue())
        self.assertIn('organisations: missing',stderr.getvalue())
        self.assertIn('merges: expected list, got NoneType',stderr.getvalue())
        self.assertNotIn('DO_NOT_LOG',stderr.getvalue())
        self.assertEqual(len(output['chats']),1)

    def test_relationship_contract_retry_is_bounded(self):
        index=InputIndex([chat('a',['ME','p'])],[msg('one','a','p','Hello')],{})
        class Fake:
            calls=0
            def call(self,*args,**kwargs):
                self.calls+=1
                return ['DO_NOT_LOG']
        client=Fake();stderr=io.StringIO()
        with redirect_stderr(stderr),self.assertRaisesRegex(ValueError,'Invalid relationship response'):
            infer_relationships(index,client)
        self.assertEqual(client.calls,2)
        self.assertIn('attempt 2',stderr.getvalue())
        self.assertIn('root: expected object, got list',stderr.getvalue())
        self.assertNotIn('DO_NOT_LOG',stderr.getvalue())

    def test_relationship_contract_still_allows_omitted_merges(self):
        index=InputIndex([chat('a',['ME','p'])],[msg('one','a','p','Hello')],{})
        class Fake:
            calls=0
            def call(self,*args,**kwargs):
                self.calls+=1
                return {'people':[],'organisations':[],'chats':[]}
        client=Fake()
        infer_relationships(index,client)
        self.assertEqual(client.calls,1)

    def index(self):
        return InputIndex([chat('a',['ME','p']),chat('b',['ME','q']),chat('g',['ME','p','q'],True)],
            [msg('one','a','p','I am your driver.',name='Alex'),
             msg('two','b','q','I am your colleague.',ts=2,name='Alex'),
             msg('three','g','ME','Alex will attend.',ts=3,name='User',quote='one')],{})

    def test_ambiguous_names_and_one_person_in_multiple_chats(self):
        index = self.index()
        out,diag = infer_relationships(index)
        people = {p['id']:p for p in out['people']}
        self.assertEqual(len(people),2)
        self.assertEqual(people['p']['circle'],'staff')
        self.assertEqual(people['q']['circle'],'colleague')
        self.assertEqual(index.person_chats['p'],['a','g'])
        self.assertEqual(index.mentions['p'],[])
        evidence = index.evidence('p')
        self.assertEqual(evidence['ambiguous_aliases'],['alex'])
        self.assertIn('three',[m['id'] for m in evidence['messages']])
        self.assertTrue(diag['chats']['g']['uncertain'])
        self.assertEqual(out['organisations'],[])  # No automatic group=>organisation.

    def test_sorting_quotes_timestamp_index_and_invalid_inputs(self):
        index = self.index()
        shuffled = InputIndex(list(reversed(index.chats)),list(reversed(index.messages)),{})
        self.assertEqual(infer_relationships(index),infer_relationships(shuffled))
        self.assertEqual([m['id'] for m in shuffled.between(2,3)],['two','three'])
        self.assertEqual(shuffled.replies['one'][0]['id'],'three')
        self.assertEqual(len(shuffled.by_timestamp[2]),1)
        with self.assertRaises(InputError):
            InputIndex(index.chats,index.messages+[index.messages[0]],{})
        with self.assertRaises(InputError):
            InputIndex(index.chats,[{**index.messages[0],'ts':float('nan')}],{})
        missing = InputIndex(index.chats,[{**index.messages[0],'quoted_id':'not-present'}],{})
        self.assertEqual(missing.warnings['unresolved_quotes'],1)

    def test_seed_preserved_and_correction_requires_confirmation(self):
        index = self.index()
        index.seed = {'circles':{'p':'vendor'}}
        out,diag = infer_relationships(index)
        self.assertEqual(out['people'][0]['circle'],'vendor')
        evidence = msg('confirm','a','ME','You are my business partner.',ts=5)
        index = InputIndex(index.chats,index.messages+[evidence],index.seed)
        correction = {'person_id':'p','circle':'partner','evidence':['confirm'],'confirmed_by_user':True}
        out,_ = infer_relationships(index,corrections=[correction])
        self.assertEqual(out['people'][0]['circle'],'partner')
        with self.assertRaises(ValueError):
            infer_relationships(index,corrections=[{**correction,'confirmed_by_user':False}])

    def test_child_group_assignment_without_child_membership(self):
        index = InputIndex([chat('d',['ME','kid']),chat('class',['ME','teacher'],True,'Robin class parents')],
                          [msg('hi','d','kid','Hello',name='Robin'),msg('schedule','class','teacher','Class tomorrow',name='Teacher')],
                          {'circles':{'kid':'child'}})
        out,diag = infer_relationships(index)
        self.assertEqual(next(c for c in out['chats'] if c['chat_id']=='class'),{'chat_id':'class','person':'kid'})
        self.assertFalse(diag['chats']['class']['uncertain'])

    def test_organisation_merging_links_and_looks_after(self):
        index = self.index()
        org = {'name':'Example Works','type':'company','side':'work','role':'owner',
               '_evidence':['one'],'_confidence':.9}
        proposal = {'organisations':[{**org,'id':'x'},{**org,'id':'y'}],
                    'people':[{'id':'p','circle':'staff','links':[{'organisation':'y','link':'supports'},
                              {'organisation':'missing','link':'supports'}],'looks_after':['q','missing'],
                              '_evidence':['one'],'_confidence':.9}],
                    'chats':[{'chat_id':'g','organisation':'x','_evidence':['three'],'_confidence':.9}]}
        out,diag = reconcile(index,[proposal])
        self.assertEqual(len(out['organisations']),1)
        self.assertEqual(out['people'][0]['links'],[{'organisation':'x','link':'supports'}])
        self.assertEqual(out['people'][0]['looks_after'],['q'])
        self.assertEqual(diag['organisations']['x']['merged_ids'],['x','y'])
        separate = copy.deepcopy(proposal)
        separate['organisations'][1]['_evidence']=['two']
        separate['people']=[]
        separate['chats']=[]
        self.assertEqual(len(reconcile(index,[separate])[0]['organisations']),2)

    def test_seed_organisation_can_gain_missing_attributes(self):
        index = self.index()
        index.seed={'organisations':['Example Works']}
        proposal={'organisations':[{'id':'seed-org-0','name':'Wrong rename','type':'company',
                     'side':'work','role':'owner','_confidence':.9,'_evidence':['one']}]}
        out,_ = reconcile(index,[proposal])
        self.assertEqual(out['organisations'][0]['name'],'Example Works')
        self.assertEqual(out['organisations'][0]['type'],'company')

    def test_conflicts_and_unknown_references_are_explicit(self):
        index = self.index()
        proposal={'people':[{'id':'p','circle':'vendor','_evidence':['one'],'_confidence':.9},
                            {'id':'p','circle':'staff','_evidence':['two'],'_confidence':.8},
                            {'id':'q','circle':'client','_evidence':['invented'],'_confidence':1}]}
        out,diag = reconcile(index,[proposal])
        self.assertEqual(out['people'][0]['circle'],'vendor')
        self.assertTrue(diag['people']['p']['uncertain'])
        self.assertTrue(diag['people']['q']['uncertain'])
        self.assertGreater(diag['rejected']['unsupported_claim'],0)

    def test_scoring_projection_excludes_validation_and_shared_people(self):
        records={'people':[{'id':'exclusive'},{'id':'shared'},{'id':'validation'}],
                 'chats':[{'chat_id':'dev','person':'shared'},{'chat_id':'val','person':'validation'}],
                 'organisations':[], 'goals':[{'private':True}], 'open_loops':[{'private':True}]}
        result=project(records,{'dev'},{'exclusive'})
        self.assertEqual(result['people'],[{'id':'exclusive'}])
        self.assertEqual(result['chats'],[{'chat_id':'dev','person':'shared'}])
        self.assertEqual(result['goals'],[])
        self.assertEqual(result['open_loops'],[])

    def test_unique_cross_chat_mentions_are_evidence_not_identity(self):
        index=InputIndex([chat('a',['ME','p']),chat('b',['ME','q'])],
            [msg('own','a','p','Hello',name='Unique Robin'),
             msg('mention','b','q','Unique Robin handles deliveries.',ts=2,name='Other')],{})
        evidence=index.evidence('p',max_messages=10,max_chars=2000)
        self.assertIn('mention',[m['id'] for m in evidence['messages']])
        self.assertIn('mentions',evidence['evidence_sources']['mention'])
        self.assertEqual(index.people,['p','q'])

    def test_model_alias_merge_uses_targeted_evidence_reconciliation(self):
        index=self.index()
        class FakeModel:
            def call(self,instruction,payload,**kwargs):
                if 'proposals' in payload:
                    orgs=payload['proposals']['organisations']
                    return {'merges':[{'left':orgs[0]['id'],'right':orgs[1]['id'],'_evidence':['one']}]}
                org={'type':'company','side':'work','role':'owner','_confidence':.9,'_evidence':['one']}
                return {'chats':[],'organisations':[{**org,'id':'x','name':'Example Works'},
                                         {**org,'id':'y','name':'EW Company'}],
                        'people':[{'id':'p','circle':'staff','_evidence':['one'],'_confidence':.9,
                                   'links':[{'organisation':'x','link':'supports'},{'organisation':'y','link':'supports'}]}]}
        output,diagnostics=infer_relationships(index,FakeModel())
        self.assertEqual(len(output['organisations']),1)
        self.assertEqual(len(output['people'][0]['links']),1)
        self.assertTrue(any(k.startswith('reconcile-') for k in diagnostics['selected_evidence']))


if __name__=='__main__':
    unittest.main()
