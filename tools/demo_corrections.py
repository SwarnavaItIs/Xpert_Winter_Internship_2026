"""An actual local CLI demonstration using entirely synthetic fixtures."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from solution import atomic_write,atomic_text
from inputs import InputIndex
from validation import validate


def fixture():
    chats=[dict(chat_id='work-room',name='Work group',is_group=True,participants=['ME','adult']),
           dict(chat_id='child-room',name='Child',is_group=False,participants=['ME','child']),
           dict(chat_id='other-room',name='Other',is_group=False,participants=['ME','other'])]
    messages=[dict(id='request',chat_id='work-room',sender_id='ME',sender_name='User',ts=1793500000,kind='text',quoted_id=None,
                   body='Please help me draft the proposal for this delegated work.'),
              dict(id='closing',chat_id='work-room',sender_id='adult',sender_name='Alex',ts=1793500100,kind='text',quoted_id='request',body='The final proposal was sent.'),
              dict(id='hello-child',chat_id='child-room',sender_id='child',sender_name='Sam',ts=1793500101,kind='text',quoted_id=None,body='Hello.'),
              dict(id='other-request',chat_id='other-room',sender_id='ME',sender_name='User',ts=1793500200,kind='text',quoted_id=None,body='I will draft the report.'),
              dict(id='other-hello',chat_id='other-room',sender_id='other',sender_name='Alex',ts=1793500201,kind='text',quoted_id=None,body='Thanks.')]
    seed={'circles':{'adult':'staff','child':'child','other':'client'}}
    base={'people':[{'id':'adult','circle':'staff','links':[{'organisation':'org','link':'supports'}],'looks_after':['child']},
                    {'id':'child','circle':'child'},{'id':'other','circle':'client'}],
          'organisations':[dict(id='org',name='Synthetic Co',type='company',side='work',role='owner')],
          'chats':[{'chat_id':'work-room','organisation':'org'},{'chat_id':'child-room','person':'child'},{'chat_id':'other-room','person':'other'}],
          'goals':[],'open_loops':[dict(title='Draft proposal',type='followup',owner='adult',chat_id='work-room',source_msg='request',
                                      status='done',closed_by='closing',due=None,help=None),
                                  dict(title='Draft report',type='commitment',owner='ME',chat_id='other-room',source_msg='other-request',
                                       status='open',closed_by=None,due=None,help={'action':'assist','next_step':'Draft privately','needs_yes':False})]}
    index=InputIndex(chats,messages,seed);validate(base,chats,messages)
    return index,base


def main():
    destination=ROOT/'.private/correction-demo'
    if destination.exists():raise SystemExit('Demo directory exists; preserve it and use the CLI directly.')
    index,base=fixture();folder=destination/'input';folder.mkdir(parents=True)
    for name,rows in [('chats',index.chats),('messages',index.messages)]:
        atomic_text(folder/(name+'.jsonl'),'\n'.join(json.dumps(r) for r in rows)+'\n')
    atomic_write(folder/'seed.json',index.seed);atomic_write(destination/'base.json',base)
    steps=[['--text','This person is my business partner, not a client.','--person-id','adult'],
           ['--text','This group belongs to my child.','--chat-id','work-room'],
           ['--text','That task is still open.','--source','request'],
           ['--text','Leave this type of delegated task alone.','--task-type','followup']]
    for i,step in enumerate(steps,1):
        process=subprocess.run([sys.executable,str(ROOT/'correction_cli.py'),str(folder),str(destination/'base.json'),
            str(destination/'ledger.json'),str(destination/f'corrected-{i}.json'),*step],capture_output=True,text=True)
        if process.returncode:raise SystemExit('Synthetic correction step failed; inspect local fixtures.')
    out=json.loads((destination/'corrected-4.json').read_text(encoding='utf-8'))
    assert next(p for p in out['people'] if p['id']=='adult')['circle']=='partner'
    assert next(c for c in out['chats'] if c['chat_id']=='work-room')['person']=='child'
    assert out['open_loops'][0]['status']=='open' and out['open_loops'][0]['help']['action']=='leave'
    assert out['open_loops'][1]==base['open_loops'][1]
    atomic_write(destination/'result.json',{'synthetic':True,'cli_steps_completed':4,'unrelated_task_unchanged':True,'validated':True})
    print('Four synthetic correction CLI steps completed; unrelated task preserved and output valid.')


if __name__=='__main__':main()
