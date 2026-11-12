"""Offline transport/coverage rehearsal. API replies are simulated, never live."""
import argparse
from collections import Counter
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from inputs import InputIndex
from llm import Client, ModelError
from solution import infer


class SimulatedEndpoint:
    def __init__(self):self.seen=set();self.calls=Counter();self.max_request_bytes=0
    def open(self,request,timeout):
        self.max_request_bytes=max(self.max_request_bytes,len(request.data))
        payload=json.loads(json.loads(request.data)['messages'][1]['content'])
        if payload.get('task_stage')=='candidates':
            self.calls['extraction']+=1
            self.seen.update(m['id'] for m in payload['chunk']['core'])
            response={'candidates':[],'goals':[]}
        elif payload.get('stage')=='goals':
            raise AssertionError('A second full-history scan was requested')
        else:
            self.calls['relationships']+=1
            response={'people':[],'organisations':[],'chats':[],'merges':[]}
        return io.BytesIO(json.dumps({'choices':[{'finish_reason':'stop','message':{
            'content':json.dumps(response)}}],
            'usage':{'prompt_tokens':0,'completion_tokens':0,'total_tokens':0}}).encode('utf-8'))


def rehearsal(message_count=17000,token_units=20000000):
    chats=[dict(chat_id='load-chat-'+str(i),name='Artificial load fixture',is_group=False,
                participants=['ME','load-peer-'+str(i)]) for i in range(20)]
    messages=[dict(id='load-message-'+str(i),chat_id=chats[i%20]['chat_id'],sender_id='ME',
                   sender_name='Artificial User',ts=1780000000+i,kind='text',quoted_id=None,
                   body='Synthetic retrospective note. '+('Observations about past context. '*5))
              for i in range(message_count)]
    index=InputIndex(chats,messages,{'circles':{'load-peer-'+str(i):'friend' for i in range(20)}})
    env={'LLM_BASE_URL':'https://simulated.invalid/v1','LLM_MODEL':'simulation',
         'LLM_API_KEY':'synthetic-placeholder','LLM_MAX_TOKEN_UNITS':str(token_units)}
    endpoint=SimulatedEndpoint()
    with tempfile.TemporaryDirectory() as temporary,patch.dict(os.environ,env,clear=True),\
         patch('solution.load_index',return_value=index):
        client=Client(time.monotonic()+1680);client.opener=endpoint
        output=Path(temporary)/'output.json'
        started=time.perf_counter()
        try:
            infer('unused',output,client)
            status='passed'
        except ModelError:
            status='budget_or_model_failure'
        report={'simulation_only':True,'status':status,'messages':message_count,
            'messages_sent_as_core':len(endpoint.seen),'attempts':len(client.usage),
            'by_stage':dict(endpoint.calls),'reserved_units':client.reserved_token_units,
            'reservation_limit':token_units,'max_request_bytes':endpoint.max_request_bytes,
            'output_created':output.exists(),'local_seconds':round(time.perf_counter()-started,3),
            'limitations':'Simulated empty extractions on artificial retrospective messages. '
            'No live latency, provider quota, model quality, task-rich lifecycle load or monetary cost measured.'}
        if status=='passed' and len(endpoint.seen)!=message_count:
            raise AssertionError('Extraction coverage incomplete')
        return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--messages',type=int,default=17000)
    parser.add_argument('--token-units',type=int,default=20000000)
    args=parser.parse_args()
    if not 1<=args.messages<=50000:parser.error('--messages must be 1..50000')
    report=rehearsal(args.messages,args.token_units)
    print(json.dumps(report,indent=2))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':raise SystemExit(main())
