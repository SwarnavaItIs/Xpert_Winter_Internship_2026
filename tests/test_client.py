import io
import json
import os
from pathlib import Path
import time
import unittest
from unittest.mock import Mock, patch
import urllib.error

from llm import Client, ModelError

ENV={'LLM_BASE_URL':'https://endpoint.invalid/v1','LLM_API_KEY':'synthetic-secret','LLM_MODEL':'synthetic-model'}


def response(content='{"ok":true}'):
    return io.BytesIO(json.dumps({'choices':[{'message':{'content':content},'finish_reason':'stop'}],
        'usage':{'prompt_tokens':12,'completion_tokens':4,'total_tokens':16}}).encode())


class ClientTests(unittest.TestCase):
    def test_selected_model_defaults_match_the_successful_windows_configuration(self):
        env={**ENV,'LLM_BASE_URL':'https://openrouter.ai/api/v1',
             'LLM_MODEL':'nvidia/nemotron-3.5-super-vl:free'}
        with patch.dict(os.environ,env,clear=True):
            client=Client(time.monotonic()+90)
        self.assertTrue(client.disable_reasoning)
        self.assertTrue(client.schema_verdicts)
        self.assertEqual(client.request_timeout,90)
        with patch.dict(os.environ,ENV,clear=True):
            client=Client(time.monotonic()+90)
        self.assertFalse(client.disable_reasoning)
        self.assertEqual(client.request_timeout,45)

    def test_selected_route_uses_strict_verdict_schema_and_can_opt_out(self):
        env={**ENV,'LLM_BASE_URL':'https://openrouter.ai/api/v1',
             'LLM_MODEL':'nvidia/nemotron-3.5-super-vl:free'}
        for stage in ('verify_help','verify_goal'):
            with patch.dict(os.environ,env,clear=True):
                client=Client(time.monotonic()+90);client.opener=Mock()
                client.opener.open.return_value=response('{"supported":false,"equivalent_index":null}' if stage=='verify_goal' else '{"supported":false}')
                client.call('Verify',{'stage':stage})
                sent=json.loads(client.opener.open.call_args.args[0].data)
                fmt=sent['response_format']
                self.assertEqual(fmt['type'],'json_schema')
                schema=fmt['json_schema']['schema']
                self.assertEqual(schema['properties']['supported'],{'type':'boolean'})
                self.assertFalse(schema['additionalProperties'])
                self.assertEqual(set(schema['required']),{'supported','equivalent_index'} if stage=='verify_goal' else {'supported'})
        with patch.dict(os.environ,{**env,'LLM_JSON_SCHEMA_VERDICTS':'0'},clear=True):
            client=Client(time.monotonic()+90);client.opener=Mock();client.opener.open.return_value=response()
            client.call('Verify',{'stage':'verify_help'})
            self.assertEqual(json.loads(client.opener.open.call_args.args[0].data)['response_format'],{'type':'json_object'})

    def test_free_route_paces_attempts_and_preserves_logical_retry(self):
        clock=[100.0]
        waits=[]
        def sleep(seconds):
            waits.append(seconds);clock[0]+=seconds
        env={**ENV,'LLM_BASE_URL':'https://openrouter.ai/api/v1','LLM_MODEL':'synthetic/model:free'}
        with patch.dict(os.environ,env,clear=True),patch('llm.time.monotonic',side_effect=lambda:clock[0]),patch('llm.time.sleep',side_effect=sleep):
            client=Client(300);client.opener=Mock()
            client.opener.open.side_effect=[response(),urllib.error.HTTPError(env['LLM_BASE_URL'],429,'private',{},None),response()]
            client.call('First',{});client.call('Second',{})
        self.assertEqual(waits,[4.0,10.0])
        self.assertEqual([u['request_id'] for u in client.usage],[1,2,2])
        self.assertEqual(client.terminal_failures,0)
        self.assertEqual(client.usage[1]['failure_kind'],'rate_limit')
        self.assertEqual(client.usage[1]['pacing_wait_seconds'],4.0)

    def test_pacing_does_not_start_a_request_past_run_deadline(self):
        with patch.dict(os.environ,{**ENV,'LLM_MIN_REQUEST_INTERVAL_SECONDS':'4'},clear=True):
            client=Client(time.monotonic()+90);client.opener=Mock();client.opener.open.return_value=response()
            client.call('First',{})
            client.deadline=time.monotonic()+5
            with self.assertRaisesRegex(ModelError,'pacing'),patch('llm.time.sleep') as sleep:
                client.call('Second',{})
            self.assertEqual(client.opener.open.call_count,1)
            sleep.assert_not_called()

    def test_incomplete_response_records_safe_reasoning_diagnostics(self):
        for finish, expected in [('length','length'),('content_filter','content_filter'),
                                 (None,'missing'),('private provider text','unknown')]:
            with self.subTest(finish=expected),patch.dict(os.environ,ENV,clear=True):
                client=Client(time.monotonic()+90)
                raw={'choices':[{'finish_reason':finish,'message':{
                    'content':'private output', 'reasoning':'private reasoning'}}],
                    'usage':{'prompt_tokens':3218,'completion_tokens':4096,'total_tokens':7314,
                             'completion_tokens_details':{'reasoning_tokens':4090}}}
                client.opener=Mock()
                client.opener.open.return_value=io.BytesIO(json.dumps(raw).encode())
                with self.assertRaises(ModelError) as error:
                    client.call('private instruction',{'evidence':['private input'],'seed_circles':{}},max_tokens=4096)
                self.assertEqual(client.opener.open.call_count,1)
                event=client.usage[0]
                self.assertEqual(event['finish_reason'],expected)
                self.assertEqual(event['reasoning_tokens'],4090)
                self.assertEqual(event['max_tokens'],4096)
                self.assertEqual(event['stage'],'relationships')
                self.assertEqual(event['failure_kind'],'finish_reason')
                self.assertIn('seconds',event)
                self.assertNotIn('private',str(error.exception)+json.dumps(event))

    def test_reasoning_disable_is_opt_in_and_preserves_request_contract(self):
        for enabled in ('0','1'):
            env={**ENV,'LLM_BASE_URL':'https://openrouter.ai/api/v1',
                 'LLM_MODEL':'nvidia/nemotron-3.5-super-vl:free','LLM_DISABLE_REASONING':enabled}
            with patch.dict(os.environ,env,clear=True):
                client=Client(time.monotonic()+90);client.opener=Mock()
                client.opener.open.return_value=response()
                self.assertEqual(client.call('test',{'complete':'evidence'},max_tokens=4096),{'ok':True})
                sent=json.loads(client.opener.open.call_args.args[0].data)
                self.assertEqual(sent.get('reasoning'),{'enabled':False} if enabled=='1' else None)
                self.assertEqual(sent['max_tokens'],4096)
                self.assertEqual(sent['response_format'],{'type':'json_object'})
                self.assertEqual(json.loads(sent['messages'][1]['content']),{'complete':'evidence'})

    def test_unverified_reasoning_override_fails_before_network(self):
        for env in ({**ENV,'LLM_DISABLE_REASONING':'1'},
                    {**ENV,'LLM_BASE_URL':'https://openrouter.ai/api/v1','LLM_DISABLE_REASONING':'1'},
                    {**ENV,'LLM_DISABLE_REASONING':'not-a-number'}):
            with patch.dict(os.environ,env,clear=True),self.assertRaises(ModelError):
                Client(time.monotonic()+30)

    def test_arbitrary_stage_and_bad_usage_do_not_leak(self):
        with patch.dict(os.environ,ENV,clear=True):
            client=Client(time.monotonic()+30);client.opener=Mock()
            raw={'choices':[{'finish_reason':'stop','message':{'content':'{}'}}],
                 'usage':{'completion_tokens':'private','completion_tokens_details':{'reasoning_tokens':True}}}
            client.opener.open.return_value=io.BytesIO(json.dumps(raw).encode())
            client.call('test',{'stage':'private'})
            self.assertEqual(client.usage[0]['stage'],'other')
            self.assertIsNone(client.usage[0]['reasoning_tokens'])
            self.assertIsNone(client.usage[0]['completion_tokens'])
            self.assertNotIn('private',json.dumps(client.usage))

    def test_transient_retry_records_attempts_and_usage_without_payload(self):
        path=Path('.private/tests/client/usage.jsonl')
        with patch.dict(os.environ,ENV,clear=True),patch('llm.time.sleep') as sleep:
            client=Client(time.monotonic()+30,path)
            client.opener=Mock()
            error=urllib.error.HTTPError(ENV['LLM_BASE_URL'],429,'private error',{'Retry-After':'2'},None)
            client.opener.open.side_effect=[error,response()]
            self.assertEqual(client.call('synthetic prompt',{},max_tokens=64),{'ok':True})
            self.assertEqual(len(client.usage),2)
            self.assertEqual(client.usage[-1]['total_tokens'],16)
            self.assertEqual(client.usage[0]['http_status'],429)
            self.assertLessEqual(client.opener.open.call_args.kwargs['timeout'],30)
            self.assertTrue(sleep.called)
            rendered=path.read_text()
            self.assertNotIn('synthetic-secret',rendered)
            self.assertNotIn('synthetic prompt',rendered)
            self.assertNotIn('private error',rendered)

    def test_authentication_error_is_not_retried(self):
        with patch.dict(os.environ,ENV,clear=True):
            client=Client(time.monotonic()+30)
            client.opener=Mock()
            client.opener.open.side_effect=urllib.error.HTTPError(ENV['LLM_BASE_URL'],401,'secret response',{},None)
            with self.assertRaisesRegex(ModelError,'Model HTTP error 401'):
                client.call('test',{})
            self.assertEqual(client.opener.open.call_count,1)
            self.assertEqual(client.usage[0]['http_status'],401)

    def test_malformed_json_is_bounded_and_can_recover(self):
        with patch.dict(os.environ,ENV,clear=True),patch('llm.time.sleep'):
            client=Client(time.monotonic()+30)
            client.opener=Mock()
            client.opener.open.side_effect=[response('bad'),response('bad'),response()]
            self.assertEqual(client.call('test',{}),{'ok':True})
            self.assertEqual(client.opener.open.call_count,3)
            self.assertEqual(client.usage[0]['status'],'invalid_response')

    def test_deadline_prevents_request(self):
        with patch.dict(os.environ,ENV,clear=True):
            client=Client(time.monotonic()-1)
            client.opener=Mock()
            with self.assertRaisesRegex(ModelError,'Runtime budget'):
                client.call('test',{})
            client.opener.open.assert_not_called()


if __name__=='__main__':
    unittest.main()
