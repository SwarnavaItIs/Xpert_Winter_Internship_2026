"""Endpoint-only client. Never log prompts, responses, keys, or provider errors."""
import json
import http.client
import math
from email.utils import parsedate_to_datetime
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


class ModelError(Exception):
    pass


def strict_json(text):
    def pairs(items):
        out={}
        for key,value in items:
            if key in out:raise ValueError('Duplicate JSON key')
            out[key]=value
        return out
    def invalid(value):
        raise ValueError('Non-finite JSON number')
    def number(value):
        result=float(value)
        if not math.isfinite(result):raise ValueError('Non-finite JSON number')
        return result
    return json.loads(text,object_pairs_hook=pairs,parse_constant=invalid,parse_float=number)


def setting(name,default,low,high):
    try:value=int(os.environ.get(name,str(default)))
    except ValueError:raise ModelError('Invalid '+name) from None
    if not low<=value<=high:raise ModelError('Out of range '+name)
    return value


def request_stage(payload):
    """Only fixed labels reach diagnostics; never copy an arbitrary payload value."""
    for key, allowed in (('stage', ('goals','verify_goal','help_facts','verify_help')),
                         ('task_stage', ('candidates','repair_candidates')), ('lifecycle_stage', ('propose','verify'))):
        value=payload.get(key)
        if value in allowed:return key+':'+value
    if 'evidence' in payload and 'seed_circles' in payload:return 'relationships'
    if 'proposals' in payload and 'messages' in payload:return 'relationship_reconciliation'
    return 'other'


def verdict_response_format(payload):
    """Constrain verification syntax, while callers still check evidence locally."""
    stage=payload.get('stage')
    if stage not in ('verify_goal','verify_help','help_facts'):
        return {'type':'json_object'}
    if stage=='help_facts':
        facts={k:{'type':'boolean'} for k in ('requested_help','sensitive','personal_attendance',
            'delegated','redundant','opt_out','useful_preparation')}
        facts.update(evidence={'type':'array','items':{'type':'string'}},
            step={'type':'string','enum':['prepare','send','book','purchase','remind','leave']},
            reason={'type':'string'})
        properties={'facts':{'type':'object','properties':facts,'required':list(facts),
                             'additionalProperties':False}}
    else:
        properties={'supported':{'type':'boolean'}}
        if stage=='verify_goal':
            properties['equivalent_index']={'type':['integer','null']}
    if 'items' in payload:
        ids=[item['item_id'] for item in payload['items']]
        properties={'item_id':{'type':'string','enum':ids},**properties}
        item_schema={'type':'object','properties':properties,'required':list(properties),
                     'additionalProperties':False}
        properties={'items':{'type':'array','items':item_schema,'minItems':len(ids),'maxItems':len(ids)}}
    return {'type':'json_schema','json_schema':{
        'name':stage,'strict':True,'schema':{
            'type':'object','properties':properties,
            'required':list(properties),'additionalProperties':False}}}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ModelError('Endpoint redirect refused')


class Client:
    def __init__(self, deadline, usage_path=None):
        required = ('LLM_BASE_URL', 'LLM_API_KEY', 'LLM_MODEL')
        if any(not os.environ.get(k) for k in required):
            raise ModelError('Set LLM_BASE_URL, LLM_API_KEY and LLM_MODEL')
        self.base, self.key, self.model = (os.environ[k] for k in required)
        url = urllib.parse.urlsplit(self.base)
        if url.scheme != 'https' or not url.hostname or url.username or url.query or url.fragment:
            raise ModelError('LLM_BASE_URL must be an HTTPS API base URL')
        self.deadline = deadline
        self.usage = []
        self.usage_path = Path(usage_path) if usage_path else None
        self.max_requests=setting('LLM_MAX_REQUESTS',1000,1,2000)
        self.max_token_units=setting('LLM_MAX_TOKEN_UNITS',20000000,1,50000000)
        selected_route=url.hostname=='openrouter.ai' and self.model=='nvidia/nemotron-3.5-super-vl:free'
        self.request_timeout=setting('LLM_REQUEST_TIMEOUT_SECONDS',90 if selected_route else 45,1,90)
        openrouter_free=url.hostname=='openrouter.ai' and self.model.endswith(':free')
        self.request_interval=setting('LLM_MIN_REQUEST_INTERVAL_SECONDS',4 if openrouter_free else 0,0,60)
        self.next_request_at=0.0
        # This selected route advertises structured_outputs. Other compatible
        # endpoints keep JSON-object mode unless explicitly configured.
        schema_default=int(selected_route)
        self.schema_verdicts=setting('LLM_JSON_SCHEMA_VERDICTS',schema_default,0,1)==1
        # Match the user's successful live configuration when only the three
        # required environment variables are supplied by the grader.
        self.disable_reasoning=setting('LLM_DISABLE_REASONING',int(selected_route),0,1)==1
        if self.disable_reasoning and (url.hostname != 'openrouter.ai' or
                self.model != 'nvidia/nemotron-3.5-super-vl:free'):
            raise ModelError('LLM_DISABLE_REASONING is only verified for the Nemotron 3.5 Super VL free OpenRouter route')
        self.reserved_token_units=0
        self.terminal_failures=0
        self.logical_requests=0
        # Ignore ambient proxies: data goes directly to the configured endpoint.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def record(self, entry):
        self.usage.append(entry)
        if os.environ.get('XPERT_PROGRESS')=='1':
            print('Model request '+str(entry['request_id'])+'.'+str(entry['attempt'])+': '+
                  entry['stage']+' '+entry['status']+' ('+str(entry['seconds'])+'s)',file=sys.stderr,flush=True)
        if self.usage_path is not None:
            self.usage_path.parent.mkdir(parents=True, exist_ok=True)
            with self.usage_path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')

    def call(self, instruction, payload, max_tokens=12000):
        self.logical_requests+=1
        try:
            return self._call(instruction,payload,max_tokens)
        except ModelError:
            self.terminal_failures+=1
            raise

    def _call(self, instruction, payload, max_tokens):
        if self.terminal_failures>=3:
            raise ModelError('Circuit open after repeated terminal request failures')
        if type(max_tokens) is not int or not 1<=max_tokens<=12000:
            raise ModelError('Invalid output token ceiling')
        # Lossless JSON compaction: no messages, fields or verification are removed.
        request_body = {'model': self.model, 'messages': [
            {'role': 'system', 'content': instruction},
            {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False,separators=(',',':'))}
        ], 'response_format': {'type': 'json_object'}, 'max_tokens': max_tokens}
        if self.schema_verdicts:
            request_body['response_format']=verdict_response_format(payload)
        # Opt-in diagnostic experiment; keep all evidence and output caps unchanged.
        # "exclude" would only hide reasoning, not stop it consuming the budget.
        if self.disable_reasoning:request_body['reasoning']={'enabled':False}
        body = json.dumps(request_body,ensure_ascii=False,separators=(',',':')).encode()
        if len(body) > 120000:
            raise ModelError('Model request exceeds configured context safety ceiling')
        for attempt in range(3):
            now=time.monotonic()
            pacing_wait=max(0.0,self.next_request_at-now) if self.request_interval else 0.0
            if pacing_wait:
                if pacing_wait+5>=self.deadline-now:
                    raise ModelError('Request pacing exceeds runtime budget')
                time.sleep(pacing_wait)
                now=time.monotonic()
            remaining = self.deadline - now
            if remaining < 5:
                raise ModelError('Runtime budget exhausted')
            # Reserve a conservative byte-based unit bound for EVERY attempt,
            # including failures with unknown usage. This is not a dollar estimate.
            reserve=len(body)+max_tokens
            if len(self.usage)>=self.max_requests or self.reserved_token_units+reserve>self.max_token_units:
                raise ModelError('Run request/token-unit budget exhausted')
            self.reserved_token_units+=reserve
            started = time.monotonic()
            self.next_request_at=started+self.request_interval
            event = {'attempt': attempt+1, 'prompt_tokens': None, 'completion_tokens': None,
                     'total_tokens': None, 'status': 'pending','request_bytes':len(body),
                     'stage':request_stage(payload),'max_tokens':max_tokens,
                     'request_id':self.logical_requests,
                     'finish_reason':'missing','reasoning_tokens':None,
                     'reasoning_disabled':self.disable_reasoning,
                     'request_timeout_seconds':self.request_timeout,
                     'pacing_wait_seconds':round(pacing_wait,3),
                     'remaining_run_seconds':round(remaining,3)}
            delay = 2 ** attempt
            request = urllib.request.Request(self.base.rstrip('/') + '/chat/completions', data=body,
                headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'})
            try:
                with self.opener.open(request, timeout=min(self.request_timeout, remaining)) as response:
                    raw=response.read(2000001)
                    if len(raw)>2000000:
                        raise ModelError('Model response exceeds byte ceiling')
                    received=time.monotonic()
                    result = strict_json(raw)
                if not isinstance(result,dict):raise ValueError('Invalid envelope')
                usage = result.get('usage') or {}
                if not isinstance(usage,dict):raise ValueError('Invalid usage')
                event.update({k: usage.get(k) if type(usage.get(k)) is int and usage[k] >= 0 else None
                              for k in ('prompt_tokens','completion_tokens','total_tokens')})
                details=usage.get('completion_tokens_details')
                reasoning=details.get('reasoning_tokens') if isinstance(details,dict) else None
                event['reasoning_tokens']=reasoning if type(reasoning) is int and reasoning>=0 else None
                choice = result['choices'][0]
                if not isinstance(choice,dict):raise ValueError('Invalid choice')
                finish=choice.get('finish_reason')
                event['finish_reason']=finish if finish in ('stop','length','content_filter','tool_calls','function_call','error') else 'missing' if finish is None else 'unknown'
                if received>min(self.deadline,started+self.request_timeout):
                    event['failure_kind']='run_deadline' if received>self.deadline else 'request_deadline'
                    raise ModelError('Model response exceeded time budget ('+event['failure_kind']+')')
                if finish != 'stop':
                    event['failure_kind']='finish_reason'
                    raise ModelError('Model response incomplete or filtered (finish_reason='+event['finish_reason']+
                                     ', stage='+event['stage']+', max_tokens='+str(max_tokens)+')')
                output = strict_json(choice['message']['content'])
                if not isinstance(output, dict):
                    raise ValueError('Model returned a non-object')
                event['status'] = 'success'
                return output
            except urllib.error.HTTPError as exc:
                event.update(status='http_error', http_status=exc.code)
                if exc.code not in (429, 500, 502, 503, 504):
                    raise ModelError('Model HTTP error ' + str(exc.code)) from None
                if exc.code==429:
                    event['failure_kind']='rate_limit'
                    delay=10*2**attempt
                try:
                    retry_after=exc.headers.get('Retry-After','0')
                    try:retry=float(retry_after)
                    except ValueError:retry=parsedate_to_datetime(retry_after).timestamp()-time.time()
                    if not math.isfinite(retry):raise ValueError('Invalid retry delay')
                    delay=max(delay,retry)
                except (TypeError, ValueError, AttributeError):
                    pass
                if delay>60 or delay+5>=self.deadline-time.monotonic():
                    raise ModelError('Rate-limit retry exceeds available wait budget') from None
            except (urllib.error.URLError, TimeoutError, OSError,http.client.IncompleteRead) as exc:
                event['status'] = 'transport_error'
                event['failure_kind']='transport_timeout' if isinstance(exc,TimeoutError) or isinstance(getattr(exc,'reason',None),TimeoutError) else 'transport_failure'
            except (ValueError, KeyError, IndexError, TypeError):
                event['status'] = 'invalid_response'
            except ModelError:
                event['status'] = 'invalid_response_or_redirect'
                raise
            finally:
                event['seconds'] = round(time.monotonic()-started, 3)
                self.record(event)
            if attempt < 2:
                time.sleep(min(delay, max(0, self.deadline - time.monotonic())))
        raise ModelError('Model request failed after three attempts')
