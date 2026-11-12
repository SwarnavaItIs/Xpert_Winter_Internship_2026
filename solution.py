"""Batched evidence extraction and global reconciliation; no answer-file access."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from llm import Client, ModelError
from validation import validate, InvalidOutput
from inputs import load_index
from relationships import infer_relationships
from task_candidates import extract_candidates
from task_lifecycle import reconcile_tasks
from goals_help import apply_goals_help, GOALS


def read_input(folder):
    index = load_index(folder)
    return index.chats, index.messages, index.seed


def runtime_budget():
    try:
        seconds=int(os.environ.get('LLM_RUNTIME_SECONDS','1680'))
    except ValueError:
        raise ModelError('LLM_RUNTIME_SECONDS must be an integer') from None
    if not 60<=seconds<=1680:
        raise ModelError('LLM_RUNTIME_SECONDS must be between 60 and 1680')
    return seconds


def batches(messages, limit=24000):
    current, size = [], 0
    for message in messages:
        n = len(json.dumps(message, ensure_ascii=False))
        if n > limit:
            raise ValueError('Single message exceeds context budget')
        if current and size + n > limit:
            yield current
            current, size = [], 0
        current.append(message)
        size += n
    if current:
        yield current


def atomic_publish(path,write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, prefix='.atomic-',suffix='.tmp.json',delete=False) as f:
            temporary = Path(f.name)
            write(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


def atomic_write(path,value):
    atomic_publish(path,lambda f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False))


def atomic_text(path,text):
    atomic_publish(path,lambda f:f.write(text))


STAGES = {'startup','input','relationships','relationship_output','task_extraction',
          'lifecycle','goals_help','final_validation','completion_gate','publication'}


def report_failure(output, exc):
    """Code locations only: never format exception messages, source text or locals."""
    status_path = Path(str(output)+'.run-status.json')
    stage = 'unknown'
    try:
        previous = json.loads(status_path.read_text(encoding='utf-8'))
        if isinstance(previous,dict) and previous.get('stage') in STAGES:
            stage = previous['stage']
    except (OSError, ValueError, TypeError):
        pass
    frames = []
    frame = exc.__traceback__
    root = Path(__file__).resolve().parent
    while frame is not None:
        source = Path(frame.tb_frame.f_code.co_filename).resolve()
        if source.parent == root and source.suffix == '.py':
            frames.append({'file':source.name,'line':frame.tb_lineno})
        frame = frame.tb_next
    location = frames[-1] if frames else {'file':'external','line':None}
    details = {'status':'failed','stage':stage,'error_type':type(exc).__name__,
               'source_file':location['file'],'source_line':location['line'],
               'code_frames':frames}
    try:
        atomic_write(status_path, details)
    except OSError:
        # A disk failure must not expose the original exception through a traceback.
        pass
    print('Inference failed: '+json.dumps(details, separators=(',',':')),
          file=sys.stderr)


def completion_report(client,relationships,tasks,lifecycle,help_):
    reasons=[]
    if getattr(client,'terminal_failures',0):reasons.append('terminal_model_request_failure')
    counts=lifecycle.get('counts',{})
    for key in ('model_unavailable','model_budget_deferred','model_tasks_deferred','model_proposals_truncated',
                'oversized_proof_deferred','malformed_proposals','malformed_verdicts'):
        if counts.get(key):reasons.append('lifecycle_'+key)
    for key in ('model_failures','deferred_model_requests','malformed_responses'):
        if help_.get(key):reasons.append('goals_help_'+key)
    task_coverage=tasks.get('coverage',{})
    if task_coverage.get('repair_contract_failures'):reasons.append('task_repair_contract_failure')
    if task_coverage.get('chunks_expected',0)!=task_coverage.get('chunks_completed',0):reasons.append('task_chunks_incomplete')
    if any(c['expected']!=c['completed'] for c in help_.get('coverage',{}).values()):reasons.append('goals_help_requests_incomplete')
    rejected=sum(tasks.get('rejected',{}).values())
    if rejected>tasks.get('candidates',0):reasons.append('majority_task_candidates_rejected')
    nonseed=[p for p in relationships.get('people',{}).values() if not p.get('_seed')]
    fallback=sum('schema-required fallback' in p.get('_reason','') for p in nonseed)
    if len(nonseed)>=5 and fallback/len(nonseed)>=.5:reasons.append('majority_people_forced_fallback')
    if relationships.get('selected_evidence',{}).get('reconciliation_jobs_not_sent',0):reasons.append('relationship_reconciliation_deferred')
    return {'status':'degraded' if reasons else 'complete','reasons':reasons,
            'nonseed_people':len(nonseed),'forced_person_fallbacks':fallback,
            'model_attempts':len(client.usage),'token_units_reserved':getattr(client,'reserved_token_units',None),
            'coverage':{'task_extraction':task_coverage,'task_rejections':tasks.get('rejected',{}),
                        'lifecycle':counts,
                        'goals_help':help_.get('coverage',{}),
                        'goals_help_failures':{k:help_.get(k,0) for k in
                            ('model_failures','deferred_model_requests','malformed_responses')},
                        'goals_help_malformed_reasons':help_.get('malformed_response_reasons',{})}}


def infer(folder, output, client):
    def stage(name):
        atomic_write(str(output)+'.run-status.json',{'status':'running','stage':name})
        if os.environ.get('XPERT_PROGRESS')=='1':
            print('Stage: '+name,file=sys.stderr,flush=True)
    stage('input')
    index = load_index(folder)
    chats, messages, seed = index.chats, index.messages, index.seed
    stage('relationships')
    state, relationship_diagnostics = infer_relationships(index, client)
    stage('relationship_output')
    atomic_write(str(output)+'.relationships.json', relationship_diagnostics)
    stage('task_extraction')
    tasks, task_diagnostics = extract_candidates(index,state,client,relationship_diagnostics,goal_instruction=GOALS)
    atomic_write(str(output)+'.candidates.json',{'candidates':tasks,'diagnostics':task_diagnostics})
    stage('lifecycle')
    state['open_loops'], lifecycle_diagnostics = reconcile_tasks(index,tasks,client,batch_model=True)
    atomic_write(str(output)+'.lifecycle.json',lifecycle_diagnostics)
    stage('goals_help')
    state, goal_help_diagnostics = apply_goals_help(index,state,client,goal_batches=task_diagnostics.get('goal_batches'))
    atomic_write(str(output)+'.goals-help.json',goal_help_diagnostics)
    circles = seed.get('circles') or {}
    for person in state['people']:
        if person['id'] in circles:
            person['circle'] = circles[person['id']]
    stage('final_validation')
    validate(state, chats, messages)
    stage('completion_gate')
    report=completion_report(client,relationship_diagnostics,task_diagnostics,lifecycle_diagnostics,goal_help_diagnostics)
    atomic_write(str(output)+'.completion.json',report)
    if report['status']!='complete':
        atomic_write(str(output)+'.run-status.json',{**report,'stage':'completion_gate'})
        raise ModelError('Incomplete model coverage; diagnostics saved, final output not replaced')
    stage('publication')
    atomic_write(str(output) + '.usage.json', {'calls': client.usage})
    atomic_write(output, state)
    atomic_write(str(output)+'.run-status.json',report)


def main():
    if len(sys.argv) == 4 and sys.argv[1] == '--worker':
        try:
            atomic_write(sys.argv[3]+'.run-status.json',{'status':'running','stage':'startup'})
            infer(sys.argv[2], sys.argv[3], Client(time.monotonic() + runtime_budget(), sys.argv[3]+'.usage.jsonl'))
            return 0
        except (ModelError, InvalidOutput) as exc:
            report_failure(sys.argv[3], exc)
            print(str(exc), file=sys.stderr)
            return 1
        except Exception as exc:
            report_failure(sys.argv[3], exc)
            return 1
    if len(sys.argv) != 3:
        print('usage: solution.py <input_dir> <output_path>', file=sys.stderr)
        return 2
    try:
        return subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', *sys.argv[1:]],
                              timeout=runtime_budget()+60).returncode
    except ModelError as exc:
        print(str(exc),file=sys.stderr)
        return 1
    except subprocess.TimeoutExpired:
        atomic_write(sys.argv[2]+'.run-status.json',{'status':'failed','error_type':'process_timeout'})
        print('Runtime limit reached; no successful output produced.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
