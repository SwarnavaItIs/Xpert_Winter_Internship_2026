"""Apply lifecycle reconciliation to saved internal candidates and write schema output."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from inputs import load_index
from llm import Client,ModelError
from relationships import infer_relationships
from solution import atomic_write,completion_report
from task_lifecycle import reconcile_tasks
from validation import validate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('candidates_path')
    parser.add_argument('output_path')
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--relationships',type=Path)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not args.worker:
        try:
            return subprocess.run([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker'],timeout=1740).returncode
        except subprocess.TimeoutExpired:
            print('Lifecycle runtime limit reached.',file=sys.stderr)
            return 1
    try:
        index=load_index(args.input_dir)
        candidates=json.loads(Path(args.candidates_path).read_text(encoding='utf-8'))['candidates']
        client=None if args.offline else Client(time.monotonic()+1680,args.output_path+'.usage.jsonl')
        if args.relationships:
            relationships=json.loads(args.relationships.read_text(encoding='utf-8'))
            validate(relationships,index.chats,index.messages)
            relationship_diagnostics={}
        else:
            relationships,relationship_diagnostics=infer_relationships(index,client)
        loops,diagnostics=reconcile_tasks(index,candidates,client)
        output={**relationships,'open_loops':loops}
        validate(output,index.chats,index.messages)
        atomic_write(args.output_path+'.lifecycle.json',diagnostics)
        if client is not None:
            report=completion_report(client,relationship_diagnostics, {},diagnostics,{})
            atomic_write(args.output_path+'.completion.json',report)
            if report['status']!='complete':raise ModelError('Lifecycle coverage incomplete')
        atomic_write(args.output_path,output)
        print('Lifecycle output validated; original-message evidence saved privately.')
        return 0
    except ModelError as exc:
        print(str(exc),file=sys.stderr)
        return 1
    except Exception:
        print('Lifecycle reconciliation failed; data omitted for privacy.',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
