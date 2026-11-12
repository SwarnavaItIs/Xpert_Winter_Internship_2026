"""Candidate-stage runner. Its output is INTERNAL, not the final output schema."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from inputs import load_index
from llm import Client, ModelError
from relationships import infer_relationships
from solution import atomic_write,completion_report
from task_candidates import extract_candidates, candidate_loop
from validation import validate


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('candidates_path')
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--relationships',type=Path)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not args.worker:
        try:
            return subprocess.run([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:],'--worker'],timeout=1740).returncode
        except subprocess.TimeoutExpired:
            print('Task extraction runtime limit reached.',file=sys.stderr)
            return 1
    try:
        index=load_index(args.input_dir)
        client=None if args.offline else Client(time.monotonic()+1680,args.candidates_path+'.usage.jsonl')
        if args.relationships:
            relationships=json.loads(args.relationships.read_text(encoding='utf-8'))
            validate(relationships,index.chats,index.messages)
            relation_diagnostics=None
        else:
            relationships,relation_diagnostics=infer_relationships(index,client)
        candidates,diagnostics=extract_candidates(index,relationships,client,relation_diagnostics)
        preview={**relationships,'goals':[],'open_loops':[candidate_loop(c) for c in candidates]}
        validate(preview,index.chats,index.messages)
        if client is not None:
            report=completion_report(client,relation_diagnostics or {},diagnostics,{}, {})
            atomic_write(args.candidates_path+'.completion.json',report)
            if report['status']!='complete':raise ModelError('Extraction coverage incomplete')
        atomic_write(args.candidates_path,{'candidates':candidates,'diagnostics':diagnostics})
        atomic_write(args.candidates_path+'.preview.json',preview)
        print('Candidate references and provisional schema preview validated. Lifecycle not evaluated.')
        return 0
    except ModelError as exc:
        print(str(exc),file=sys.stderr)
        return 1
    except Exception:
        print('Task extraction failed; data omitted for privacy.',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
