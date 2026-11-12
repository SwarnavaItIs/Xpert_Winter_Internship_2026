"""Additional local correction interface; the official run.sh contract is unchanged."""
import argparse
import json
from pathlib import Path
import sys

from inputs import load_index
from corrections import Clarification,new_ledger,normalize_request,replay,readable_diff,text_request,fingerprint
from solution import atomic_write,atomic_text


def main():
    p=argparse.ArgumentParser()
    p.add_argument('input_dir');p.add_argument('base_output');p.add_argument('ledger');p.add_argument('output')
    group=p.add_mutually_exclusive_group();group.add_argument('--request');group.add_argument('--text')
    for name in ('person-id','chat-id','child-id','source','title','owner','task-type','owner-circle'):p.add_argument('--'+name)
    args=p.parse_args()
    committed=False
    try:
        if Path(args.base_output).resolve()==Path(args.output).resolve():raise ValueError('Output must not overwrite inferred base')
        if len({Path(p).resolve() for p in (args.base_output,args.output,args.ledger)})!=3:raise ValueError('Correction paths must be distinct')
        index=load_index(args.input_dir);base=json.loads(Path(args.base_output).read_text(encoding='utf-8'))
        ledger_path=Path(args.ledger)
        ledger=json.loads(ledger_path.read_text(encoding='utf-8')) if ledger_path.exists() else new_ledger(index,base)
        previous_diff=ledger.get('last_diff')
        # A single atomic ledger bundle is the source of truth. Output/diff are derived exports.
        if 'ledger' in ledger:ledger=ledger['ledger']
        before,_=replay(index,base,ledger)
        request=None
        if args.request:request=json.loads(Path(args.request).read_text(encoding='utf-8'))
        elif args.text:request=text_request(args.text,args.person_id,args.chat_id,args.child_id,args.source,args.title,args.owner,args.task_type,args.owner_circle)
        if request is not None:
            ledger['events'].append(normalize_request(index,before,request))
        after,diagnostics=replay(index,base,ledger)
        diff=previous_diff if request is None and previous_diff is not None else readable_diff(before,after)
        atomic_write(ledger_path,{'ledger':ledger,'materialized_output':after,'output_sha256':fingerprint(after),
                                  'last_diff':diff,'diagnostics':diagnostics})
        committed=True
        atomic_write(args.output,after)
        # JSON contains a readable Markdown string; no raw data is printed to console.
        atomic_text(str(args.output)+'.diff.md',diff)
        print('Correction committed locally; valid output and private before/after diff exported.')
        return 0
    except Clarification as exc:
        atomic_write(str(args.output)+'.clarification.json',{'question':exc.question,'choices':exc.choices})
        print('Clarification required: '+exc.question+' Choices saved privately; no correction committed.',file=sys.stderr)
        return 2
    except Exception:
        print('Correction committed; export failed. Replay without a new request to recover.' if committed else
              'Correction failed; inferred base unchanged. Check references/ledger; data omitted.',file=sys.stderr)
        return 1


if __name__=='__main__':raise SystemExit(main())
