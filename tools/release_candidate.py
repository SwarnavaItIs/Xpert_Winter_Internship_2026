"""Freeze RC1 before one-shot holdout scoring. Never send data to a network."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from solution import atomic_write


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def worker():
    # Isolated Python, no site packages, no development imports or cached outputs.
    root = Path.cwd().resolve()
    stdlib = Path(os.__file__).resolve().parent
    def guard(event, args):
        if event.startswith('socket.') or event in ('subprocess.Popen', 'os.system'):
            raise PermissionError('Offline rehearsal denies networking and subprocesses')
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if not any(path == p or p in path.parents for p in (root, stdlib)):
                raise PermissionError('Read/write outside clean rehearsal denied')
    sys.addaudithook(guard)
    from inputs import load_index
    from relationships import infer_relationships
    from task_candidates import extract_candidates
    from task_lifecycle import reconcile_tasks
    from goals_help import apply_goals_help
    from validation import validate
    import check
    started = time.perf_counter()
    index = load_index(root/'input')
    timings = {}
    def timed(name, fn):
        start = time.perf_counter(); value = fn()
        timings[name] = time.perf_counter()-start
        return value
    rel, rd = timed('relationships', lambda: infer_relationships(index))
    candidates, _ = timed('extraction', lambda: extract_candidates(index, rel, None, rd))
    loops, _ = timed('lifecycle', lambda: reconcile_tasks(index, candidates))
    output, _ = timed('goals_help', lambda: apply_goals_help(index, {**rel, 'open_loops': loops}))
    validate(output, index.chats, index.messages)
    assert not check.validate(output, index.chats, index.messages)
    atomic_write(root/'output.json', output)
    atomic_write(root/'measurement.json', {'mode': 'offline fallback, not run.sh model pipeline',
        'seconds': time.perf_counter()-started, 'stage_seconds': timings,
        'tokens': 0, 'model': None, 'api_cost_usd': 0, 'validation': 'strict and official passed'})


def run():
    release = ROOT/'.private/releases/rc1'
    release.mkdir(parents=True, exist_ok=False)  # No accidental second holdout evaluation.
    source = release/'source'; source.mkdir()
    files = sorted(set([*ROOT.glob('*.py'), *ROOT.glob('*.sh'), ROOT/'schema.json',
                       * (ROOT/'tools').glob('*.py'), * (ROOT/'tests').glob('*.py')]))
    manifest = {p.relative_to(ROOT).as_posix(): digest(p) for p in files}
    for p in files:
        dest = source/p.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    split_path = ROOT/'.private/splits/day1-v1.json'
    from tools.day1 import freeze_split, load_inputs
    data = ROOT/'.private/official-day1/data/dev'
    freeze_split(data)
    atomic_write(release/'freeze.json', {'version': 'rc1', 'source_sha256': fingerprint,
        'files': manifest, 'split_sha256': digest(split_path), 'configuration': 'offline, client=None',
        'frozen_before_label_access': True})
    split = json.loads(split_path.read_text(encoding='utf-8'))
    chats, messages, seed = load_inputs(data)
    val = set(split['validation_chat_ids']); dev = set(split['development_chat_ids'])
    vc = [c for c in chats if c['chat_id'] in val]
    vm = [m for m in messages if m['chat_id'] in val]
    vp = {p for c in vc for p in c['participants']}-{'ME'}
    dp = {p for c in chats if c['chat_id'] in dev for p in c['participants']}-{'ME'}
    vs = {**seed, 'circles': {p: v for p, v in seed.get('circles', {}).items() if p in vp}}
    clean = release/'clean'; clean.mkdir()
    for p in source.glob('*'):
        if p.is_file(): shutil.copyfile(p, clean/p.name)
    shutil.copyfile(source/'tools/release_candidate.py', clean/'release_worker.py')
    inputs = clean/'input'; inputs.mkdir()
    for name, rows in [('chats', vc), ('messages', vm)]:
        (inputs/(name+'.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
    atomic_write(inputs/'seed.json', vs)
    # -I -S -B: ignore environment imports, no site initialization, no bytecode cache.
    env = {k: v for k, v in os.environ.items() if k in ('SYSTEMROOT','WINDIR','PATH','TEMP','TMP')}
    bootstrap = "import sys; from pathlib import Path; sys.path.insert(0,str(Path.cwd())); import release_worker; release_worker.worker()"
    start = time.perf_counter()
    result = subprocess.run([sys.executable, '-I', '-S', '-B', '-c', bootstrap], cwd=clean,
        env=env, capture_output=True, timeout=120)
    (release/'worker.stderr.log').write_bytes(result.stderr)
    if result.returncode: raise RuntimeError('Clean offline worker failed; inspect private stderr')
    wall = time.perf_counter()-start
    # First and only holdout label access, AFTER frozen inference completed.
    from tools.evaluate_relationships import project
    from tools.evaluate_goals_help import project_labels
    from tools.task_metrics import detection_metrics
    from tools.lifecycle_metrics import lifecycle_metrics
    import check
    answers = json.loads((data/'answers/answers.json').read_text(encoding='utf-8'))
    visible = {m['id'] for m in vm}; other = {m['id'] for m in messages if m['chat_id'] in dev}
    boundary = sum(bool(set(t.get('source_msgs', [])) & visible) and bool(set(t.get('source_msgs', [])) & other)
                   for t in answers.get('open_loops', []))
    gold = project(answers, val, vp-dp)
    gold.update({k: v for k, v in project_labels(answers, visible).items() if k in ('goals','open_loops')})
    del answers
    output = json.loads((clean/'output.json').read_text(encoding='utf-8'))
    scoped = project(output, val, vp-dp)
    scoped.update(goals=output['goals'], open_loops=output['open_loops'])
    report = json.loads((clean/'measurement.json').read_text(encoding='utf-8'))
    report.update(version='rc1', source_sha256=fingerprint, process_seconds=wall,
        scores=check.score(scoped, gold, vs, vm),
        detection=detection_metrics(output['open_loops'], gold['open_loops'])[0],
        lifecycle=lifecycle_metrics(output['open_loops'], gold['open_loops'], visible),
        samples={'messages': len(vm), 'chats': len(vc), 'exclusive_people': len(vp-dp),
            'scored_people': sum(p['id'] not in vs['circles'] for p in gold['people']),
            'labelled_chats': len(gold['chats']), 'task_labels': len(gold['open_loops']),
            'boundary_tasks_excluded': boundary},
        holdout_evaluations=1, post_holdout_implementation_changes=False)
    atomic_write(release/'validation.json', report)
    print(json.dumps(report, indent=2))  # Aggregate statistics only.


if __name__ == '__main__':
    run()
