#!/usr/bin/env python3
"""Check an output.json for the Cold Start challenge, and score it against an answer key.

    python3 check.py <input_dir> <output.json>

<input_dir> is a data folder (chats.jsonl, messages.jsonl, seed.json). If it has answers/answers.json,
the output is scored with the same rules we use on the hidden set. Those rules cover 74 of the 100
points; cost, the write-up and "runs with one command" are scored by us.

Standard library only, Python 3.9 or later.
"""
import json
import sys
from pathlib import Path

CIRCLES = {"spouse", "child", "parent", "in-law", "sibling", "relative", "friend", "colleague", "client", "vendor", "partner", "staff"}
ORG_TYPES = {"company", "residence", "vacation home", "workspace", "society", "school", "other"}
ORG_ROLES = {"owner", "partner", "helper", "resident", "parent", "member", "other"}
AREAS = {"self", "spouse", "kids", "parents", "work", "friends", "experiences", "finances", "health", "home"}
GOAL_TYPES = {"habit", "milestone", "experience"}
HORIZONS = {"3mo", "1yr", "3yr", "5yr"}
LOOP_TYPES = {"commitment", "followup", "deadline", "question"}
STATUSES = {"open", "done", "dropped"}
ACTIONS = {"assist", "remind", "leave"}

POINTS = {
    "circles": 12,
    "organisations": 8,
    "chats": 8,
    "goals": 8,
    "loops_found": 14,
    "loops_closed": 14,
    "help": 10,
}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_input(folder):
    folder = Path(folder)
    chats = read_jsonl(folder / "chats.jsonl")
    messages = read_jsonl(folder / "messages.jsonl")
    seed = json.loads((folder / "seed.json").read_text(encoding="utf-8")) if (folder / "seed.json").exists() else {}
    answers_path = folder / "answers" / "answers.json"
    answers = json.loads(answers_path.read_text(encoding="utf-8")) if answers_path.exists() else None
    return chats, messages, seed, answers


# ---------------------------------------------------------------- shape

def validate(out, chats, messages):
    """Every problem with the output's shape, as plain sentences. Empty means it is valid."""
    errs = []
    if not isinstance(out, dict):
        return ["The output must be a JSON object."]
    for key in ("people", "organisations", "chats", "goals", "open_loops"):
        if not isinstance(out.get(key), list):
            errs.append(f'"{key}" must be a list.')
    if errs:
        return errs
    extra = set(out) - {"people", "organisations", "chats", "goals", "open_loops"}
    if extra:
        errs.append(f"Unknown top-level keys: {sorted(extra)}.")

    chat_ids = {c["chat_id"] for c in chats}
    msg_ids = {m["id"] for m in messages}
    person_ids = {p for c in chats for p in c.get("participants", [])} - {"ME"}
    org_ids = set()
    for i, o in enumerate(out["organisations"]):
        where = f"organisations[{i}]"
        if not isinstance(o, dict) or not o.get("id"):
            errs.append(f"{where} needs an id.")
            continue
        if o["id"] in org_ids:
            errs.append(f'{where}: the id "{o["id"]}" is used twice.')
        org_ids.add(o["id"])
        if not o.get("name"):
            errs.append(f"{where} needs a name.")
        if o.get("type") not in ORG_TYPES:
            errs.append(f"{where}: type must be one of {sorted(ORG_TYPES)}.")
        if o.get("side") not in ("home", "work"):
            errs.append(f'{where}: side must be "home" or "work".')
        if o.get("role") not in ORG_ROLES:
            errs.append(f"{where}: role must be one of {sorted(ORG_ROLES)}.")

    seen_people = set()
    for i, p in enumerate(out["people"]):
        where = f"people[{i}]"
        pid = p.get("id") if isinstance(p, dict) else None
        if pid not in person_ids:
            errs.append(f'{where}: "{pid}" is not a person in the input (and ME is never listed).')
            continue
        if pid in seen_people:
            errs.append(f'{where}: "{pid}" is listed twice.')
        seen_people.add(pid)
        if p.get("circle") not in CIRCLES:
            errs.append(f"{where}: circle must be one of {sorted(CIRCLES)}.")
        for link in p.get("links", []) or []:
            if link.get("organisation") not in org_ids:
                errs.append(f'{where}: links to an unknown organisation "{link.get("organisation")}".')
            if link.get("link") not in ("runs", "supports"):
                errs.append(f'{where}: a link must be "runs" or "supports".')
        for other in p.get("looks_after", []) or []:
            if other not in person_ids:
                errs.append(f'{where}: looks_after names an unknown person "{other}".')

    tied = set()
    for i, c in enumerate(out["chats"]):
        where = f"chats[{i}]"
        cid = c.get("chat_id") if isinstance(c, dict) else None
        if cid not in chat_ids:
            errs.append(f'{where}: "{cid}" is not a chat in the input.')
            continue
        if cid in tied:
            errs.append(f'{where}: "{cid}" is tied twice. Tie each chat once.')
        tied.add(cid)
        has_p, has_o = "person" in c, "organisation" in c
        if has_p == has_o:
            errs.append(f"{where}: give exactly one of person or organisation.")
        elif has_p and c["person"] not in person_ids:
            errs.append(f'{where}: unknown person "{c["person"]}".')
        elif has_o and c["organisation"] not in org_ids:
            errs.append(f'{where}: unknown organisation "{c["organisation"]}".')
    missing = chat_ids - tied
    if missing:
        errs.append(f"{len(missing)} chat(s) are not tied to anyone, for example {sorted(missing)[:3]}.")

    if len(out["goals"]) > 10:
        errs.append("At most 10 goals.")
    for i, g in enumerate(out["goals"]):
        where = f"goals[{i}]"
        if not g.get("title"):
            errs.append(f"{where} needs a title.")
        if g.get("area") not in AREAS:
            errs.append(f"{where}: area must be one of {sorted(AREAS)}.")
        if g.get("type") not in GOAL_TYPES:
            errs.append(f"{where}: type must be one of {sorted(GOAL_TYPES)}.")
        if g.get("horizon") not in HORIZONS:
            errs.append(f"{where}: horizon must be one of {sorted(HORIZONS)}.")
        ev = g.get("evidence") or []
        if not ev:
            errs.append(f"{where}: give at least one evidence message.")
        for m in ev:
            if m not in msg_ids:
                errs.append(f'{where}: evidence "{m}" is not a message in the input.')

    for i, l in enumerate(out["open_loops"]):
        where = f"open_loops[{i}]"
        if not l.get("title"):
            errs.append(f"{where} needs a title.")
        if l.get("type") not in LOOP_TYPES:
            errs.append(f"{where}: type must be one of {sorted(LOOP_TYPES)}.")
        if l.get("owner") != "ME" and l.get("owner") not in person_ids:
            errs.append(f'{where}: owner must be "ME" or a person id.')
        if l.get("chat_id") not in chat_ids:
            errs.append(f"{where}: unknown chat_id.")
        if l.get("source_msg") not in msg_ids:
            errs.append(f"{where}: source_msg is not a message in the input.")
        if l.get("status") not in STATUSES:
            errs.append(f"{where}: status must be one of {sorted(STATUSES)}.")
        if l.get("closed_by") is not None and l.get("closed_by") not in msg_ids:
            errs.append(f"{where}: closed_by is not a message in the input.")
        help_ = l.get("help")
        if help_ is not None:
            if l.get("status") != "open":
                errs.append(f"{where}: help is only for open loops.")
            if not isinstance(help_, dict) or help_.get("action") not in ACTIONS:
                errs.append(f"{where}: help.action must be one of {sorted(ACTIONS)}.")
    return errs


# ---------------------------------------------------------------- scoring

def f1(matched, predicted, actual):
    if predicted == 0 or actual == 0:
        return 0.0
    p, r = matched / predicted, matched / actual
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


def org_members(org_id, chats_out, people_out):
    members = {"c:" + c["chat_id"] for c in chats_out if c.get("organisation") == org_id}
    for p in people_out:
        if any(link.get("organisation") == org_id for link in p.get("links", []) or []):
            members.add("p:" + p["id"])
    return members


def match_orgs(out, ans):
    """Pair our organisations with yours one to one, by the chats and people attached to each."""
    ours = {o["id"]: org_members(o["id"], ans["chats"], ans["people"]) for o in ans["organisations"]}
    yours = {o["id"]: org_members(o["id"], out["chats"], out["people"]) for o in out["organisations"]}
    pairs = sorted(((jaccard(a, b), x, y) for x, a in ours.items() for y, b in yours.items()), reverse=True)
    matched, used_x, used_y = {}, set(), set()
    for score, x, y in pairs:
        if score > 0 and x not in used_x and y not in used_y:
            matched[x] = (y, score)
            used_x.add(x)
            used_y.add(y)
    return matched, len(ours), len(yours)


def score(out, ans, seed, messages):
    s = {}
    seeded = set((seed.get("circles") or {}).keys())
    circle_out = {p["id"]: p.get("circle") for p in out["people"]}
    scored = [p for p in ans["people"] if p["id"] not in seeded]
    s["circles"] = sum(circle_out.get(p["id"]) == p["circle"] for p in scored) / len(scored) if scored else None

    orgs, n_ours, n_yours = match_orgs(out, ans)
    s["organisations"] = (sum(v[1] for v in orgs.values()) / max(n_ours, n_yours)) if n_ours else 0.0

    tie_out = {c["chat_id"]: c for c in out["chats"]}
    right = 0
    for c in ans["chats"]:
        mine = tie_out.get(c["chat_id"], {})
        if "person" in c:
            right += mine.get("person") == c["person"]
        else:
            right += mine.get("organisation") is not None and orgs.get(c["organisation"], (None,))[0] == mine.get("organisation")
    s["chats"] = right / len(ans["chats"]) if ans["chats"] else 0.0

    # The seeded goal is given, so it counts neither way: repeating it is not a wrong guess.
    seeded_titles = {g.get("title") for g in seed.get("goals", [])}
    seeded = [g for g in ans["goals"] if g.get("title") in seeded_titles]
    goals_ans = [g for g in ans["goals"] if g.get("title") not in seeded_titles]
    goals_out = [
        g
        for g in out["goals"]
        if not any(g.get("area") == a["area"] and set(g.get("evidence", [])) & set(a["evidence"]) for a in seeded)
    ]
    used, hit = set(), 0
    for g in goals_out:
        for j, a in enumerate(goals_ans):
            if j not in used and g.get("area") == a["area"] and set(g.get("evidence", [])) & set(a["evidence"]):
                used.add(j)
                hit += 1
                break
    s["goals"] = f1(hit, len(goals_out), len(goals_ans)) if goals_ans else None

    # Open loops can be scored on a closing window only (the hidden set's last two weeks): loops whose
    # source is earlier are ignored on both sides.
    since = ans.get("loops_scored_from")
    loops_out = out["open_loops"]
    if since is not None:
        ts = {m["id"]: m["ts"] for m in messages}
        loops_out = [l for l in loops_out if ts.get(l.get("source_msg"), 0) >= since]
        ans = {**ans, "open_loops": [a for a in ans["open_loops"] if min(ts.get(x, 0) for x in a["source_msgs"]) >= since]}

    # A loop is found when its source message is one we accept for that loop.
    pairs, used = [], set()
    for l in loops_out:
        for j, a in enumerate(ans["open_loops"]):
            if j not in used and l.get("source_msg") in a["source_msgs"]:
                used.add(j)
                pairs.append((l, a))
                break
    n_ans = len(ans["open_loops"])
    s["loops_found"] = f1(len(pairs), len(loops_out), n_ans) if n_ans else None

    # Status right counts 1. Closing a loop that is still open costs 1: a false "done" hides real work.
    closed = 0
    for l, a in pairs:
        if l.get("status") == a["status"]:
            closed += 1
        elif a["status"] == "open":
            closed -= 1
    s["loops_closed"] = max(0.0, closed / n_ans) if n_ans else None

    # Help: the right call counts 1. Stepping in where Arjun wants you out costs 1.
    labelled = [a for a in ans["open_loops"] if a["status"] == "open" and a.get("help")]
    by_ans = {id(a): l for l, a in pairs}
    helped = 0
    for a in labelled:
        l = by_ans.get(id(a))
        action = ((l or {}).get("help") or {}).get("action") if l and l.get("status") == "open" else None
        if action == a["help"]:
            helped += 1
        elif a["help"] == "leave" and action in ("assist", "remind"):
            helped -= 1
    s["help"] = max(0.0, helped / len(labelled)) if labelled else None
    return s


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    chats, messages, seed, answers = load_input(argv[1])
    try:
        out = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"Could not read {argv[2]}: {e}")
        return 1
    errs = validate(out, chats, messages)
    if errs:
        print(f"The output has {len(errs)} problem(s):")
        for e in errs[:50]:
            print("  -", e)
        if len(errs) > 50:
            print(f"  ... and {len(errs) - 50} more")
        return 1
    print("The output is valid.")
    if answers is None:
        print("No answers/answers.json in this folder, so there is nothing to score against.")
        return 0
    s = score(out, answers, seed, messages)
    total = 0.0
    print()
    print(f"{'Part':<26}{'Score':>8}{'Points':>10}")
    labels = {
        "circles": "Circles",
        "organisations": "Organisations",
        "chats": "Chats",
        "goals": "Goals",
        "loops_found": "Open loops found",
        "loops_closed": "Open loops closed right",
        "help": "Where to help",
    }
    possible = 0
    for key, pts in POINTS.items():
        if s[key] is None:
            print(f"{labels[key]:<26}{'not labelled in this set':>25}")
            continue
        got = s[key] * pts
        total += got
        possible += pts
        print(f"{labels[key]:<26}{s[key]:>8.2f}{got:>7.1f} / {pts}")
    print(f"{'Total from the data':<26}{'':>8}{total:>7.1f} / {possible}")
    print("Cost (12), the write-up (7) and runs with one command (7) are scored by us.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
