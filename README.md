# Xpert Cold Start

Builds a first picture of someone's life from a WhatsApp export: the people around
them, the organisations they belong to, their goals, and the commitments that are
still open. For each open task, it suggests whether an assistant should help,
remind the user, or leave it alone.

Built for the [Xpert internship challenge](https://goals.xpert.chat/w/intern).
The project uses Python's standard library and an OpenAI-compatible model API.

## How it works

1. Index the chats and messages in time order, preserving the original IDs.
2. Infer relationships and organisation links, using signup information as a starting point.
3. Extract tasks and lasting goals together from overlapping message chunks.
4. Review later messages for completion, cancellation, rescheduling or reopening.
5. Verify goal and help proposals, validate the final JSON, and write the output.

Tasks retain their source message. Proposed closures are checked against the
original conversation; acknowledgements and unfinished drafts do not count as
completed work. The assistant only proposes actions—it does not send messages,
make bookings or purchase anything. Local corrections can override inferred state
through a separate, replayable ledger.

## Setup

You need Python 3.9 or later and access to the configured model. On Windows, install
Git for Windows as well; its Bash executable runs `run.sh`. There are no Python
packages to install.

The required environment variables are `LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL`.
`.env.example` lists the settings, but the program reads the terminal environment
and does not load that file automatically.

The model used for the recorded live test was
`nvidia/nemotron-3.5-super-vl:free` through OpenRouter. For this route, the client
defaults to reasoning disabled, strict goal/help schemas, a 90-second request
timeout and four seconds between request starts. Other routes use their normal
reasoning settings and JSON-object mode unless configured otherwise.

### Windows: run the included example

Open PowerShell in the folder containing `run.sh` and paste:

```powershell
$env:LLM_BASE_URL = 'https://openrouter.ai/api/v1'
$env:LLM_MODEL = 'nvidia/nemotron-3.5-super-vl:free'
$env:LLM_JSON_SCHEMA_VERDICTS = '1'
$env:LLM_MIN_REQUEST_INTERVAL_SECONDS = '4'
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tools\test_live_synthetic.ps1 -PromptForKey
```

Paste the API key at the masked prompt. The helper handles the Windows `python3`
shim, copies the three example inputs into a fresh run folder, and runs inference,
strict validation and the official checker. It prints progress and saves results
under `.private/live-synthetic/`.

This is a live model test, so check your available quota before starting. The
recorded successful run used 21 API requests; the count can vary. Success requires
`run_exit`, `strict_validation_exit` and `checker_exit` to all be `0`.

### Linux: run the included example

```bash
export LLM_BASE_URL='https://openrouter.ai/api/v1'
export LLM_MODEL='nvidia/nemotron-3.5-super-vl:free'
read -r -s -p 'OpenRouter API key: ' LLM_API_KEY
export LLM_API_KEY
mkdir -p .private/input
cp example/chats.jsonl example/messages.jsonl example/seed.json .private/input/
./run.sh .private/input .private/output.json
python3 check.py example .private/output.json
```

For another dataset, place these three files in your input folder:

| File | Contents |
|---|---|
| `chats.jsonl` | Chat IDs, names, group flags and participants |
| `messages.jsonl` | Message IDs, senders, timestamps, text and quoted references |
| `seed.json` | Known circles, organisation names and signup goals |

The general interface is `./run.sh INPUT_DIR OUTPUT_PATH`. The output contains
`people`, `organisations`, `chats`, `goals` and `open_loops`, following `schema.json`.
Inference reads only the three input files and the local schema. Run `check.py`
after inference, using the matching dataset if it includes answers.

### Settings and failures

Optional settings are in `.env.example`. The worker allows 1,680 seconds, plus
60 seconds in the parent process. Defaults are 1,000 attempts and 20,000,000
reservation units. Units count request bytes plus output allowance on each
attempt, not billed tokens or dollars. Terminal environment values override defaults.

Output is replaced only after validation and completion checks pass. A failed run
exits nonzero and may leave an older output untouched. Check the adjacent
`.run-status.json`, `.completion.json` and `.usage.jsonl` files to identify the
fresh run's status. `XPERT_PROGRESS=1` enables stage and request progress without
printing message bodies or keys. Network and response failures have bounded retries.

## Tests and local tools

```bash
python -S -B -m unittest discover -s tests -q
python -S -B tools/check_capacity.py
```

The 146 tests cover source references, lifecycles, corrections, malformed replies,
API failures and atomic output. HTTP responses are mocked. The capacity check
processes 17,000 artificial messages with simulated replies; it measures transport
coverage, not live speed or accuracy.

`python tools/demo_corrections.py` demonstrates corrections without API calls;
`python correction_cli.py --help` shows the interface. Evaluation scripts in
`tools/` expect the original local development data and saved split, which are
not included here.

The main implementation is in `inputs.py`, `relationships.py`, `task_candidates.py`,
`task_lifecycle.py` and `goals_help.py`. `solution.py` coordinates the pipeline;
`llm.py` handles model requests and usage; `validation.py` checks the final state.
The supplied `check.py` and `schema.json` are unchanged.

## Recorded result and limitations

The live Windows/Git Bash run on 12 November 2026 used the public example:
6 chats and 29 messages. It completed in **396.672 seconds**, made **21 requests**,
and recorded **44,819 input + 7,721 output = 52,540 tokens**. All three exit checks
passed. Dollar cost was not independently measured.

| Component | Points |
|---|---:|
| Circles | 8.6 / 12 |
| Organisations | 6.0 / 8 |
| Chats | 8.0 / 8 |
| Goals | 5.3 / 8 |
| Loops found | 6.2 / 14 |
| Loop status | 4.2 / 14 |
| Help decisions | 2.0 / 10 |
| **Total** | **40.3 / 74** |

This small example was used during development. It is not an independent estimate
of hidden-set performance. Task recall, closure decisions and appropriate help
remain the main weaknesses. A successful full-size live run and execution on Linux
with network access restricted to the model endpoint have not been demonstrated.
Relative dates assume UTC+05:30 and use message timestamps rather than today's date.

## Submission files and private data

[`WRITEUP.md`](WRITEUP.md) is the challenge's short approach and results write-up.
Genuine AI-session transcripts still need to be reviewed and added under
`ai-logs/` before submission; they are not bundled here. `tools/preserve_session.py`
exports recorded user/assistant messages for local review. Track reviewed copies
with `git add -f ai-logs/FILENAME`.

Keep real chats, answers, outputs and credentials out of Git. The public labelled
fixture in `example/` is included for evaluation after inference. AI tools helped
with implementation and review.
