# WhatsApp cold start

## What I focused on, and why

I focused on keeping a useful connection between an inferred task and the
conversation that created it. An assistant that confidently marks unfinished work
as done is difficult to trust. The same applies to reminders about work that
somebody else already owns. I therefore kept source evidence, task ownership and
intervention decisions separate, with checks before publishing the final state.

This makes failures easier to inspect, but it has a cost: conservative evidence
rules can reject real obligations. Recall and help decisions remain weaker than
the relationship and chat mapping in the recorded live example.

## My approach

The implementation uses Python's standard library. It indexes messages in time
order, identifies people by stable IDs and preserves known signup facts.
Relationship inference uses sampled direct and group conversations, reconciles
organisation proposals and retains supported links across samples. Names are
retrieval cues; identical display names do not establish identity.

Overlapping chunks extract tasks and goals in one scan. Each task needs an exact
source span, a valid owner and supporting involvement evidence. Relative dates are
anchored to message time with UTC+05:30 assumed. Group obligations can involve the
user through a supported household or work affiliation even when the user is silent.

Lifecycle review proposes events from original messages, then uses a second model
pass to check them. Events are applied chronologically. Local guards distinguish
acknowledgements, plans and drafts from completion, and keep recurring or ambiguous
obligations separate. Semantic review can consider multilingual replies without
requiring shared words. These checks reduce some mistakes; agreement between two
passes of the same model is not independent proof.

Goals and help facts are reviewed in bounded batches with a result for every item.
The final policy chooses assist, remind or leave. Private preparation and an
external action have different approval requirements. The program proposes actions
only. Explicit local corrections use a replayable ledger and a separate output.

The client bounds time, attempts and response size. Malformed or incomplete
responses can block publication. Strict schema/reference validation precedes an
atomic output replacement. Inference never reads answer files or executes chat
instructions. The official checker and schema are unchanged.

## What it costs

The recorded live run used `nvidia/nemotron-3.5-super-vl:free` through OpenRouter,
with reasoning disabled. On the public 29-message example it took **396.672 seconds**
and **21 requests**, recording **44,819 input and 7,721 output tokens**. Relationship
inference/reconciliation and task/goal extraction accounted for most of the tokens.
Batching avoids a separate full-history goal scan and separate help calls per task.

I do not have a verified dollar-per-1,000-message measurement on the real development
set. The small example has considerable fixed overhead, so I would not extrapolate
its time or token use linearly. An artificial 17,000-message rehearsal completed
with 248 simulated requests, but its empty task replies establish transport
coverage only. Task-rich live throughput, provider quota and monetary cost remain
unverified at that scale.

## Where it's wrong

The live example scored **40.3/74**: circles 8.6/12, organisations 6.0/8, chats 8.0/8,
goals 5.3/8, loops found 6.2/14, loop status 4.2/14 and help 2.0/10. It passed strict
validation and the official checker. The local suite also passed all 146 tests;
model responses in those tests are mocked.

The main problem is still missed or poorly anchored tasks. Three proposals were
rejected for insufficient user involvement in this run. Relationship uncertainty
can therefore propagate into task recall. Lifecycle review rejected four proposals
and deferred an ambiguous closure; cautious filtering can leave real completions
open. Help decisions remain particularly weak, and structural validity says little
about whether a reminder or intervention is appropriate.

The public example was used for debugging and prompt changes, so its score is
development feedback rather than independent validation. An earlier offline version
had one small holdout evaluation; this model-backed revision changed afterwards
and has not been evaluated on a fresh holdout. I have not demonstrated a full-size
live run or endpoint-restricted Linux execution. The most useful next evaluation
would label more group obligations, closures and help decisions before adjusting
these rules further.
