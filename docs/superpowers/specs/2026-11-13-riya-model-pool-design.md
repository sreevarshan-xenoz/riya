# Riya's model pool: free online models that take over for each other

## Why

Riya's online brain is one free opencode model. Free tiers have limits nobody publishes, so she might go quiet for
hard questions mid-day. The user wants her to keep talking and working with them all day by drawing on several free
sources, switching on her own when one runs out and going back to it once its limit resets.

Everyday small talk stays on her local model (unlimited, runs on the laptop's MX250); the pool backs the online
work: hard questions, memory consolidation when online, her hands' tool calls and her eyes.

## Goals

- When a model hits a limit, the next reply goes to another one without the user seeing an error.
- A limited model is benched, re-tried after a cooldown, and put back when it answers again.
- opencode's free models each have their own limit, so they rotate one by one, not as a block.
- Slow sources (CLI agents) only take over when every fast one is benched, and the wait is covered by a short
  filler line ("hmm…") and streamed speech.
- The Health page shows which models are live or benched.

## Non-goals

- Paid models, or accounts the user hasn't logged into.
- Picking the "best" model per question. Order is fixed; the pool only routes around failures.
- Switching models in the middle of a reply. Once the first words are out, a failure is a failure.

## Anti-goals

- A CLI agent that can run commands or edit files on the laptop. Every CLI member runs in its read-only (plan or
  ask) mode, in a scratch folder, with tools off where the CLI allows it.
- Hammering a limited service: re-checks are spaced out and grow longer on repeated limits.
- Hogging the laptop: warm CLI processes are capped like the model server.

## Approaches

1. **A pool inside Riya (chosen).** One `ModelPool` that looks like a single online model to the rest of Riya.
   Fast members first (opencode's free models over HTTP), slow members after (agy, Kilo, Cline in ask mode with a
   warm process). Picked because it builds on the opencode client she already has, keeps live chat fast, and lets
   Health show what is benched.
2. Calling Kilo's and Cline's web APIs directly. Faster than their CLIs, but it means lifting login tokens out of
   their apps: fragile, and more clearly against their terms.
3. A separate router program in front of Riya. One more thing to run, and Riya can't see what is benched.

## Design

- **Members.** Each is a `ChatModel` with a name and flags: `vision` (accepts images), `tools` (has `call_tools`),
  `slow`. opencode's free models come from its `/models` list at start (ids ending in `-free`; vision from
  `modalities.input`), with the current model first; a built-in list is the fallback when the list can't be read.
  `RIYA_ONLINE_MODEL`, when set, goes first.
- **Order.** Sticky: the pool starts from the member that last answered (better caching, a steadier voice), then
  walks the rest in order; slow members come after every fast one.
- **Failover.** `stream()` tries members in order until one yields its first piece; after that it is committed.
  `call_tools()` skips members without tools.
- **Benching.** A limit (HTTP 429, or 402/403 with a quota or limit message) benches a member for `Retry-After` when
  given, else 10 minutes, doubling on each repeat up to 6 hours. A server error or timeout benches it for 1 minute.
  Bad requests (400) don't bench; a member that rejects tools is marked `tools=False`. One good answer clears the
  member's record. A benched member is re-tried lazily, the first time the pool reaches it after its cooldown.
- **Vision.** Her eyes get their own pool over the vision members (same benching state).
- **Slow tier (step 2).** agy (and Kilo, Cline once installed and logged in) run as one warm process each in its
  print/stream-json mode, read-only, in a scratch folder, with a spare process pre-warmed so each reply is about
  3 s rather than 15. Before a slow member answers, Riya says a short filler line.
- **Health.** A "models" row: how many are live, which are benched and until when.

## Left to the builder

- The exact filler lines and when the voice says them.
- How much of each CLI's own instructions can be switched off to shorten its replies.
- Whether bench state survives a restart (start without; add if limits turn out to be daily).
