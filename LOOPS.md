# Project loops

## Riya Champion Loop

Saved: 2026-10-10

Adaptation of published Loop Library loops:
- The Revolve versioned-experiment loop — https://signals.forwardfuture.com/loop-library/loops/revolve-self-improvement-loop/ (modified date not shown at save time)
- The self-improving champion loop — https://signals.forwardfuture.com/loop-library/loops/self-improving-champion-loop/ (modified date not shown at save time)

Each pass researches and builds one improvement to Riya (base model, fine-tune, voice, latency or architecture), checks it automatically on the laptop, and hands it to the user to keep or reject; it stops when the user is happy or when it stops finding improvements.

Prompt:
> Keep one current-best Riya. Each pass, pick the single most promising change from fresh research and past results (base model, fine-tune data, voice, latency, architecture), build it on Colab, and test it on the laptop: it must fit the 6 GB GPU with voice running, be at least as fast to respond, stay stable across a long chat, and never refuse. Then give me the candidate to try; promote it only if I say keep. Log every result in Notion. Stop and report when I'm satisfied or two passes in a row fail to produce a candidate.
