"use strict";
/** Pure helpers for the first-run welcome window: the steps and her lines, the messages each step sends (built here,
 *  validated by the core), and the display-only agents rows. There is no "finished" builder on purpose: the main
 *  process's `closed` handler for this window is the only sender. Everything the core says is shown through textContent
 *  by the caller. Loads in Node (tests) and as a plain script in the welcome page, after health-logic.js. */

const Health = typeof require !== "undefined" ? require("./health-logic") : globalThis.RiyaHealth;

const STEPS = ["intro", "name", "key", "voice", "mode", "eyes", "agents", "done"];
const MODES = ["buddy", "panel", "window"];
const MAX_NAME = 40;
const AGENT_KEYS = ["claude_code", "colab"];
const FALLBACK_HOTKEY = "the don't-look hotkey";

// Her voice; {hotkey} in the eyes line is filled by eyesLine().
const LINES = {
  intro: "Hi! I'm Riya~ I live on your desktop, I chat, I keep you company, and I keep an eye on your agents. Let's set me up. It takes a minute, and you can skip anything.",
  name: "What should I call you?",
  key: "I can think harder with an online brain. Paste a free Gemini key if you'd like, or skip and I'll stay fully local.",
  voice: "Pick the voice you like best~ Press Play to hear me.",
  mode: "How should I show up? A little buddy on your desktop, a side panel, or a normal window.",
  eyes: "Can I peek at your screen now and then? I never look at private things, and {hotkey} makes me look away any time.",
  agents: "I can tell you when Claude Code or your Colab runs need you. Here's how that looks right now.",
  done: "All set! I'm so happy you're here~ You can open this again from Settings any time.",
};

/** The step after `step`, or null at the end (or for an unknown step). */
function nextStep(step) {
  const i = STEPS.indexOf(step);
  return i === -1 || i === STEPS.length - 1 ? null : STEPS[i + 1];
}

/** The step before `step`, or null at the start (or for an unknown step). */
function prevStep(step) {
  const i = STEPS.indexOf(step);
  return i <= 0 ? null : STEPS[i - 1];
}

/** What was typed as a name: trimmed, whitespace collapsed; null when empty, over 40 characters, or holding a "?" or a
 *  control character (a question is not a name, and the remember path ignores one). */
function cleanName(raw) {
  if (typeof raw !== "string") return null;
  const trimmed = raw.trim();
  if (!trimmed || /[?\p{C}]/u.test(trimmed)) return null;
  const name = trimmed.replace(/\s+/g, " ");
  return name.length <= MAX_NAME ? name : null;
}

/** The existing remember path: she acks aloud and the fact is saved as `user_said`. */
function nameMessage(raw) {
  const name = cleanName(raw);
  return name === null ? null : { type: "user_text", text: `remember that my name is ${name}` };
}

function modeMessage(mode) {
  return typeof mode === "string" && MODES.includes(mode) ? { type: "set_settings", patch: { mode } } : null;
}

function eyesMessage(on) {
  return typeof on === "boolean" ? { type: "set_settings", patch: { eyes_enabled: on } } : null;
}

/** The eyes line with the don't-look accelerator in it, or "the don't-look hotkey" when none is registered. */
function eyesLine(hotkey) {
  const name = typeof hotkey === "string" && hotkey.trim() ? hotkey.trim() : FALLBACK_HOTKEY;
  return LINES.eyes.replace("{hotkey}", () => name);
}

/** The Claude Code and Colab rows of a `health` message, without their fix buttons: this step only shows status (the
 *  hooks are connected, and the folders edited, in Settings). */
function agentRows(healthMsg) {
  return Health.healthRows(healthMsg).filter((row) => AGENT_KEYS.includes(row.key)).map((row) => ({ ...row, fix: null }));
}

const api = { STEPS, LINES, MAX_NAME, nextStep, prevStep, cleanName, nameMessage, modeMessage, eyesMessage, eyesLine, agentRows };
if (typeof module !== "undefined" && module.exports) module.exports = api;
else globalThis.RiyaWelcome = api;
