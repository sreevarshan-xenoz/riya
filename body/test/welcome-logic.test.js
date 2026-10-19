"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const Ajv = require("ajv");
const W = require("../src/welcome-logic");

const fixture = (name) => JSON.parse(fs.readFileSync(path.join(__dirname, "fixtures", name), "utf8"));
const read = (f) => fs.readFileSync(path.join(__dirname, "..", f), "utf8");
const schema = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "..", "docs", "protocol.schema.json"), "utf8"));
const validateBody = new Ajv({ strict: false }).compile(schema.body_to_core);

test("the welcome has eight steps, intro first and done last, and next/prev stop at the ends", () => {
  assert.equal(W.STEPS.length, 8);
  assert.deepEqual(W.STEPS, ["intro", "name", "key", "voice", "mode", "eyes", "agents", "done"]);
  assert.equal(W.nextStep("intro"), "name");
  assert.equal(W.nextStep("agents"), "done");
  assert.equal(W.nextStep("done"), null);
  assert.equal(W.prevStep("name"), "intro");
  assert.equal(W.prevStep("intro"), null);
  assert.equal(W.nextStep("nope"), null);
  assert.equal(W.prevStep("nope"), null);
});

test("cleanName trims, collapses whitespace, and refuses empty, long, question or control input", () => {
  assert.equal(W.cleanName("  Sree  "), "Sree");
  assert.equal(W.cleanName("Mary   Jane"), "Mary Jane");
  assert.equal(W.cleanName("a".repeat(40)), "a".repeat(40));
  for (const bad of ["", "   ", "who?", "a".repeat(41), "Mary\nJane", "x\u0000", "x\u007f", 5, null, undefined]) {
    assert.equal(W.cleanName(bad), null, JSON.stringify(bad));
  }
});

test("the name goes through the remember path", () => {
  assert.deepEqual(W.nameMessage("Sree"), { type: "user_text", text: "remember that my name is Sree" });
  assert.equal(W.nameMessage(""), null);
  assert.equal(W.nameMessage("who?"), null);
});

test("modeMessage and eyesMessage accept only their own values", () => {
  for (const mode of ["buddy", "panel", "window"]) assert.deepEqual(W.modeMessage(mode), { type: "set_settings", patch: { mode } });
  for (const bad of ["x", "", null, undefined, 3, "toString"]) assert.equal(W.modeMessage(bad), null, String(bad));
  assert.deepEqual(W.eyesMessage(true), { type: "set_settings", patch: { eyes_enabled: true } });
  assert.deepEqual(W.eyesMessage(false), { type: "set_settings", patch: { eyes_enabled: false } });
  for (const bad of ["yes", 1, 0, null, undefined]) assert.equal(W.eyesMessage(bad), null, String(bad));
});

test("eyesLine names the hotkey, or says the don't-look hotkey", () => {
  assert.match(W.eyesLine(null), /the don't-look hotkey/);
  assert.ok(W.eyesLine("Control+Alt+H").includes("Control+Alt+H"));
  assert.ok(!W.eyesLine("Control+Alt+H").includes("{hotkey}"));
  assert.ok(!W.eyesLine(null).includes("{hotkey}"));
  assert.match(W.eyesLine(7), /the don't-look hotkey/);
  assert.match(W.eyesLine("  "), /the don't-look hotkey/);
});

test("agentRows are the Claude Code and Colab rows only, with no fix button", () => {
  const rows = W.agentRows(fixture("core_health.json"));
  assert.deepEqual(rows.map((r) => r.key), ["claude_code"]);
  assert.equal(rows[0].fix, null);
  assert.equal(rows[0].label, "Claude Code");
  const both = W.agentRows({ items: [
    { key: "online", status: "ok", detail: "x", fix: "online_test" },
    { key: "colab", status: "ok", detail: "1 folder", fix: "x" },
    { key: "claude_code", status: "ok", detail: "ok", fix: "hooks_disconnect" },
  ] });
  assert.deepEqual(both.map((r) => [r.key, r.fix]), [["colab", null], ["claude_code", null]]);
  assert.deepEqual(W.agentRows(null), []);
});

test("every line is her voice: a non-empty string for every step", () => {
  assert.deepEqual(Object.keys(W.LINES), W.STEPS);
  for (const step of W.STEPS) assert.ok(typeof W.LINES[step] === "string" && W.LINES[step].length > 0, step);
  assert.match(W.LINES.eyes, /\{hotkey\}/);
});

test("every message builder validates against the protocol schema", () => {
  for (const msg of [W.nameMessage("Sree"), W.modeMessage("window"), W.eyesMessage(true)]) {
    assert.ok(validateBody(msg), JSON.stringify(validateBody.errors));
  }
});

test("the welcome never builds onboarding_done: the main process's closed handler is the single sender", () => {
  for (const f of ["welcome.js", "src/welcome-logic.js"]) assert.doesNotMatch(read(f), /onboarding_done/, f);
  assert.ok(!("onboardingDone" in W));
});

test("welcome.html: a password field for the key, the settings CSP, no inline script, and the scripts in order", () => {
  const html = read("welcome.html");
  assert.match(html, /type="password"/);
  assert.match(html, /default-src 'none'; script-src 'self'; style-src 'self'/);
  assert.doesNotMatch(html, /<script(?![^>]*\bsrc=)/);
  const i = (s) => html.indexOf(s);
  assert.ok(i("src/health-logic.js") !== -1 && i("src/health-logic.js") < i("src/welcome-logic.js"));
  assert.ok(i("src/welcome-logic.js") < i("welcome.js"));
});

test("welcome.js reuses the settings key path: submitKey, saveKeyEnabled, and the hotkey and key-page bridges", () => {
  const js = read("welcome.js");
  assert.match(js, /submitKey\(/);
  assert.match(js, /saveKeyEnabled\(/);
  assert.match(js, /openKeyPage\(\)/);
  assert.match(js, /health_get/);
  assert.match(js, /window\.close\(\)/);
  assert.doesNotMatch(js, /localStorage|sessionStorage|console\./, "the key is never stored or logged");
});

test("Settings has a button to open the welcome again, through the argument-free bridge", () => {
  assert.match(read("settings.html"), /<button id="welcome-open"[^>]*>Open the welcome again<\/button>/);
  assert.match(read("settings.js"), /\$\("welcome-open"\)\.addEventListener\("click", \(\) => window\.riya\.openWelcome\(\)\)/);
});

test("cleanName refuses a pasted key, and nameError says so clearly", () => {
  const keys = [
    "AIzaSyD-0123456789abcdefghijklmnopqrstu", // a 39-character Gemini-shaped key: under the 40 limit
    "AIzaSyDUMMY12",
    "sk-abcdefghijklmnop",
    "ab12cd34ef56gh78ij90kl12mn34", // a long mixed token
    "Riya friend ab12cd34ef56gh78ij90kl12", // inside a longer name
  ];
  for (const k of keys) {
    assert.equal(W.looksLikeKey(k), true, k);
    assert.equal(W.cleanName(k), null, k);
    assert.equal(W.nameMessage(k), null, k);
    assert.equal(W.nameError(k), W.KEY_IN_NAME, k);
  }
  assert.match(W.KEY_IN_NAME, /looks like a key/);
  assert.match(W.KEY_IN_NAME, /next step/);
  for (const name of ["Sree", "Mary Jane", "a".repeat(40), "Sreevarshan Xenoz", "R2-D2", "Anna-Maria 3rd"]) {
    assert.equal(W.looksLikeKey(name), false, name);
    assert.equal(W.nameError(name), null, name);
  }
  assert.equal(W.nameError("who?"), W.NAME_RULES);
  assert.equal(W.nameError("ab ".repeat(14)), W.NAME_RULES); // long, but words: not a key
  assert.equal(W.nameError("a".repeat(41)), W.KEY_IN_NAME); // over 40 with no space: key-shaped
});

test("cleanName keeps ZWJ and ZWNJ (Malayalam, Hindi, Persian spellings) and still refuses other control characters", () => {
  for (const name of ["ശ്രീ\u200D", "क्\u200Dष", "می\u200Cخواهم", "Sree\u200Cvarshan"]) {
    assert.equal(W.cleanName(name), name, JSON.stringify(name));
  }
  for (const bad of ["x\u200B", "x\u202E", "x\u2066", "x\u00AD", "x\u0007", "x\uE000"]) {
    assert.equal(W.cleanName(bad), null, JSON.stringify(bad));
  }
});
