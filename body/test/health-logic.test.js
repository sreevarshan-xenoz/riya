"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const Ajv = require("ajv");
const H = require("../src/health-logic");

const fixture = (name) => JSON.parse(fs.readFileSync(path.join(__dirname, "fixtures", name), "utf8"));
const schema = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "..", "docs", "protocol.schema.json"), "utf8"));
const validateBody = new Ajv({ strict: false }).compile(schema.body_to_core);

test("healthRows keeps the core's order and labels, and skips bad rows", () => {
  const rows = H.healthRows(fixture("core_health.json"));
  assert.deepEqual(rows.map((r) => [r.key, r.label, r.status, r.fix]), [
    ["online", "Online brain", "warn", "online_test"],
    ["claude_code", "Claude Code", "off", "hooks_connect"],
    ["look", "Look", "warn", null],
  ]);
  assert.match(rows[0].detail, /^Key ••••1a2b saved in Riya/);
  const bad = H.healthRows({ items: [
    { key: "mic", status: "<b>", detail: "x" },
    { key: "nope", status: "ok", detail: "x" },
    { key: "voice", status: "ok", detail: 5 },
    null,
    { key: "memory", status: "ok", detail: "x".repeat(300), fix: 7 },
  ] });
  assert.deepEqual(bad.map((r) => r.key), ["memory"]);
  assert.equal(bad[0].detail.length, 120);
  assert.equal(bad[0].fix, null);
  assert.deepEqual(H.healthRows(null), []);
  assert.deepEqual(H.healthRows({ items: "x" }), []);
});

test("every row key has a label, in the core's order", () => {
  assert.deepEqual(Object.keys(H.ROW_LABELS), ["local", "online", "voice", "mic", "look", "eyes", "claude_code", "colab", "memory", "autostart"]);
  assert.deepEqual(H.CONFIRM_GATED, ["online_test", "eyes_test", "hooks_connect", "hooks_disconnect"]);
  assert.equal(H.REFRESH_MS, 30000);
  assert.equal(H.MAX_KEY, 200);
});

test("fixRequest builds the setup_request, with confirm for the gated actions only", () => {
  const row = (key, status, fix) => ({ key, label: "x", status, detail: "", fix });
  assert.deepEqual(H.fixRequest(row("voice", "ok", "voice_sample"), { voice: "eve" }), {
    label: "Play sample", confirm: false, message: { type: "setup_request", action: "voice_sample", args: { voice: "eve" } },
  });
  assert.equal(H.fixRequest(row("voice", "ok", "voice_sample"), {}).message.args.voice, null);
  const on = H.fixRequest(row("autostart", "off", "autostart"), {});
  assert.equal(on.label, "Turn on");
  assert.deepEqual(on.message.args, { on: true });
  const off = H.fixRequest(row("autostart", "ok", "autostart"), {});
  assert.equal(off.label, "Turn off");
  assert.deepEqual(off.message.args, { on: false });
  const connect = H.fixRequest(row("claude_code", "off", "hooks_connect"), {});
  assert.equal(connect.confirm, true);
  assert.equal(connect.label, "Connect");
  assert.deepEqual(connect.message, { type: "setup_request", action: "hooks_connect", args: {} });
  assert.equal(H.fixRequest(row("mic", "ok", "mic_check"), {}).confirm, false);
  assert.equal(H.fixRequest(row("look", "warn", null), {}), null);
  assert.equal(H.fixRequest(row("look", "warn", "online_key_set"), {}), null, "an action with no button is not offered");
});

test("keyRequest trims, and refuses empty, long, spaced or control-character keys", () => {
  assert.deepEqual(H.keyRequest(" AIzaKEY12345678 "), { type: "setup_request", action: "online_key_set", args: { key: "AIzaKEY12345678" } });
  for (const bad of ["a b", "a​b", "a".repeat(201), "", "   ", null, 5]) assert.equal(H.keyRequest(bad), null, String(bad));
  assert.ok(H.keyRequest("a".repeat(200)));
});

test("test_submitKey_sends_once_and_clears_the_field", () => {
  const sent = [];
  const send = (m) => sent.push(m);
  const field = { value: " AIzaCANARY123456 " };
  assert.equal(H.submitKey(field, send, true), true);
  assert.deepEqual(sent, [{ type: "setup_request", action: "online_key_set", args: { key: "AIzaCANARY123456" } }]);
  assert.equal(field.value, "");

  const bad = { value: "has space" };
  assert.equal(H.submitKey(bad, send, true), false);
  assert.equal(sent.length, 1);
  assert.equal(bad.value, "has space");

  for (const connected of [false, undefined, "yes", 1]) {
    const f = { value: "AIzaCANARY123456" };
    assert.equal(H.submitKey(f, send, connected), false);
    assert.equal(f.value, "AIzaCANARY123456", "the key stays in the field while the core is away");
  }
  assert.equal(sent.length, 1, "nothing was sent while disconnected");
  assert.equal(H.submitKey(null, send, true), false);
});

test("saveKeyEnabled and clearKeyEnabled", () => {
  assert.equal(H.saveKeyEnabled(true), true);
  assert.equal(H.saveKeyEnabled(false), false);
  assert.equal(H.saveKeyEnabled(undefined), false);
  assert.equal(H.saveKeyEnabled("true"), false);
  assert.equal(H.clearKeyEnabled("app"), true);
  for (const s of [".env", "env", null, undefined, ""]) assert.equal(H.clearKeyEnabled(s), false, String(s));
});

test("keySourceText says where the key comes from", () => {
  assert.equal(H.keySourceText("app"), "Saved in Riya");
  assert.equal(H.keySourceText(".env"), "Set via .env (edit that file to remove it)");
  assert.equal(H.keySourceText("env"), "Set via the RIYA_ONLINE_API_KEY environment variable");
  assert.equal(H.keySourceText(null), "No key yet");
  assert.equal(H.keySourceText("<b>"), "No key yet");
});

test("voiceItems takes the string voices of a voices message", () => {
  assert.deepEqual(H.voiceItems(fixture("core_voices.json")), { items: ["alba", "cosette", "eve"], current: "alba" });
  assert.deepEqual(H.voiceItems({ type: "voices", items: ["alba", 3, null, "eve"], current: 4 }), { items: ["alba", "eve"], current: null });
  assert.deepEqual(H.voiceItems({ type: "health", items: [] }), { items: [], current: null });
  assert.deepEqual(H.voiceItems(null), { items: [], current: null });
});

test("ConfirmGate only accepts a confirm this window asked for, once, with a sane nonce", () => {
  const confirm = (over) => ({ type: "setup_confirm", nonce: "n0nce-abc", action: "hooks_connect", text: "Sure?", ...over });
  const gate = new H.ConfirmGate();
  assert.equal(gate.accepts(confirm()), false, "no request here: the welcome's");
  gate.request("hooks_connect");
  assert.equal(gate.accepts(confirm()), true);
  assert.equal(gate.accepts(confirm()), false, "the request was consumed");

  gate.request("hooks_connect");
  assert.equal(gate.accepts(confirm({ action: "online_test" })), false, "asked for elsewhere");
  assert.equal(gate.accepts(confirm({ nonce: "" })), false);
  assert.equal(gate.accepts(confirm({ nonce: "x".repeat(65) })), false);
  assert.equal(gate.accepts(confirm({ nonce: 7 })), false);
  assert.equal(gate.accepts(confirm({ text: 7 })), false);
  assert.equal(gate.accepts(confirm({ nonce: "x".repeat(64) })), true);

  gate.request("hooks_connect");
  gate.clear();
  assert.equal(gate.accepts(confirm()), false);
  gate.request("voice_sample");
  gate.request("not_an_action");
  assert.equal(gate.accepts(confirm({ action: "voice_sample" })), false, "plain actions never get a confirm here");
  assert.equal(gate.accepts(null), false);

  assert.deepEqual(gate.answer("n", true), { type: "setup_action", nonce: "n", yes: true });
  assert.equal(gate.answer("n", "yes").yes, false);
  assert.equal(gate.answer("n", 1).yes, false);
  assert.equal(gate.answer("n", false).yes, false);
});

test("resultText and shouldRefresh", () => {
  assert.equal(H.resultText({ ok: true, detail: "Connected." }), "Connected.");
  assert.equal(H.resultText({ ok: false, detail: "No mic." }), "Didn't work: No mic.");
  assert.equal(H.resultText({ ok: true, detail: 5 }), "");
  assert.equal(H.shouldRefresh({ hidden: false, connected: true }), true);
  assert.equal(H.shouldRefresh({ hidden: true, connected: true }), false);
  assert.equal(H.shouldRefresh({ hidden: false, connected: false }), false);
});

test("every message the Health section builds validates against body_to_core", () => {
  const rows = [
    ["voice", "ok", "voice_sample"], ["autostart", "off", "autostart"], ["claude_code", "off", "hooks_connect"],
    ["claude_code", "ok", "hooks_disconnect"], ["online", "warn", "online_test"], ["eyes", "ok", "eyes_test"], ["mic", "ok", "mic_check"],
  ];
  for (const [key, status, fix] of rows) {
    const req = H.fixRequest({ key, label: "x", status, detail: "", fix }, { voice: "alba" });
    assert.ok(validateBody(req.message), `${fix}: ${JSON.stringify(validateBody.errors)}`);
  }
  const msgs = [
    H.keyRequest("AIzaCANARY123456"),
    { type: "setup_request", action: "online_key_clear", args: {} },
    { type: "health_get" },
    new H.ConfirmGate().answer("n0nce-abc", true),
    new H.ConfirmGate().answer("n0nce-abc", false),
  ];
  for (const m of msgs) assert.ok(validateBody(m), `${m.type}: ${JSON.stringify(validateBody.errors)}`);
});
