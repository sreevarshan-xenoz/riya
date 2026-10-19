"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const A = require("../src/agents-logic");

const agent = (over) => ({ id: "a", kind: "claude", name: "riya", status: "busy", since: 0, detail: "", needs_attention: false, ...over });

test("agentChip covers every status with a calm tone, and falls back to unknown", () => {
  for (const s of ["busy", "waiting", "idle", "finished", "failed", "unknown", "ended", "running", "lost", "stalled"]) {
    const chip = A.agentChip(s);
    assert.equal(chip.label, s);
    assert.match(chip.tone, /^[a-z]+$/);
  }
  assert.equal(A.agentChip("waiting").tone, "alert");
  for (const s of ["failed", "lost", "stalled"]) assert.equal(A.agentChip(s).tone, "bad", s);
  assert.equal(A.agentChip("finished").tone, "good");
  assert.deepEqual(A.agentChip("<img src=x onerror=1>"), { label: "unknown", tone: "dim" }, "the label never echoes an unknown string");
  assert.deepEqual(A.agentChip(undefined), { label: "unknown", tone: "dim" });
  assert.deepEqual(A.agentChip("toString"), { label: "unknown", tone: "dim" });
});

test("relativeAge is short, coarse, and blank for junk", () => {
  assert.equal(A.relativeAge(0), "just now");
  assert.equal(A.relativeAge(44_000), "just now");
  assert.equal(A.relativeAge(45_000), "1m");
  assert.equal(A.relativeAge(5 * 60_000 + 1), "5m");
  assert.equal(A.relativeAge(59 * 60_000), "59m");
  assert.equal(A.relativeAge(60 * 60_000), "1h");
  assert.equal(A.relativeAge(26 * 3600_000), "1d");
  assert.equal(A.relativeAge(-5000), "just now", "a clock a little ahead is not an error");
  for (const bad of [NaN, Infinity, null, undefined, "5"]) assert.equal(A.relativeAge(bad), "", String(bad));
});

test("badgeCount counts needs_attention === true only", () => {
  assert.equal(A.badgeCount([]), 0);
  assert.equal(A.badgeCount([agent({ needs_attention: true }), agent({}), agent({ needs_attention: true })]), 2);
  assert.equal(A.badgeCount([agent({ needs_attention: "yes" }), null, 3]), 0);
  assert.equal(A.badgeCount(undefined), 0);
});

test("sortAgents: attention, then live, then ended, stable inside each group, without mutating", () => {
  const items = [
    agent({ id: "e1", status: "ended" }), agent({ id: "l1" }), agent({ id: "a1", status: "waiting", needs_attention: true }),
    agent({ id: "l2", status: "idle" }), agent({ id: "a2", status: "failed", needs_attention: true }), agent({ id: "e2", status: "ended" }),
  ];
  const copy = JSON.parse(JSON.stringify(items));
  assert.deepEqual(A.sortAgents(items).map((a) => a.id), ["a1", "a2", "l1", "l2", "e1", "e2"]);
  assert.deepEqual(items, copy);
  assert.deepEqual(A.sortAgents(undefined), []);
  assert.deepEqual(A.sortAgents([null, 3, agent({ id: "x" })]).map((a) => a.id), ["x"]);
});

test("hookStatus says whether events came in the last 24 h", () => {
  const now = Date.UTC(2026, 9, 18, 12, 0);
  const sec = (ms) => ms / 1000;
  assert.equal(A.hookStatus(null, now), "Events received in the last 24 h: never");
  assert.equal(A.hookStatus(undefined, now), "Events received in the last 24 h: never");
  assert.equal(A.hookStatus(sec(now - 3600_000), now), "Events received in the last 24 h: yes");
  assert.equal(A.hookStatus(sec(now - 24 * 3600_000), now), "Events received in the last 24 h: yes");
  assert.equal(A.hookStatus(sec(now - 24 * 3600_000 - 1000), now), "Events received in the last 24 h: no");
  assert.equal(A.hookStatus(sec(now + 5000), now), "Events received in the last 24 h: yes");
  assert.equal(A.hookStatus("x", now), "Events received in the last 24 h: never");
});

test("agentTitle is the hover text: detail, else the name and status", () => {
  assert.equal(A.agentTitle(agent({ detail: "needs approval to run npm test" })), "needs approval to run npm test");
  assert.equal(A.agentTitle(agent({ detail: "  " })), "riya: busy");
  assert.equal(A.agentTitle(agent({ detail: 5 })), "riya: busy");
});

test("ageRefreshDue: at most every 30 s, and never while hidden or empty", () => {
  assert.equal(A.AGE_REFRESH_MS, 30_000);
  assert.equal(A.ageRefreshDue({ last: 0, now: 29_999, hidden: false, count: 2 }), false);
  assert.equal(A.ageRefreshDue({ last: 0, now: 30_000, hidden: false, count: 2 }), true);
  assert.equal(A.ageRefreshDue({ last: 0, now: 60_000, hidden: true, count: 2 }), false);
  assert.equal(A.ageRefreshDue({ last: 0, now: 60_000, hidden: false, count: 0 }), false);
  assert.equal(A.ageRefreshDue({ last: null, now: 1, hidden: false, count: 1 }), true);
});

const good = { name: "colab", glob: "G:/runs/**/*.out", done_re: "DONE", fail_re: "FAILED|Traceback", lost_re: "" };

test("validateAgentLogs mirrors the core: shape, limits, unique names, compiling patterns", () => {
  assert.deepEqual(A.validateAgentLogs([good]), { ok: true, logs: [good] });
  assert.deepEqual(A.validateAgentLogs([]), { ok: true, logs: [] });
  const trimmed = A.validateAgentLogs([{ ...good, name: "  colab ", glob: " x/*.out ", done_re: " DONE " }]);
  assert.deepEqual(trimmed.logs[0], { ...good, glob: "x/*.out", done_re: "DONE" }, "name and glob are trimmed, patterns kept as typed except for edges");
  const bad = (list, re) => {
    const r = A.validateAgentLogs(list);
    assert.equal(r.ok, false, JSON.stringify(list));
    assert.match(r.error, re);
  };
  bad("nope", /list/i);
  bad([{ ...good, name: "" }], /name/i);
  bad([{ ...good, name: "n".repeat(41) }], /name/i);
  bad([{ ...good, glob: "  " }], /glob/i);
  bad([good, { ...good }], /twice|unique|already/i);
  bad([{ ...good, done_re: "(" }], /done_re|done/i);
  bad([{ ...good, fail_re: "[" }], /fail/i);
  bad([{ ...good, lost_re: "x".repeat(301) }], /300/);
  bad([{ ...good, extra: 1 }], /entry 1/i);
  bad([null], /entry 1/i);
  bad([{ ...good, name: 5 }], /entry 1/i);
  bad(Array.from({ length: 21 }, (_, i) => ({ ...good, name: `n${i}` })), /at most 20/i);
});

test("an agents fixture is a valid shape for the helpers", () => {
  const msg = JSON.parse(fs.readFileSync(path.join(__dirname, "fixtures", "core_agents.json"), "utf8"));
  assert.equal(msg.type, "agents");
  assert.equal(A.badgeCount(msg.items), 2);
  const sorted = A.sortAgents(msg.items);
  assert.equal(sorted[sorted.length - 1].status, "ended");
  assert.ok(sorted.slice(0, 2).every((a) => a.needs_attention));
  assert.match(A.hookStatus(msg.hooks_seen_at, msg.hooks_seen_at * 1000 + 1000), /yes$/);
});

test("agent text only reaches the DOM through textContent: no HTML sinks in the body scripts, and the scripts load in order", () => {
  const read = (f) => fs.readFileSync(path.join(__dirname, "..", f), "utf8");
  for (const f of ["renderer.js", "settings.js", "src/agents-logic.js", "src/settings-logic.js", "src/hands-logic.js", "src/bubble.js"]) {
    assert.doesNotMatch(read(f), /innerHTML|outerHTML|insertAdjacentHTML|document\.write|createContextualFragment/, f);
  }
  const before = (html, a, b) => html.indexOf(a) !== -1 && html.indexOf(a) < html.indexOf(b);
  assert.ok(before(read("index.html"), "src/agents-logic.js", "renderer.js"));
  assert.ok(before(read("settings.html"), "src/agents-logic.js", "src/settings-logic.js"));
  assert.ok(before(read("index.html"), "src/hands-logic.js", "renderer.js"));
  assert.ok(before(read("index.html"), "src/bubble.js", "renderer.js"));
  assert.ok(before(read("settings.html"), "src/hands-logic.js", "settings.js"));
  assert.ok(before(read("settings.html"), "src/hands-logic.js", "src/settings-logic.js"), "settings-logic reads RiyaHands at load");
});

test("the token path and setup command are shown exactly, backslashes intact", () => {
  assert.equal(A.TOKEN_PATH, String.raw`%LOCALAPPDATA%\Riya\hook_token`);
  assert.equal(A.SETUP_COMMAND, "python -m riya.agents.install_hooks");
});

test("a dropped core empties the list (so no stale badge) and hook status turns unknown", () => {
  const items = [agent({ needs_attention: true })];
  assert.deepEqual(A.agentsWhileConnected(items, false), []);
  assert.equal(A.agentsWhileConnected(items, true), items);
  assert.equal(A.badgeCount(A.agentsWhileConnected(items, false)), 0);
  assert.match(A.hookStatus(1000, 2000, false), /unknown/);
  assert.match(A.hookStatus(1000, 2000, true), /yes$/);
});

test("relativeAge never shows 0h: minutes stay minutes until a full hour", () => {
  assert.equal(A.relativeAge(59.6 * 60_000), "59m");
  assert.equal(A.relativeAge(3600_000 - 1), "59m");
  assert.equal(A.relativeAge(3600_000), "1h");
  assert.equal(A.relativeAge(45_000), "1m");
  assert.equal(A.relativeAge(23.9 * 3600_000), "23h");
  assert.equal(A.relativeAge(24 * 3600_000), "1d");
  for (let ms = 0; ms < 3 * 3600_000; ms += 7_000) assert.doesNotMatch(A.relativeAge(ms), /^0/, String(ms));
});

test("receiverProblems: a busy port and wrong-token rejections, only while connected (final review M4, M5)", () => {
  assert.deepEqual(A.receiverProblems({ receiver_ok: true, rejected_recent: 0 }, true), []);
  assert.deepEqual(A.receiverProblems({ receiver_ok: false, rejected_recent: 0 }, true),
    ["receiver not running (port 8766 busy)"]);
  assert.deepEqual(A.receiverProblems({ receiver_ok: true, rejected_recent: 3 }, true),
    ["events rejected (wrong token): 3, re-run the installer"]);
  assert.deepEqual(A.receiverProblems({ receiver_ok: false, rejected_recent: 1 }, true).length, 2);
  assert.deepEqual(A.receiverProblems({ receiver_ok: false, rejected_recent: 3 }, false), []);
  assert.deepEqual(A.receiverProblems({}, true), [], "an older core without the fields: nothing to say");
  assert.deepEqual(A.receiverProblems({ receiver_ok: "no", rejected_recent: "<b>9</b>" }, true), []);
  assert.deepEqual(A.receiverProblems(null, true), []);
});

test("the settings page shows the receiver problems in their own row", () => {
  const html = fs.readFileSync(path.join(__dirname, "..", "settings.html"), "utf8");
  const js = fs.readFileSync(path.join(__dirname, "..", "settings.js"), "utf8");
  assert.match(html, /id="agents-receiver"/);
  assert.match(js, /receiverProblems\(/);
});
