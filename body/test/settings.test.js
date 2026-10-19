const test = require("node:test");
const assert = require("node:assert/strict");
const L = require("../src/settings-logic");

test("sourceLabel maps the fact source to a badge and conflict wins", () => {
  assert.deepEqual(L.sourceLabel({ source: "user_said", status: "active" }), { label: "user said", tone: "user" });
  assert.deepEqual(L.sourceLabel({ source: "inferred", status: "active" }), { label: "inferred", tone: "inferred" });
  assert.deepEqual(L.sourceLabel({ source: "looked_up", status: "active" }), { label: "looked up", tone: "lookup" });
  assert.deepEqual(L.sourceLabel({ source: "user_said", status: "conflict" }), { label: "conflicting", tone: "conflict" });
  assert.deepEqual(L.sourceLabel({ source: "weird", status: "active" }), { label: "weird", tone: "inferred" });
  assert.deepEqual(L.sourceLabel({}), { label: "unknown", tone: "inferred" });
});

test("formatWhen prints a short local-or-utc date and time, and dashes for junk", () => {
  const ts = Date.UTC(2026, 9, 18, 14, 5) / 1000;
  assert.equal(L.formatWhen(ts, { utc: true }), "18 Oct 2026, 14:05");
  assert.equal(L.formatWhen(Date.UTC(2026, 0, 3, 0, 9) / 1000, { utc: true }), "3 Jan 2026, 00:09");
  assert.equal(L.formatWhen(null), "unknown");
  assert.equal(L.formatWhen("x"), "unknown");
  assert.equal(L.formatWhen(Infinity), "unknown");
});

test("validateQuietHours mirrors the core's HH:MM-HH:MM and treats blank as off", () => {
  assert.deepEqual(L.validateQuietHours("23:30-08:00"), { ok: true, value: "23:30-08:00" });
  assert.deepEqual(L.validateQuietHours("  00:00-23:59 "), { ok: true, value: "00:00-23:59" });
  assert.deepEqual(L.validateQuietHours(""), { ok: true, value: null });
  assert.deepEqual(L.validateQuietHours("   "), { ok: true, value: null });
  for (const bad of ["24:00-08:00", "23:60-08:00", "9:00-17:00", "23:30–08:00", "23:30-08", "abc", "23:30-08:00-"]) {
    const r = L.validateQuietHours(bad);
    assert.equal(r.ok, false, bad);
    assert.match(r.error, /HH:MM-HH:MM/);
  }
});

test("settingPatch builds one-key patches and rejects invalid input", () => {
  assert.deepEqual(L.settingPatch("mode", "panel"), { ok: true, patch: { mode: "panel" } });
  assert.equal(L.settingPatch("mode", "fullscreen").ok, false);
  assert.deepEqual(L.settingPatch("voice", "  marius "), { ok: true, patch: { voice: "marius" } });
  assert.equal(L.settingPatch("voice", "  ").ok, false);
  assert.deepEqual(L.settingPatch("volume", "0.456"), { ok: true, patch: { volume: 0.46 } });
  assert.deepEqual(L.settingPatch("volume", 7), { ok: true, patch: { volume: 1 } });
  assert.deepEqual(L.settingPatch("volume", -1), { ok: true, patch: { volume: 0 } });
  assert.equal(L.settingPatch("volume", "loud").ok, false);
  assert.deepEqual(L.settingPatch("autostart", 1), { ok: true, patch: { autostart: true } });
  assert.deepEqual(L.settingPatch("mute_mic", false), { ok: true, patch: { mute_mic: false } });
  assert.deepEqual(L.settingPatch("quiet_hours", ""), { ok: true, patch: { quiet_hours: null } });
  assert.equal(L.settingPatch("quiet_hours", "25:00-01:00").ok, false);
  assert.equal(L.settingPatch("buddy_position", [1, 2]).ok, false);
});

test("hotkeyText names the shortcut or the conflict", () => {
  assert.equal(L.hotkeyText("Control+Alt+F9"), "Control+Alt+F9");
  assert.equal(L.hotkeyText(null), "none — another app is using the shortcuts");
  assert.equal(L.hotkeyText(undefined), "none — another app is using the shortcuts");
});

test("ErrorRouter sends an error to the setting it names, else to the latest pending action", () => {
  const r = new L.ErrorRouter();
  r.sent("setting:mode");
  assert.equal(r.route("invalid value for setting 'quiet_hours': 'x'"), "setting:quiet_hours");
  r.sent("setting:autostart");
  assert.equal(r.route("couldn't change autostart: denied"), "setting:autostart");
  r.sent("fact:3");
  r.sent("fact:4");
  assert.equal(r.route("fact 4 not found"), "fact:4");
  assert.equal(r.route("anything"), "global");
});

test("ErrorRouter forgets pending actions once the core answers", () => {
  const r = new L.ErrorRouter();
  r.sent("fact:3");
  r.settle();
  assert.equal(r.route("late"), "global");
  assert.equal(r.route("unknown setting 'zzz'"), "global");
});

test("episodeText and dueText", () => {
  const t = Date.UTC(2026, 9, 18, 9, 0) / 1000;
  assert.equal(L.episodeText({ summary: " we talked ", started_at: t }), "we talked");
  assert.equal(L.episodeText({ summary: "", started_at: t }), "(no summary)");
  assert.equal(L.dueText({ due_at: t }, { utc: true }), "due 18 Oct 2026, 09:00");
});

test("memory fixtures with episodes and threads are tolerated", () => {
  const msg = { type: "memories", facts: [], transcript_days: [], episodes: [{ id: 1, started_at: 1, ended_at: 2, summary: "s" }] };
  assert.deepEqual(L.memoryLists(msg).threads, []);
  assert.equal(L.memoryLists(msg).episodes.length, 1);
  assert.deepEqual(L.memoryLists({ type: "memories" }), { facts: [], episodes: [], threads: [], days: [], notes: [] });
});

test("voiceOptions lists the presets and keeps a non-preset current voice as a custom entry", () => {
  assert.deepEqual(L.VOICES, ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]);
  const presets = L.VOICES.map((v) => ({ value: v, label: v }));
  assert.deepEqual(L.voiceOptions("alba"), presets);
  assert.deepEqual(L.voiceOptions(undefined), presets);
  assert.deepEqual(L.voiceOptions(""), presets);
  assert.deepEqual(L.voiceOptions("my_clip.wav"), [{ value: "my_clip.wav", label: "my_clip.wav (custom)" }, ...presets]);
});

test("voiceOptions takes the core's voice list, with the static presets as the fallback", () => {
  const fallback = L.VOICES.map((v) => ({ value: v, label: v }));
  assert.deepEqual(L.voiceOptions("alba", ["alba", "eve"]).map((o) => o.value), ["alba", "eve"]);
  assert.deepEqual(L.voiceOptions("marius", ["alba", "eve"]), [
    { value: "marius", label: "marius (custom)" }, { value: "alba", label: "alba" }, { value: "eve", label: "eve" }]);
  assert.deepEqual(L.voiceOptions("alba", []), fallback);
  assert.deepEqual(L.voiceOptions("alba"), fallback);
  assert.deepEqual(L.voiceOptions("alba", [3]), fallback);
  assert.deepEqual(L.voiceOptions("alba", "alba"), fallback);
});

test("the settings page no longer says the voice needs a restart", () => {
  const html = require("node:fs").readFileSync(require("node:path").join(__dirname, "..", "settings.html"), "utf8");
  assert.ok(!html.includes("Applies after a restart."));
  assert.ok(html.includes("Takes effect on her next line."));
});

test("DeferWhileFocused applies at once when unfocused, and holds only the latest value while focused", () => {
  const d = new L.DeferWhileFocused();
  assert.deepEqual(d.offer("alba", false), { value: "alba" });
  assert.equal(d.take(), null);
  assert.equal(d.offer("jean", true), null);
  assert.equal(d.offer(null, true), null);
  assert.deepEqual(d.take(), { value: null }, "the latest offer wins, even a null voice");
  assert.equal(d.take(), null, "taken once");
  d.offer("jean", true);
  assert.deepEqual(d.offer("marius", false), { value: "marius" });
  assert.equal(d.take(), null, "an applied offer clears the stale pending one");
});

test("parsePatternLines splits lines, trims, drops blanks and duplicates", () => {
  assert.deepEqual(L.parsePatternLines(" bank \r\n\n(?:secret|vault)\nbank\n  "), { ok: true, patterns: ["bank", "(?:secret|vault)"] });
  assert.deepEqual(L.parsePatternLines(""), { ok: true, patterns: [] });
  assert.deepEqual(L.parsePatternLines(undefined), { ok: true, patterns: [] });
});

test("parsePatternLines rejects a line that is not a JS regex and names it", () => {
  const r = L.parsePatternLines("fine\n(unclosed\nalso fine");
  assert.equal(r.ok, false);
  assert.match(r.error, /line 2/i);
  assert.match(r.error, /\(unclosed/);
  assert.equal(L.parsePatternLines("[a-").ok, false);
});

test("parsePatternLines enforces the core's count and length limits", () => {
  assert.equal(L.parsePatternLines("x".repeat(301)).ok, false);
  assert.equal(L.parsePatternLines("x".repeat(300)).ok, true);
  const many = Array.from({ length: 101 }, (_, i) => `p${i}`).join("\n");
  assert.equal(L.parsePatternLines(many).ok, false);
  assert.equal(L.parsePatternLines(many.split("\n").slice(0, 100).join("\n")).ok, true);
});

test("settingPatch handles the Eyes settings", () => {
  assert.deepEqual(L.settingPatch("eyes_enabled", false), { ok: true, patch: { eyes_enabled: false } });
  assert.deepEqual(L.settingPatch("glance_daily_cap", "150"), { ok: true, patch: { glance_daily_cap: 150 } });
  assert.deepEqual(L.settingPatch("glance_daily_cap", 1), { ok: true, patch: { glance_daily_cap: 1 } });
  assert.deepEqual(L.settingPatch("glance_daily_cap", 1000), { ok: true, patch: { glance_daily_cap: 1000 } });
  for (const bad of [0, 1001, 1.5, "", "  ", "many", NaN, null]) assert.equal(L.settingPatch("glance_daily_cap", bad).ok, false, String(bad));
  assert.deepEqual(L.settingPatch("glance_min_gap_s", "45"), { ok: true, patch: { glance_min_gap_s: 45 } });
  assert.deepEqual(L.settingPatch("glance_min_gap_s", 10), { ok: true, patch: { glance_min_gap_s: 10 } });
  assert.deepEqual(L.settingPatch("glance_min_gap_s", 3600), { ok: true, patch: { glance_min_gap_s: 3600 } });
  for (const bad of [9, 3601, 12.5, "", "slow"]) assert.equal(L.settingPatch("glance_min_gap_s", bad).ok, false, String(bad));
  assert.deepEqual(L.settingPatch("eyes_deny_titles", "bank\n\nvault"), { ok: true, patch: { eyes_deny_titles: ["bank", "vault"] } });
  assert.deepEqual(L.settingPatch("eyes_deny_processes", ""), { ok: true, patch: { eyes_deny_processes: [] } });
  assert.equal(L.settingPatch("eyes_deny_titles", "(oops").ok, false);
  assert.equal(L.settingPatch("dont_look", true).ok, false);
});

test("ErrorRouter sends a bad deny pattern to its textarea", () => {
  const r = new L.ErrorRouter();
  assert.equal(r.route("invalid pattern in 'eyes_deny_titles': '(x'"), "setting:eyes_deny_titles");
  assert.equal(r.route("invalid value for setting 'glance_daily_cap': 0"), "setting:glance_daily_cap");
});

test("eyesCountText says how many looks she has used today", () => {
  assert.equal(L.eyesCountText({ glances_today: 12, cap: 150 }), "12 of 150 today");
  assert.equal(L.eyesCountText({ glances_today: 0, cap: 0 }), "0 of 0 today");
  assert.equal(L.eyesCountText(null), "—");
  assert.equal(L.eyesCountText({ glances_today: "x", cap: 150 }), "—");
});

test("eyesStatusText reflects off and dont-look", () => {
  assert.equal(L.eyesStatusText({ enabled: true, dont_look: false }), "");
  assert.match(L.eyesStatusText({ enabled: true, dont_look: true }), /look/i);
  assert.match(L.eyesStatusText({ enabled: false, dont_look: false }), /off/i);
});

test("glanceRows shows time, app and activity, newest first as given, tolerating junk", () => {
  const t = Date.UTC(2026, 9, 18, 14, 5) / 1000;
  const msg = { type: "memories", glances: [
    { id: 7, at: t, app: "Code.exe", title: "secret.py", activity: " fixing a bug ", details: "d", notable: null },
    { id: 8, at: t, app: "", title: "", activity: "", details: "", notable: "x" },
    null, { id: "nope" },
  ] };
  assert.deepEqual(L.glanceRows(msg, { utc: true }), [
    { id: 7, text: "fixing a bug", meta: "18 Oct 2026, 14:05 · Code.exe" },
    { id: 8, text: "(no description)", meta: "18 Oct 2026, 14:05 · unknown app" },
  ]);
  assert.deepEqual(L.glanceRows({ type: "memories" }), []);
});

test("agent_logs patches are validated like the core and errors route to their control", () => {
  const entry = { name: "colab", glob: "G:/runs/**/*.out", done_re: "DONE", fail_re: "FAILED", lost_re: "" };
  assert.deepEqual(L.settingPatch("agent_logs", [entry]), { ok: true, patch: { agent_logs: [entry] } });
  assert.deepEqual(L.settingPatch("agent_logs", []), { ok: true, patch: { agent_logs: [] } });
  assert.equal(L.settingPatch("agent_logs", [{ ...entry, done_re: "(oops" }]).ok, false);
  assert.equal(L.settingPatch("agent_logs", [{ ...entry, name: "" }]).ok, false);
  const r = new L.ErrorRouter();
  assert.equal(r.route("invalid pattern in agent_logs 'colab' done_re: missing )"), "setting:agent_logs");
  assert.equal(r.route("invalid value for setting 'agent_logs': []"), "setting:agent_logs");
});

test("memoryLists returns the notes, and tolerates a core without them", () => {
  const notes = [{ id: 1, at: 5, text: "buy tea" }];
  assert.deepEqual(L.memoryLists({ type: "memories", facts: [], transcript_days: [], notes }).notes, notes);
  assert.deepEqual(L.memoryLists({ type: "memories", notes: "x" }).notes, []);
});

test("hands_online is a boolean setting; weather_place trims, empties to null and is capped at 80", () => {
  assert.deepEqual(L.settingPatch("hands_online", false), { ok: true, patch: { hands_online: false } });
  assert.deepEqual(L.settingPatch("hands_online", 1), { ok: true, patch: { hands_online: true } });
  assert.deepEqual(L.settingPatch("weather_place", "  Chennai "), { ok: true, patch: { weather_place: "Chennai" } });
  assert.deepEqual(L.settingPatch("weather_place", "  "), { ok: true, patch: { weather_place: null } });
  assert.equal(L.settingPatch("weather_place", "x".repeat(81)).ok, false);
  assert.equal(L.settingPatch("weather_place", 5).ok, false);
});

test("apps and commands go through the hands validators", () => {
  assert.deepEqual(L.settingPatch("apps", [{ name: "Notes", target: "notepad.exe" }]),
    { ok: true, patch: { apps: [{ name: "Notes", target: "notepad.exe" }] } });
  assert.equal(L.settingPatch("apps", [{ name: "", target: "x" }]).ok, false);
  const r = L.settingPatch("commands", [{ name: "b", line: "robocopy a b", cwd: "", timeout_s: 20 }]);
  assert.deepEqual(r, { ok: true, patch: { commands: [{ name: "b", argv: ["robocopy", "a", "b"], cwd: null, timeout_s: 20 }] } });
  assert.equal(L.settingPatch("commands", [{ name: "b", line: 'x "y', cwd: "", timeout_s: 20 }]).ok, false);
});

test("a core error about a hands setting is routed to its control", () => {
  for (const key of ["hands_online", "weather_place", "apps", "commands", "tools_enabled", "tools_ask"]) {
    const r = new L.ErrorRouter();
    assert.equal(r.route(`invalid value for setting '${key}': 5`), `setting:${key}`, key);
  }
});

test("an actions fixture is a valid shape for actionText", () => {
  const fs = require("node:fs");
  const path = require("node:path");
  const H = require("../src/hands-logic");
  const msg = JSON.parse(fs.readFileSync(path.join(__dirname, "fixtures", "core_actions.json"), "utf8"));
  assert.equal(msg.type, "actions");
  assert.ok(msg.items.length >= 2);
  for (const a of msg.items) assert.ok(H.actionText(a).includes(a.summary));
});

test("learning_enabled is a boolean setting", () => {
  assert.deepEqual(L.settingPatch("learning_enabled", false), { ok: true, patch: { learning_enabled: false } });
  assert.deepEqual(L.settingPatch("learning_enabled", 1), { ok: true, patch: { learning_enabled: true } });
});
