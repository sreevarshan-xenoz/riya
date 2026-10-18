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
  assert.deepEqual(L.memoryLists({ type: "memories" }), { facts: [], episodes: [], threads: [], days: [] });
});

test("voiceOptions lists the presets and keeps a non-preset current voice as a custom entry", () => {
  assert.deepEqual(L.VOICES, ["alba", "marius", "javert", "jean", "fantine", "cosette", "eponine", "azelma"]);
  const presets = L.VOICES.map((v) => ({ value: v, label: v }));
  assert.deepEqual(L.voiceOptions("alba"), presets);
  assert.deepEqual(L.voiceOptions(undefined), presets);
  assert.deepEqual(L.voiceOptions(""), presets);
  assert.deepEqual(L.voiceOptions("my_clip.wav"), [{ value: "my_clip.wav", label: "my_clip.wav (custom)" }, ...presets]);
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
