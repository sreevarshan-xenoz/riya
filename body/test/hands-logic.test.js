"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const H = require("../src/hands-logic");

test("TOOLS lists all 17 tools with their tiers", () => {
  assert.equal(H.TOOLS.length, 17);
  const tier = (t) => H.TOOLS.filter((x) => x.tier === t).map((x) => x.name);
  assert.equal(tier(0).length, 10);
  assert.deepEqual(tier(1), ["open_url", "open_app", "media", "volume", "clipboard_copy"]);
  assert.deepEqual(tier(2), ["run_command", "write_file"]);
  assert.ok(H.TOOLS.every((t) => typeof t.label === "string" && t.label));
});

test("splitCommandLine splits on spaces, keeps quoted spans whole and returns null for an unbalanced quote", () => {
  const line = String.raw`robocopy "C:\My Src" D:\bak /MIR`;
  assert.deepEqual(H.splitCommandLine(line), ["robocopy", String.raw`C:\My Src`, String.raw`D:\bak`, "/MIR"]);
  assert.equal(H.splitCommandLine(line).length, 4);
  assert.deepEqual(H.splitCommandLine("  a   b  "), ["a", "b"]);
  assert.deepEqual(H.splitCommandLine(""), []);
  assert.deepEqual(H.splitCommandLine('echo "" x'), ["echo", "", "x"]);
  assert.deepEqual(H.splitCommandLine("it's fine"), ["it's", "fine"]);
  assert.deepEqual(H.splitCommandLine('a"b c"d'), ["ab cd"]);
  assert.equal(H.splitCommandLine('robocopy "C:\\My Src D:\\bak'), null);
  assert.equal(H.splitCommandLine(5), null);
});

test("joinCommandLine quotes what needs it and round-trips through splitCommandLine", () => {
  const argv = ["robocopy", String.raw`C:\My Src`, "D:\\bak", "/MIR", ""];
  assert.equal(H.joinCommandLine(argv), String.raw`robocopy "C:\My Src" D:\bak /MIR ""`);
  assert.deepEqual(H.splitCommandLine(H.joinCommandLine(argv)), argv);
});

test("toolRows: the ask toggle is only on T1 rows, T2 rows say they always ask", () => {
  const rows = H.toolRows({ tools_enabled: { calc: false, run_command: true }, tools_ask: ["media", "calc"] });
  assert.equal(rows.length, 17);
  const by = Object.fromEntries(rows.map((r) => [r.name, r]));
  assert.equal(by.calc.enabled, false);
  assert.equal(by.time_now.enabled, true, "missing entries take the default");
  assert.equal(by.write_file.enabled, false, "T2 tools are off by default");
  assert.equal(by.run_command.enabled, true);
  assert.equal(by.media.askToggle, true);
  assert.equal(by.media.ask, true);
  assert.equal(by.volume.askToggle, true);
  assert.equal(by.volume.ask, false);
  assert.equal(by.calc.askToggle, false);
  assert.equal(by.calc.ask, false, "a non-T1 name in tools_ask never shows as asking");
  assert.equal(by.run_command.askToggle, false);
  assert.equal(by.run_command.note, "always asks");
  assert.equal(by.write_file.note, "always asks");
  assert.equal(by.media.note, "");
  assert.equal(H.toolRows(null).length, 17);
  assert.equal(H.toolRows({}).find((r) => r.name === "weather").enabled, true);
});

test("toolsPatch: enabled is a partial map, ask edits the T1 list and anything else is refused", () => {
  const s = { tools_enabled: {}, tools_ask: ["media"] };
  assert.deepEqual(H.toolsPatch(s, "calc", "enabled", false), { tools_enabled: { calc: false } });
  assert.deepEqual(H.toolsPatch(s, "run_command", "enabled", true), { tools_enabled: { run_command: true } });
  assert.deepEqual(H.toolsPatch(s, "volume", "ask", true), { tools_ask: ["media", "volume"] });
  assert.deepEqual(H.toolsPatch(s, "media", "ask", false), { tools_ask: [] });
  assert.deepEqual(H.toolsPatch(s, "media", "ask", true), { tools_ask: ["media"] });
  assert.equal(H.toolsPatch(s, "calc", "ask", true), null, "only T1 tools may ask");
  assert.equal(H.toolsPatch(s, "run_command", "ask", true), null, "T2 always asks");
  assert.equal(H.toolsPatch(s, "nope", "enabled", true), null);
  assert.equal(H.toolsPatch(s, "calc", "weird", true), null);
  assert.deepEqual(H.toolsPatch(null, "media", "ask", true), { tools_ask: ["media"] });
  assert.deepEqual(H.toolsPatch(s, "calc", "enabled", "yes"), { tools_enabled: { calc: true } }, "always a real boolean");
});

test("validateApps mirrors the core: keys, 40/500 char limits, 50 entries, unique names", () => {
  const ok = H.validateApps([{ name: " Notes ", target: " notepad.exe " }]);
  assert.deepEqual(ok, { ok: true, apps: [{ name: "Notes", target: "notepad.exe" }] });
  assert.deepEqual(H.validateApps([]), { ok: true, apps: [] });
  const bad = (list, re) => { const r = H.validateApps(list); assert.equal(r.ok, false); assert.match(r.error, re); };
  bad("x", /list/i);
  bad([{ name: "", target: "x" }], /app 1.*name/i);
  bad([{ name: "a", target: "  " }], /app 1.*target/i);
  bad([{ name: "x".repeat(41), target: "t" }], /40/);
  bad([{ name: "n", target: "t".repeat(501) }], /500/);
  bad([{ name: "Notes", target: "a" }, { name: "notes", target: "b" }], /twice|unique/i);
  bad(Array.from({ length: 51 }, (_, i) => ({ name: `n${i}`, target: "t" })), /at most 50/i);
  bad([null], /app 1/i);
  bad([{ name: 5, target: "t" }], /app 1/i);
  assert.equal(H.validateApps(Array.from({ length: 50 }, (_, i) => ({ name: `n${i}`, target: "t" }))).ok, true);
});

test("validateCommands mirrors the core: argv from the line, cwd, timeout 1-60, 20 entries, unique names", () => {
  const ok = H.validateCommands([{ name: " backup ", line: String.raw`robocopy "C:\My Src" D:\bak /MIR`, cwd: " ", timeout_s: "30" }]);
  assert.deepEqual(ok, { ok: true, commands: [{ name: "backup", argv: ["robocopy", String.raw`C:\My Src`, "D:\\bak", "/MIR"], cwd: null, timeout_s: 30 }] });
  const withArgv = H.validateCommands([{ name: "a", argv: ["x", "y z"], cwd: " C:\\w ", timeout_s: 60 }]);
  assert.deepEqual(withArgv.commands, [{ name: "a", argv: ["x", "y z"], cwd: "C:\\w", timeout_s: 60 }]);
  const bad = (list, re) => { const r = H.validateCommands(list); assert.equal(r.ok, false, JSON.stringify(list)); assert.match(r.error, re); };
  const c = { name: "a", line: "x", cwd: "", timeout_s: 10 };
  bad("x", /list/i);
  bad([{ ...c, name: "" }], /command 1.*name/i);
  bad([{ ...c, name: "n".repeat(41) }], /40/);
  bad([{ ...c, line: "" }], /command 1.*command line/i);
  bad([{ ...c, line: 'x "y' }], /quote/i);
  bad([{ ...c, line: "x ".repeat(21) }], /at most 20/i);
  bad([{ ...c, line: "x " + "y".repeat(501) }], /500/);
  bad([{ ...c, line: 'x ""' }], /empty/i);
  bad([{ ...c, cwd: "c".repeat(501) }], /500/);
  bad([{ ...c, timeout_s: 0 }], /1.*60/);
  bad([{ ...c, timeout_s: 61 }], /1.*60/);
  bad([{ ...c, timeout_s: "abc" }], /1.*60/);
  bad([{ ...c, timeout_s: "" }], /1.*60/);
  bad([c, { ...c, name: "A" }], /twice|unique/i);
  bad(Array.from({ length: 21 }, (_, i) => ({ ...c, name: `n${i}` })), /at most 20/i);
  bad([null], /command 1/i);
});

const at = (y, mo, d, h, mi) => new Date(y, mo, d, h, mi).getTime();

test("reminderDue: 'in N min' within the hour, then today or tomorrow with a 12-hour time", () => {
  const now = at(2026, 9, 18, 14, 0);
  assert.equal(H.reminderDue({ dueAtMs: now + 5 * 60_000 }, now), "in 5 min");
  assert.equal(H.reminderDue({ dueAtMs: now + 30_000 }, now), "in 1 min");
  assert.equal(H.reminderDue({ dueAtMs: now + 59 * 60_000 }, now), "in 59 min");
  assert.equal(H.reminderDue({ dueAtMs: at(2026, 9, 18, 17, 0) }, now), "today 5:00 pm");
  assert.equal(H.reminderDue({ dueAtMs: at(2026, 9, 18, 23, 45) }, now), "today 11:45 pm");
  assert.equal(H.reminderDue({ dueAtMs: at(2026, 9, 19, 9, 0) }, now), "tomorrow 9:00 am");
  assert.equal(H.reminderDue({ dueAtMs: at(2026, 9, 19, 0, 5) }, now), "tomorrow 12:05 am");
  assert.equal(H.reminderDue({ dueAtMs: at(2026, 9, 19, 12, 0) }, now), "tomorrow 12:00 pm");
  assert.equal(H.reminderDue({ dueAtMs: at(2026, 9, 21, 9, 0) }, now), "21 Oct 9:00 am");
  assert.equal(H.reminderDue({ dueAtMs: now }, now), "due now");
  assert.equal(H.reminderDue({ dueAtMs: now - 60_000 }, now), "due now");
  assert.equal(H.reminderDue({ dueAtMs: NaN }, now), "");
  assert.equal(H.reminderDue({}, now), "");
  assert.equal(H.reminderDue(null, now), "");
});

test("reminderItems stores due_at (epoch s) as dueAtMs, and reminderDue compares in epoch ms", () => {
  const items = H.reminderItems({ items: [{ id: 3, kind: "timer", due_at: 1792327800.5, text: "tea" }, { id: "x" }, null,
    { id: 4, kind: "weird", due_at: "soon", text: 5 }] });
  assert.equal(items.length, 1, "bad rows are dropped");
  assert.deepEqual(items[0], { id: 3, kind: "timer", text: "tea", dueAtMs: 1792327800500 });
  assert.equal(H.reminderDue(items[0], items[0].dueAtMs - 5 * 60_000), "in 5 min");
  assert.deepEqual(H.reminderItems({}), []);
  assert.deepEqual(H.reminderItems(null), []);
});

test("actionText is the summary and its outcome, capped, and tolerates junk", () => {
  assert.equal(H.actionText({ summary: 'Opened "notes"', outcome: "ran" }), 'Opened "notes" · ran');
  assert.ok(H.actionText({ summary: "x".repeat(300), outcome: "failed" }).length <= 230);
  assert.equal(H.actionText({ summary: "Ran backup", outcome: "declined" }), "Ran backup · declined");
  assert.equal(H.actionText({}), "(no summary)");
  assert.equal(H.actionText(null), "(no summary)");
  assert.equal(H.actionText({ summary: "hi" }), "hi");
  assert.equal(H.actionText({ summary: "<b>x</b>", outcome: "ran" }), "<b>x</b> · ran", "kept as text, never markup");
});

test("every element id the renderer and settings scripts look up exists in their page", () => {
  const fs = require("node:fs");
  const path = require("node:path");
  const read = (f) => fs.readFileSync(path.join(__dirname, "..", f), "utf8");
  for (const [js, html] of [["renderer.js", "index.html"], ["settings.js", "settings.html"]]) {
    const page = read(html);
    const ids = new Set([...read(js).matchAll(/\$\("([\w-]+)"\)/g)].map((m) => m[1]));
    assert.ok(ids.size > 10, js);
    for (const id of ids) assert.match(page, new RegExp(`id="${id}"`), `${js} looks up #${id}, which ${html} lacks`);
  }
});
