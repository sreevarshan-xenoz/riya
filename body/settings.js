"use strict";
/** The settings window: one `set_settings` patch per control, her hands (tools, apps, commands), the memory inspector
 *  (facts, notes, follow-ups, conversation summaries, transcript days) and her action log. Text from the core is only ever put in with textContent. Pure logic is in
 *  src/settings-logic.js. */
(() => {
  const {
    sourceLabel, formatWhen, settingPatch, voiceOptions, hotkeyText, episodeText, dueText, memoryLists, ErrorRouter,
    DeferWhileFocused, eyesCountText, eyesStatusText, glanceRows,
  } = RiyaSettings;
  const { hookStatus, receiverProblems, TOKEN_PATH, SETUP_COMMAND, LOG_KEYS } = RiyaAgents;
  const { toolRows, toolsPatch, joinCommandLine, actionText } = RiyaHands;
  const $ = (id) => document.getElementById(id);
  const router = new ErrorRouter();
  const errorEls = new Map(); // target -> its inline error element
  const send = (msg) => window.riya.send(msg);

  function h(tag, className, text) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text !== undefined) el.textContent = text;
    return el;
  }

  // ---- inline errors ----------------------------------------------------------------------------------------------

  function showError(target, message) {
    const el = errorEls.get(target) || $("global-err");
    el.textContent = message;
    el.hidden = !message;
  }

  function clearError(target) {
    const el = errorEls.get(target);
    if (el) showError(target, "");
  }

  /** Send an action, remembering which control it came from so a later error can be shown next to it. */
  function act(target, msg) {
    clearError(target);
    $("global-err").hidden = true;
    router.sent(target);
    send(msg);
  }

  // ---- agents: hook status, fixed token and setup text, and the editable log-watch entries ------------------------

  const LOG_LABELS = { name: "Name", glob: "Glob", done_re: "Done", fail_re: "Failed", lost_re: "Lost" };
  let logsDirty = false; // edited and not yet saved: a settings message must not overwrite it
  let hooksSeenAt = null;
  let lastAgents = null; // the last agents message: the receiver's status
  let coreConnected = false;

  function logEntry(entry) {
    const box = h("div", "log-entry");
    for (const key of LOG_KEYS) {
      const input = h("input");
      input.type = "text";
      input.dataset.key = key;
      input.value = typeof entry[key] === "string" ? entry[key] : "";
      input.spellcheck = false;
      input.autocomplete = "off";
      input.maxLength = 300;
      input.id = `log-${Math.random().toString(36).slice(2)}-${key}`;
      const label = h("label", null, LOG_LABELS[key]);
      label.htmlFor = input.id;
      input.addEventListener("input", () => { logsDirty = true; });
      box.append(label, input);
    }
    const remove = h("button", "small log-remove", "Remove");
    remove.type = "button";
    remove.addEventListener("click", () => { box.remove(); logsDirty = true; });
    box.append(remove);
    return box;
  }

  function renderLogs(list) {
    $("agent-logs").replaceChildren(...list.filter((e) => e && typeof e === "object").map(logEntry));
    logsDirty = false;
  }

  /** The entries as typed; validation (and trimming) is settingPatch's job. */
  function readLogs() {
    return [...$("agent-logs").children].map((box) => {
      const entry = {};
      for (const input of box.querySelectorAll("input[data-key]")) entry[input.dataset.key] = input.value;
      return entry;
    });
  }

  function renderHooks() {
    $("agents-hooks").textContent = hookStatus(hooksSeenAt, Date.now(), coreConnected);
    const problems = receiverProblems(lastAgents, coreConnected);
    $("agents-receiver").textContent = problems.join("; ");
    $("agents-receiver").hidden = problems.length === 0;
  }

  $("agents-token").textContent = TOKEN_PATH;
  $("agents-setup").textContent = SETUP_COMMAND;
  setInterval(() => { if (!document.hidden) renderHooks(); }, 60_000);

  // ---- hands: tool toggles, and the apps and commands editors -----------------------------------------------------

  let latest = {}; // the core's latest settings, for the tool patches
  let appsDirty = false; // edited and not yet saved: a settings message must not overwrite it
  let commandsDirty = false;
  const toolBoxes = new Map(); // tool name -> { enabled, ask }

  function labelled(box, text, input) {
    input.id = `hand-${Math.random().toString(36).slice(2)}`;
    const label = h("label", null, text);
    label.htmlFor = input.id;
    box.append(label, input);
  }

  function textInput(key, value, maxLength, onInput) {
    const input = h("input");
    input.type = "text";
    input.dataset.key = key;
    input.value = value;
    input.spellcheck = false;
    input.autocomplete = "off";
    input.maxLength = maxLength;
    input.addEventListener("input", onInput);
    return input;
  }

  function removeButton(box, onRemove) {
    const remove = h("button", "small hand-remove", "Remove");
    remove.type = "button";
    remove.addEventListener("click", () => { box.remove(); onRemove(); });
    return remove;
  }

  function buildTools() {
    const rows = toolRows({}).map((row) => {
      const line = h("div", "tool");
      const enabled = h("input");
      enabled.type = "checkbox";
      enabled.id = `tool-${row.name}`;
      enabled.addEventListener("change", () => changeTool(row.name, "enabled", enabled.checked));
      const label = h("label", null, row.label);
      label.htmlFor = enabled.id;
      label.append(h("span", `tier${row.tier === 2 ? " t2" : ""}`, `T${row.tier}`));
      const boxes = { enabled, ask: null };
      line.append(enabled, label);
      if (row.askToggle) {
        const ask = h("input");
        ask.type = "checkbox";
        ask.id = `tool-ask-${row.name}`;
        ask.addEventListener("change", () => changeTool(row.name, "ask", ask.checked));
        const askLabel = h("label", "tool-ask", "ask first");
        askLabel.htmlFor = ask.id;
        askLabel.prepend(ask);
        line.append(askLabel);
        boxes.ask = ask;
      } else {
        line.append(h("span", "tool-ask", row.note));
      }
      toolBoxes.set(row.name, boxes);
      return line;
    });
    $("hands-tools").replaceChildren(...rows);
  }

  function changeTool(name, field, value) {
    const patch = toolsPatch(latest, name, field, value);
    if (!patch) return;
    // the next toggle builds on this one even before the core's echo arrives (the core replaces both with its own)
    latest = { ...latest, ...patch, ...(patch.tools_enabled ? { tools_enabled: { ...latest.tools_enabled, ...patch.tools_enabled } } : {}) };
    act(`setting:${field === "ask" ? "tools_ask" : "tools_enabled"}`, { type: "set_settings", patch });
  }

  function showTools(s) {
    for (const row of toolRows(s)) {
      const boxes = toolBoxes.get(row.name);
      boxes.enabled.checked = row.enabled;
      if (boxes.ask) boxes.ask.checked = row.ask;
    }
  }

  function appEntry(app) {
    const box = h("div", "hand-entry");
    const dirty = () => { appsDirty = true; };
    labelled(box, "Name", textInput("name", typeof app.name === "string" ? app.name : "", 40, dirty));
    labelled(box, "Target", textInput("target", typeof app.target === "string" ? app.target : "", 500, dirty));
    box.append(removeButton(box, dirty));
    return box;
  }

  function renderApps(list) {
    $("hands-apps").replaceChildren(...list.filter((a) => a && typeof a === "object").map(appEntry));
    appsDirty = false;
  }

  /** The entries as typed; validation (and trimming) is settingPatch's job. */
  function readEntries(container) {
    return [...container.children].map((box) => {
      const entry = {};
      for (const input of box.querySelectorAll("input[data-key]")) entry[input.dataset.key] = input.value;
      return entry;
    });
  }

  const readApps = () => readEntries($("hands-apps"));

  function commandEntry(cmd) {
    const box = h("div", "hand-entry");
    const argv = Array.isArray(cmd.argv) ? cmd.argv.filter((a) => typeof a === "string") : [];
    box._argv = argv; // kept while the line is untouched, so an argument holding a quote survives a round trip
    box._line = joinCommandLine(argv);
    const dirty = () => { commandsDirty = true; };
    labelled(box, "Name", textInput("name", typeof cmd.name === "string" ? cmd.name : "", 40, dirty));
    labelled(box, "Command", textInput("line", box._line, 4000, dirty));
    labelled(box, "Folder", textInput("cwd", typeof cmd.cwd === "string" ? cmd.cwd : "", 500, dirty));
    const timeout = h("input");
    timeout.type = "number";
    timeout.min = "1";
    timeout.max = "60";
    timeout.step = "1";
    timeout.dataset.key = "timeout_s";
    timeout.value = Number.isFinite(cmd.timeout_s) ? String(cmd.timeout_s) : "30";
    timeout.addEventListener("input", dirty);
    labelled(box, "Timeout (s)", timeout);
    box.append(removeButton(box, dirty));
    return box;
  }

  function renderCommands(list) {
    $("hands-commands").replaceChildren(...list.filter((c) => c && typeof c === "object").map(commandEntry));
    commandsDirty = false;
  }

  function readCommands() {
    const boxes = [...$("hands-commands").children];
    return readEntries($("hands-commands")).map((entry, i) => (entry.line === boxes[i]._line ? { ...entry, argv: boxes[i]._argv } : entry));
  }

  buildTools();

  // ---- settings controls ------------------------------------------------------------------------------------------

  const SETTING_ROWS = document.querySelectorAll(".row[data-key]");
  for (const row of SETTING_ROWS) errorEls.set(`setting:${row.dataset.key}`, row.querySelector(".err"));
  errorEls.set("setting:tools_ask", errorEls.get("setting:tools_enabled")); // one error line for both tool controls

  function changeSetting(key, raw) {
    const target = `setting:${key}`;
    const result = settingPatch(key, raw);
    if (!result.ok) {
      showError(target, result.error);
      return false;
    }
    act(target, { type: "set_settings", patch: result.patch });
    return true;
  }

  const CHECK_KEYS = ["mute_voice", "mute_mic", "autostart", "online_enabled", "online_consolidation", "may_start_conversations",
    "eyes_enabled", "hands_online"];
  for (const key of CHECK_KEYS) $(key).addEventListener("change", (e) => changeSetting(key, e.target.checked));
  for (const key of ["glance_daily_cap", "glance_min_gap_s"]) $(key).addEventListener("change", (e) => changeSetting(key, e.target.value));
  for (const [key, save] of [["eyes_deny_titles", "deny-titles-save"], ["eyes_deny_processes", "deny-processes-save"]]) {
    $(save).addEventListener("click", () => changeSetting(key, $(key).value));
  }
  $("agent-logs-save").addEventListener("click", () => {
    if (changeSetting("agent_logs", readLogs())) logsDirty = false; // the core's echo then replaces the entries
  });
  $("agent-log-add").addEventListener("click", () => {
    $("agent-logs").append(logEntry({ name: "", glob: "", done_re: "", fail_re: "", lost_re: "" }));
    logsDirty = true;
  });
  $("weather-save").addEventListener("click", () => changeSetting("weather_place", $("weather_place").value));
  $("weather_place").addEventListener("keydown", (e) => { if (e.key === "Enter") changeSetting("weather_place", e.target.value); });
  $("apps-save").addEventListener("click", () => {
    if (changeSetting("apps", readApps())) appsDirty = false; // the core's echo then replaces the rows
  });
  $("app-add").addEventListener("click", () => { $("hands-apps").append(appEntry({ name: "", target: "" })); appsDirty = true; });
  $("commands-save").addEventListener("click", () => {
    if (changeSetting("commands", readCommands())) commandsDirty = false;
  });
  $("command-add").addEventListener("click", () => {
    $("hands-commands").append(commandEntry({ name: "", argv: [], cwd: null, timeout_s: 30 }));
    commandsDirty = true;
  });
  $("actions-clear").addEventListener("click", clearActions);
  $("mode").addEventListener("change", (e) => changeSetting("mode", e.target.value));
  $("voice").addEventListener("change", (e) => changeSetting("voice", e.target.value));
  $("volume").addEventListener("input", (e) => { $("volume-out").textContent = `${Math.round(e.target.value * 100)}%`; });
  $("volume").addEventListener("change", (e) => changeSetting("volume", e.target.value));
  $("quiet-save").addEventListener("click", () => changeSetting("quiet_hours", $("quiet_hours").value));
  $("quiet_hours").addEventListener("keydown", (e) => { if (e.key === "Enter") changeSetting("quiet_hours", e.target.value); });

  const voiceDefer = new DeferWhileFocused();

  function rebuildVoice(current) {
    const select = $("voice");
    select.replaceChildren(...voiceOptions(current).map(({ value, label }) => {
      const option = h("option", null, label);
      option.value = value;
      return option;
    }));
    select.value = typeof current === "string" ? current : "";
  }

  /** The voice presets, plus the current voice as "(custom)" when it isn't one (e.g. a reference WAV). Rebuilding the
   *  options closes an open dropdown, so while the select has focus the rebuild waits for blur. */
  function showVoice(current) {
    const now = voiceDefer.offer(current, document.activeElement === $("voice"));
    if (now) rebuildVoice(now.value);
  }

  $("voice").addEventListener("blur", () => {
    const pending = voiceDefer.take();
    if (pending) rebuildVoice(pending.value);
  });

  /** Show the core's settings in the controls (a control being typed in keeps its text). */
  function applySettings(s) {
    const set = (id, prop, value) => { if (document.activeElement !== $(id)) $(id)[prop] = value; };
    if (typeof s.mode === "string") $("mode").value = s.mode;
    if (typeof s.voice === "string") showVoice(s.voice);
    if (typeof s.volume === "number") {
      set("volume", "value", String(s.volume));
      $("volume-out").textContent = `${Math.round(s.volume * 100)}%`;
    }
    for (const key of CHECK_KEYS) if (typeof s[key] === "boolean") $(key).checked = s[key];
    for (const key of ["glance_daily_cap", "glance_min_gap_s"]) if (Number.isFinite(s[key])) set(key, "value", String(s[key]));
    for (const key of ["eyes_deny_titles", "eyes_deny_processes"]) {
      if (Array.isArray(s[key])) set(key, "value", s[key].filter((p) => typeof p === "string").join("\n"));
    }
    if (Array.isArray(s.agent_logs) && !logsDirty) renderLogs(s.agent_logs);
    latest = s;
    showTools(s);
    if (Array.isArray(s.apps) && !appsDirty) renderApps(s.apps);
    if (Array.isArray(s.commands) && !commandsDirty) renderCommands(s.commands);
    if (s.weather_place === null || typeof s.weather_place === "string") set("weather_place", "value", s.weather_place || "");
    if (s.quiet_hours === null || typeof s.quiet_hours === "string") set("quiet_hours", "value", s.quiet_hours || "");
  }

  // ---- confirm dialog ---------------------------------------------------------------------------------------------

  /** Resolves { ok, checked }; `checkLabel` adds an unchecked checkbox. */
  function confirmDialog({ title, body, okLabel, checkLabel }) {
    const dlg = $("confirm");
    return new Promise((resolve) => {
      $("confirm-title").textContent = title;
      $("confirm-body").textContent = body;
      $("confirm-ok").textContent = okLabel;
      $("confirm-check").checked = false;
      $("confirm-check-label").textContent = checkLabel || "";
      $("confirm-check-row").hidden = !checkLabel;
      dlg.returnValue = "";
      $("confirm-cancel").onclick = () => dlg.close("cancel");
      dlg.onclose = () => resolve({ ok: dlg.returnValue === "ok", checked: $("confirm-check").checked });
      dlg.showModal();
      $("confirm-cancel").focus();
    });
  }

  // ---- memory lists -----------------------------------------------------------------------------------------------

  function button(label, onClick) {
    const b = h("button", "small", label);
    b.type = "button";
    b.addEventListener("click", onClick);
    return b;
  }

  /** One list item: `text`, a meta line (optional badge, `meta` text) and action buttons; its error line is registered. */
  function item(target, text, { badge, meta, actions }) {
    const li = h("li", "item");
    const body = h("div", "item-text", text);
    const metaLine = h("div", "item-meta");
    if (badge) metaLine.append(h("span", `badge ${badge.tone}`, badge.label));
    if (meta) metaLine.append(h("span", null, meta));
    const bar = h("span", "item-actions");
    bar.append(...actions);
    metaLine.append(bar);
    const err = h("p", "err");
    err.hidden = true;
    err.setAttribute("role", "alert");
    errorEls.set(target, err);
    li.append(body, metaLine, err);
    return { li, body, metaLine, bar };
  }

  function editFact(fact, parts) {
    const target = `fact:${fact.id}`;
    const input = h("input");
    input.type = "text";
    input.value = fact.text;
    input.maxLength = 500;
    input.setAttribute("aria-label", "Corrected memory");
    const row = h("div", "edit");
    const close = () => { row.remove(); parts.bar.hidden = false; };
    const save = () => {
      const text = input.value.trim();
      if (!text) return showError(target, "A memory can't be empty. Delete it instead.");
      if (text === fact.text) return close();
      close();
      act(target, { type: "memory_correct", id: fact.id, text });
    };
    row.append(input, button("Save", save), button("Cancel", close));
    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") save();
      else if (e.key === "Escape") close();
    });
    parts.bar.hidden = true;
    parts.metaLine.after(row);
    input.focus();
    input.select();
  }

  async function deleteFact(fact) {
    const r = await confirmDialog({
      title: "Delete this memory?",
      body: fact.text,
      okLabel: "Delete",
      checkLabel: "also delete conversations that mention this",
    });
    if (r.ok) act(`fact:${fact.id}`, { type: "memory_delete", id: fact.id, delete_episodes: r.checked });
  }

  async function deleteEpisode(episode) {
    const r = await confirmDialog({
      title: "Delete this conversation summary?",
      body: episodeText(episode),
      okLabel: "Delete",
    });
    if (r.ok) act(`episode:${episode.id}`, { type: "episode_delete", id: episode.id });
  }

  async function deleteDay(day) {
    const r = await confirmDialog({
      title: "Delete this day's transcript?",
      body: `Everything said on ${day} will be removed.`,
      okLabel: "Delete",
    });
    if (r.ok) act(`day:${day}`, { type: "transcript_delete", day });
  }

  function fill(listId, emptyId, rows) {
    const ul = $(listId);
    ul.replaceChildren(...rows);
    $(emptyId).hidden = rows.length > 0;
  }

  async function deleteGlance(glance) {
    const r = await confirmDialog({
      title: "Delete this note?",
      body: glance.text,
      okLabel: "Delete",
    });
    if (r.ok) act(`glance:${glance.id}`, { type: "glance_delete", id: glance.id });
  }

  async function deleteNote(note) {
    const r = await confirmDialog({ title: "Delete this note?", body: String(note.text), okLabel: "Delete" });
    if (r.ok) act(`note:${note.id}`, { type: "note_delete", id: note.id });
  }

  async function clearActions() {
    const r = await confirmDialog({
      title: "Clear her action log?",
      body: "This only clears the list. It does not undo anything she did.",
      okLabel: "Clear",
    });
    if (r.ok) act("actions", { type: "actions_clear" });
  }

  function renderActions(msg) {
    const list = msg && Array.isArray(msg.items) ? msg.items.filter((a) => a && typeof a === "object") : [];
    fill("actions", "actions-empty", list.map((a) => {
      const li = h("li", "item");
      li.append(h("div", "item-text", actionText(a)));
      li.append(h("div", "item-meta", `${formatWhen(a.at)} · ${typeof a.tool === "string" ? a.tool : "unknown"}`));
      return li;
    }));
  }

  function renderEyes(eyes) {
    $("eyes-count").textContent = eyesCountText(eyes);
    const status = eyesStatusText(eyes);
    $("eyes-status").textContent = status;
    $("eyes-status").hidden = !status;
  }

  function renderMemories(msg) {
    for (const target of [...errorEls.keys()]) if (!target.startsWith("setting:")) errorEls.delete(target);
    const { facts, episodes, threads, days, notes } = memoryLists(msg);

    fill("facts", "facts-empty", facts.map((fact) => {
      const parts = item(`fact:${fact.id}`, String(fact.text), {
        badge: sourceLabel(fact),
        meta: `learned ${formatWhen(fact.valid_from)}`,
        actions: [button("Correct", () => editFact(fact, parts)), button("Delete", () => deleteFact(fact))],
      });
      return parts.li;
    }));

    fill("notes", "notes-empty", notes.filter((n) => n && Number.isInteger(n.id)).map((note) => item(`note:${note.id}`, String(note.text), {
      meta: formatWhen(note.at),
      actions: [button("Delete", () => deleteNote(note))],
    }).li));

    fill("glances", "glances-empty", glanceRows(msg).map((glance) => item(`glance:${glance.id}`, glance.text, {
      meta: glance.meta,
      actions: [button("Delete", () => deleteGlance(glance))],
    }).li));

    fill("threads", "threads-empty", threads.map((thread) => item(`thread:${thread.id}`, String(thread.text), {
      meta: dueText(thread),
      actions: [button("Resolve", () => act(`thread:${thread.id}`, { type: "thread_resolve", id: thread.id }))],
    }).li));

    fill("episodes", "episodes-empty", episodes.map((episode) => item(`episode:${episode.id}`, episodeText(episode), {
      meta: formatWhen(episode.started_at),
      actions: [button("Delete", () => deleteEpisode(episode))],
    }).li));

    fill("days", "days-empty", days.map((day) => item(`day:${day}`, String(day), {
      actions: [button("Delete", () => deleteDay(day))],
    }).li));
  }

  // ---- core wiring ------------------------------------------------------------------------------------------------

  window.riya.onConnection((connected) => {
    $("conn").textContent = connected ? "connected" : "connecting…";
    $("conn").classList.toggle("off", !connected);
    coreConnected = Boolean(connected);
    renderHooks();
    if (!connected) renderActions(null); // a stale log would pass for live; the core sends it again on change
    if (connected) send({ type: "get_memories" });
  });

  window.riya.onCoreMessage((msg) => {
    if (!msg || typeof msg.type !== "string") return;
    if (msg.type === "settings" && msg.settings && typeof msg.settings === "object") {
      router.settle();
      applySettings(msg.settings);
    } else if (msg.type === "actions") {
      renderActions(msg);
    } else if (msg.type === "eyes") {
      renderEyes(msg);
    } else if (msg.type === "agents") {
      hooksSeenAt = typeof msg.hooks_seen_at === "number" ? msg.hooks_seen_at : null;
      lastAgents = msg;
      renderHooks();
    } else if (msg.type === "memories") {
      router.settle();
      renderMemories(msg);
    } else if (msg.type === "error") {
      const target = router.route(msg.message);
      if (target === "setting:agent_logs") logsDirty = true; // the core refused them: keep what was typed
      if (target === "setting:apps") appsDirty = true;
      if (target === "setting:commands") commandsDirty = true;
      showError(target, String(msg.message));
    }
  });

  showVoice(null);
  window.riya.dontLookHotkey().then((accelerator) => { $("dontlook-hotkey").textContent = hotkeyText(accelerator); },
    () => { $("dontlook-hotkey").textContent = hotkeyText(null); });
  window.riya.hotkey().then((accelerator) => { $("hotkey").textContent = hotkeyText(accelerator); },
    () => { $("hotkey").textContent = hotkeyText(null); });
  send({ type: "get_memories" });
})();
