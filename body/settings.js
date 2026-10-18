"use strict";
/** The settings window: one `set_settings` patch per control, and the memory inspector (facts, follow-ups, conversation
 *  summaries, transcript days). Text from the core is only ever put in with textContent. Pure logic is in
 *  src/settings-logic.js. */
(() => {
  const {
    sourceLabel, formatWhen, settingPatch, voiceOptions, hotkeyText, episodeText, dueText, memoryLists, ErrorRouter,
    DeferWhileFocused, eyesCountText, eyesStatusText, glanceRows,
  } = RiyaSettings;
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

  // ---- settings controls ------------------------------------------------------------------------------------------

  const SETTING_ROWS = document.querySelectorAll(".row[data-key]");
  for (const row of SETTING_ROWS) errorEls.set(`setting:${row.dataset.key}`, row.querySelector(".err"));

  function changeSetting(key, raw) {
    const target = `setting:${key}`;
    const result = settingPatch(key, raw);
    if (!result.ok) return showError(target, result.error);
    act(target, { type: "set_settings", patch: result.patch });
  }

  const CHECK_KEYS = ["mute_voice", "mute_mic", "autostart", "online_enabled", "online_consolidation", "may_start_conversations",
    "eyes_enabled"];
  for (const key of CHECK_KEYS) $(key).addEventListener("change", (e) => changeSetting(key, e.target.checked));
  for (const key of ["glance_daily_cap", "glance_min_gap_s"]) $(key).addEventListener("change", (e) => changeSetting(key, e.target.value));
  for (const [key, save] of [["eyes_deny_titles", "deny-titles-save"], ["eyes_deny_processes", "deny-processes-save"]]) {
    $(save).addEventListener("click", () => changeSetting(key, $(key).value));
  }
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

  function renderEyes(eyes) {
    $("eyes-count").textContent = eyesCountText(eyes);
    const status = eyesStatusText(eyes);
    $("eyes-status").textContent = status;
    $("eyes-status").hidden = !status;
  }

  function renderMemories(msg) {
    for (const target of [...errorEls.keys()]) if (!target.startsWith("setting:")) errorEls.delete(target);
    const { facts, episodes, threads, days } = memoryLists(msg);

    fill("facts", "facts-empty", facts.map((fact) => {
      const parts = item(`fact:${fact.id}`, String(fact.text), {
        badge: sourceLabel(fact),
        meta: `learned ${formatWhen(fact.valid_from)}`,
        actions: [button("Correct", () => editFact(fact, parts)), button("Delete", () => deleteFact(fact))],
      });
      return parts.li;
    }));

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
    if (connected) send({ type: "get_memories" });
  });

  window.riya.onCoreMessage((msg) => {
    if (!msg || typeof msg.type !== "string") return;
    if (msg.type === "settings" && msg.settings && typeof msg.settings === "object") {
      router.settle();
      applySettings(msg.settings);
    } else if (msg.type === "eyes") {
      renderEyes(msg);
    } else if (msg.type === "memories") {
      router.settle();
      renderMemories(msg);
    } else if (msg.type === "error") {
      showError(router.route(msg.message), String(msg.message));
    }
  });

  showVoice(null);
  window.riya.dontLookHotkey().then((accelerator) => { $("dontlook-hotkey").textContent = hotkeyText(accelerator); },
    () => { $("dontlook-hotkey").textContent = hotkeyText(null); });
  window.riya.hotkey().then((accelerator) => { $("hotkey").textContent = hotkeyText(accelerator); },
    () => { $("hotkey").textContent = hotkeyText(null); });
  send({ type: "get_memories" });
})();
