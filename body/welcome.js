"use strict";
/** The first-run welcome: one step at a time, in her voice, every step skippable. Each step sends the same messages the
 *  Settings window would. Text from the core is only ever put in with textContent. Pure logic is in src/welcome-logic.js
 *  and src/health-logic.js. The window never says it is finished: closing it (Finish, Skip setup or the X) is all it does,
 *  and the main process's `closed` handler decides what the core hears. The key is typed into a password field, sent once
 *  through submitKey and never kept here. */
(() => {
  const { STEPS, LINES, nextStep, prevStep, nameMessage, cleanName, modeMessage, eyesMessage, eyesLine, agentRows } = RiyaWelcome;
  const { submitKey, saveKeyEnabled, keySourceText, voiceItems, resultText, STATUS_TEXT } = RiyaHealth;
  const $ = (id) => document.getElementById(id);
  const send = (msg) => window.riya.send(msg);
  const SETUP_ACTIONS = ["online_key_set", "voice_sample", "voice_choose"]; // results shown here: the others are Settings'
  const MODE_LABELS = { buddy: "Buddy", panel: "Panel", window: "Window" };

  let step = STEPS[0];
  let coreConnected = false;
  let healthMsg = null; // the core's latest `health`; null while it is away, so stale rows never pass for live
  let voices = { items: [], current: null };
  let settings = {};
  let sentName = null; // the name already sent, so going Back and Next again doesn't remember it twice
  let renderToken = 0;
  let keyUi = null; // { field, save, source, err } while the key step shows
  let nameUi = null; // { field, err } while the name step shows

  function h(tag, className, text) {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text !== undefined) el.textContent = text;
    return el;
  }

  function button(label, onClick, className) {
    const b = h("button", className, label);
    b.type = "button";
    b.addEventListener("click", onClick);
    return b;
  }

  const refreshHealth = () => { if (coreConnected) send({ type: "health_get" }); };

  function showResult(text) {
    $("result").textContent = text;
  }

  // ---- the steps --------------------------------------------------------------------------------------------------

  function nameStep(box) {
    const field = h("input");
    field.type = "text";
    field.maxLength = 40;
    field.autocomplete = "off";
    field.spellcheck = false;
    field.placeholder = "your name";
    field.setAttribute("aria-label", "Your name");
    if (sentName !== null) field.value = sentName;
    field.addEventListener("keydown", (e) => { if (e.key === "Enter") goNext(); });
    const err = h("p", "err");
    err.setAttribute("role", "alert");
    err.hidden = true;
    box.append(field, err);
    nameUi = { field, err };
    setTimeout(() => field.focus(), 0);
  }

  /** The key controls come from welcome.html's template (a password field), so the page itself shows what the field is. */
  function keyStep(box) {
    box.append($("key-template").content.cloneNode(true));
    const field = box.querySelector("#online-key");
    const save = box.querySelector("#online-key-save");
    save.addEventListener("click", saveKey);
    box.querySelector("#online-key-page").addEventListener("click", () => window.riya.openKeyPage());
    field.addEventListener("keydown", (e) => { if (e.key === "Enter") saveKey(); });
    keyUi = { field, save, source: box.querySelector("#online-key-source"), err: box.querySelector("#online-key-err") };
    updateKey();
  }

  function saveKey() {
    if (!keyUi) return;
    const sent = submitKey(keyUi.field, send, coreConnected);
    keyUi.err.textContent = sent ? "" : coreConnected ? "That doesn't look like a key (no spaces, at most 200 characters)."
      : "Her core isn't connected, so the key wasn't sent.";
    keyUi.err.hidden = sent;
  }

  function updateKey() {
    if (!keyUi) return;
    keyUi.save.disabled = !saveKeyEnabled(coreConnected);
    keyUi.source.textContent = keySourceText(healthMsg && typeof healthMsg.key_source === "string" ? healthMsg.key_source : null);
  }

  function voiceStep(box) {
    if (voices.items.length === 0) {
      box.append(h("p", "note", coreConnected ? "Looking for my voices…" : "I'll list my voices once my core is connected."));
      return;
    }
    for (const name of voices.items) {
      const line = h("div", "voice");
      line.append(h("span", "name", name));
      if (name === voices.current) line.append(h("span", "tag", "chosen"));
      line.append(
        button("Play", () => send({ type: "setup_request", action: "voice_sample", args: { voice: name } }), "small"),
        button("Choose", () => send({ type: "setup_request", action: "voice_choose", args: { voice: name } }), "small"),
      );
      box.append(line);
    }
  }

  function modeStep(box) {
    const row = h("div", "inline");
    for (const mode of Object.keys(MODE_LABELS)) {
      const b = button(MODE_LABELS[mode], () => { const m = modeMessage(mode); if (m) send(m); }, "choice");
      b.setAttribute("aria-pressed", String(settings.mode === mode));
      row.append(b);
    }
    box.append(row);
  }

  function eyesStep(box) {
    const token = renderToken;
    const note = h("p", "note");
    const row = h("div", "inline");
    for (const [label, on] of [["Yes, she may look", true], ["No, keep her eyes closed", false]]) {
      const b = button(label, () => { const m = eyesMessage(on); if (m) send(m); }, "choice");
      b.setAttribute("aria-pressed", String(settings.eyes_enabled === on));
      row.append(b);
    }
    box.append(row, note);
    // The hotkey text is the step's line; it arrives asynchronously, and only the step still showing takes it.
    Promise.resolve(window.riya.dontLookHotkey()).catch(() => null).then((hotkey) => {
      if (token === renderToken) $("line").textContent = eyesLine(hotkey);
    });
    note.textContent = "You can change this any time in Settings.";
  }

  function agentsStep(box) {
    const rows = agentRows(healthMsg);
    if (rows.length === 0) box.append(h("p", "note", coreConnected ? "Checking…" : "I'll check once my core is connected."));
    for (const row of rows) {
      const item = h("div", "agent");
      item.append(h("span", `chip ${row.status}`, STATUS_TEXT[row.status]));
      const text = h("div");
      text.append(h("span", "label", row.label), h("span", "detail", row.detail));
      item.append(text);
      box.append(item);
    }
    box.append(h("p", "note", "Edit in Settings: connect Claude Code and choose the Colab folders there."));
  }

  const BUILDERS = { name: nameStep, key: keyStep, voice: voiceStep, mode: modeStep, eyes: eyesStep, agents: agentsStep };

  // ---- rendering and navigation -----------------------------------------------------------------------------------

  function render() {
    renderToken += 1;
    keyUi = null;
    nameUi = null;
    $("progress").textContent = `Step ${STEPS.indexOf(step) + 1} of ${STEPS.length}`;
    $("line").textContent = step === "eyes" ? eyesLine(null) : LINES[step]; // eyesStep puts the real hotkey in when it arrives
    const box = h("div", "controls-inner");
    if (BUILDERS[step]) BUILDERS[step](box);
    $("controls").replaceChildren(box);
    $("back").disabled = prevStep(step) === null;
    $("back").hidden = step === "intro";
    $("skip").hidden = step === "done" || step === "intro";
    $("next").textContent = step === "done" ? "Finish" : "Next";
  }

  /** A message from the core changed what the current step shows. The name and key steps hold typed text, so they are
   *  never rebuilt under the user; the key step only refreshes its own source line and Save button. */
  function refreshStep() {
    if (step === "name") return;
    if (step === "key") { updateKey(); return; }
    render();
  }

  function goNext() {
    if (step === "done") { window.close(); return; }
    if (step === "name" && nameUi) {
      const raw = nameUi.field.value;
      if (raw.trim() !== "" && raw !== sentName) {
        if (cleanName(raw) === null) {
          nameUi.err.textContent = "I can only remember a short name, up to 40 letters and no question marks.";
          nameUi.err.hidden = false;
          return;
        }
        if (!coreConnected) {
          nameUi.err.textContent = "Her core isn't connected, so I couldn't remember that. Skip for now, or try again in a moment.";
          nameUi.err.hidden = false;
          return;
        }
        send(nameMessage(raw));
        sentName = cleanName(raw);
      }
    }
    moveTo(nextStep(step));
  }

  function goBack() {
    moveTo(prevStep(step));
  }

  /** Skip this step: on to the next one, sending nothing. */
  function skipStep() {
    moveTo(nextStep(step));
  }

  function moveTo(to) {
    if (!to) return;
    step = to;
    showResult("");
    render();
  }

  $("next").addEventListener("click", goNext);
  $("back").addEventListener("click", goBack);
  $("skip").addEventListener("click", skipStep);
  $("skip-setup").addEventListener("click", () => window.close());

  // ---- core wiring ------------------------------------------------------------------------------------------------

  window.riya.onConnection((connected) => {
    coreConnected = Boolean(connected);
    $("conn").textContent = coreConnected ? "connected" : "connecting…";
    $("conn").classList.toggle("off", !coreConnected);
    if (!coreConnected) healthMsg = null;
    refreshStep();
    refreshHealth(); // on load and on every reconnect: the status replay on load arrives here
  });

  window.riya.onCoreMessage((msg) => {
    if (!msg || typeof msg.type !== "string") return;
    if (msg.type === "settings" && msg.settings && typeof msg.settings === "object") {
      settings = msg.settings;
      if (step === "mode" || step === "eyes") refreshStep();
    } else if (msg.type === "health") {
      healthMsg = msg;
      if (step === "key" || step === "agents") refreshStep();
    } else if (msg.type === "voices") {
      voices = voiceItems(msg);
      if (step === "voice") refreshStep();
    } else if (msg.type === "setup_result") {
      if (SETUP_ACTIONS.includes(msg.action)) showResult(resultText(msg));
      refreshHealth(); // after every result, so the key step's source line follows a save
    }
  });

  render();
})();
