"use strict";
/** Electron main process: the three window modes, click-through and dragging, tray, hotkeys and the core relay.
 *
 *  Electron 33's main process has no WebSocket (Node 20), so the single core connection runs in a hidden bridge window
 *  (bridge.html) built on the reusable CoreClient and is relayed here over IPC; this process fans it out to the stage
 *  windows over `core:message` / `core:status` and forwards their `core:send` back. */
const fs = require("node:fs");
const path = require("node:path");
const { execFile } = require("node:child_process");
const {
  app, BrowserWindow, Tray, Menu, nativeImage, screen, globalShortcut, ipcMain, session, shell, powerMonitor,
} = require("electron");
const { boundsFor, clampToArea, DEFAULT_SIZE } = require("./src/geometry");
const { Pet, petSize, makeRng, reducedMotionOn, OFF_VIEW, PET } = require("./src/pet-logic");
const { loadFirstPack } = require("./src/pack-path");
const { makeTrayPng } = require("./src/tray-icon");
const {
  pickHotkey, modeToApply, repositionTarget, shouldSendToCore, quitDelay, rememberForReplay, toPhysical, toDip, displayConverter,
  forgetOnDisconnect, DONT_LOOK_HOTKEYS, KEY_URL, shouldOpenWelcome, welcomeClosedMessage, ipcAllowed, welcomeWebPreferences,
  QUIT_EVENTS, rendererMayRequest, isClick, focusTarget, clickAction, decideFocus, focusCommand, petMenuItems,
  reducedMotionPatch, moveRect, FOCUS_WAIT_MS, petVisibilityAction, runFocusHelper,
} = require("./src/main-logic");

const SMOKE = process.env.RIYA_SMOKE === "1";
const OPEN_SETTINGS = process.env.RIYA_OPEN_SETTINGS === "1"; // dev/smoke: open the settings window at startup
const REPO_ROOT = path.resolve(__dirname, "..");
const TALK_HOTKEYS = ["Control+Alt+Space", "Control+Alt+F9"]; // first one that registers wins
const FIRST_SETTINGS_WAIT_MS = 1500;
const DRAG_INTERVAL_MS = 1000 / 60;
const STAGE_RESPAWN_MS = 1000;
const IDLE_POLL_MS = 5000; // the pet looks at the user's idle time this often, for sleep
const FORCE_WALK = process.env.RIYA_PET_FORCE_WALK === "1"; // dev override for evaluation/pet_cpu.py

const state = {
  settings: {},
  mode: null,
  win: null,
  bridge: null,
  bridgeReady: false,
  bridgeQueue: [],
  connected: false,
  readyPending: false,
  hotkey: null,
  dontLookHotkey: null,
  dontLook: false, // the core's latest eyes.dont_look
  userDragged: false,
  ignoring: false,
  drag: null,
  quitting: false,
  tray: null,
  settingsWin: null,
  welcomeWin: null,
  welcomeShown: false, // the welcome opens on its own at most once per session
  cache: new Map(),
  // Little Riya (buddy mode)
  pet: null, // the Pet state machine, while she is on
  petTimer: null, // the 20 Hz interval, only while she walks or falls
  petWake: null, // the one timeout to her next deadline otherwise
  idlePoll: null,
  presence: null, // the core's latest presence, null while unknown or disconnected
  agents: [], // the core's latest agents.items
  lastActivity: null, // { session, atMs } of the latest agent_activity
  pendingFocus: null, // { id, atMs, timer } while a click waits for its agent_window
  lastPetView: null, // the view a new or recreated window draws at once
  osReducedMotion: false, // what the renderer reports from matchMedia
  petPaused: false, // her window is hidden: no timers, no steps, no saves
  petMoved: false, // she has walked, fallen or been dropped this session: a late saved position leaves her alone
  focusToken: 0, // counts clicks, so a helper that fails after a newer click opens nothing
  press: null, // { cursor, atMs, pos } from drag-start, to tell a click from a drag
  canvas: null, // the active sprite pack's canvas, for her size
  settingsLoaded: false,
  settingsSection: null, // a section the settings window should scroll to once it has loaded
};
const stageWindows = new Set();

function log(...args) {
  console.log("[riya]", ...args);
}

function fail(what, err) {
  console.error(`[riya] ${what}:`, err && err.stack ? err.stack : err);
  if (SMOKE) process.exit(1);
}

process.on("uncaughtException", (err) => fail("uncaught exception", err));
process.on("unhandledRejection", (err) => fail("unhandled rejection", err));

// ---- security: every webContents ----------------------------------------------------------------------------------

app.on("web-contents-created", (_event, contents) => {
  contents.on("will-navigate", (event) => event.preventDefault());
  contents.on("will-attach-webview", (event) => event.preventDefault());
  contents.setWindowOpenHandler(() => ({ action: "deny" }));
  if (SMOKE) {
    contents.on("console-message", (_e, level, message) => log(`renderer console[${level}]`, message));
    contents.on("did-fail-load", (_e, code, desc, url) => fail("did-fail-load", `${code} ${desc} ${url}`));
    contents.on("render-process-gone", (_e, details) => fail("render-process-gone", details.reason));
  }
});

// ---- core relay -----------------------------------------------------------------------------------------------------

function sendToCore(msg) {
  if (!shouldSendToCore(msg, state.connected)) return;
  if (state.bridge && state.bridgeReady) state.bridge.webContents.send("bridge:send", msg);
  else state.bridgeQueue.push(msg);
}

/** Every window that shows the core: the stage windows, the settings window and the welcome. */
function uiWindows() {
  return [...stageWindows, state.settingsWin, state.welcomeWin].filter((win) => win && !win.isDestroyed());
}

function broadcast(channel, payload) {
  for (const win of uiWindows()) win.webContents.send(channel, payload);
}

function createBridge() {
  state.bridgeReady = false;
  const bridge = new BrowserWindow({
    show: false,
    width: 1,
    height: 1,
    webPreferences: {
      preload: path.join(__dirname, "bridge-preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
      backgroundThrottling: false,
    },
  });
  state.bridge = bridge;
  bridge.webContents.on("did-finish-load", () => {
    state.bridgeReady = true;
    for (const msg of state.bridgeQueue.splice(0)) bridge.webContents.send("bridge:send", msg);
  });
  bridge.webContents.on("render-process-gone", () => {
    state.bridgeReady = false;
    onCoreStatus(false);
    if (!state.quitting) setTimeout(createBridge, 1000);
  });
  bridge.loadFile(path.join(__dirname, "bridge.html"));
}

function onCoreStatus(connected) {
  state.connected = connected;
  state.readyPending = connected;
  if (!connected) {
    forgetOnDisconnect(state.cache); // the core sends a fresh look and agents on hello
    state.presence = null; // unknown until the core says so again: she does not walk
    feedPet({ presence: null });
  }
  broadcast("core:status", connected);
  sendBodyReady();
}

/** body_ready {mode} after every (re)connect; held back until a window (and so a mode) exists on a cold start. */
function sendBodyReady() {
  if (!state.readyPending || !state.mode) return;
  state.readyPending = false;
  sendToCore({ type: "body_ready", mode: state.mode });
}

function onCoreMessage(msg) {
  if (!msg || typeof msg.type !== "string") return;
  rememberForReplay(state.cache, msg);
  if (msg.type === "eyes") {
    const dontLook = msg.dont_look === true;
    if (dontLook !== state.dontLook) {
      state.dontLook = dontLook;
      refreshTrayMenu();
    }
  }
  if (msg.type === "state" && typeof msg.presence === "string") {
    state.presence = msg.presence;
    feedPet({ presence: state.presence });
  }
  if (msg.type === "presence" && typeof msg.state === "string") {
    state.presence = msg.state;
    feedPet({ presence: state.presence });
  }
  if (msg.type === "agents") state.agents = Array.isArray(msg.items) ? msg.items : [];
  if (msg.type === "agent_activity" && typeof msg.session === "string") {
    state.lastActivity = { session: msg.session, atMs: Date.now() };
    petReact(msg.kind, msg.label);
  }
  if (msg.type === "agent_window") {
    const d = decideFocus(state.pendingFocus, msg, Date.now());
    if (d) {
      clearTimeout(state.pendingFocus.timer);
      state.pendingFocus = null;
      const cmd = d.spawn ? focusCommand(REPO_ROOT, d.spawn, fs.existsSync) : null;
      const token = state.focusToken;
      const started = runFocusHelper({
        cmd, spawn: execFile, cwd: REPO_ROOT, isCurrent: () => token === state.focusToken, onFailed: openAgentsList,
      });
      if (!started) openAgentsList(); // no window found, or no safe command for it
    }
  }
  if (msg.type === "settings" && msg.settings && typeof msg.settings === "object") {
    const before = { win: state.win, mode: state.mode, size: buddySize(), pack: state.settings.sprite_pack };
    state.settings = { ...msg.settings };
    if (state.settings.sprite_pack !== before.pack) refreshCanvas();
    ensureMode(state.settings.mode);
    applyLateSavedPosition();
    applyPetSettings(before);
    if (shouldOpenWelcome(msg.settings, state.welcomeShown)) {
      state.welcomeShown = true;
      openWelcome();
    }
  }
  broadcast("core:message", msg);
}

ipcMain.on("bridge:message", (event, msg) => {
  if (state.bridge && event.sender === state.bridge.webContents) onCoreMessage(msg);
});
ipcMain.on("bridge:status", (event, connected) => {
  if (state.bridge && event.sender === state.bridge.webContents) onCoreStatus(Boolean(connected));
});
ipcMain.on("core:send", (event, msg) => {
  if (!isUiSender(event.sender)) return;
  if (!rendererMayRequest(msg)) return; // typed messages only, and never one that is main's alone
  sendToCore(msg);
});

function isStageSender(sender) {
  for (const win of stageWindows) if (!win.isDestroyed() && win.webContents === sender) return true;
  return false;
}

function isUiSender(sender) {
  return uiWindows().some((win) => win.webContents === sender);
}

// ---- sprite pack ----------------------------------------------------------------------------------------------------

/** The first valid pack (settings.sprite_pack, sprites/riya-v1, placeholder-pack) with frame paths as file URLs. */
function loadPack() {
  return loadFirstPack({ spritePack: state.settings.sprite_pack, repoRoot: REPO_ROOT, bodyDir: __dirname, log });
}

/** The active pack's canvas (for her size), re-read when a window is made and when sprite_pack changes. */
function refreshCanvas() {
  try {
    const pack = loadPack();
    state.canvas = pack && pack.manifest ? pack.manifest.canvas || null : null;
  } catch (e) {
    state.canvas = null;
  }
}

/** The active talk-hotkey accelerator (e.g. "Control+Alt+F9" when the preferred one was taken), or null. */
ipcMain.handle("hotkey:get", (event) => (isUiSender(event.sender) ? state.hotkey : null));
/** The active don't-look accelerator (Control+Alt+H, or Control+Alt+F10 when that was taken), or null. */
ipcMain.handle("hotkey:dontlook", (event) => (isUiSender(event.sender) ? state.dontLookHotkey : null));
ipcMain.handle("pack:get", (event) => (isStageSender(event.sender) ? loadPack() : null));

// ---- windows --------------------------------------------------------------------------------------------------------

/** The saved buddy_position is in physical pixels (what the core compares with window rects); Electron places in DIPs. */
/** Electron's own DIP <-> physical conversion on Windows (exact with mixed-DPI monitors); a scale-factor fallback elsewhere. */
function converter() {
  if (process.platform === "win32") return screen;
  return displayConverter(screen.getAllDisplays().map((d) => ({ bounds: d.bounds, scaleFactor: d.scaleFactor })));
}

function savedPosDip() {
  return toDip(state.settings.buddy_position, converter());
}

function workAreaFor(mode, pos) {
  const finite = Array.isArray(pos) && pos.length === 2 && pos.every(Number.isFinite);
  const display = mode === "buddy" && finite
    ? screen.getDisplayNearestPoint({ x: pos[0], y: pos[1] })
    : screen.getPrimaryDisplay();
  return display.workArea;
}

function createStageWindow(mode) {
  if (mode === "buddy") refreshCanvas();
  const savedPos = savedPosDip();
  const area = workAreaFor(mode, savedPos);
  const bounds = boundsFor(mode, area, savedPos, buddyBoundsSize());
  const hasSaved = Array.isArray(savedPos) && savedPos.length === 2 && savedPos.every(Number.isFinite);
  if (mode === "buddy" && petOn() && !hasSaved) bounds.y = area.y + area.height - bounds.height; // she starts on the floor
  const overlay = mode === "buddy" || mode === "panel";
  const win = new BrowserWindow({
    ...bounds,
    show: false,
    title: "Riya",
    frame: !overlay,
    alwaysOnTop: overlay,
    skipTaskbar: overlay,
    resizable: !overlay,
    fullscreenable: !overlay,
    transparent: mode === "buddy",
    hasShadow: mode !== "buddy",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      additionalArguments: [`--riya-mode=${mode}`],
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  stageWindows.add(win);
  win.removeMenu();
  state.ignoring = false;
  if (mode === "buddy") {
    win.setIgnoreMouseEvents(true, { forward: true });
    state.ignoring = true;
  }
  win.once("ready-to-show", () => {
    if (win.isDestroyed()) return;
    if (mode === "buddy") win.showInactive();
    else win.show();
    if (SMOKE && !state.smokeTimer) {
      state.smokeTimer = setTimeout(() => {
        log(`smoke ok mode=${mode}`);
        app.quit();
      }, 2000);
    }
  });
  win.webContents.on("did-finish-load", () => {
    if (win.isDestroyed()) return;
    win.webContents.send("core:status", state.connected);
    for (const msg of state.cache.values()) win.webContents.send("core:message", msg);
    if (state.lastPetView) win.webContents.send("pet:view", state.lastPetView); // a new window draws her at once
  });
  win.on("close", (event) => {
    if (!state.quitting) {
      event.preventDefault();
      win.hide();
    }
  });
  for (const event of ["hide", "blur"]) win.on(event, () => cancelDragOf(win));
  win.on("hide", () => setPetVisible(win, false));
  win.on("show", () => setPetVisible(win, true));
  win.webContents.on("render-process-gone", (_event, details) => {
    log(`stage renderer gone (${details.reason}), recreating`);
    cancelDragOf(win);
    if (state.win === win) state.win = null;
    stageWindows.delete(win);
    if (!win.isDestroyed()) win.destroy();
    setTimeout(() => { if (!state.quitting && !state.win) ensureMode(state.mode); }, STAGE_RESPAWN_MS);
  });
  win.on("closed", () => {
    stageWindows.delete(win);
    if (state.win === win) state.win = null;
  });
  win.loadFile(path.join(__dirname, "index.html"));
  return win;
}

/** (Re)create the stage window for `mode` unless it is already showing. An unknown mode keeps the current one (buddy if none). */
function ensureMode(requested) {
  clearTimeout(state.firstSettingsTimer);
  const mode = modeToApply(requested, state.mode);
  if (state.win && state.mode === mode) return;
  stopDrag(false);
  state.press = null;
  stopPet(); // a new window means a new pet, started from where the window is
  const old = state.win;
  log(`mode -> ${mode}`);
  state.mode = mode;
  state.win = createStageWindow(mode);
  if (old && !old.isDestroyed()) old.destroy();
  if (petOn()) startPet();
  refreshTrayMenu();
  sendBodyReady();
}

/** The core's settings can arrive after buddy is already shown: move her to the saved position (unless she was dragged). */
function applyLateSavedPosition() {
  const win = state.win;
  if (!win || win.isDestroyed()) return;
  const current = win.getPosition();
  const savedPos = savedPosDip();
  const target = repositionTarget({
    mode: state.mode,
    dragging: Boolean(state.drag),
    userDragged: state.userDragged,
    moved: state.petMoved,
    current,
    savedPos,
    workArea: workAreaFor(state.mode, savedPos),
    size: buddyBoundsSize(),
  });
  if (target) {
    log(`saved position arrived late, moving to ${target.x},${target.y}`);
    win.setBounds(target);
    if (state.pet) {
      state.pet.setPos(target);
      stepPet();
    }
  }
}

function showWindow() {
  const win = state.win;
  if (!win || win.isDestroyed()) return;
  if (win.isMinimized()) win.restore();
  win.show();
  win.focus();
}

/** Keep buddy/panel on screen when monitors or the taskbar change. */
function relayout() {
  const win = state.win;
  if (!win || win.isDestroyed() || state.drag || state.mode === "window") return;
  if (state.pet) { // a changed work area re-clamps her at once, and she falls if the floor moved up to meet her
    state.pet.setArea(petArea(win), Date.now());
    stepPet();
    return;
  }
  const [x, y] = win.getPosition();
  win.setBounds(boundsFor(state.mode, workAreaFor(state.mode, [x, y]), [x, y], buddyBoundsSize()));
}

// ---- Little Riya --------------------------------------------------------------------------------------------------

/** Buddy mode with the pet setting on (the default). Off is exactly the v0.8 buddy. */
function petOn() {
  return state.mode === "buddy" && state.settings.pet_enabled !== false;
}

/** The buddy window's size: hers (from the pack's canvas and pet_scale) with the pet on, else v0.8's 300x520. */
function buddySize() {
  if (petOn()) return petSize({ canvas: state.canvas, scale: state.settings.pet_scale });
  return { width: DEFAULT_SIZE.buddy[0], height: DEFAULT_SIZE.buddy[1] };
}

/** buddySize() in the shape geometry's boundsFor takes. */
function buddyBoundsSize() {
  const { width, height } = buddySize();
  return { buddy: [width, height] };
}

/** The work area of the monitor she is on (a drop on another monitor uses that monitor's bounds and floor). */
function petArea(win) {
  return screen.getDisplayMatching(win.getBounds()).workArea;
}

/** The settings-derived part of her context. */
function petSettingsContext() {
  const s = state.settings;
  return {
    enabled: s.pet_enabled !== false,
    walksWhen: s.pet_walks_when,
    quietHours: s.quiet_hours === undefined ? null : s.quiet_hours,
    reducedMotion: reducedMotionOn(s.reduced_motion, state.osReducedMotion),
  };
}

function startPet() {
  const win = state.win;
  if (state.pet || !win || win.isDestroyed() || !petOn()) return;
  const now = Date.now();
  const [x, y] = win.getPosition();
  const ctx = { ...petSettingsContext(), presence: state.presence, idleS: powerMonitor.getSystemIdleTime(), forceWalk: FORCE_WALK };
  state.pet = new Pet({ rng: makeRng(now), nowMs: now, area: petArea(win), size: buddySize(), pos: { x, y }, ctx });
  state.idlePoll = setInterval(pollIdle, IDLE_POLL_MS);
  stepPet(); // sends her first view
}

/** She is off: every timer, a click still waiting for its answer, and the pet itself are dropped, and the window is told
 *  the whole neutral view so it never keeps a stale pose. */
function stopPet() {
  clearInterval(state.petTimer);
  clearTimeout(state.petWake);
  clearInterval(state.idlePoll);
  state.petTimer = null;
  state.petWake = null;
  state.idlePoll = null;
  clearPendingFocus();
  state.pet = null;
  state.petPaused = false;
  state.press = null;
  state.focusToken += 1;
  state.lastPetView = OFF_VIEW;
  const win = state.win;
  if (win && !win.isDestroyed()) win.webContents.send("pet:view", OFF_VIEW);
}

/** Every 5 s: user idle time, for sleep. */
function pollIdle() {
  if (!state.pet) return;
  state.pet.setContext({ idleS: powerMonitor.getSystemIdleTime() }, Date.now());
  stepPet();
}

/** One step of her clock: idle time as of now, then move the window, send a changed view, save a position, and reschedule. */
function stepPet() {
  const pet = state.pet;
  if (!pet || state.petPaused) return;
  const now = Date.now();
  pet.setContext({ idleS: powerMonitor.getSystemIdleTime() }, now);
  const f = pet.step(now);
  const win = state.win;
  if (win && !win.isDestroyed()) {
    if (f.moved && !state.drag) {
      win.setBounds(moveRect(f, buddySize()));
      state.petMoved = true;
    }
    if (f.viewChanged) {
      state.lastPetView = f.view;
      win.webContents.send("pet:view", f.view);
    }
    if (f.save) saveBuddyPosition(win);
  } else if (f.viewChanged) state.lastPetView = f.view;
  schedulePet();
}

/** The 20 Hz interval exists only while she walks or falls; otherwise one timeout to her next deadline (none when off). */
function schedulePet() {
  const pet = state.pet;
  if (!pet) return;
  clearTimeout(state.petWake);
  state.petWake = null;
  if (pet.ticking()) {
    if (!state.petTimer) state.petTimer = setInterval(stepPet, PET.MOVE_MS);
    return;
  }
  clearInterval(state.petTimer);
  state.petTimer = null;
  const wake = pet.nextWakeMs(Date.now());
  if (wake !== null) state.petWake = setTimeout(stepPet, Math.max(0, wake - Date.now()));
}

/** The buddy window hid or showed: a hidden pet has no timers, no steps and no saves, and a shown one picks up from where
 *  she was (her clock caps a long gap, so she does not lurch). */
function setPetVisible(win, visible) {
  if (state.win !== win || !state.pet) return;
  const action = petVisibilityAction(state.petPaused, visible);
  if (action === "pause") {
    state.petPaused = true;
    clearInterval(state.petTimer);
    clearTimeout(state.petWake);
    clearInterval(state.idlePoll);
    state.petTimer = null;
    state.petWake = null;
    state.idlePoll = null;
  } else if (action === "resume") {
    state.petPaused = false;
    state.idlePoll = setInterval(pollIdle, IDLE_POLL_MS);
    stepPet();
  }
}

/** Feed her a context change and step at once, so a gate change never waits for the next wake. */
function feedPet(patch) {
  if (!state.pet) return;
  state.pet.setContext(patch, Date.now());
  stepPet();
}

function petReact(kind, label) {
  if (!state.pet) return;
  state.pet.react(kind, label, Date.now());
  stepPet();
}

function petSleep() {
  if (!state.pet) return;
  state.pet.sleepNow(Date.now());
  stepPet();
}

/** A drag that ended without drag-end (the window hid, blurred or crashed): let go of her where she is. */
function releasePet() {
  state.press = null;
  const win = state.win;
  if (!state.pet || state.pet.state !== "drag" || !win || win.isDestroyed()) return;
  const [x, y] = win.getPosition();
  state.pet.setArea(petArea(win), Date.now());
  state.pet.drop({ x, y }, Date.now(), { save: false });
  stepPet();
}

/** Settings changed: the pet turns off or on, and a size change keeps her feet where they were. `before` is what the
 *  window, mode and size were before the new settings. */
function applyPetSettings(before) {
  const win = state.win;
  const same = Boolean(win) && !win.isDestroyed() && win === before.win && state.mode === before.mode;
  if (!petOn() && state.pet) stopPet();
  if (same && state.mode === "buddy") resizeBuddy(before.size);
  if (petOn() && !state.pet) startPet();
  else feedPet(petSettingsContext());
}

/** Resize the buddy window in place (pet_scale, pet_enabled or the pack changed): her feet stay put, and she stays in the
 *  work area. */
function resizeBuddy(old) {
  const win = state.win;
  const size = buddySize();
  if (!win || win.isDestroyed() || (size.width === old.width && size.height === old.height)) return;
  const [x, y] = win.getPosition();
  const rect = clampToArea({ x, y: y + old.height - size.height, width: size.width, height: size.height }, petArea(win));
  win.setBounds(rect);
  if (state.pet) {
    state.pet.setPos(rect);
    state.pet.setSize(size, Date.now());
    stepPet();
  }
}

/** A click on her (a press that barely moved): the agent that needs the user comes forward, else she is poked. */
function petClick() {
  const now = Date.now();
  const action = clickAction(focusTarget(state.agents, state.lastActivity, now), state.connected);
  if (action.poke) {
    if (!state.pet) return;
    state.pet.poke(now);
    stepPet();
  } else if (action.list) {
    openAgentsList();
  } else {
    clearPendingFocus();
    state.focusToken += 1;
    if (state.win && !state.win.isDestroyed()) state.win.focus(); // Riya holds foreground rights when the helper starts
    sendToCore({ type: "focus_agent", id: action.focus });
    state.pendingFocus = { id: action.focus, atMs: now, timer: setTimeout(focusTimedOut, FOCUS_WAIT_MS) };
  }
}

/** No answer within FOCUS_WAIT_MS counts as no window found: she opens the agents list. */
function focusTimedOut() {
  state.pendingFocus = null;
  openAgentsList();
}

function clearPendingFocus() {
  if (state.pendingFocus) clearTimeout(state.pendingFocus.timer);
  state.pendingFocus = null;
}

// ---- click-through and dragging ---------------------------------------------------------------------------------

function isBuddySender(sender) {
  return state.mode === "buddy" && state.win && !state.win.isDestroyed() && state.win.webContents === sender;
}

ipcMain.on("click-through", (event, ignore) => {
  if (!isBuddySender(event.sender) || typeof ignore !== "boolean") return;
  const want = ignore && !state.drag;
  if (want === state.ignoring) return;
  state.ignoring = want;
  if (want) state.win.setIgnoreMouseEvents(true, { forward: true });
  else state.win.setIgnoreMouseEvents(false);
});

ipcMain.on("pet:reduced-motion", (event, on) => {
  if (!isBuddySender(event.sender) || typeof on !== "boolean") return;
  state.osReducedMotion = on;
  feedPet({ reducedMotion: reducedMotionOn(state.settings.reduced_motion, state.osReducedMotion) });
});

/** Her right-click menu: the items come from petMenuItems, main adds what each one does. */
ipcMain.on("pet:menu", (event) => {
  if (!isBuddySender(event.sender)) return;
  const items = petMenuItems({
    mode: state.mode,
    dontLook: state.dontLook,
    reducedMotion: reducedMotionOn(state.settings.reduced_motion, state.osReducedMotion),
  });
  const clicks = {
    "mode:buddy": () => requestMode("buddy"),
    "mode:panel": () => requestMode("panel"),
    "mode:window": () => requestMode("window"),
    dont_look: toggleDontLook,
    reduced_motion: (menuItem) => sendToCore({ type: "set_settings", patch: reducedMotionPatch(menuItem.checked) }),
    sleep: petSleep,
    settings: openSettings,
    quit: quitRiya,
  };
  const template = items.map((item) => {
    if (item.type === "separator") return { type: "separator" };
    const entry = { label: item.label, type: item.type, click: (menuItem) => clicks[item.id](menuItem) };
    if (item.checked !== undefined) entry.checked = item.checked;
    return entry;
  });
  Menu.buildFromTemplate(template).popup({ window: state.win });
});

ipcMain.on("drag-start", (event) => {
  if (!isBuddySender(event.sender) || state.drag) return;
  const win = state.win;
  const cursor = screen.getCursorScreenPoint();
  const [x, y] = win.getPosition();
  const size = state.pet ? buddySize() : null; // her window has a fixed size, whatever the OS reports mid-drag
  const [width, height] = size ? [size.width, size.height] : win.getSize();
  win.setIgnoreMouseEvents(false);
  state.ignoring = false;
  const drag = { win, cursor, x, y, width, height, timer: null };
  drag.timer = setInterval(() => {
    if (win.isDestroyed()) return stopDrag(false);
    const now = screen.getCursorScreenPoint();
    const rect = { x: x + now.x - cursor.x, y: y + now.y - cursor.y, width, height };
    win.setBounds(clampToArea(rect, screen.getDisplayNearestPoint(now).workArea));
  }, DRAG_INTERVAL_MS);
  state.drag = drag;
  if (state.pet) {
    state.press = { cursor, atMs: Date.now(), pos: { x, y } };
    state.pet.grab(Date.now());
    stepPet();
  }
});

ipcMain.on("drag-end", (event) => {
  if (!state.drag || !isBuddySender(event.sender)) return;
  if (!state.pet) { // pet off: the v0.8 buddy, a drag is only a drag
    stopDrag(true);
    return;
  }
  const now = Date.now();
  const press = state.press;
  state.press = null;
  stopDrag(false);
  const win = state.win;
  if (!win || win.isDestroyed()) return;
  const [x, y] = win.getPosition();
  // A press that barely moved is a click: she stands back where she was, so it never hops, falls or saves. The drop
  // re-reads the monitor first, so a drop on another monitor uses that monitor's floor.
  const cursor = screen.getCursorScreenPoint();
  const click = press !== null && isClick({ dx: cursor.x - press.cursor.x, dy: cursor.y - press.cursor.y, ms: now - press.atMs });
  state.pet.setArea(screen.getDisplayMatching(win.getBounds()).workArea, now);
  if (click) {
    win.setBounds(moveRect(press.pos, buddySize()));
    state.pet.drop(press.pos, now, { save: false });
    petClick();
  } else {
    state.pet.drop({ x, y }, now);
    state.petMoved = true;
  }
  stepPet();
});

/** A drag that loses its window (hidden, blurred, crashed) ends unsaved, and buddy goes back to click-through. */
function cancelDragOf(win) {
  if (!state.drag || state.drag.win !== win) return;
  stopDrag(false);
  releasePet();
  if (state.mode === "buddy" && state.win === win && !win.isDestroyed()) {
    win.setIgnoreMouseEvents(true, { forward: true });
    state.ignoring = true;
  }
}

function stopDrag(save) {
  const drag = state.drag;
  if (!drag) return;
  clearInterval(drag.timer);
  state.drag = null;
  if (save && !drag.win.isDestroyed()) saveBuddyPosition(drag.win);
}

/** Remember where buddy is (physical pixels, what the core compares with window rects); the late saved position then
 *  leaves her alone this session. */
function saveBuddyPosition(win) {
  const [x, y] = win.getPosition();
  const position = toPhysical([x, y], converter());
  state.settings.buddy_position = position;
  state.userDragged = true;
  sendToCore({ type: "set_settings", patch: { buddy_position: position } });
}

// ---- tray, settings and hotkey ----------------------------------------------------------------------------------

/** The one settings window: a normal framed window for the settings controls and the memory list. */
function openSettings() {
  const existing = state.settingsWin;
  if (existing && !existing.isDestroyed()) {
    if (existing.isMinimized()) existing.restore();
    existing.show();
    existing.focus();
    return;
  }
  const win = new BrowserWindow({
    width: 520,
    height: 720,
    show: false,
    title: "Riya settings",
    autoHideMenuBar: true,
    fullscreenable: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      additionalArguments: ["--riya-mode=settings"],
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  state.settingsWin = win;
  state.settingsLoaded = false;
  win.removeMenu();
  win.once("ready-to-show", () => {
    if (win.isDestroyed()) return;
    win.show();
  });
  win.webContents.on("did-finish-load", () => {
    if (win.isDestroyed()) return;
    win.webContents.send("core:status", state.connected);
    for (const msg of state.cache.values()) win.webContents.send("core:message", msg);
    state.settingsLoaded = true;
    sendSettingsSection();
  });
  win.on("closed", () => {
    if (state.settingsWin === win) {
      state.settingsWin = null;
      state.settingsLoaded = false;
      state.settingsSection = null;
    }
  });
  win.loadFile(path.join(__dirname, "settings.html"));
}

/** Tell the settings window which section to scroll to, once it has loaded (right away when it already has). */
function sendSettingsSection() {
  const win = state.settingsWin;
  if (!win || win.isDestroyed() || !state.settingsLoaded || !state.settingsSection) return;
  win.webContents.send("settings:show", state.settingsSection);
  state.settingsSection = null;
}

/** "The agents list" is Settings scrolled to its Agents section. */
function openAgentsList() {
  state.settingsSection = "agents";
  openSettings();
  sendSettingsSection();
}

/** The first-run welcome: the openSettings() pattern, a small window of its own. Closing it by any route (Finish, Skip, the
 *  X) counts as skipping, so the `closed` handler is the one place that tells the core she is done; welcomeClosedMessage
 *  holds it back while Riya quits. The renderer never sends it itself. */
function openWelcome() {
  const existing = state.welcomeWin;
  if (existing && !existing.isDestroyed()) {
    if (existing.isMinimized()) existing.restore();
    existing.show();
    existing.focus();
    return;
  }
  const win = new BrowserWindow({
    width: 460,
    height: 600,
    show: false,
    title: "Welcome to Riya",
    autoHideMenuBar: true,
    fullscreenable: false,
    webPreferences: welcomeWebPreferences(path.join(__dirname, "preload.js")),
  });
  state.welcomeWin = win;
  // a Windows shutdown or logoff emits no before-quit: it counts as quitting, so closing marks nothing done
  for (const event of QUIT_EVENTS.window) win.on(event, () => { state.quitting = true; });
  win.removeMenu();
  win.once("ready-to-show", () => {
    if (win.isDestroyed()) return;
    win.show();
  });
  win.webContents.on("did-finish-load", () => {
    if (win.isDestroyed()) return;
    win.webContents.send("core:status", state.connected);
    for (const msg of state.cache.values()) win.webContents.send("core:message", msg);
  });
  win.on("closed", () => {
    const m = welcomeClosedMessage({ quitting: state.quitting, onboarded: state.settings.onboarded });
    if (m) sendToCore(m);
    if (state.welcomeWin === win) state.welcomeWin = null;
  });
  win.loadFile(path.join(__dirname, "welcome.html"));
}

/** "settings" / "welcome" for those two windows' webContents, else "other": what ipcAllowed decides on. */
function senderKind(sender) {
  if (state.settingsWin && !state.settingsWin.isDestroyed() && state.settingsWin.webContents === sender) return "settings";
  if (state.welcomeWin && !state.welcomeWin.isDestroyed() && state.welcomeWin.webContents === sender) return "welcome";
  return "other";
}

ipcMain.on("welcome:open", (event) => {
  if (ipcAllowed("welcome:open", senderKind(event.sender))) openWelcome();
});
// No argument is read: the page opened is always KEY_URL, never anything the renderer says.
ipcMain.on("welcome:key-page", (event) => {
  if (ipcAllowed("welcome:key-page", senderKind(event.sender))) shell.openExternal(KEY_URL).catch(() => {});
});

function requestMode(mode) {
  state.settings.mode = mode;
  ensureMode(mode);
  sendToCore({ type: "set_settings", patch: { mode } });
}

function toggleDontLook() {
  sendToCore({ type: "dont_look", on: !state.dontLook });
  refreshTrayMenu(); // the checkbox flips by itself when clicked; put it back until the core's eyes message says so
}

/** The tray's Quit: the core stops too (quit_core), so nothing keeps glancing with no body; then the body exits. */
function quitRiya() {
  if (state.quitting) return;
  sendToCore({ type: "quit_core" });
  setTimeout(() => app.quit(), quitDelay(state.connected));
}

function refreshTrayMenu() {
  if (!state.tray) return;
  const modeItem = (mode, label) => ({
    label,
    type: "radio",
    checked: state.mode === mode,
    click: () => requestMode(mode),
  });
  state.tray.setContextMenu(Menu.buildFromTemplate([
    modeItem("buddy", "Buddy"),
    modeItem("panel", "Panel"),
    modeItem("window", "Window"),
    { type: "separator" },
    { label: "Don't look", type: "checkbox", checked: state.dontLook, click: toggleDontLook },
    { label: "Settings…", click: openSettings },
    { label: "Quit", click: quitRiya },
  ]));
}

function createTray() {
  const icon = nativeImage.createFromBuffer(makeTrayPng(16));
  icon.addRepresentation({ scaleFactor: 2, buffer: makeTrayPng(32) });
  state.tray = new Tray(icon);
  state.tray.setToolTip("Riya");
  state.tray.on("click", showWindow);
  refreshTrayMenu();
}

// ---- lifecycle --------------------------------------------------------------------------------------------------

if (!app.requestSingleInstanceLock()) {
  log("another Riya body is already running; focusing it");
  app.quit();
} else {
  app.on("second-instance", showWindow);
  app.on("window-all-closed", () => {}); // a tray app: stay alive with no windows
  for (const event of QUIT_EVENTS.app) app.on(event, () => { state.quitting = true; });
  app.on("will-quit", () => {
    globalShortcut.unregisterAll();
    stopDrag(false);
    stopPet();
  });

  app.whenReady().then(() => {
    session.defaultSession.setPermissionRequestHandler((_wc, _permission, callback) => callback(false));
    session.defaultSession.setPermissionCheckHandler(() => false);
    createBridge();
    createTray();
    state.hotkey = pickHotkey(TALK_HOTKEYS, (accelerator) =>
      globalShortcut.register(accelerator, () => sendToCore({ type: "talk_hotkey" })));
    state.dontLookHotkey = pickHotkey(DONT_LOOK_HOTKEYS, (accelerator) =>
      globalShortcut.register(accelerator, toggleDontLook));
    if (OPEN_SETTINGS) openSettings();
    log(state.hotkey ? `talk hotkey: ${state.hotkey}` : `no talk hotkey registered (tried ${TALK_HOTKEYS.join(", ")})`);
    log(state.dontLookHotkey ? `don't-look hotkey: ${state.dontLookHotkey}`
      : `no don't-look hotkey registered (tried ${DONT_LOOK_HOTKEYS.join(", ")})`);
    for (const event of ["display-added", "display-removed", "display-metrics-changed"]) screen.on(event, relayout);
    // Wait briefly for the core's settings so she opens in the right mode; show buddy if the core isn't there.
    state.firstSettingsTimer = setTimeout(() => ensureMode(state.settings.mode), FIRST_SETTINGS_WAIT_MS);
  }).catch((err) => fail("startup failed", err));
}
