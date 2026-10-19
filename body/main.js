"use strict";
/** Electron main process: the three window modes, click-through and dragging, tray, hotkeys and the core relay.
 *
 *  Electron 33's main process has no WebSocket (Node 20), so the single core connection runs in a hidden bridge window
 *  (bridge.html) built on the reusable CoreClient and is relayed here over IPC; this process fans it out to the stage
 *  windows over `core:message` / `core:status` and forwards their `core:send` back. */
const path = require("node:path");
const { app, BrowserWindow, Tray, Menu, nativeImage, screen, globalShortcut, ipcMain, session, shell } = require("electron");
const { boundsFor, clampToArea } = require("./src/geometry");
const { loadFirstPack } = require("./src/pack-path");
const { makeTrayPng } = require("./src/tray-icon");
const {
  pickHotkey, modeToApply, repositionTarget, shouldSendToCore, quitDelay, rememberForReplay, toPhysical, toDip, displayConverter,
  forgetOnDisconnect, DONT_LOOK_HOTKEYS, KEY_URL, shouldOpenWelcome, welcomeClosedMessage, ipcAllowed,
} = require("./src/main-logic");

const SMOKE = process.env.RIYA_SMOKE === "1";
const OPEN_SETTINGS = process.env.RIYA_OPEN_SETTINGS === "1"; // dev/smoke: open the settings window at startup
const REPO_ROOT = path.resolve(__dirname, "..");
const TALK_HOTKEYS = ["Control+Alt+Space", "Control+Alt+F9"]; // first one that registers wins
const FIRST_SETTINGS_WAIT_MS = 1500;
const DRAG_INTERVAL_MS = 1000 / 60;
const STAGE_RESPAWN_MS = 1000;

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
  if (!connected) forgetOnDisconnect(state.cache); // the core sends a fresh look and agents on hello
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
  if (msg.type === "settings" && msg.settings && typeof msg.settings === "object") {
    state.settings = { ...msg.settings };
    ensureMode(state.settings.mode);
    applyLateSavedPosition();
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
  if (msg && typeof msg === "object" && typeof msg.type === "string") sendToCore(msg);
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
  const savedPos = savedPosDip();
  const bounds = boundsFor(mode, workAreaFor(mode, savedPos), savedPos);
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
  });
  win.on("close", (event) => {
    if (!state.quitting) {
      event.preventDefault();
      win.hide();
    }
  });
  for (const event of ["hide", "blur"]) win.on(event, () => cancelDragOf(win));
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
  const old = state.win;
  log(`mode -> ${mode}`);
  state.mode = mode;
  state.win = createStageWindow(mode);
  if (old && !old.isDestroyed()) old.destroy();
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
    current,
    savedPos,
    workArea: workAreaFor(state.mode, savedPos),
  });
  if (target) {
    log(`saved position arrived late, moving to ${target.x},${target.y}`);
    win.setBounds(target);
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
  const [x, y] = win.getPosition();
  win.setBounds(boundsFor(state.mode, workAreaFor(state.mode, [x, y]), [x, y]));
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

ipcMain.on("drag-start", (event) => {
  if (!isBuddySender(event.sender) || state.drag) return;
  const win = state.win;
  const cursor = screen.getCursorScreenPoint();
  const [x, y] = win.getPosition();
  const [width, height] = win.getSize();
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
});

ipcMain.on("drag-end", (event) => {
  if (state.drag && isBuddySender(event.sender)) stopDrag(true);
});

/** A drag that loses its window (hidden, blurred, crashed) ends unsaved, and buddy goes back to click-through. */
function cancelDragOf(win) {
  if (!state.drag || state.drag.win !== win) return;
  stopDrag(false);
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
  if (!save || drag.win.isDestroyed()) return;
  const [x, y] = drag.win.getPosition();
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
    if (state.settingsWin === win) state.settingsWin = null;
  });
  win.loadFile(path.join(__dirname, "settings.html"));
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
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      additionalArguments: ["--riya-mode=welcome"],
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  state.welcomeWin = win;
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
  app.on("before-quit", () => { state.quitting = true; });
  app.on("will-quit", () => {
    globalShortcut.unregisterAll();
    stopDrag(false);
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
