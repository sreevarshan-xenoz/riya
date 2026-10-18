"use strict";
/** Electron main process: the three window modes, click-through and dragging, tray, talk hotkey and the core relay.
 *
 *  Electron 33's main process has no WebSocket (Node 20), so the single core connection runs in a hidden bridge window
 *  (bridge.html) built on the reusable CoreClient and is relayed here over IPC; this process fans it out to the stage
 *  windows over `core:message` / `core:status` and forwards their `core:send` back. */
const fs = require("node:fs");
const path = require("node:path");
const { app, BrowserWindow, Tray, Menu, nativeImage, screen, globalShortcut, ipcMain, session } = require("electron");
const { boundsFor, clampToArea } = require("./src/geometry");
const { getPackPath, resolveManifestFrames } = require("./src/pack-path");
const { makeTrayPng } = require("./src/tray-icon");

const SMOKE = process.env.RIYA_SMOKE === "1";
const REPO_ROOT = path.resolve(__dirname, "..");
const MODES = ["buddy", "panel", "window"];
const TALK_HOTKEY = "Control+Alt+Space";
const FIRST_SETTINGS_WAIT_MS = 1500;
const DRAG_INTERVAL_MS = 1000 / 60;
const REPLAY_TYPES = ["hello", "settings", "state", "expression"];

const state = {
  settings: {},
  mode: null,
  win: null,
  bridge: null,
  bridgeReady: false,
  bridgeQueue: [],
  connected: false,
  readyPending: false,
  ignoring: false,
  drag: null,
  quitting: false,
  tray: null,
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
  if (state.bridge && state.bridgeReady) state.bridge.webContents.send("bridge:send", msg);
  else state.bridgeQueue.push(msg);
}

function broadcast(channel, payload) {
  for (const win of stageWindows) if (!win.isDestroyed()) win.webContents.send(channel, payload);
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
  if (REPLAY_TYPES.includes(msg.type)) state.cache.set(msg.type, msg);
  if (msg.type === "settings" && msg.settings && typeof msg.settings === "object") {
    state.settings = { ...msg.settings };
    ensureMode(state.settings.mode);
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
  if (!isStageSender(event.sender)) return;
  if (msg && typeof msg === "object" && typeof msg.type === "string") sendToCore(msg);
});

function isStageSender(sender) {
  for (const win of stageWindows) if (!win.isDestroyed() && win.webContents === sender) return true;
  return false;
}

// ---- sprite pack ----------------------------------------------------------------------------------------------------

/** The first usable pack (settings.sprite_pack, sprites/riya-v1, placeholder-pack) with frame paths as file URLs. */
function loadPack() {
  const skipped = new Set();
  const existsSync = (p) => !skipped.has(path.dirname(p)) && fs.existsSync(p);
  for (;;) {
    const dir = getPackPath({ spritePack: state.settings.sprite_pack, repoRoot: REPO_ROOT, bodyDir: __dirname, existsSync });
    if (!dir) return null;
    try {
      const raw = JSON.parse(fs.readFileSync(path.join(dir, "manifest.json"), "utf8"));
      const manifest = resolveManifestFrames(raw, dir, fs.existsSync);
      if (manifest && manifest.expressions.neutral && manifest.expressions.neutral.closed) return { dir, manifest };
      log("sprite pack has no usable neutral.closed frame, skipping", dir);
    } catch (e) {
      log("sprite pack manifest unreadable, skipping", dir, e.message);
    }
    skipped.add(dir);
  }
}

ipcMain.handle("pack:get", (event) => (isStageSender(event.sender) ? loadPack() : null));

// ---- windows --------------------------------------------------------------------------------------------------------

function workAreaFor(mode, pos) {
  const finite = Array.isArray(pos) && pos.length === 2 && pos.every(Number.isFinite);
  const display = mode === "buddy" && finite
    ? screen.getDisplayNearestPoint({ x: pos[0], y: pos[1] })
    : screen.getPrimaryDisplay();
  return display.workArea;
}

function createStageWindow(mode) {
  const savedPos = state.settings.buddy_position;
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
      backgroundThrottling: false,
    },
  });
  stageWindows.add(win);
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
  win.on("closed", () => {
    stageWindows.delete(win);
    if (state.win === win) state.win = null;
  });
  win.loadFile(path.join(__dirname, "index.html"));
  return win;
}

/** (Re)create the stage window for `mode` unless it is already showing. Ignores unknown modes. */
function ensureMode(mode) {
  clearTimeout(state.firstSettingsTimer);
  if (!MODES.includes(mode) || (state.win && state.mode === mode)) return;
  stopDrag(false);
  const old = state.win;
  log(`mode -> ${mode}`);
  state.mode = mode;
  state.win = createStageWindow(mode);
  if (old && !old.isDestroyed()) old.destroy();
  refreshTrayMenu();
  sendBodyReady();
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

function stopDrag(save) {
  const drag = state.drag;
  if (!drag) return;
  clearInterval(drag.timer);
  state.drag = null;
  if (!save || drag.win.isDestroyed()) return;
  const [x, y] = drag.win.getPosition();
  state.settings.buddy_position = [x, y];
  sendToCore({ type: "set_settings", patch: { buddy_position: [x, y] } });
}

// ---- tray, settings and hotkey ----------------------------------------------------------------------------------

function openSettings() {
  log("settings window is not implemented yet");
}

function requestMode(mode) {
  state.settings.mode = mode;
  ensureMode(mode);
  sendToCore({ type: "set_settings", patch: { mode } });
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
    { label: "Settings…", click: openSettings },
    { label: "Quit", click: () => app.quit() },
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
    if (!globalShortcut.register(TALK_HOTKEY, () => sendToCore({ type: "talk_hotkey" }))) {
      log(`could not register ${TALK_HOTKEY} (already taken?)`);
    }
    for (const event of ["display-added", "display-removed", "display-metrics-changed"]) screen.on(event, relayout);
    // Wait briefly for the core's settings so she opens in the right mode; show buddy if the core isn't there.
    state.firstSettingsTimer = setTimeout(() => ensureMode(MODES.includes(state.settings.mode) ? state.settings.mode : "buddy"),
      FIRST_SETTINGS_WAIT_MS);
  }).catch((err) => fail("startup failed", err));
}
