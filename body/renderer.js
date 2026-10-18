"use strict";
/** The stage: sprite crossfade, blink/breathing/lean loop, mouth, speech bubble, chat log, text input and (buddy mode)
 *  alpha-based click-through and dragging. Pure decisions live in src/ (sprite-logic, bubble, stage-logic, hit-test). */
(() => {
  const { frameFor, breathing, leanFor, BlinkScheduler } = RiyaSprite;
  const { isOpaqueAt, toImageCoords } = RiyaHitTest;
  const { BubbleModel } = RiyaBubble;
  const { ExpressionModel, MouthTracker, StatusLine, cropBox } = RiyaStage;

  const mode = ["buddy", "panel", "window"].includes(window.riya.mode) ? window.riya.mode : "buddy";
  const MAX_LOG_ENTRIES = 300;
  const MASK_MAX_WIDTH = 256;
  const FADE_MS = 150;
  const now = () => performance.now();
  const $ = (id) => document.getElementById(id);

  document.body.classList.add(`mode-${mode}`);
  const els = {
    box: $("sprite-box"), sprite: $("sprite"), bubble: $("bubble"), status: $("status"),
    log: $("log"), form: $("composer"), input: $("input"), layers: [$("layer-a"), $("layer-b")],
  };

  const bubble = new BubbleModel();
  const expression = new ExpressionModel();
  const mouth = new MouthTracker();
  const errorLine = new StatusLine();
  const blink = new BlinkScheduler();
  const feelings = { presence: null, curiosity: 0 };
  const lean = { rotateDeg: 0, dx: 0 };
  let connected = false;
  expression.setConnected(false);

  let manifest = null;
  let front = 0;
  let wantedUrl = null;
  let frameToken = 0;
  let lastExpression = null;
  let fadeUntil = 0;
  const preloaded = [];

  // ---- sprite frames --------------------------------------------------------------------------------------------

  /** Show `url` on the hidden layer, then swap. Expression changes crossfade (150 ms, in CSS); mouth and blink frames swap at once. */
  function showFrame(url, fade) {
    const token = ++frameToken;
    if (fade) fadeUntil = now() + FADE_MS;
    else if (now() < fadeUntil) fade = true; // a mouth or blink swap mid-crossfade must not cut it short
    const next = els.layers[1 - front];
    const apply = () => {
      if (token !== frameToken) return;
      const prev = els.layers[front];
      for (const layer of els.layers) layer.style.transition = fade ? "" : "none";
      next.style.opacity = "1";
      prev.style.opacity = "0";
      front = 1 - front;
      refreshHitMask(next);
    };
    if (next.getAttribute("src") === url) return apply();
    next.src = url;
    (next.decode ? next.decode() : Promise.resolve()).then(apply, apply);
  }

  function layoutSprite() {
    if (!manifest) return;
    const { width, height } = manifest.canvas;
    els.box.style.setProperty("--ar", String(width / height));
    if (mode !== "panel") return;
    const box = cropBox($("stage").clientWidth, manifest.canvas, manifest.crops.upper);
    els.box.style.width = `${box.width}px`;
    els.box.style.height = `${box.height}px`;
    for (const layer of els.layers) {
      layer.style.left = `${box.left}px`;
      layer.style.top = `${box.top}px`;
      layer.style.width = `${box.imgWidth}px`;
      layer.style.height = `${box.imgHeight}px`;
    }
  }

  function preload(pack) {
    for (const frames of Object.values(pack.expressions)) {
      for (const url of Object.values(frames)) {
        const img = new Image();
        img.src = url;
        preloaded.push(img);
      }
    }
  }

  // ---- buddy: click-through only on her opaque pixels ----------------------------------------------------------

  const hit = { canvas: document.createElement("canvas"), ctx: null, data: null, w: 0, h: 0, blocked: false };
  let clickable = false;
  let dragging = false;

  /** Copy the current frame's alpha into a small mask, once per frame change. A tainted canvas disables the mask. */
  function refreshHitMask(img) {
    if (mode !== "buddy" || !manifest) return;
    try {
      const scale = Math.min(1, MASK_MAX_WIDTH / manifest.canvas.width);
      hit.w = Math.max(1, Math.round(manifest.canvas.width * scale));
      hit.h = Math.max(1, Math.round(manifest.canvas.height * scale));
      hit.canvas.width = hit.w;
      hit.canvas.height = hit.h;
      hit.ctx = hit.ctx || hit.canvas.getContext("2d", { willReadFrequently: true });
      hit.ctx.clearRect(0, 0, hit.w, hit.h);
      hit.ctx.drawImage(img, 0, 0, hit.w, hit.h);
      hit.data = hit.ctx.getImageData(0, 0, hit.w, hit.h).data;
      hit.blocked = false;
    } catch (e) {
      hit.data = null;
      hit.blocked = true; // pixels unreadable: the whole sprite box counts as her
    }
  }

  function overSprite(clientX, clientY) {
    if (!manifest) return false;
    const rect = els.box.getBoundingClientRect();
    const p = toImageCoords(rect, hit.w || 1, hit.h || 1, clientX, clientY);
    if (!p) return false;
    if (hit.blocked) return p.x >= 0 && p.y >= 0 && p.x < (hit.w || 1) && p.y < (hit.h || 1);
    return hit.data ? isOpaqueAt(hit.data, hit.w, p.x, p.y) : false;
  }

  function setClickable(value) {
    if (value === clickable) return;
    clickable = value;
    window.riya.setClickThrough(!value);
  }

  function endDrag() {
    if (!dragging) return;
    dragging = false;
    window.riya.dragEnd();
  }

  if (mode === "buddy") {
    document.addEventListener("mousemove", (e) => {
      if (dragging) return;
      const onBubble = e.target instanceof Element && e.target.closest("#bubble") !== null;
      setClickable(onBubble || overSprite(e.clientX, e.clientY));
    });
    document.addEventListener("mousedown", (e) => {
      if (e.button !== 0 || dragging) return;
      if (e.target instanceof Element && e.target.closest("#bubble")) return;
      if (!overSprite(e.clientX, e.clientY)) return;
      dragging = true;
      setClickable(true);
      window.riya.dragStart();
    });
    document.addEventListener("mouseup", endDrag);
    window.addEventListener("blur", () => { endDrag(); setClickable(false); });
    document.documentElement.addEventListener("mouseleave", () => { if (!dragging) setClickable(false); });
  }

  // ---- chat log (panel and window) -----------------------------------------------------------------------------

  let live = { id: null, el: null };

  function addEntry(kind, text, extraClass) {
    const el = document.createElement("div");
    el.className = `entry ${kind}${extraClass ? ` ${extraClass}` : ""}`;
    el.textContent = text;
    const stick = els.log.scrollHeight - els.log.scrollTop - els.log.clientHeight < 40;
    els.log.appendChild(el);
    while (els.log.childElementCount > MAX_LOG_ENTRIES) els.log.removeChild(els.log.firstElementChild);
    if (stick) els.log.scrollTop = els.log.scrollHeight;
    return el;
  }

  /** Her utterance appears in the log as soon as its first sentence arrives and grows to the complete text. */
  function logUtterance() {
    if (mode === "buddy" || bubble.id === null) return;
    if (live.id !== bubble.id) live = { id: bubble.id, el: addEntry("riya", bubble.text(), bubble.kind() === "bid" ? "bid" : "") };
    else live.el.textContent = bubble.text();
    els.log.scrollTop = els.log.scrollHeight;
  }

  els.form.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = els.input.value.trim();
    if (!text) return;
    window.riya.send({ type: "user_text", text });
    bubble.onUserInput();
    addEntry("user", text);
    els.input.value = "";
  });

  // ---- core messages -------------------------------------------------------------------------------------------

  window.riya.onCoreMessage((msg) => {
    if (!msg || typeof msg.type !== "string") return;
    const t = now();
    switch (msg.type) {
      case "say": {
        const fresh = bubble.id !== msg.utterance_id;
        bubble.onSay(msg);
        if (fresh && msg.kind === "bid" && bubble.id === msg.utterance_id) expression.onBid();
        logUtterance();
        break;
      }
      case "say_end":
        bubble.onSayEnd(msg.utterance_id, t);
        break;
      case "mouth":
        mouth.onLevel(Number(msg.level) || 0, t);
        break;
      case "expression":
        expression.onExpression(msg.name);
        break;
      case "state":
        expression.onState(msg);
        feelings.presence = msg.presence;
        feelings.curiosity = Number(msg.feelings && msg.feelings.curiosity) || 0;
        break;
      case "error":
        errorLine.show(msg.message, t);
        break;
      default:
        break;
    }
  });

  window.riya.onConnection((value) => {
    connected = Boolean(value);
    expression.setConnected(connected);
    if (!connected) bubble.endCurrent(now());
  });

  // ---- the frame loop ------------------------------------------------------------------------------------------

  let shownBubble = { visible: false, text: "", kind: "" };
  let shownStatus = null;

  function renderBubble(t) {
    if (mode !== "buddy") return;
    const visible = bubble.visible(t);
    const text = bubble.text();
    const kind = bubble.kind();
    if (visible !== shownBubble.visible) els.bubble.hidden = !visible;
    if (visible && text !== shownBubble.text) {
      els.bubble.textContent = text;
      els.bubble.scrollTop = els.bubble.scrollHeight;
    }
    if (kind !== shownBubble.kind) els.bubble.classList.toggle("bid", kind === "bid");
    shownBubble = { visible, text, kind };
  }

  function renderStatus(t) {
    const text = connected ? errorLine.text(t) : "connecting…";
    if (text === shownStatus) return;
    shownStatus = text;
    els.status.textContent = text;
    els.status.hidden = !text;
  }

  function tick(t) {
    requestAnimationFrame(tick);
    renderBubble(t);
    renderStatus(t);
    if (!manifest) return;
    const bubbleVisible = bubble.visible(t);
    const name = expression.current(bubbleVisible);
    const url = frameFor({ manifest, expression: name, mouthOpen: connected && mouth.isOpen(t), blinking: blink.isClosed(t) });
    if (url && url !== wantedUrl) {
      showFrame(url, lastExpression !== null && name !== lastExpression);
      wantedUrl = url;
    }
    lastExpression = name;

    const target = leanFor({ presence: feelings.presence, curiosity: feelings.curiosity });
    lean.rotateDeg += (target.rotateDeg - lean.rotateDeg) * 0.08;
    lean.dx += (target.dx - lean.dx) * 0.08;
    const b = breathing(t);
    els.sprite.style.transform = `translate(${lean.dx}px, ${b.dy}px) rotate(${lean.rotateDeg}deg) scale(${b.scale})`;
  }

  // ---- start ---------------------------------------------------------------------------------------------------

  window.addEventListener("resize", layoutSprite);
  window.riya.hotkey().then((key) => {
    if (key && mode !== "buddy") els.input.placeholder = `Say something… (or press ${key} to talk)`;
  }).catch(() => {});
  window.riya.getPack().then((pack) => {
    if (pack && pack.manifest) {
      manifest = pack.manifest;
      layoutSprite();
      preload(manifest);
    } else {
      errorLine.show("no sprite pack found", now());
    }
  }).catch(() => errorLine.show("couldn't load the sprite pack", now()));
  if (mode !== "buddy") els.input.focus();
  requestAnimationFrame(tick);
})();
