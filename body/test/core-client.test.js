const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const Ajv = require("ajv");
const { CoreClient } = require("../src/core-client");

class FakeWS {
  static OPEN = 1;
  static last = null;
  constructor(url) {
    this.url = url;
    this.sent = [];
    this.readyState = 0;
    FakeWS.last = this;
  }
  send(data) { this.sent.push(data); }
  close() { this.readyState = 3; }
  open() { this.readyState = 1; this.onopen && this.onopen(); }
  message(obj) { this.onmessage && this.onmessage({ data: typeof obj === "string" ? obj : JSON.stringify(obj) }); }
  drop() { this.readyState = 3; this.onclose && this.onclose(); }
}

function make() {
  const delays = [];
  const timers = [];
  const setTimeoutFn = (fn, ms) => { delays.push(ms); timers.push(fn); return timers.length; };
  const client = new CoreClient({ url: "ws://x/ws", WebSocketImpl: FakeWS, setTimeoutFn });
  const fire = () => timers.shift()();
  return { client, delays, fire };
}

test("queues sends until open then flushes in order", () => {
  const { client } = make();
  client.connect();
  client.send({ type: "a" });
  client.send({ type: "b" });
  assert.deepEqual(FakeWS.last.sent, []);
  FakeWS.last.open();
  assert.deepEqual(FakeWS.last.sent.map((s) => JSON.parse(s)), [{ type: "a" }, { type: "b" }]);
  client.send({ type: "c" });
  assert.equal(FakeWS.last.sent.length, 3);
});

test("dispatches by type", () => {
  const { client } = make();
  const got = [];
  client.on("say", (m) => got.push(m));
  client.connect();
  FakeWS.last.open();
  const msg = { type: "say", utterance_id: "u1", text: "hi", spoken: true, kind: "reply" };
  FakeWS.last.message(msg);
  FakeWS.last.message({ type: "mouth", level: 0.1 });
  assert.deepEqual(got, [msg]);
});

test("ignores bad JSON", () => {
  const { client } = make();
  const logs = [];
  client.log = (...a) => logs.push(a);
  client.on("say", () => assert.fail("should not dispatch"));
  client.connect();
  FakeWS.last.open();
  FakeWS.last.message("{not json");
  assert.equal(logs.length, 1);
});

test("emits open and close", () => {
  const { client } = make();
  const events = [];
  client.on("open", () => events.push("open"));
  client.on("close", () => events.push("close"));
  client.connect();
  FakeWS.last.open();
  FakeWS.last.drop();
  assert.deepEqual(events, ["open", "close"]);
});

test("reconnect backoff", () => {
  const { client, delays, fire } = make();
  client.connect();
  for (let i = 0; i < 7; i++) {
    FakeWS.last.drop();
    fire();
  }
  assert.deepEqual(delays, [500, 1000, 2000, 4000, 8000, 10000, 10000]);
  FakeWS.last.open();
  FakeWS.last.drop();
  assert.equal(delays[delays.length - 1], 500);
});

test("close stops reconnecting", () => {
  const { client, delays } = make();
  client.connect();
  client.close();
  FakeWS.last.drop();
  assert.deepEqual(delays, []);
});

test("fixtures match the committed protocol schema", () => {
  const schema = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "..", "docs", "protocol.schema.json"), "utf8"));
  const ajv = new Ajv({ strict: false });
  const validators = { core: ajv.compile(schema.core_to_body), body: ajv.compile(schema.body_to_core) };
  const dir = path.join(__dirname, "fixtures");
  const names = fs.readdirSync(dir).filter((n) => n.endsWith(".json"));
  assert.equal(names.filter((n) => n.startsWith("core_")).length, 25);
  assert.equal(names.filter((n) => n.startsWith("body_")).length, 26);
  for (const n of names) {
    const validate = n.startsWith("core_") ? validators.core : validators.body;
    const obj = JSON.parse(fs.readFileSync(path.join(dir, n), "utf8"));
    assert.ok(validate(obj), `${n}: ${JSON.stringify(validate.errors)}`);
  }
});

/** Every string in a parsed fixture, keys and values, at any depth. */
function* allStrings(x) {
  if (typeof x === "string") yield x;
  else if (Array.isArray(x)) for (const v of x) yield* allStrings(v);
  else if (x && typeof x === "object") for (const [k, v] of Object.entries(x)) { yield k; yield* allStrings(v); }
}

test("no fixture carries a key", () => {
  const dir = path.join(__dirname, "fixtures");
  const keyShaped = (t) => /AIza[\w-]{8,}|sk-[A-Za-z0-9_-]{8,}/.test(t)
    || t.split(/\s+/).some((word) => word.length >= 24 && /^[A-Za-z0-9_-]+$/.test(word) && /\d/.test(word) && /[A-Za-z]/.test(word));
  for (const n of fs.readdirSync(dir).filter((f) => f.endsWith(".json"))) {
    const text = fs.readFileSync(path.join(dir, n), "utf8");
    const obj = JSON.parse(text);
    assert.ok(!(obj.type === "setup_request" && obj.action === "online_key_set"), n);
    assert.ok(!text.includes("online_key_set"), n);
    for (const t of allStrings(obj)) assert.ok(!keyShaped(t), `${n} holds a key-shaped string`);
  }
  assert.ok(keyShaped("x AIzaSyDUMMYDUMMYDUMMY1234 y") && keyShaped("w sk-abcdefghijklmnop"), "the scan catches what it should");
  assert.ok(keyShaped("ab12cd34ef56gh78ij90kl12mn34"), "and a long mixed token");
  assert.ok(!keyShaped("Key ••••1a2b saved in Riya"), "but not the mask");
});

test("confirm_done validates for all seven outcomes, and an unknown one is refused", () => {
  const schema = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "..", "docs", "protocol.schema.json"), "utf8"));
  const validate = new Ajv({ strict: false }).compile(schema.core_to_body);
  for (const outcome of ["ran", "started", "failed", "declined", "expired", "replaced", "cancelled"]) {
    assert.ok(validate({ type: "confirm_done", id: "c1", outcome }), outcome);
  }
  assert.equal(validate({ type: "confirm_done", id: "c1", outcome: "exploded" }), false);
});

test("a now-only message is never queued across a drop: a stale setup_request can't reach the next connection", () => {
  const { NOW_ONLY: CLIENT_NOW_ONLY } = require("../src/core-client");
  const { NOW_ONLY } = require("../src/main-logic");
  assert.deepEqual(CLIENT_NOW_ONLY, NOW_ONLY, "the core client drops what main drops while the core is away");
  const { client, fire } = make();
  const logs = [];
  client.log = (...a) => logs.push(a);
  client.connect();
  client.send({ type: "setup_request", action: "hooks_connect", args: {} }); // still connecting: dropped
  FakeWS.last.open();
  client.send({ type: "setup_request", action: "mic_check", args: {} }); // open: sent at once
  assert.deepEqual(FakeWS.last.sent.map((s) => JSON.parse(s).action), ["mic_check"]);
  FakeWS.last.drop();
  client.send({ type: "setup_action", nonce: "n1", yes: true });
  client.send({ type: "onboarding_done" });
  client.send({ type: "get_memories" }); // not now-only: still queued for the next connection
  fire();
  FakeWS.last.open();
  assert.deepEqual(FakeWS.last.sent.map((s) => JSON.parse(s).type), ["get_memories"]);
  assert.equal(logs.length, 3);
});
