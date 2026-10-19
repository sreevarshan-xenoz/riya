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
  assert.equal(names.filter((n) => n.startsWith("core_")).length, 18);
  assert.equal(names.filter((n) => n.startsWith("body_")).length, 21);
  for (const n of names) {
    const validate = n.startsWith("core_") ? validators.core : validators.body;
    const obj = JSON.parse(fs.readFileSync(path.join(dir, n), "utf8"));
    assert.ok(validate(obj), `${n}: ${JSON.stringify(validate.errors)}`);
  }
});

test("confirm_done validates for all seven outcomes, and an unknown one is refused", () => {
  const schema = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "..", "docs", "protocol.schema.json"), "utf8"));
  const validate = new Ajv({ strict: false }).compile(schema.core_to_body);
  for (const outcome of ["ran", "started", "failed", "declined", "expired", "replaced", "cancelled"]) {
    assert.ok(validate({ type: "confirm_done", id: "c1", outcome }), outcome);
  }
  assert.equal(validate({ type: "confirm_done", id: "c1", outcome: "exploded" }), false);
});
