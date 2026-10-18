const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { CORE_TYPES } = require("../src/core-client");

const schema = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "..", "docs", "protocol.schema.json"), "utf8"));
const coreTypes = Object.keys(schema.core_to_body.discriminator.mapping).sort();

/** Runs the real bridge.js in a sandbox: returns the message types it relays and what reached the main process. */
function runBridge() {
  const handlers = new Map();
  const relayed = [];
  class FakeClient {
    on(type, fn) { handlers.set(type, fn); }
    send() {}
    connect() {}
  }
  const sandbox = {
    window: { riyaBridge: { message: (msg) => relayed.push(msg), status() {}, onSend() {} } },
    RiyaCore: { CoreClient: FakeClient, CORE_TYPES },
    WebSocket: function WebSocket() {},
  };
  vm.runInNewContext(fs.readFileSync(path.join(__dirname, "..", "bridge.js"), "utf8"), sandbox);
  return { handlers, relayed };
}

test("the schema lists core-to-body types (so the next check can't pass on an empty list)", () => {
  assert.ok(coreTypes.includes("eyes") && coreTypes.includes("look") && coreTypes.length >= 11);
});

test("CORE_TYPES is exactly every core_to_body type in the committed schema", () => {
  assert.deepEqual([...CORE_TYPES].sort(), coreTypes);
});

test("the bridge relays every core_to_body type in the schema to the main process", () => {
  const { handlers, relayed } = runBridge();
  for (const type of coreTypes) {
    assert.ok(handlers.has(type), `bridge.js doesn't relay ${type}`);
    handlers.get(type)({ type });
  }
  assert.deepEqual(relayed.map((m) => m.type), coreTypes);
});
