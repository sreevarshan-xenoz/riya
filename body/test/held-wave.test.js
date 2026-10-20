"use strict";
const test = require("node:test");
const assert = require("node:assert");
const { HELD_WAVE_MS } = require("../src/main-logic.js");
const { REACTIONS } = require("../src/pet-logic.js");

test("a held wave lives as long as the wave pose", () => {
  assert.strictEqual(HELD_WAVE_MS, REACTIONS.wave.ms);
});
