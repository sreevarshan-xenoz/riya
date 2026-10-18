"use strict";
const test = require("node:test");
const assert = require("node:assert/strict");
const { BubbleModel, HIDE_AFTER_MS, BID_HIDE_MS } = require("../src/bubble");

const say = (utterance_id, text, kind = "reply") => ({ utterance_id, text, kind });

test("accumulates sentences of one utterance", () => {
  const b = new BubbleModel();
  b.onSay(say("u1", "Hi."));
  b.onSay(say("u1", "I missed you."));
  assert.equal(b.text(), "Hi. I missed you.");
  assert.equal(b.kind(), "reply");
  assert.equal(b.visible(0), true);
});

test("hides 6s after say_end", () => {
  const b = new BubbleModel();
  b.onSay(say("u1", "Hi."));
  b.onSayEnd("u1", 1000);
  assert.equal(HIDE_AFTER_MS, 6000);
  assert.equal(b.visible(1000 + 5999), true);
  assert.equal(b.visible(1000 + 6000), false);
});

test("stays visible while she is still talking", () => {
  const b = new BubbleModel();
  b.onSay(say("u1", "Hi."));
  assert.equal(b.visible(10 * 60 * 1000), true);
});

test("new utterance replaces old", () => {
  const b = new BubbleModel();
  b.onSay(say("u1", "Old."));
  b.onSayEnd("u1", 100);
  b.onSay(say("u2", "New."));
  assert.equal(b.text(), "New.");
  assert.equal(b.visible(100 + 6000), true, "the old utterance's timer must not hide the new one");
  b.onSayEnd("u1", 200);
  assert.equal(b.visible(200 + 6000), true, "a stray say_end of the old utterance is ignored");
});

test("bid stays until user input or 60s", () => {
  const b = new BubbleModel();
  b.onSay(say("b1", "You there?", "bid"));
  b.onSayEnd("b1", 0);
  assert.equal(b.kind(), "bid");
  assert.equal(BID_HIDE_MS, 60000);
  assert.equal(b.visible(59999), true);
  assert.equal(b.visible(60000), false);

  const c = new BubbleModel();
  c.onSay(say("b2", "Hey.", "bid"));
  c.onSayEnd("b2", 0);
  c.onUserInput();
  assert.equal(c.visible(1000), false);
});

test("user input leaves a reply bubble alone and does not pre-dismiss the next bid", () => {
  const b = new BubbleModel();
  b.onSay(say("u1", "Answer."));
  b.onUserInput();
  assert.equal(b.visible(0), true);
  b.onSay(say("b1", "Hey.", "bid"));
  assert.equal(b.visible(0), true);
});

test("empty text and a say_end with nothing said are harmless", () => {
  const b = new BubbleModel();
  b.onSay(say("u1", "   "));
  b.onSayEnd("u1", 0);
  assert.equal(b.visible(0), false);
  assert.equal(b.text(), "");
  assert.equal(b.kind(), "reply");
});

test("endCurrent closes an unfinished utterance (core dropped)", () => {
  const b = new BubbleModel();
  b.onSay(say("u1", "Hi."));
  b.endCurrent(500);
  assert.equal(b.visible(500 + 5999), true);
  assert.equal(b.visible(500 + 6000), false);
});

test("hidesAt is when a finished bubble disappears, so it can be hidden by a timer", () => {
  const b = new BubbleModel();
  assert.equal(b.hidesAt(0), null);
  b.onSay(say("u1", "Hi."));
  assert.equal(b.hidesAt(500), null, "still talking: no deadline");
  b.onSayEnd("u1", 1000);
  assert.equal(b.hidesAt(1000), 1000 + HIDE_AFTER_MS);
  assert.equal(b.hidesAt(1000 + HIDE_AFTER_MS), null, "already hidden");
  b.onSay(say("u2", "psst", "bid"));
  b.onSayEnd("u2", 2000);
  assert.equal(b.hidesAt(2000), 2000 + BID_HIDE_MS);
  b.onUserInput();
  assert.equal(b.hidesAt(2001), null, "an answered bid is gone");
});
