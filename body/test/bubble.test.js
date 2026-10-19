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

// ---- the confirmation (epoch-ms clock: expires_at arrives in seconds) ----

const conf = (id, summary = 'Run "backup" (robocopy a b)? (yes / no)', expires_at = 1000) => ({ id, summary, expires_at });

test("a confirm request shows until expiry, which is stored as epoch ms", () => {
  const b = new BubbleModel();
  assert.equal(b.confirm(0), null);
  b.onConfirmRequest(conf("c1", "Run it?", 1000));
  assert.deepEqual(b.confirm(999_999), { id: "c1", summary: "Run it?" });
  assert.equal(b.confirm(1_000_000), null, "hides at expiresAtMs");
  assert.equal(b.confirmExpiresAt(), 1_000_000);
});

test("confirm_done hides it, and only the pending one's done does", () => {
  const b = new BubbleModel();
  b.onConfirmRequest(conf("c1"));
  b.onConfirmDone("other");
  assert.equal(b.confirm(0).id, "c1", "a done for a replaced confirmation doesn't clear the new one");
  b.onConfirmDone("c1");
  assert.equal(b.confirm(0), null);
  assert.equal(b.confirmExpiresAt(), null);
});

test("a newer confirm replaces the pending one", () => {
  const b = new BubbleModel();
  b.onConfirmRequest(conf("c1", "one"));
  b.onConfirmRequest(conf("c2", "two", 2000));
  assert.deepEqual(b.confirm(0), { id: "c2", summary: "two" });
  b.onConfirmDone("c1");
  assert.equal(b.confirm(0).id, "c2");
});

test("answering takes the confirmation once: a second click gets nothing", () => {
  const b = new BubbleModel();
  b.onConfirmRequest(conf("c1"));
  assert.deepEqual(b.answerConfirm(0), { id: "c1", summary: conf("c1").summary });
  assert.equal(b.answerConfirm(0), null);
  assert.equal(b.confirm(0), null);
  b.onConfirmRequest(conf("c2"));
  assert.equal(b.answerConfirm(1_000_000), null, "an expired one can't be answered");
});

test("malformed confirm requests are ignored, and clearConfirm drops the pending one", () => {
  const b = new BubbleModel();
  b.onConfirmRequest({ id: "", summary: "x", expires_at: 5 });
  b.onConfirmRequest({ id: "c", summary: "x", expires_at: "soon" });
  b.onConfirmRequest({ id: "c", summary: 5, expires_at: 5 });
  b.onConfirmRequest(null);
  assert.equal(b.confirm(0), null);
  b.onConfirmRequest(conf("c1"));
  b.clearConfirm();
  assert.equal(b.confirm(0), null);
});

test("every confirm_done outcome takes the buttons away, and none of them can be clicked afterwards", () => {
  for (const outcome of ["ran", "started", "failed", "declined", "expired", "replaced", "cancelled"]) {
    const b = new BubbleModel();
    b.onConfirmRequest(conf("c1"));
    b.onConfirmDone("c1", outcome);
    assert.equal(b.confirm(0), null, outcome);
    assert.equal(b.answerConfirm(0), null, outcome);
    assert.equal(b.confirmExpiresAt(), null, outcome);
  }
});

test("a click answers only the confirmation its buttons were drawn for", () => {
  const b = new BubbleModel();
  b.onConfirmRequest(conf("c1"));
  b.onConfirmRequest(conf("c2", "two", 2000));
  assert.equal(b.answerConfirm(0, "c1"), null, "the old buttons can't answer the new question");
  assert.equal(b.confirm(0).id, "c2", "and it stays pending");
  assert.deepEqual(b.answerConfirm(0, "c2"), { id: "c2", summary: "two" });
  assert.equal(b.answerConfirm(0, "c2"), null);
});

test("a click just after a swap is ignored, and the new question stays answerable after the guard", () => {
  const { CONFIRM_SWAP_GUARD_MS } = require("../src/bubble");
  const b = new BubbleModel();
  b.onConfirmRequest(conf("c1", "one", 1000), 10_000);
  assert.equal(b.answerConfirm(10_000, "c1")?.id, "c1", "a first confirmation is answerable at once");
  b.onConfirmRequest(conf("c3", "three", 1000), 10_000);
  b.onConfirmRequest(conf("c4", "four", 1000), 20_000);
  assert.equal(b.answerConfirm(20_000 + CONFIRM_SWAP_GUARD_MS - 1, "c4"), null);
  assert.equal(b.confirm(20_000 + CONFIRM_SWAP_GUARD_MS).id, "c4");
  assert.equal(b.answerConfirm(20_000 + CONFIRM_SWAP_GUARD_MS, "c4").id, "c4");
});
