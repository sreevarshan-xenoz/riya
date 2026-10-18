"use strict";
/** Pure speech-bubble model: one utterance at a time, shown while she talks and for a while after.
 *  Loads in Node (tests) and as a plain script in the renderer. */

const HIDE_AFTER_MS = 6000;
const BID_HIDE_MS = 60000;

class BubbleModel {
  constructor() {
    this.id = null;
    this._text = "";
    this._kind = "reply";
    this.endedAt = null;
    this.dismissed = false;
  }

  /** A `say` message: the first sentence of a new utterance replaces the old one, later ones append. */
  onSay({ utterance_id: id, text, kind }) {
    const piece = typeof text === "string" ? text.trim() : "";
    if (!piece) return;
    if (id !== this.id) {
      this.id = id;
      this._text = "";
      this.endedAt = null;
      this.dismissed = false;
    }
    this._kind = kind || "reply";
    this._text = this._text ? `${this._text} ${piece}` : piece;
  }

  onSayEnd(id, nowMs) {
    if (id === this.id && this.endedAt === null) this.endedAt = nowMs;
  }

  /** The core went away mid-utterance: no say_end is coming. */
  endCurrent(nowMs) {
    if (this.id !== null && this.endedAt === null) this.endedAt = nowMs;
  }

  /** The user typed or spoke: an unprompted bid has been answered. */
  onUserInput() {
    if (this._kind === "bid") this.dismissed = true;
  }

  visible(nowMs) {
    if (!this._text || this.dismissed) return false;
    if (this.endedAt === null) return true;
    return nowMs < this.endedAt + (this._kind === "bid" ? BID_HIDE_MS : HIDE_AFTER_MS);
  }

  /** When a finished, still-visible bubble hides (for a timer), else null. */
  hidesAt(nowMs) {
    if (this.endedAt === null || !this.visible(nowMs)) return null;
    return this.endedAt + (this._kind === "bid" ? BID_HIDE_MS : HIDE_AFTER_MS);
  }

  text() {
    return this._text;
  }

  kind() {
    return this._kind;
  }
}

if (typeof module !== "undefined" && module.exports) module.exports = { BubbleModel, HIDE_AFTER_MS, BID_HIDE_MS };
else globalThis.RiyaBubble = { BubbleModel, HIDE_AFTER_MS, BID_HIDE_MS };
