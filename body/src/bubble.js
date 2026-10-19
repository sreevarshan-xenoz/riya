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
    this._confirm = null; // { id, summary, expiresAtMs (epoch ms), answered }
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

  /** A `confirm_request`: replaces any pending one. `expires_at` is epoch seconds, kept as epoch ms; malformed ones are
   *  ignored. The summary is shown with textContent only. */
  onConfirmRequest(msg) {
    if (!msg || typeof msg.id !== "string" || !msg.id || typeof msg.summary !== "string") return;
    if (typeof msg.expires_at !== "number" || !Number.isFinite(msg.expires_at)) return;
    this._confirm = { id: msg.id, summary: msg.summary, expiresAtMs: msg.expires_at * 1000, answered: false };
  }

  /** A `confirm_done`: only the pending confirmation's own ends it (a stale one must not clear a newer request). */
  onConfirmDone(id) {
    if (this._confirm && this._confirm.id === id) this._confirm = null;
  }

  clearConfirm() {
    this._confirm = null;
  }

  /** The confirmation to show at epoch-ms `nowMs` ({ id, summary }), or null: none, answered, or expired. */
  confirm(nowMs) {
    const c = this._confirm;
    return c && !c.answered && nowMs < c.expiresAtMs ? { id: c.id, summary: c.summary } : null;
  }

  /** A button was pressed: the confirmation to answer, once (the buttons go away at once, so there is no second click). */
  answerConfirm(nowMs) {
    const shown = this.confirm(nowMs);
    if (shown) this._confirm.answered = true;
    return shown;
  }

  /** When the pending, unanswered confirmation expires (epoch ms), else null (for a timer). */
  confirmExpiresAt() {
    return this._confirm && !this._confirm.answered ? this._confirm.expiresAtMs : null;
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
