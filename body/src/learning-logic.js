"use strict";
/** Pure helpers for the "She learns you" settings section: the dial bars, the change history, style notes and the
 *  decision twin's engagement rates, and the three messages it sends. Every string here ends up in the DOM through
 *  textContent only. Loads in Node (tests) and as a plain script in the settings page. */

const DIALS = [ // a test keeps the names and bands equal to riya/core/learn/dials.py BANDS
  { name: "sweetness", label: "Sweetness", lo: 0.6, hi: 1.0 },
  { name: "teasing", label: "Teasing", lo: 0.1, hi: 0.8 },
  { name: "clinginess", label: "Clinginess", lo: 0.4, hi: 1.0 },
  { name: "verbosity", label: "How much she says", lo: 0.15, hi: 0.7 },
  { name: "emoji", label: "Emoji", lo: 0.0, hi: 0.6 },
];
const TWIN_KINDS = [ // a test keeps the kinds equal to riya/core/learn/twin.py KINDS
  { kind: "bid", label: "Break check-ins" },
  { kind: "game_offer", label: "Game offers" },
  { kind: "report", label: "Agent news" },
  { kind: "greeting", label: "Greetings" },
];
const HISTORY_CAP = 20;
const MIN_BUCKET_N = 0.5; // thinner buckets are noise
const BY_DIAL = new Map(DIALS.map((d) => [d.name, d]));
const isObj = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
const pct = (x) => Math.round(x * 100);
const str = (v) => (typeof v === "string" ? v : "");

/** The dials in DIALS order for the bars; a dial with no finite value is skipped, a missing band falls back to the core's. */
function dialRows(msg) {
  const dials = isObj(msg && msg.dials) ? msg.dials : {};
  const rows = [];
  for (const d of DIALS) {
    const info = dials[d.name];
    if (!isObj(info) || typeof info.value !== "number" || !Number.isFinite(info.value)) continue;
    const lo = Number.isFinite(info.lo) ? info.lo : d.lo;
    const hi = Number.isFinite(info.hi) ? info.hi : d.hi;
    rows.push({ name: d.name, label: d.label, value: info.value, lo, hi, valuePct: pct(info.value), loPct: pct(lo), hiPct: pct(hi) });
  }
  return rows;
}

/** The last changes, newest first as the core sends them: "Teasing 0.50 → 0.35". Rows for an unknown dial are skipped. */
function historyRows(msg) {
  const list = Array.isArray(msg && msg.history) ? msg.history : [];
  const rows = [];
  for (const h of list) {
    const dial = isObj(h) ? BY_DIAL.get(h.dial) : null;
    if (!dial || !Number.isFinite(h.old) || !Number.isFinite(h.new)) continue;
    rows.push({ text: `${dial.label} ${h.old.toFixed(2)} → ${h.new.toFixed(2)}`, at: h.at, source: str(h.source), reason: str(h.reason) });
    if (rows.length >= HISTORY_CAP) break;
  }
  return rows;
}

/** The style notes she keeps: integer ids and string texts only. */
function noteRows(msg) {
  const list = Array.isArray(msg && msg.notes) ? msg.notes : [];
  return list.filter((n) => isObj(n) && Number.isInteger(n.id) && typeof n.text === "string")
    .map((n) => ({ id: n.id, text: n.text, polarity: str(n.polarity) }));
}

/** One row per twin kind: the engagement rate (Σp·n / Σn, null with no evidence), the evidence count and the buckets
 *  with real evidence, most first. */
function twinRows(msg) {
  const list = Array.isArray(msg && msg.twin) ? msg.twin : [];
  return TWIN_KINDS.map(({ kind, label }) => {
    const cells = list.filter((c) => isObj(c) && c.kind === kind && typeof c.bucket === "string"
      && Number.isFinite(c.p) && Number.isFinite(c.n) && c.n > 0);
    const total = cells.reduce((sum, c) => sum + c.n, 0);
    const rate = total > 0 ? Math.round((cells.reduce((sum, c) => sum + c.p * c.n, 0) / total) * 100) / 100 : null;
    const buckets = cells.filter((c) => c.n >= MIN_BUCKET_N).sort((a, b) => b.n - a.n).map((c) => ({ bucket: c.bucket, p: c.p, n: c.n }));
    return { kind, label, rate, count: Math.round(total), buckets };
  });
}

/** "72% of 9", or "no data yet". */
function rateText(row) {
  return row && row.rate !== null && Number.isFinite(row.rate) ? `${pct(row.rate)}% of ${row.count}` : "no data yet";
}

const noteDeleteMessage = (id) => ({ type: "note_delete_style", id });
const twinResetMessage = (kind) => ({ type: "twin_reset", kind });
const resetMessage = () => ({ type: "learning_reset" });

const api = { DIALS, TWIN_KINDS, dialRows, historyRows, noteRows, twinRows, rateText, noteDeleteMessage, twinResetMessage, resetMessage };
if (typeof module !== "undefined" && module.exports) module.exports = api;
else globalThis.RiyaLearning = api;
