"use strict";
/** Pure helpers for "She notices your routines" in the settings page: the routines she learned (her own words, never
 *  anything the user typed), her honesty line (the Health row's texts, riya/core/health.py honesty_detail) and the
 *  forget message. Every string ends up in the DOM through textContent only. Loads in Node (tests) and as a plain
 *  script in the settings page; it keeps its names to itself (the page's scripts share one global scope). */
(() => {
  const isObj = (v) => v !== null && typeof v === "object" && !Array.isArray(v);
  const count = (v) => (Number.isInteger(v) && v >= 0 ? v : 0);
  const FEW = 10; // audit.FEW
  const WARN_N = 20; // audit.WARN_N
  const WARN_RATE = 0.9; // audit.WARN_RATE

  /** The cells in the core's order: {line, sure} (sure: "sure", "fairly sure" or "still learning", the line's end). */
  function cellRows(msg) {
    const cells = msg && Array.isArray(msg.cells) ? msg.cells : [];
    return cells.filter((c) => isObj(c) && typeof c.line === "string" && c.line).map((c) => {
      const sure = ["still learning", "fairly sure", "sure"].find((s) => c.line.endsWith(s)) || "";
      return { line: c.line, sure };
    });
  }

  /** Her honesty over the last 30 days, the user reading (a disputed claim counts false). */
  function honestyText(msg) {
    if (!msg || msg.enabled !== true) return "Off: she isn't noticing your routines.";
    const h = isObj(msg.honesty) ? msg.honesty : {};
    const hr = count(h.hunches_right);
    const hn = count(h.hunches_scored);
    if (h.claims_enabled !== true) return `Hunches only in this version · hunches right ${hr} of ${hn} · 30 days`;
    const t = count(h.claims_true);
    const n = count(h.claims_scored);
    if (n === 0) return "No claims checked yet: she needs a few more days";
    if (n < FEW) return `${n} claims checked so far (${t === n ? "all true" : `${t} true`}): she needs a few more days`;
    const p = Math.round((100 * t) / n);
    if (n >= WARN_N && t / n < WARN_RATE) return `Only ${t} of ${n} claims held up (${p}%): tell Claude`;
    return `Honest ${t} of ${n} claims (${p}%) · hunches right ${hr} of ${hn} · 30 days`;
  }

  const forgetMessage = () => ({ type: "routine_forget" });

  const routineExports = { cellRows, honestyText, forgetMessage };
  if (typeof module !== "undefined" && module.exports) module.exports = routineExports;
  else globalThis.RiyaRoutine = routineExports;
})();
