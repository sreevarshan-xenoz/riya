const test = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
const { resolveInPack, packCandidates, getPackPath, resolveManifestFrames } = require("../src/pack-path");

const PACK = path.resolve("/riya/pack");
const REPO = path.resolve("/repo");
const BODY = path.join(REPO, "body");

test("resolveInPack accepts relative paths inside the pack", () => {
  assert.equal(resolveInPack(PACK, "a.svg"), path.join(PACK, "a.svg"));
  assert.equal(resolveInPack(PACK, "sub/b.svg"), path.join(PACK, "sub", "b.svg"));
  assert.equal(resolveInPack(PACK, "sub/../c.svg"), path.join(PACK, "c.svg"));
});

test("resolveInPack rejects escapes, absolute paths and junk", () => {
  const bad = ["../x.svg", "sub/../../x.svg", "..", "/etc/passwd", "C:\\Windows\\x.png", "C:/x.png", "\\\\host\\share\\x.png",
    "..\\x.svg", "", ".", "a\0b.svg", null, undefined, 5, {}];
  for (const p of bad) assert.equal(resolveInPack(PACK, p), null, String(p));
});

test("resolveInPack does not confuse a sibling directory sharing the prefix", () => {
  assert.equal(resolveInPack(PACK, "../pack-evil/x.svg"), null);
});

test("packCandidates orders configured pack, sprites pack, placeholder", () => {
  const c = packCandidates({ spritePack: "mine", repoRoot: REPO, bodyDir: BODY });
  assert.deepEqual(c, [path.join(REPO, "mine"), path.join(REPO, "sprites", "riya-v1"), path.join(BODY, "placeholder-pack")]);
  const abs = path.resolve("/elsewhere/pack");
  assert.equal(packCandidates({ spritePack: abs, repoRoot: REPO, bodyDir: BODY })[0], abs);
  assert.equal(packCandidates({ spritePack: null, repoRoot: REPO, bodyDir: BODY }).length, 2);
  assert.equal(packCandidates({ spritePack: "  ", repoRoot: REPO, bodyDir: BODY }).length, 2);
});

test("getPackPath picks the first directory containing manifest.json", () => {
  const has = (dirs) => (p) => dirs.map((d) => path.join(d, "manifest.json")).includes(p);
  const mine = path.join(REPO, "mine");
  const sprites = path.join(REPO, "sprites", "riya-v1");
  const placeholder = path.join(BODY, "placeholder-pack");
  const args = { spritePack: "mine", repoRoot: REPO, bodyDir: BODY };
  assert.equal(getPackPath({ ...args, existsSync: has([mine, sprites]) }), mine);
  assert.equal(getPackPath({ ...args, existsSync: has([sprites]) }), sprites);
  assert.equal(getPackPath({ ...args, existsSync: has([placeholder]) }), placeholder);
  assert.equal(getPackPath({ ...args, existsSync: () => false }), null);
});

test("resolveManifestFrames maps frames to file URLs and drops escapes and missing files", () => {
  const manifest = {
    name: "x", version: 1, canvas: { width: 1, height: 1 }, crops: { upper: [0, 0, 1, 1] },
    expressions: {
      neutral: { closed: "n_closed.svg", open: "../outside.svg", blink: "sub/n blink#1.svg" },
      happy: { closed: "/abs.svg", open: "gone.svg" },
      sad: { closed: "s.svg" },
    },
  };
  const exists = (p) => !p.endsWith("gone.svg");
  const out = resolveManifestFrames(manifest, PACK, exists);
  assert.deepEqual(out.expressions.neutral, {
    closed: pathToFileURL(path.join(PACK, "n_closed.svg")).href,
    blink: pathToFileURL(path.join(PACK, "sub", "n blink#1.svg")).href,
  });
  assert.deepEqual(out.expressions.happy, {});
  assert.deepEqual(out.expressions.sad, { closed: pathToFileURL(path.join(PACK, "s.svg")).href });
  assert.equal(out.name, "x");
  assert.equal(manifest.expressions.neutral.open, "../outside.svg", "input untouched");
});

test("resolveManifestFrames tolerates a manifest without expressions", () => {
  assert.deepEqual(resolveManifestFrames({ name: "x" }, PACK).expressions, {});
  assert.equal(resolveManifestFrames(null, PACK), null);
});
