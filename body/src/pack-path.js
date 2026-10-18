"use strict";
/** Sprite pack location and containment: manifest frame paths may never leave their pack directory. */
const fs = require("node:fs");
const path = require("node:path");
const { pathToFileURL } = require("node:url");
const { validateManifest } = require("./sprite-logic");

/** Absolute path of `rel` inside `packDir`, or null if it is not a plain relative path that stays inside. */
function resolveInPack(packDir, rel) {
  if (typeof rel !== "string" || !rel || rel.includes("\0")) return null;
  if (path.win32.isAbsolute(rel) || path.posix.isAbsolute(rel) || /^[a-zA-Z]:/.test(rel)) return null;
  const root = path.resolve(packDir);
  const full = path.resolve(root, rel);
  const inside = path.relative(root, full);
  if (!inside || inside.startsWith("..") || path.isAbsolute(inside)) return null;
  return full;
}

/** settings.sprite_pack, then <repo>/sprites/riya-v1, then body/placeholder-pack, in order of preference. */
function packCandidates({ spritePack, repoRoot, bodyDir }) {
  const list = [];
  if (typeof spritePack === "string" && spritePack.trim()) list.push(path.resolve(repoRoot, spritePack.trim()));
  list.push(path.join(repoRoot, "sprites", "riya-v1"), path.join(bodyDir, "placeholder-pack"));
  return list;
}

function getPackPath({ spritePack, repoRoot, bodyDir, existsSync = fs.existsSync }) {
  return packCandidates({ spritePack, repoRoot, bodyDir }).find((dir) => existsSync(path.join(dir, "manifest.json"))) || null;
}

/** Copy of `manifest` whose frame paths are file URLs; escaping or non-existent frames are dropped (treated as missing). */
function resolveManifestFrames(manifest, packDir, exists = () => true) {
  if (!manifest || typeof manifest !== "object") return null;
  const expressions = {};
  for (const [name, frames] of Object.entries(manifest.expressions || {})) {
    expressions[name] = {};
    for (const [variant, rel] of Object.entries(frames || {})) {
      const full = resolveInPack(packDir, rel);
      if (full && exists(full)) expressions[name][variant] = pathToFileURL(full).href;
    }
  }
  return { ...manifest, expressions };
}

/** The first candidate pack whose manifest parses, passes validateManifest and has a neutral.closed file, as
 *  { dir, manifest } with frame file URLs, or null. Each skipped pack is reported through `log`. */
function loadFirstPack({ spritePack, repoRoot, bodyDir, existsSync = fs.existsSync,
  readText = (p) => fs.readFileSync(p, "utf8"), log = () => {} }) {
  for (const dir of packCandidates({ spritePack, repoRoot, bodyDir })) {
    const file = path.join(dir, "manifest.json");
    if (!existsSync(file)) continue;
    try {
      const raw = JSON.parse(readText(file));
      validateManifest(raw, []);
      const manifest = resolveManifestFrames(raw, dir, existsSync);
      if (manifest.expressions.neutral && manifest.expressions.neutral.closed) return { dir, manifest };
      log("sprite pack has no neutral.closed file, skipping", dir);
    } catch (e) {
      log("sprite pack manifest invalid, skipping", dir, e.message);
    }
  }
  return null;
}

module.exports = { resolveInPack, packCandidates, getPackPath, resolveManifestFrames, loadFirstPack };
