const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const DIR = path.join(__dirname, "..");

// A page loads its src/ scripts as classic scripts, which share ONE global lexical scope: a top-level const (or let, or
// class) repeated across two of them is a SyntaxError that stops the second script from running at all. Node's require()
// gives each file its own scope, so the unit tests alone can't see it. This loads each page's scripts the way it does.
for (const page of ["index.html", "settings.html", "welcome.html"]) {
  test(`${page}: its src scripts load together in one global scope`, () => {
    const html = fs.readFileSync(path.join(DIR, page), "utf8");
    const scripts = [...html.matchAll(/<script src="(src\/[^"]+)"><\/script>/g)].map((m) => m[1]);
    assert.ok(scripts.length >= 2, "found the page's scripts");
    const g = {};
    g.globalThis = g;
    const context = vm.createContext(g); // no `require` and no `module`, as in a renderer
    for (const file of scripts) vm.runInContext(fs.readFileSync(path.join(DIR, file), "utf8"), context, { filename: file });
    assert.ok(Object.keys(g).filter((n) => n.startsWith("Riya")).length >= scripts.length - 1, "each script exported itself");
  });
}
