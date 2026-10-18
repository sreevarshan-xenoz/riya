const test = require("node:test");
const assert = require("node:assert/strict");
const zlib = require("node:zlib");
const { makeTrayPng } = require("../src/tray-icon");

function decode(png) {
  assert.deepEqual([...png.subarray(0, 8)], [137, 80, 78, 71, 13, 10, 26, 10]);
  const width = png.readUInt32BE(16);
  const height = png.readUInt32BE(20);
  assert.equal(png[24], 8, "bit depth");
  assert.equal(png[25], 6, "RGBA colour type");
  let off = 8;
  const idat = [];
  let sawEnd = false;
  while (off < png.length) {
    const len = png.readUInt32BE(off);
    const type = png.toString("latin1", off + 4, off + 8);
    if (type === "IDAT") idat.push(png.subarray(off + 8, off + 8 + len));
    if (type === "IEND") sawEnd = true;
    off += 12 + len;
  }
  assert.equal(off, png.length, "no trailing bytes");
  assert.ok(sawEnd);
  const raw = zlib.inflateSync(Buffer.concat(idat));
  assert.equal(raw.length, height * (1 + width * 4));
  const alphaAt = (x, y) => raw[y * (1 + width * 4) + 1 + x * 4 + 3];
  return { width, height, alphaAt };
}

test("tray png is a valid 16x16 RGBA png with an opaque disc and transparent corners", () => {
  const { width, height, alphaAt } = decode(makeTrayPng(16));
  assert.deepEqual([width, height], [16, 16]);
  assert.equal(alphaAt(8, 8), 255);
  assert.equal(alphaAt(0, 0), 0);
  assert.equal(alphaAt(15, 15), 0);
});

test("tray png scales to 32", () => {
  const { width, alphaAt } = decode(makeTrayPng(32));
  assert.equal(width, 32);
  assert.equal(alphaAt(16, 16), 255);
  assert.equal(alphaAt(0, 31), 0);
});

test("tray png chunk CRCs are correct", () => {
  const png = makeTrayPng(16);
  const crcTable = Array.from({ length: 256 }, (_, n) => {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    return c >>> 0;
  });
  const crc32 = (buf) => {
    let c = 0xffffffff;
    for (const b of buf) c = crcTable[(c ^ b) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  };
  let off = 8;
  while (off < png.length) {
    const len = png.readUInt32BE(off);
    assert.equal(png.readUInt32BE(off + 8 + len), crc32(png.subarray(off + 4, off + 8 + len)));
    off += 12 + len;
  }
});
