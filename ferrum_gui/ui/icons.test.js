// The bundle icons are real files of the right format: tauri.conf.json lists them, the PNGs have
// their stated size, icon.ico is a container of PNG images and icon.icns a chain of typed chunks
// whose lengths add up. Windows and macOS bundles fail without the last two. Regenerate every file
// from the logo with `cd ferrum_gui && cargo tauri icon ../docs/assets/logo.png -o /tmp/icons` and copy
// the ones listed in tauri.conf.json. Run with: node ui/icons.test.js
"use strict";
const assert = require("node:assert");
const fs = require("node:fs");
const path = require("node:path");

const root = path.join(__dirname, "..");
const conf = JSON.parse(fs.readFileSync(path.join(root, "tauri.conf.json"), "utf8"));
const listed = conf.bundle.icon;
const PNG_SIG = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

// Width and height of a PNG held in `buf` at `at`; throws on a bad signature or first chunk.
function pngSize(buf, at = 0) {
  assert.ok(buf.subarray(at, at + 8).equals(PNG_SIG), "PNG signature");
  assert.strictEqual(buf.readUInt32BE(at + 8), 13, "IHDR length");
  assert.strictEqual(buf.toString("latin1", at + 12, at + 16), "IHDR");
  return [buf.readUInt32BE(at + 16), buf.readUInt32BE(at + 20)];
}
const read = (rel) => fs.readFileSync(path.join(root, rel));

// ── The list names the formats each platform needs, and every file exists ──
for (const need of ["icons/icon.ico", "icons/icon.icns", "icons/32x32.png", "icons/128x128.png", "icons/128x128@2x.png"]) {
  assert.ok(listed.includes(need), `bundle.icon must list ${need}`);
}
for (const f of listed) assert.ok(fs.existsSync(path.join(root, f)), `${f} is listed but missing`);
assert.ok(fs.existsSync(path.join(root, "icons/icon.svg")), "the vector source icons/icon.svg is kept in the repo");

// ── PNGs: the size in the name (or in the data, for icon.png) ──
for (const [f, size] of [["icons/32x32.png", 32], ["icons/128x128.png", 128], ["icons/128x128@2x.png", 256], ["icons/icon.png", 512]]) {
  assert.deepStrictEqual(pngSize(read(f)), [size, size], f);
}

// ── ICO: ICONDIR (0, 1, count) then 16-byte entries pointing at PNG images inside the file ──
{
  const b = read("icons/icon.ico");
  assert.deepStrictEqual([b.readUInt16LE(0), b.readUInt16LE(2)], [0, 1], "ICO header");
  const n = b.readUInt16LE(4);
  assert.ok(n >= 4, "an ICO with several sizes, got " + n);
  const sizes = [];
  for (let i = 0; i < n; i++) {
    const e = 6 + 16 * i;
    const [len, off] = [b.readUInt32LE(e + 8), b.readUInt32LE(e + 12)];
    assert.ok(off + len <= b.length, `ICO image ${i} lies inside the file`);
    const [w, h] = pngSize(b, off);
    assert.strictEqual(w, h);
    assert.strictEqual(b[e] || 256, w, `ICO entry ${i} states its width`);
    sizes.push(w);
  }
  assert.ok(sizes.includes(32) && sizes.includes(256), "ICO has 32 and 256 pixel images: " + sizes);
}

// ── ICNS: 'icns' + total length, then [type(4) length(4) data] chunks that fill the file ──
{
  const b = read("icons/icon.icns");
  assert.strictEqual(b.toString("latin1", 0, 4), "icns");
  assert.strictEqual(b.readUInt32BE(4), b.length, "ICNS length field equals the file size");
  const types = [];
  let at = 8;
  while (at < b.length) {
    const len = b.readUInt32BE(at + 4);
    assert.ok(len >= 8 && at + len <= b.length, "ICNS chunk lies inside the file");
    types.push(b.toString("latin1", at, at + 4));
    at += len;
  }
  assert.strictEqual(at, b.length, "ICNS chunks fill the file exactly");
  assert.ok(types.length >= 4, "ICNS has several images: " + types);
}

console.log("icons.test.js: all assertions passed");
