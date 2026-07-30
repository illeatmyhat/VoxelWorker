// Export the design sheet's RESOLVED geometry as Rust `Mark` data, so a hand-authored glyph can
// be diffed against the drawing it is supposed to be a transposition of.
//
// The sheet is the authority for geometry; the Rust file is the authority for prose. Neither is
// generated from the other — this only makes disagreement a test failure instead of a surprise.
import fs from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..', '..');

const ROOT = path.join(REPO, 'docs', 'design', 'sketch-marks');
const OUT = path.join(REPO, 'crates', 'ui', 'src', 'icons', 'design_reference.rs');

// Sheet name -> the kebab id the Rust `Icon` answers to. Only marks listed here are exported;
// anything absent is either already shipped under an older drawing or not yet authored.
const IDS = {
  'Line|2 points': 'line',
};

const SCALE = 0.5;                       // the sheet draws on 36 units, the glyph grid is 18
const INK = { '#f2f6fa': 'LineArt', '#9cb4d8': 'Accent', '#dda06a': 'Construction' };

function load(file) {
  const src = fs.readFileSync(path.join(ROOT, file), 'utf8');
  const body = src.slice(src.lastIndexOf('<script>') + 8, src.lastIndexOf('</script>'));
  const sandbox = {
    Math, console, Array, Number, String, JSON,
    document: { getElementById: () => ({ appendChild() {} }), createElement: () => ({ set innerHTML(_) {} }) },
  };
  vm.createContext(sandbox);
  vm.runInContext(body + '\n;globalThis.__X = { CREATE, MODIFY, SHIPPED, chosen };', sandbox);
  return sandbox.__X;
}

// ---- SVG elliptical-arc endpoint form -> centre form (SVG 1.1 F.6.5) -------
// Implemented from the spec rather than from a sign heuristic: the flags interact, and getting
// the interaction subtly wrong produces an arc that is still tangent and still the right length.
function arcCentre(x1, y1, rx, ry, fA, fS, x2, y2) {
  const dx2 = (x1 - x2) / 2, dy2 = (y1 - y2) / 2;
  const num = rx * rx * ry * ry - rx * rx * dy2 * dy2 - ry * ry * dx2 * dx2;
  const den = rx * rx * dy2 * dy2 + ry * ry * dx2 * dx2;
  const k = Math.sqrt(Math.max(0, num / den)) * (fA === fS ? -1 : 1);
  const cxp = k * (rx * dy2) / ry, cyp = k * -(ry * dx2) / rx;
  const cx = cxp + (x1 + x2) / 2, cy = cyp + (y1 + y2) / 2;

  const ang = (ux, uy, vx, vy) => {
    const d = Math.sign(ux * vy - uy * vx) || 1;
    const c = (ux * vx + uy * vy) / (Math.hypot(ux, uy) * Math.hypot(vx, vy));
    return d * Math.acos(Math.min(1, Math.max(-1, c)));
  };
  const ax = (dx2 - cxp) / rx, ay = (dy2 - cyp) / ry;
  const bx = (-dx2 - cxp) / rx, by = (-dy2 - cyp) / ry;
  const from = ang(1, 0, ax, ay);
  let sweep = ang(ax, ay, bx, by);
  if (!fS && sweep > 0) sweep -= 2 * Math.PI;
  if (fS && sweep < 0) sweep += 2 * Math.PI;
  return { cx, cy, from, to: from + sweep };
}

// ---- walk one resolved fragment into ordered primitives --------------------
function parse(svg, name) {
  const out = [];
  const attrs = (s) => Object.fromEntries([...s.matchAll(/([a-z-]+)="([^"]*)"/g)].map(m => [m[1], m[2]]));

  for (const el of svg.matchAll(/<(path|rect|circle|ellipse)\s([^>]*?)\/>/g)) {
    const tag = el[1], a = attrs(el[2]);
    const dashed = 'stroke-dasharray' in a;
    const role = INK[a.stroke] || INK[a.fill];
    if (!role) throw new Error(`${name}: element inked ${a.stroke || a.fill}, which is not one of the three`);
    const ink = { role, dashed };

    if (tag === 'rect') {
      const [x, y, w, h] = ['x', 'y', 'width', 'height'].map(k => Number(a[k]));
      // A filled rect with no stroke is a node square; a stroked one is an outline box.
      if (a.stroke === 'none') out.push({ k: 'Node', c: [x + w / 2, y + h / 2], size: w, ink });
      else out.push({ k: 'Rect', a: [x, y], b: [x + w, y + h], ink });
      continue;
    }
    if (tag === 'circle') {
      out.push({ k: 'Circle', c: [Number(a.cx), Number(a.cy)], r: Number(a.r), ink });
      continue;
    }
    if (tag === 'ellipse') {
      out.push({ k: 'Ellipse', c: [Number(a.cx), Number(a.cy)], rx: Number(a.rx), ry: Number(a.ry), ink });
      continue;
    }

    // path: walk the commands in order, flushing the polyline buffer whenever a curve interrupts
    const toks = a.d.match(/[MLHVACQZ]|-?\d+(?:\.\d+)?/g) || [];
    let i = 0, cmd = null, x = 0, y = 0, start = null, run = [], closed = false, curved = false;
    const flush = () => { if (run.length >= 2) out.push({ k: 'Line', pts: run, ink }); run = []; };
    while (i < toks.length) {
      if (/^[A-Z]$/.test(toks[i])) { cmd = toks[i++]; if (cmd === 'Z') { closed = true; } continue; }
      const n = () => Number(toks[i++]);
      if (cmd === 'M') { x = n(); y = n(); start = [x, y]; run = [[x, y]]; }
      else if (cmd === 'L') { x = n(); y = n(); run.push([x, y]); }
      else if (cmd === 'H') { x = n(); run.push([x, y]); }
      else if (cmd === 'V') { y = n(); run.push([x, y]); }
      else if (cmd === 'A') {
        curved = true;
        const rx = n(), ry = n(); n();
        const fA = n(), fS = n(), nx = n(), ny = n();
        flush();
        const c = arcCentre(x, y, rx, ry, fA, fS, nx, ny);
        out.push({ k: 'Arc', c: [c.cx, c.cy], rx, ry, from: c.from, to: c.to, ink });
        x = nx; y = ny; run = [[x, y]];
      } else if (cmd === 'C') {
        curved = true;
        const p1 = [n(), n()], p2 = [n(), n()], p3 = [n(), n()];
        flush();
        out.push({ k: 'Cubic', p: [[x, y], p1, p2, p3], ink });
        x = p3[0]; y = p3[1]; run = [[x, y]];
      } else if (cmd === 'Q') {
        // Mark has no quadratic; a Q would have to be raised to a cubic by hand, so refuse it
        // rather than export a shape the glyph cannot actually be.
        throw new Error(`${name}: quadratic segment — Mark has no Q, so this one cannot be exported`);
      } else { i++; }
    }
    if (closed && !curved && run.length >= 2) { out.push({ k: 'Closed', pts: run, ink }); run = []; }
    else if (closed && start) { run.push(start); }
    flush();
  }
  return out;
}

// ---- emit -----------------------------------------------------------------
const f = (v) => {
  const s = (v * SCALE).toFixed(4).replace(/0+$/, '').replace(/\.$/, '.0');
  return s.includes('.') ? s : s + '.0';
};
const rad = (v) => v.toFixed(6);
const pts = (ps) => '&[' + ps.map(p => `(${f(p[0])}, ${f(p[1])})`).join(', ') + ']';
const ink = (i) => {
  if (i.role === 'LineArt') return i.dashed ? 'Ink::DASHED' : 'Ink::SOLID';
  if (i.role === 'Accent') return 'Ink::ACCENT';
  return 'Ink::CONSTRUCTION';
};

function emit(p) {
  switch (p.k) {
    case 'Line': return `Mark::Line { points: ${pts(p.pts)}, ink: ${ink(p.ink)} }`;
    case 'Closed': return `Mark::Closed { points: ${pts(p.pts)}, ink: ${ink(p.ink)} }`;
    case 'Rect': return `Mark::Rect { a: (${f(p.a[0])}, ${f(p.a[1])}), b: (${f(p.b[0])}, ${f(p.b[1])}), ink: ${ink(p.ink)} }`;
    case 'Node': return `Mark::Node { center: (${f(p.c[0])}, ${f(p.c[1])}), size: ${f(p.size)}, ink: ${ink(p.ink)} }`;
    case 'Circle': return `Mark::Circle { center: (${f(p.c[0])}, ${f(p.c[1])}), radius: ${f(p.r)}, ink: ${ink(p.ink)} }`;
    case 'Ellipse': return `Mark::Ellipse { center: (${f(p.c[0])}, ${f(p.c[1])}), rx: ${f(p.rx)}, ry: ${f(p.ry)}, ink: ${ink(p.ink)} }`;
    case 'Arc': return `Mark::Arc { center: (${f(p.c[0])}, ${f(p.c[1])}), rx: ${f(p.rx)}, ry: ${f(p.ry)}, ` +
      `from: ${rad(p.from)}, to: ${rad(p.to)}, ink: ${ink(p.ink)} }`;
    case 'Cubic': return `Mark::Cubic { p0: (${f(p.p[0][0])}, ${f(p.p[0][1])}), p1: (${f(p.p[1][0])}, ${f(p.p[1][1])}), ` +
      `p2: (${f(p.p[2][0])}, ${f(p.p[2][1])}), p3: (${f(p.p[3][0])}, ${f(p.p[3][1])}), ink: ${ink(p.ink)} }`;
  }
  throw new Error('unknown primitive ' + p.k);
}

const { CREATE, MODIFY, SHIPPED, chosen } = load('tool-marks.html');
const MARKS = CREATE.concat(MODIFY, SHIPPED);

const rows = [];
for (const m of MARKS) {
  const key = `${m.name}|${m.hint}`;
  const id = IDS[key];
  if (!id) continue;
  const parsed = parse(chosen(m).draw(), key);
  rows.push(`    // ${m.name} — ${m.hint}\n    ("${id}", &[\n` +
    parsed.map(p => `        ${emit(p)},`).join('\n') + '\n    ]),');
}

const missing = Object.keys(IDS).filter(k => !MARKS.some(m => `${m.name}|${m.hint}` === k));
if (missing.length) { console.error('IDS names no such mark: ' + missing.join(', ')); process.exit(1); }

fs.writeFileSync(OUT, `//! The design sheet's resolved geometry, as data. GENERATED — do not hand-edit.
//!
//! Regenerated by \`reference.mjs\` from \`sketch/tool-marks.html\`. Every glyph in the sketch set
//! is a hand transposition of a drawing on that sheet: the sheet carries the geometry and the
//! argument for it, the glyph file carries the prose, and neither is compiled from the other.
//! This table is what makes a transposition slip a test failure rather than a subtly wrong icon.
//!
//! Coordinates are the sheet's 36-unit canvas halved onto the glyph grid. Arcs arrive in
//! endpoint form and are converted here by the SVG 1.1 F.6.5 centre formula, because the large
//! and sweep flags interact and a sign heuristic gets a plausible wrong answer.

use super::{Ink, Mark};

/// Kebab id -> the marks that id's glyph must draw, in order.
///
/// Left unformatted on purpose: the layout is the generator's, so a rustfmt pass would make
/// regeneration and formatting two steps that have to be run in the right order.
#[rustfmt::skip]
pub(super) const REFERENCE: &[(&str, &[Mark])] = &[
${rows.join('\n')}
];
`);
console.log(`wrote ${rows.length} reference glyph(s)`);
