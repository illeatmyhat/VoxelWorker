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

// Sheet name -> the kebab id the Rust `Icon` answers to. Only marks listed here are exported, and
// the parity test panics on an id no glyph answers to, so an entry here is a claim that the glyph
// is authored. The queue below it is the transposition backlog: uncomment a line as its shelf
// lands. Anything in neither list is shipped under an older drawing.
const IDS = {
  'Line|2 points': 'line',
  'Fillet|rounds a corner': 'fillet',
  'Chamfer|equal distance': 'chamfer-equal',
  'Chamfer|distance and angle': 'chamfer-distance-angle',
  'Chamfer|two distance': 'chamfer-two-distance',
  'Trim|to nearest crossing': 'trim',
  'Extend|to nearest boundary': 'extend',
  'Break|split at a point': 'break-curve',
  'Offset|parallel copy': 'offset-curve',
  'Move / copy|free transform': 'move-copy',
  'Sketch scale|uniform': 'sketch-scale',
  'Blend curve|tangent join': 'blend-curve',

  'Coincident|2 DOF': 'constraint-coincident',
  'Collinear|2 DOF': 'constraint-collinear',
  'Concentric|2 DOF': 'constraint-concentric',
  'Midpoint|2 DOF': 'constraint-midpoint',
  'Fix|all remaining': 'constraint-fix',
  'Parallel|1 DOF': 'constraint-parallel',
  'Perpendicular|1 DOF': 'constraint-perpendicular',
  'Horizontal|1 DOF': 'constraint-horizontal',
  'Vertical|1 DOF': 'constraint-vertical',
  'Tangent|1 DOF': 'constraint-tangent',
  'Equal|1 DOF': 'constraint-equal',
  'Symmetry|2 DOF': 'constraint-symmetry',
  'Curvature|+1 on tangency': 'constraint-curvature',
  'Quantize|1 → integer': 'constraint-quantize',

  'Mirror|generator': 'mirror',
  'Rectangular pattern|generator': 'rectangular-pattern',
  'Circular pattern|generator': 'circular-pattern',

  // Sketch · create.
  'Midpoint line|centre · end': 'midpoint-line',
  'Circle|centre · diameter': 'circle-center-diameter',
  'Circle|2-point': 'circle-2-point',
  'Circle|3-point': 'circle-3-point',
  'Circle|2-tangent': 'circle-2-tangent',
  'Circle|3-tangent': 'circle-3-tangent',
  'Arc|centre · endpoints': 'arc-center-endpoints',
  'Arc|tangent': 'arc-tangent',
  'Ellipse|centre · 2 axes': 'ellipse-sketch',
  'Slot|centre-to-centre': 'slot-center-to-center',
  'Slot|overall': 'slot-overall',
  'Slot|centre-point': 'slot-center-point',
  'Slot|centre-point arc': 'slot-center-point-arc',
  'Slot|3-point arc': 'slot-3-point-arc',
  'Spline|fit point': 'spline-fit-point',
  'Spline|control point': 'spline-control-point',
  'Conic|apex · rho': 'conic',
  'Polygon|inscribed': 'polygon-inscribed',
  'Polygon|circumscribed': 'polygon-circumscribed',
  'Polygon|edge': 'polygon-edge',
  'Rectangle|3-point': 'rectangle-3-point',
  'Rectangle|centre · corner': 'rectangle-center-corner',
  'Sketch dimension|drive a distance': 'sketch-dimension',
  'Text|profile from glyphs': 'sketch-text',
  'Construction|role toggle': 'construction-toggle',

  // Four the sheet re-draws. Each of these shipped under an older drawing, so transposing it
  // REPLACES a glyph rather than adding one — held back until the owner ruled the sheet's
  // version the one to keep (2026-07-30).
  'Select|entity pick': 'select-vertex',
  'Add point|free vertex': 'add-point',
  'Rectangle|2-point': 'rectangle',
  'Three-point arc|ends + through': 'three-point-arc',

  // The one mark still outside the gate. Its geometry IS transposed; only its ink is not, so
  // there is nothing here to compare. The sheet draws snap-voxel entirely in tool blue, where
  // blue means "this is a mode" — but the rail says that with an armed button, so an all-accent
  // glyph reads as permanently lit while its two siblings are line art. See `snap_voxel.rs`.
  // 'Snap to voxel|modal': 'snap-voxel',
};

const SCALE = 0.5;                       // the sheet draws on 36 units, the glyph grid is 18

// The sheets draw in four colours and `Mark` now has four ink ROLES, one for one:
//
//   #f2f6fa  white       the reference entity          -> LineArt
//   #9cb4d8  tool blue   a generator, or a mode        -> Accent
//   #e2564b  constraint  the DRIVEN entity             -> Constraint
//   #dda06a  amber       construction geometry         -> Construction
//
// Red and blue were collapsed onto Accent when the constraint shelf first landed, on the argument
// that no glyph uses both so the merge is lossless and the Signal language has one accent. Lossless
// per glyph, but not per SET: the accent means "picked" on every other shelf, so a constraint drawn
// in it said nothing its white reference entity did not also say. Ruled back out 2026-07-30.
const INK = {
  '#f2f6fa': 'LineArt',
  '#9cb4d8': 'Accent',
  '#e2564b': 'Constraint',
  '#dda06a': 'Construction',
};

function load(file, exports) {
  const src = fs.readFileSync(path.join(ROOT, file), 'utf8');
  const body = src.slice(src.lastIndexOf('<script>') + 8, src.lastIndexOf('</script>'));
  const sandbox = {
    Math, console, Array, Number, String, JSON,
    document: { getElementById: () => ({ appendChild() {} }), createElement: () => ({ set innerHTML(_) {} }) },
  };
  vm.createContext(sandbox);
  vm.runInContext(`${body}\n;globalThis.__X = { ${exports.join(', ')} };`, sandbox);
  return sandbox.__X;
}

// A mark authored on a non-square canvas is CENTRED on the glyph grid rather than stretched: the
// sheet draws Horizontal 36x22 and Vertical 22x36 on purpose, so that the pair reads as a
// quarter-turn of each other, and padding either back to a square would break exactly that.
function centred(list, w, h) {
  const dx = (Math.max(w, h) - w) / 2, dy = (Math.max(w, h) - h) / 2;
  if (!dx && !dy) return list;
  const p = ([x, y]) => [x + dx, y + dy];
  return list.map(m => ({
    ...m,
    ...(m.pts ? { pts: m.pts.map(p) } : {}),
    ...(m.c ? { c: p(m.c) } : {}),
    ...(m.a ? { a: p(m.a), b: p(m.b) } : {}),
    ...(m.p ? { p: m.p.map(p) } : {}),
  }));
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
      // Same filled/stroked split the rects get: a filled circle is a disc, a stroked one a ring.
      const k = a.stroke === 'none' ? 'Disc' : 'Circle';
      out.push({ k, c: [Number(a.cx), Number(a.cy)], r: Number(a.r), ink });
      continue;
    }
    if (tag === 'ellipse') {
      out.push({ k: 'Ellipse', c: [Number(a.cx), Number(a.cy)], rx: Number(a.rx), ry: Number(a.ry), ink });
      continue;
    }

    // path: walk the commands in order, flushing the polyline buffer whenever a curve interrupts
    const toks = a.d.match(/[MLHVACQZ]|-?\d+(?:\.\d+)?/g) || [];
    let i = 0, cmd = null, x = 0, y = 0, start = null, run = [], closed = false, curved = false;
    // A `Z` after an arc closes back onto the point the arc already ended at, which leaves a
    // zero-length run. It is a fact about the path syntax, not a mark, so it is dropped here
    // rather than transposed into a glyph that draws nothing.
    const flush = () => {
      const moved = run.some(p => p[0] !== run[0][0] || p[1] !== run[0][1]);
      if (run.length >= 2 && moved) out.push({ k: 'Line', pts: run, ink });
      run = [];
    };
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
        // `Mark` has no quadratic, but every quadratic IS a cubic: degree elevation is exact,
        // not an approximation, so the glyph draws the same curve rather than a fitted one.
        // C1 = P0 + 2/3(Q1 - P0), C2 = P2 + 2/3(Q1 - P2).
        curved = true;
        const q1 = [n(), n()], p2 = [n(), n()];
        const lift = (a, b) => [a[0] + (2 / 3) * (b[0] - a[0]), a[1] + (2 / 3) * (b[1] - a[1])];
        const p0 = [x, y];
        flush();
        out.push({ k: 'Cubic', p: [p0, lift(p0, q1), lift(p2, q1), p2], ink });
        x = p2[0]; y = p2[1]; run = [[x, y]];
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
// One arm per role, and a throw rather than a fallthrough: the fallthrough used to hand back
// CONSTRUCTION for anything it did not recognise, so adding the Constraint role silently emitted
// every red mark as amber and the parity gate compared two copies of the same mistake.
const ink = (i) => {
  switch (i.role) {
    case 'LineArt': return i.dashed ? 'Ink::DASHED' : 'Ink::SOLID';
    case 'Accent': return 'Ink::ACCENT';
    case 'Constraint': return i.dashed ? 'Ink::CONSTRAINT_DASHED' : 'Ink::CONSTRAINT';
    case 'Construction': return 'Ink::CONSTRUCTION';
    default: throw new Error(`no Ink const for role ${i.role}`);
  }
};

function emit(p) {
  switch (p.k) {
    case 'Line': return `Mark::Line { points: ${pts(p.pts)}, ink: ${ink(p.ink)} }`;
    case 'Closed': return `Mark::Closed { points: ${pts(p.pts)}, ink: ${ink(p.ink)} }`;
    case 'Rect': return `Mark::Rect { a: (${f(p.a[0])}, ${f(p.a[1])}), b: (${f(p.b[0])}, ${f(p.b[1])}), ink: ${ink(p.ink)} }`;
    case 'Node': return `Mark::Node { center: (${f(p.c[0])}, ${f(p.c[1])}), size: ${f(p.size)}, ink: ${ink(p.ink)} }`;
    case 'Circle': return `Mark::Circle { center: (${f(p.c[0])}, ${f(p.c[1])}), radius: ${f(p.r)}, ink: ${ink(p.ink)} }`;
    case 'Disc': return `Mark::Disc { center: (${f(p.c[0])}, ${f(p.c[1])}), radius: ${f(p.r)}, ink: ${ink(p.ink)} }`;
    case 'Ellipse': return `Mark::Ellipse { center: (${f(p.c[0])}, ${f(p.c[1])}), rx: ${f(p.rx)}, ry: ${f(p.ry)}, ink: ${ink(p.ink)} }`;
    case 'Arc': return `Mark::Arc { center: (${f(p.c[0])}, ${f(p.c[1])}), rx: ${f(p.rx)}, ry: ${f(p.ry)}, ` +
      `from: ${rad(p.from)}, to: ${rad(p.to)}, ink: ${ink(p.ink)} }`;
    case 'Cubic': return `Mark::Cubic { p0: (${f(p.p[0][0])}, ${f(p.p[0][1])}), p1: (${f(p.p[1][0])}, ${f(p.p[1][1])}), ` +
      `p2: (${f(p.p[2][0])}, ${f(p.p[2][1])}), p3: (${f(p.p[3][0])}, ${f(p.p[3][1])}), ink: ${ink(p.ink)} }`;
  }
  throw new Error('unknown primitive ' + p.k);
}

// Both sheets, flattened to one list of `{ key, hint, w, h, svg }`. The tool sheet carries several
// candidate drawings per mark and `chosen` picks the shipped one; the constraint sheet has one
// drawing per mark and sizes its own canvas.
const tools = load('tool-marks.html', ['CREATE', 'MODIFY', 'SHIPPED', 'chosen']);
const relations = load('constraint-marks.html', ['MARKS', 'TOOLS']);

const SHEET = [
  ...tools.CREATE.concat(tools.MODIFY, tools.SHIPPED).map(m => ({
    key: `${m.name}|${m.hint}`, hint: m.hint, w: 36, h: 36, svg: () => tools.chosen(m).draw(),
  })),
  ...relations.MARKS.concat(relations.TOOLS).map(m => ({
    key: `${m.name}|${m.dof}`, hint: m.dof, w: m.w, h: m.h, svg: () => m.draw(),
  })),
];

const rows = [];
for (const m of SHEET) {
  const id = IDS[m.key];
  if (!id) continue;
  const parsed = centred(parse(m.svg(), m.key), m.w, m.h);
  rows.push(`    // ${m.key.replace('|', ' — ')}\n    ("${id}", &[\n` +
    parsed.map(p => `        ${emit(p)},`).join('\n') + '\n    ]),');
}

const missing = Object.keys(IDS).filter(k => !SHEET.some(m => m.key === k));
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
///
/// Angles are literal radians rather than \`consts::PI\` expressions: the sheet resolves them
/// numerically and the glyph is free to name them however it reads best, so the two agree to a
/// tolerance rather than by sharing a symbol.
#[rustfmt::skip]
#[allow(clippy::approx_constant)]
pub(super) const REFERENCE: &[(&str, &[Mark])] = &[
${rows.join('\n')}
];
`);
console.log(`wrote ${rows.length} reference glyph(s)`);
