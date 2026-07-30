// Headless check of the tool sheet, across every variant. No rendering, no eyeballing.
import fs from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..', '..');

const HTML = path.join(REPO, 'docs', 'design', 'sketch-marks', 'tool-marks.html');
const src = fs.readFileSync(HTML, 'utf8');
const body = src.slice(src.lastIndexOf('<script>') + 8, src.lastIndexOf('</script>'));

const sandbox = {
  Math, console, Array, Number, String, JSON,
  document: { getElementById: () => ({ appendChild() {} }), createElement: () => ({ set innerHTML(_) {} }) },
};
vm.createContext(sandbox);
vm.runInContext(body + '\n;globalThis.__X = { CREATE, MODIFY, SHIPPED, variantsOf, chosen, OVERRIDE };', sandbox);
const { CREATE, MODIFY, SHIPPED, variantsOf, chosen, OVERRIDE } = sandbox.__X;
const MARKS = CREATE.concat(MODIFY, SHIPPED);

const BLUE = '#9cb4d8', WHITE = '#f2f6fa';
let fails = 0;
const bad = (m) => { console.log('  FAIL ' + m); fails++; };
const ok = (c, m) => { if (!c) bad(m); };
const near = (a, b, tol, m) => ok(Math.abs(a - b) <= tol, `${m}: ${a} vs ${b}`);

// ---- walk an SVG fragment into points and arcs ----------------------------
function pieces(s) {
  const pts = [], arcs = [];
  for (const d of [...s.matchAll(/ d="([^"]+)"/g)].map(m => m[1])) {
    const toks = d.match(/[MLHVAQCZ]|-?\d+(\.\d+)?/g) || [];
    let i = 0, cmd = null, x = 0, y = 0;
    while (i < toks.length) {
      if (/[A-Z]/.test(toks[i])) { cmd = toks[i++]; continue; }
      const n = () => Number(toks[i++]);
      if (cmd === 'M' || cmd === 'L') { x = n(); y = n(); }
      else if (cmd === 'H') { x = n(); }
      else if (cmd === 'V') { y = n(); }
      else if (cmd === 'A') {
        const r = n(); n(); n(); n(); n();
        const nx = n(), ny = n();
        arcs.push({ r, chord: Math.hypot(nx - x, ny - y) });
        x = nx; y = ny;
      } else if (cmd === 'Q') { pts.push([n(), n()]); x = n(); y = n(); }
      else if (cmd === 'C') { pts.push([n(), n()]); pts.push([n(), n()]); x = n(); y = n(); }
      else { i++; continue; }
      pts.push([x, y]);
    }
  }
  for (const r of s.matchAll(/<rect x="([\d.-]+)" y="([\d.-]+)" width="([\d.-]+)" height="([\d.-]+)"/g)) {
    const [, x, y, w, h] = r.map(Number); pts.push([x, y], [x + w, y + h]);
  }
  for (const c of s.matchAll(/<circle cx="([\d.-]+)" cy="([\d.-]+)" r="([\d.-]+)"/g)) {
    const [, x, y, r] = c.map(Number); pts.push([x - r, y - r], [x + r, y + r]);
  }
  for (const e of s.matchAll(/<ellipse cx="([\d.-]+)" cy="([\d.-]+)" rx="([\d.-]+)" ry="([\d.-]+)"/g)) {
    const [, x, y, rx, ry] = e.map(Number); pts.push([x - rx, y - ry], [x + rx, y + ry]);
  }
  return { pts, arcs };
}

// ---- structural checks over every variant ---------------------------------
let variantCount = 0;
for (const m of MARKS) {
  const vs = variantsOf(m);
  // Every mark has A/B/C; the seven redrawn ones add a D.
  // A mark carries a variant per PASS it actually went through. The four-pass provenance was a
  // critique of a ported set; a mark drawn fresh for this app never had a B to argue with, and
  // inventing two rivals to reject would be theatre. `fresh` is that claim, made explicitly.
  const wantVariants = m.fresh ? 1 : m.d ? 4 : 3;
  ok(vs.length === wantVariants, `${m.name} (${m.hint}): ${vs.length} variants, expected ${wantVariants}`);
  for (const v of vs) {
    variantCount++;
    const tag = `${m.name} (${m.hint}) ${v.key}`;
    const s = v.draw();
    ok(!/NaN|undefined/.test(s), `${tag}: NaN/undefined in output`);
    ok(s.includes(WHITE) || s.includes(BLUE), `${tag}: draws nothing`);
    ok(!/#e2564b/.test(s), `${tag}: uses constraint red`);
    ok(v.w <= 36 && v.h <= 36, `${tag}: canvas over 36`);
    ok(v.note && v.note.length > 20, `${tag}: note missing or too thin to argue with`);

    const { pts, arcs } = pieces(s);
    for (const [px, py] of pts) {
      if (px < 0 || px > v.w || py < 0 || py > v.h) {
        bad(`${tag}: point (${px}, ${py}) outside ${v.w}x${v.h}`);
      }
    }
    // a semicircular cap is exactly chord = 2r, so this is <= with rounding room
    for (const a of arcs) {
      ok(a.chord <= 2 * a.r + 1e-6, `${tag}: arc chord ${a.chord.toFixed(4)} > 2r=${2 * a.r}`);
    }
  }

  // The variants must actually differ — a pass that renders identically is not a pass. A fresh
  // mark has no rivals to differ from, so the whole comparison is skipped rather than faked.
  const [A, Bv, Cv] = m.fresh ? ['a', 'b', 'c'] : vs.map(v => v.draw());
  // A declared convergence is a finding, not a defect — but it must be DECLARED.
  if (A === Bv && m.converges !== 'B') bad(`${m.name} (${m.hint}): A and B identical, undeclared`);
  if (A === Cv && m.converges !== 'C') bad(`${m.name} (${m.hint}): A and C identical, undeclared`);
  if (Bv === Cv) bad(`${m.name} (${m.hint}): B and C are byte-identical`);
  if (m.converges && vs.map(v => v.draw()).filter(d => d === A).length < 2) {
    bad(`${m.name} (${m.hint}): declares convergence on ${m.converges} but the marks differ`);
  }
}
console.log(`marks: ${CREATE.length} create + ${MODIFY.length} modify + ${SHIPPED.length} shipped ` +
            `= ${MARKS.length}, ${variantCount} variants`);

// C's rule: a tool with no sibling carries no accent. Text is the case that tests it.
{
  const text = CREATE.find(m => m.name === 'Text');
  ok(!text.c.draw().includes(BLUE), 'Text C: rule C says no sibling means no accent, but it has blue');
  ok(!text.b.draw().includes(BLUE), 'Text B: Fusion draws this monochrome, but it has blue');
}

// ---- named geometry claims the notes make ---------------------------------
const dist = (p, q) => Math.hypot(p[0] - q[0], p[1] - q[1]);

// 2-tangent, A: parallels at y=8 and y=28 with r=10 about (18,18)
near(18 - 10, 8, 1e-9, '2-tangent A top');
near(18 + 10, 28, 1e-9, '2-tangent A bottom');

// 2-tangent, B: circle (15,21) r=11 against x=4 and y=32
near(15 - 4, 11, 1e-9, '2-tangent B vertical line');
near(32 - 21, 11, 1e-9, '2-tangent B horizontal line');

// 2-tangent, C: incircle of the V at (4,30)
{
  const C = [19.27, 22], r = 8;
  near(30 - C[1], r, 5e-3, '2-tangent C horizontal leg');
  const A = [4, 30], d = [(22.2 - 4) / Math.hypot(18.2, -26.3), (3.7 - 30) / Math.hypot(18.2, -26.3)];
  const perp = Math.abs((C[0] - A[0]) * d[1] - (C[1] - A[1]) * d[0]);
  near(perp, r, 5e-3, '2-tangent C diagonal leg');
}

// 3-tangent: all three lines at distance r from the centre, tangent points on the segments
{
  const C = [11, 23], r = 7;
  const segs = [[[2, 30], [34, 30]], [[4, 7], [4, 32]], [[2.4, 7.8], [33.6, 31.2]]];
  for (const [a, b] of segs) {
    const vx = b[0] - a[0], vy = b[1] - a[1], len = Math.hypot(vx, vy);
    const along = ((C[0] - a[0]) * vx + (C[1] - a[1]) * vy) / len;
    const perp = Math.abs((C[0] - a[0]) * vy - (C[1] - a[1]) * vx) / len;
    near(perp, r, 5e-3, `3-tangent line ${a}-${b} distance`);
    ok(along > 0 && along < len, `3-tangent: tangent point off the drawn segment ${a}-${b}`);
  }
  near((28 + 21 - 35) / 2, r, 1e-9, '3-tangent inradius formula');
}

// fit-spline B: four points at t = 0, 1/3, 2/3, 1 must lie ON the cubic
{
  const P = [[5, 24], [11, 4], [19, 26], [31, 8]];
  const at = (t) => [0, 1].map(k =>
    (1 - t) ** 3 * P[0][k] + 3 * (1 - t) ** 2 * t * P[1][k] +
    3 * (1 - t) * t * t * P[2][k] + t ** 3 * P[3][k]);
  const drawn = [[5, 24], [11.74, 14.96], [20.26, 15.7], [31, 8]];
  [0, 1 / 3, 2 / 3, 1].forEach((t, i) => {
    near(dist(drawn[i], at(t)), 0, 6e-3, `fit-spline B point ${i} is off the curve`);
  });
  // fit and control C must share ONE curve, which is C's whole argument
  const fit = CREATE.find(m => m.name === 'Spline' && m.hint === 'fit point');
  const ctl = CREATE.find(m => m.name === 'Spline' && m.hint === 'control point');
  const curveOf = (s) => (s.match(/M 5 24 C 11 4, 19 26, 31 8/g) || []).length;
  ok(curveOf(fit.c.draw()) === 1 && curveOf(ctl.c.draw()) === 1,
     'spline C: the two marks do not share one base curve');
  // ...and their accents must be on opposite sides of it
  const onCurve = at(0.5);
  near(dist([15.75, 15.25], onCurve), 0, 1e-6, 'fit-spline C accent is not on the curve');
  for (const p of [[11, 4], [19, 26]]) {
    ok(dist(p, at(0.5)) > 6, `control-spline C accent ${p} is too close to the curve to read as off it`);
  }
}

// pentagons: inscribed sits on r=13, circumscribed's edge midpoints sit on r=10
{
  const pent = (r, rot) => Array.from({ length: 5 }, (_, i) => {
    const a = (rot + i * 72) * Math.PI / 180;
    return [18 + r * Math.cos(a), 18 + r * Math.sin(a)];
  });
  for (const v of pent(13, -90)) near(dist(v, [18, 18]), 13, 1e-6, 'inscribed pentagon vertex');
  const C = pent(12.36, -90);
  for (let i = 0; i < 5; i++) {
    const j = (i + 1) % 5;
    const mid = [(C[i][0] + C[j][0]) / 2, (C[i][1] + C[j][1]) / 2];
    near(dist(mid, [18, 18]), 10, 5e-3, 'circumscribed pentagon edge midpoint');
  }
  near(12.36 * Math.cos(36 * Math.PI / 180), 10, 5e-3, 'circumscribed: R*cos36 is not the inradius');
  // the accented midpoint used in B and C
  near(dist([23.88, 9.91], [18, 18]), 10, 5e-3, 'circumscribed accent is not on the inscribed circle');
}

// polygon-by-edge: the open path must omit the bottom edge in every variant
{
  const m = CREATE.find(x => x.name === 'Polygon' && x.hint === 'edge');
  for (const [k, s] of [['A', m.draw()], ['B', m.b.draw()], ['C', m.c.draw()]]) {
    ok(!/Z"/.test(s), `polygon edge ${k}: path is closed, so the drawn edge is redundant`);
    // no polyline segment may run along the base line
    const runsAlongBase = [...s.matchAll(/ d="M ([^"]+)"/g)].some(d => {
      const n = d[1].split(/[ L]+/).map(Number).filter(x => !Number.isNaN(x));
      if (n.length <= 4) return false;   // the base segment itself is SUPPOSED to be there
      for (let i = 0; i + 3 < n.length; i += 2) {
        if (Math.abs(n[i + 1] - 27.52) < 0.01 && Math.abs(n[i + 3] - 27.52) < 0.01 &&
            Math.abs(n[i] - n[i + 2]) > 1) return true;
      }
      return false;
    });
    ok(!runsAlongBase, `polygon edge ${k}: the outline still draws the bottom edge`);
  }
}

// offset B: two nested U profiles, uniform 7 apart on both the caps and the legs
{
  near(12 - 5, 7, 1e-9, 'offset B cap radii differ by the offset');
  near(11 - 4, 7, 1e-9, 'offset B upper leg');
  near(28 - 21, 7, 1e-9, 'offset B lower leg');
  near((11 + 21) / 2, 16, 1e-9, 'offset B inner cap centre');
  near((4 + 28) / 2, 16, 1e-9, 'offset B outer cap centre — not concentric with the inner');
}

// offset A: a true miter, equal distance on both legs
near(13 - 6, 7, 1e-9, 'offset A vertical leg');
near(25 - 18, 7, 1e-9, 'offset A horizontal leg');

// trim B: the scissor blades must actually cross, and the rings sit on their axes
{
  const b1 = [[11, 12], [23, 30]], b2 = [[25, 12], [13, 30]];
  const t = (25 - 11) / ((23 - 11) - (13 - 25));
  const cross = [11 + t * 12, 12 + t * 18];
  near(cross[0], 18, 1e-6, 'trim B blades cross x');
  near(cross[1], 22.5, 1e-6, 'trim B blades cross y');
  for (const [b, ring] of [[b1, [24.55, 32.33]], [b2, [11.45, 32.33]]]) {
    const d = [b[1][0] - b[0][0], b[1][1] - b[0][1]];
    const len = Math.hypot(d[0], d[1]);
    const want = [b[1][0] + 2.8 * d[0] / len, b[1][1] + 2.8 * d[1] / len];
    near(dist(ring, want), 0, 0.02, 'trim B ring is off the blade axis');
  }
  // the trimmed half and the surviving half must meet exactly, with no overlap
  near(18, 18, 1e-9, 'trim B line split point');
}

// trim A: the crossing is computed
near(10 + ((30 - 20) / (30 - 6)) * 16, 16.67, 5e-3, 'trim A crossing x');

// three-point arc: radius solved from the three points, in every variant
{
  const [p0, p2] = [[5, 26], [18, 8]];
  const k = (p0[0] ** 2 + p0[1] ** 2 - p2[0] ** 2 - p2[1] ** 2 - 36 * (p0[0] - p2[0]))
          / (2 * (p0[1] - p2[1]));
  const r = Math.hypot(p0[0] - 18, p0[1] - k);
  near(r, 13.6944, 5e-4, 'three-point arc radius does not solve the three points');
  const m = SHIPPED.find(x => x.name === 'Three-point arc');
  for (const [key, s] of [['A', m.draw()], ['B', m.b.draw()], ['C', m.c.draw()]]) {
    ok(s.includes('13.6944'), `three-point arc ${key}: radius is not the solved one`);
  }
  // A and C keep the through-point a DISC per ADR 0030 sec 5; B is the Fusion port and may not
  ok(/<circle cx="18" cy="8"/.test(m.draw()), 'three-point arc A: through-point is not a disc');
  ok(/<circle cx="18" cy="8"/.test(m.c.draw()), 'three-point arc C: through-point is not a disc');
  ok(/<rect[^>]*x="15.4"/.test(m.b.draw()), 'three-point arc B: through-point is not a square');
}

// arc slot: three concentric radii about one centre, exact semicircular caps
{
  const C = [18, 26];
  const at = (r, deg) => [C[0] + r * Math.cos(deg * Math.PI / 180), C[1] + r * Math.sin(deg * Math.PI / 180)];
  const drawn = { oL: [2.4115, 17], oR: [33.5885, 17], iL: [9.3397, 21], iR: [26.6603, 21],
                  sL: [5.8756, 19], sM: [18, 12], sR: [30.1244, 19] };
  const want = { oL: at(18, -150), oR: at(18, -30), iL: at(10, -150), iR: at(10, -30),
                 sL: at(14, -150), sM: at(14, -90), sR: at(14, -30) };
  for (const k of Object.keys(drawn)) near(dist(drawn[k], want[k]), 0, 5e-4, `arc slot ${k}`);
  near(dist(drawn.oR, drawn.iR), 8, 5e-4, 'arc slot right cap is not a semicircle of radius 4');
  near(dist(drawn.oL, drawn.iL), 8, 5e-4, 'arc slot left cap is not a semicircle of radius 4');
  for (const [s, i, o] of [[drawn.sR, drawn.iR, drawn.oR], [drawn.sL, drawn.iL, drawn.oL]]) {
    near(dist(s, i), 4, 5e-4, 'arc slot spine not at the cap centre (inner)');
    near(dist(s, o), 4, 5e-4, 'arc slot spine not at the cap centre (outer)');
  }
}

// fillet/chamfer C must be the SAME white corner with one part swapped — C's whole argument
{
  const f = MODIFY.find(m => m.name === 'Fillet').c.draw();
  const c = MODIFY.find(m => m.name === 'Chamfer').c.draw();
  const corner = 'M 6 4 L 6 26 L 31 26';
  ok(f.includes(corner) && c.includes(corner), 'fillet/chamfer C do not share one corner');
  const fBlue = f.slice(f.indexOf(corner) + corner.length);
  const cBlue = c.slice(c.indexOf(corner) + corner.length);
  ok(fBlue !== cBlue, 'fillet/chamfer C are identical after the shared corner');
  ok(/A 10 10/.test(fBlue), 'fillet C insert is not an arc');
  ok(!/A /.test(cBlue), 'chamfer C insert is not a straight line');
  // both inserts must span the same two points, or they are not answers to one situation
  ok(fBlue.includes('6 16') && fBlue.includes('16 26'), 'fillet C insert does not span (6,16)-(16,26)');
  ok(cBlue.includes('6 16') && cBlue.includes('16 26'), 'chamfer C insert does not span (6,16)-(16,26)');
}

// fillet tangency, in every variant that draws the arc
{
  const C = [16, 16], r = 10;
  near(dist([6, 16], C), r, 1e-9, 'fillet tangent point A off the circle');
  near(dist([16, 26], C), r, 1e-9, 'fillet tangent point B off the circle');
  near(16 - C[1], 0, 1e-9, 'fillet: radius at A not perpendicular to the vertical leg');
  near(16 - C[0], 0, 1e-9, 'fillet: radius at B not perpendicular to the horizontal leg');
}

// arrowheads land their base on the line that feeds them
{
  const cases = [
    { n: 'extend A', line: [[18, 22], [28, 22]], tip: [28, 22], dir: [1, 0] },
    { n: 'move A', line: [[14.5, 17.5], [19, 13]], tip: [21, 11], dir: [0.7071, -0.7071] },
    { n: 'move C', line: [[14.5, 17.5], [19, 13]], tip: [21, 11], dir: [0.7071, -0.7071] },
    { n: 'scale A', line: [[25, 7], [28, 4]], tip: [30, 2], dir: [0.7071, -0.7071] },
    { n: 'extend B', line: [[8, 13], [23, 13]], tip: [26, 13], dir: [1, 0] },
  ];
  for (const c of cases) {
    const base = [c.tip[0] - c.dir[0] * 6.5, c.tip[1] - c.dir[1] * 6.5];
    const [a, b] = c.line, vx = b[0] - a[0], vy = b[1] - a[1], len = Math.hypot(vx, vy);
    near(((base[0] - a[0]) * vy - (base[1] - a[1]) * vx) / len, 0, 0.02, `${c.n}: head base off the line`);
    const tipAlong = ((c.tip[0] - a[0]) * vx + (c.tip[1] - a[1]) * vy) / len;
    ok(len <= tipAlong + 0.01, `${c.n}: line overshoots the arrow tip`);
  }
}

// scale A: both boxes square, sharing the anchor, handle on the shared 45 deg diagonal
{
  const w = [3, 18, 14, 29], b = [3, 9, 23, 29];
  near(w[2] - w[0], w[3] - w[1], 1e-9, 'scale A white box not square');
  near(b[2] - b[0], b[3] - b[1], 1e-9, 'scale A blue box not square');
  ok(w[0] === b[0] && w[3] === b[3], 'scale A boxes do not share the anchor');
  for (const p of [[25, 7], [28, 4], [30, 2]]) {
    near((p[0] - 3) - (29 - p[1]), 0, 1e-9, `scale A handle point ${p} off the diagonal`);
  }
}

// move A/C: the arrow must clear both boxes
{
  const inside = (bx, p) => p[0] > bx[0] && p[0] < bx[2] && p[1] > bx[1] && p[1] < bx[3];
  for (const p of [[14.5, 17.5], [19, 13], [21, 11]]) {
    ok(!inside([3, 18, 14, 29], p) && !inside([22, 3, 33, 14], p), `move: arrow point ${p} inside a box`);
  }
}

// select C: the accent square must sit on the segment it marks
{
  const a = [5, 26], b = [30, 12], p = [17.5, 19];
  const vx = b[0] - a[0], vy = b[1] - a[1], len = Math.hypot(vx, vy);
  near(((p[0] - a[0]) * vy - (p[1] - a[1]) * vx) / len, 0, 1e-6, 'select C accent is off the segment');
}

// rectangle B: the outline must be GAPPED at the two accented corners, not drawn through them
{
  const s = SHIPPED.find(m => m.name === 'Rectangle').b.draw();
  ok(!/<rect[^>]*stroke="#f2f6fa"/.test(s), 'rectangle B: outline is a closed rect, so it has no gaps');
  ok((s.match(/<rect /g) || []).length === 2, 'rectangle B: expected exactly two accent squares');
}

// ---- the resolved set -----------------------------------------------------
const ORANGE = '#dda06a';
{
  // An OVERRIDE key that matches no mark would fall back to B in SILENCE — the one failure
  // mode of a lookup-table ruling, and the whole reason this check exists.
  const keys = new Set(MARKS.map(m => `${m.name}|${m.hint}`));
  for (const k of Object.keys(OVERRIDE)) {
    ok(keys.has(k), `OVERRIDE key "${k}" matches no mark — it silently resolves to B`);
  }

  // The ruling, restated independently of the page: exactly these take A, one takes orange.
  const WANT_A = ['Spline|fit point', 'Spline|control point', 'Text|profile from glyphs',
                  'Trim|to nearest crossing', 'Sketch scale|uniform', 'Add point|free vertex',
                  'Rectangle|2-point', 'Chamfer|equal distance'];
  for (const m of MARKS) {
    const k = `${m.name}|${m.hint}`, v = chosen(m);
    // Same precedence as the page: an A-override outranks a D redraw, D outranks the B default.
    // A fresh mark has only an A, so it resolves there by having nowhere else to go.
    const want = WANT_A.includes(k) || m.fresh ? 'A' : m.d ? 'D' : 'B';
    ok(v.key === want, `${k}: resolved to ${v.key}, expected ${want}`);
    ok(typeof v.note === 'string' && v.note.length > 20, `${k}: resolved mark has no note`);

    const d = v.draw();
    ok(!/NaN|undefined/.test(d), `${k}: resolved draw emits NaN/undefined`);
    const { pts } = pieces(d);
    ok(pts.length > 0, `${k}: resolved mark draws nothing`);
    for (const [x, y] of pts) {
      ok(x >= -0.01 && x <= v.w + 0.01 && y >= -0.01 && y <= v.h + 0.01,
         `${k}: resolved point ${x},${y} outside ${v.w}x${v.h}`);
    }
  }

  // The A-overrides must genuinely differ from the B they replace, or the override is a no-op.
  for (const k of WANT_A) {
    const m = MARKS.find(x => `${x.name}|${x.hint}` === k);
    ok(m.b && m.draw() !== m.b.draw(), `${k}: overridden to A but A and B are identical`);
  }
  // A fresh mark must NOT be in OVERRIDE: it has no B to be overridden away from, so an entry
  // there would be a reason given for a choice that was never available.
  for (const m of MARKS.filter(x => x.fresh)) {
    const k = `${m.name}|${m.hint}`;
    ok(!OVERRIDE[k], `${k}: fresh, so an OVERRIDE entry argues against a variant it never had`);
    ok(!m.b && !m.c && !m.d, `${k}: flagged fresh but carries a rival variant`);
  }

  // Orange is confined to exactly one mark, and that mark spends no blue.
  const orange = MARKS.filter(m => chosen(m).draw().includes(ORANGE));
  ok(orange.length === 1 && orange[0].name === 'Construction',
     `orange appears in ${orange.length} resolved marks, expected 1 (Construction)`);
  const con = chosen(MARKS.find(m => m.name === 'Construction')).draw();
  ok(!con.includes(BLUE), 'Construction resolved: still spends blue as well as orange');
  ok(/stroke-dasharray/.test(con), 'Construction resolved: the orange linetype is not dashed');

  // The two cut tools are gone from every table.
  for (const gone of ['Polyline', 'Close loop']) {
    ok(!MARKS.some(m => m.name === gone), `${gone} was cut but is still in the set`);
  }
  // 33 after the two cuts, plus the 8 the owner's authoritative list named and the sheet had
  // never drawn: Midpoint line, Rectangle 3-point and centre, Circle 2-point, the centre-point
  // arc slot, Sketch dimension, and the two chamfers past equal distance.
  ok(MARKS.length === 41, `expected 41 marks, found ${MARKS.length}`);
  ok(MARKS.filter(m => m.fresh).length === 8,
     `expected 8 fresh marks, found ${MARKS.filter(m => m.fresh).length}`);
  // Every command on the owner's list is drawn, under the hint the list implies.
  for (const need of ['Line|2 points', 'Midpoint line|centre · end',
                      'Rectangle|2-point', 'Rectangle|3-point', 'Rectangle|centre · corner',
                      'Circle|centre · diameter', 'Circle|2-point', 'Circle|3-point',
                      'Circle|2-tangent', 'Circle|3-tangent',
                      'Arc|centre · endpoints', 'Arc|tangent', 'Three-point arc|ends + through',
                      'Polygon|circumscribed', 'Polygon|inscribed', 'Polygon|edge',
                      'Ellipse|centre · 2 axes',
                      'Slot|centre-to-centre', 'Slot|overall', 'Slot|centre-point',
                      'Slot|3-point arc', 'Slot|centre-point arc',
                      'Spline|fit point', 'Spline|control point',
                      'Add point|free vertex', 'Text|profile from glyphs',
                      'Sketch dimension|drive a distance',
                      'Fillet|rounds a corner', 'Chamfer|equal distance',
                      'Chamfer|distance and angle', 'Chamfer|two distance',
                      'Blend curve|tangent join', 'Offset|parallel copy',
                      'Trim|to nearest crossing', 'Extend|to nearest boundary',
                      'Break|split at a point', 'Sketch scale|uniform',
                      'Move / copy|free transform']) {
    ok(MARKS.some(m => `${m.name}|${m.hint}` === need), `the owner's list names "${need}", which is not drawn`);
  }
}

// ---- pass D: the redraws, and the claims their notes make -----------------
{
  // Rectangle carries a D that was explored and REJECTED in favour of A — it must stay drawn on
  // the sheet and must not resolve.
  const WANT_D = ['Line|2 points', 'Circle|centre · diameter', 'Conic|apex · rho',
                  'Construction|role toggle', 'Offset|parallel copy', 'Break|split at a point'];
  const REJECTED_D = ['Rectangle|2-point'];
  const has = MARKS.filter(m => m.d).map(m => `${m.name}|${m.hint}`);
  const all = WANT_D.concat(REJECTED_D);
  ok(has.length === all.length && all.every(k => has.includes(k)),
     `D covers [${has}], expected [${all}]`);
  for (const k of all) {
    const m = MARKS.find(x => `${x.name}|${x.hint}` === k);
    const want = WANT_D.includes(k) ? 'D' : 'A';
    ok(chosen(m).key === want, `${k}: resolved to ${chosen(m).key}, expected ${want}`);
    // A redraw that matches the thing it is clearing is not a redraw.
    ok(m.d.draw() !== m.b.draw(), `${k}: D is byte-identical to B, so it clears nothing`);
  }

  const D = (n, h) => MARKS.find(m => m.name === n && m.hint === h).d.draw();

  // Line D: our Line DOES drag into a tangent arc, so the arc must be there — and must actually
  // be tangent at the seam, which is the only part of the drawing that carries meaning.
  {
    const s = D('Line', '2 points');
    const seg = [...s.matchAll(/d="M ([\d.]+) ([\d.]+) L ([\d.]+) ([\d.]+)"/g)][0].slice(1).map(Number);
    const arc = [...s.matchAll(/d="M ([\d.]+) ([\d.]+) A ([\d.]+) [\d.]+ 0 (\d) (\d) ([\d.]+) ([\d.]+)"/g)][0]
      .slice(1).map(Number);
    const [sx0, sy0, sx1, sy1] = seg;
    const [ax, ay, r, large, sweep, ex, ey] = arc;
    ok(large === 1, 'Line D: arc is minor — it is meant to carry two thirds of a circle');
    near(ax, sx1, 1e-9, 'Line D: arc does not start where the segment ends');
    near(ay, sy1, 1e-9, 'Line D: arc does not start where the segment ends');

    // Both centres consistent with (start, end, r); (large, sweep) together select one.
    const dx = ex - ax, dy = ey - ay, d = Math.hypot(dx, dy);
    const h = Math.sqrt(Math.max(0, r * r - (d / 2) ** 2));
    const mid = [(ax + ex) / 2, (ay + ey) / 2], u = [-dy / d, dx / d];
    const cands = [[mid[0] + h * u[0], mid[1] + h * u[1]], [mid[0] - h * u[0], mid[1] - h * u[1]]];
    // sweep=1 is clockwise on screen, so for a MINOR arc the centre lies where
    // cross(chord, centre - start) > 0. A major arc bulges the other way, so large flips it.
    const wantPositive = (sweep === 1) !== (large === 1);
    const centre = cands.find(c => (dx * (c[1] - ay) - dy * (c[0] - ax) > 0) === wantPositive);
    ok(centre, 'Line D: neither candidate centre matches the drawn sweep flag');
    near(Math.hypot(centre[0] - ax, centre[1] - ay), r, 5e-3, 'Line D: centre is not r from the seam');
    // Tangency: the radius at the seam must be perpendicular to the incoming segment.
    const v = [sx1 - sx0, sy1 - sy0], w = [centre[0] - ax, centre[1] - ay];
    near((v[0] * w[0] + v[1] * w[1]) / (Math.hypot(...v) * r), 0, 5e-3,
         'Line D: the arc is NOT tangent to the segment at the seam');
    // Two thirds of a circle, to the degree. sweep=1 runs clockwise, which is increasing angle
    // once y grows downward, so the swept amount is (end - start) taken mod a full turn.
    const a0 = Math.atan2(ay - centre[1], ax - centre[0]);
    const a1 = Math.atan2(ey - centre[1], ex - centre[0]);
    const swept = ((a1 - a0) * (sweep === 1 ? 1 : -1) + 2 * Math.PI) % (2 * Math.PI);
    near(swept * 180 / Math.PI, 240, 5e-3, 'Line D: arc is not two thirds of a circle');

    // The radius has a floor, and it is the squares that set it, not taste. A 240° sweep puts
    // the seam and the end 1.732r apart; two 5.2-unit squares closer than that in BOTH axes
    // overlap and read as one lozenge, which silently costs the glyph a pick.
    const boxes = [...s.matchAll(/<rect x="([-\d.]+)" y="([-\d.]+)" width="([\d.]+)"/g)]
      .map(g => [Number(g[1]), Number(g[2]), Number(g[3])]);
    ok(boxes.length === 3, `Line D: ${boxes.length} squares — the tool asks for three picks`);
    for (let i = 0; i < boxes.length; i++)
      for (let j = i + 1; j < boxes.length; j++) {
        const [x0, y0, w0] = boxes[i], [x1, y1, w1] = boxes[j];
        const over = x0 < x1 + w1 && x1 < x0 + w0 && y0 < y1 + w1 && y1 < y0 + w0;
        ok(!over, `Line D: squares ${i} and ${j} overlap — the radius is under the lozenge floor`);
      }

    // And it is not their composition: theirs ends directly above the junction (same x).
    ok(Math.abs(ex - ax) > 1, 'Line D: arc ends over the junction, which is their candy cane');

    // Line and Arc-tangent are both "straight run into a tangent arc" and collided in B, where
    // both stems ran horizontally. Orientation is what separates them now, so it has to hold.
    const at = chosen(MARKS.find(m => m.name === 'Arc' && m.hint === 'tangent')).draw();
    const ats = [...at.matchAll(/d="M ([\d.]+) ([\d.]+) L ([\d.]+) ([\d.]+)"/g)][0].slice(1).map(Number);
    const a = [sx1 - sx0, sy1 - sy0], b = [ats[2] - ats[0], ats[3] - ats[1]];
    const cos = Math.abs(a[0] * b[0] + a[1] * b[1]) / (Math.hypot(...a) * Math.hypot(...b));
    const deg = Math.acos(Math.min(1, cos)) * 180 / Math.PI;
    ok(deg >= 30, `Line D vs Arc-tangent: stems only ${deg.toFixed(1)}° apart — the pair collides`);
  }

  // Circle D: the stub runs centre→rim and STOPS there (a radius, not a diameter chord).
  {
    const s = D('Circle', 'centre · diameter');
    const seg = [...s.matchAll(/d="M ([\d.]+) ([\d.]+) L ([\d.]+) ([\d.]+)"/g)][0].slice(1).map(Number);
    const [x0, y0, x1, y1] = seg;
    near(x0, 18, 1e-9, 'circle D: stub does not start at the centre');
    near(y0, 18, 1e-9, 'circle D: stub does not start at the centre');
    near(Math.hypot(x1 - 18, y1 - 18), 12, 5e-3, 'circle D: stub does not end ON the rim');
  }

  // Conic D: the apex is a real intersection of the two tangent legs, and rho is ON the curve.
  {
    const s = D('Conic', 'apex · rho');
    ok(/d="M 5 26 L 18 4 L 31 26"/.test(s), 'conic D: control triangle is not the two tangent legs');
    // Quadratic Bezier at t=0.5 with control (18,4): the rho point must sit exactly there.
    const bez = (t) => [
      (1 - t) ** 2 * 5 + 2 * (1 - t) * t * 18 + t * t * 31,
      (1 - t) ** 2 * 26 + 2 * (1 - t) * t * 4 + t * t * 26,
    ];
    const [rx, ry] = bez(0.5);
    ok(s.includes(`x="${rx - 5.2 / 2}"`), `conic D: rho x is not the curve's t=0.5 (${rx})`);
    ok(s.includes(`y="${ry - 5.2 / 2}"`), `conic D: rho y is not the curve's t=0.5 (${ry})`);
  }

  // Construction D: the dash is the DIAGONAL of the box, and the node is its midpoint.
  {
    const s = D('Construction', 'role toggle');
    const r = [...s.matchAll(/<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"[^>]*fill="none"/g)][0]
      .slice(1).map(Number);
    const [bx, by, bw, bh] = r;
    const dg = [...s.matchAll(/d="M ([\d.]+) ([\d.]+) L ([\d.]+) ([\d.]+)"/g)][0].slice(1).map(Number);
    near(dg[0], bx, 1e-9, 'construction D: diagonal does not start at a box corner');
    near(dg[1], by, 1e-9, 'construction D: diagonal does not start at a box corner');
    near(dg[2], bx + bw, 1e-9, 'construction D: diagonal does not end at the opposite corner');
    near(dg[3], by + bh, 1e-9, 'construction D: diagonal does not end at the opposite corner');
    ok(s.includes(`x="${bx + bw / 2 - 5.2 / 2}"`), 'construction D: node is not the diagonal midpoint');
    ok(/stroke-dasharray/.test(s) && s.includes(ORANGE), 'construction D: diagonal is not an orange dash');
  }

  // Offset D: the corner arc radius EQUALS the straight-run offset distance. The note's claim.
  {
    const s = D('Offset', 'parallel copy');
    const src = [...s.matchAll(/d="M 8 6 L 8 24 L 30 24"/g)];
    ok(src.length === 1, 'offset D: source L profile is not the expected path');
    const a = [...s.matchAll(/d="M ([\d.]+) ([\d.]+) A ([\d.]+) [\d.]+ 0 \d \d ([\d.]+) ([\d.]+)"/g)][0]
      .slice(1).map(Number);
    const [ax, ay, ar, aex, aey] = a;
    // The vertical run offsets from x=8 to x=2, the horizontal from y=24 to y=30: both 6.
    near(8 - ax, ar, 1e-9, 'offset D: vertical offset distance != corner radius');
    near(aey - 24, ar, 1e-9, 'offset D: horizontal offset distance != corner radius');
    // And the arc is genuinely centred on the source corner (8,24).
    near(Math.hypot(ax - 8, ay - 24), ar, 1e-9, 'offset D: arc start is not r from the source corner');
    near(Math.hypot(aex - 8, aey - 24), ar, 1e-9, 'offset D: arc end is not r from the source corner');
  }

  // Break D: ONE unbroken run, no gap anywhere. That is the entire claim of the redraw.
  {
    const s = D('Break', 'split at a point');
    const runs = [...s.matchAll(/d="M ([\d.]+) 18 L ([\d.]+) 18"/g)];
    ok(runs.length === 1, `break D: ${runs.length} horizontal runs — the line is still gapped`);
    ok((s.match(/<rect /g) || []).length === 1, 'break D: expected exactly one accent square');
  }

  // Rectangle D: outline UNBROKEN (a real rect), brackets are open Ls entirely OUTSIDE it.
  {
    const s = D('Rectangle', '2-point');
    const r = [...s.matchAll(/<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"[^>]*fill="none"/g)];
    ok(r.length === 1, 'rectangle D: outline is not a single unbroken rect');
    const [bx, by, bw, bh] = r[0].slice(1).map(Number);
    const brs = [...s.matchAll(/d="M ([\d.]+) ([\d.]+) L ([\d.]+) ([\d.]+) L ([\d.]+) ([\d.]+)"/g)];
    ok(brs.length === 2, `rectangle D: expected 2 brackets, found ${brs.length}`);
    for (const br of brs) {
      const p = br.slice(1).map(Number);
      const corner = [p[2], p[3]];
      ok(corner[0] < bx || corner[0] > bx + bw || corner[1] < by || corner[1] > by + bh,
         `rectangle D: bracket corner ${corner} is inside the outline, not outside it`);
      // An L, not a straight line: the two legs must be perpendicular.
      const v1 = [p[0] - p[2], p[1] - p[3]], v2 = [p[4] - p[2], p[5] - p[3]];
      near(v1[0] * v2[0] + v1[1] * v2[1], 0, 1e-9, 'rectangle D: bracket legs are not perpendicular');
    }
  }
}

// ---- the eight fresh marks: their notes make claims, so the claims get proved --------------
{
  const R = (n, h) => chosen(MARKS.find(m => m.name === n && m.hint === h)).draw();
  const segs = (d) => [...d.matchAll(/<path d="M ([\d.-]+) ([\d.-]+) L ([\d.-]+) ([\d.-]+)"/g)]
    .map(g => g.slice(1).map(Number));
  const sqs = (d) => [...d.matchAll(/<rect x="([\d.-]+)" y="([\d.-]+)" width="([\d.-]+)"[^>]*fill="([^"]+)"/g)]
    .map(g => [Number(g[1]), Number(g[2]), Number(g[3]), g[4]])
    .filter(r => r[3] !== 'none');   // an outline box is a rect too, and it is not a vertex
  const centres = (d, tint) => sqs(d).filter(r => !tint || r[3] === tint)
    .map(r => [r[0] + r[2] / 2, r[1] + r[2] / 2]);
  const has = (list, p, tol = 1e-6) => list.some(q => Math.hypot(q[0] - p[0], q[1] - p[1]) <= tol);

  // Midpoint line: the accented node really is the MIDPOINT, the ticks really are equal and
  // really are perpendicular, and the run leans the opposite way to Line so the pair separates.
  {
    const d = R('Midpoint line', 'centre · end');
    const [x0, y0, x1, y1] = segs(d)[0];
    const mid = [(x0 + x1) / 2, (y0 + y1) / 2];
    ok(has(centres(d, BLUE), mid), 'Midpoint line: no accent on the segment midpoint');
    ok(has(centres(d, BLUE), [x1, y1]), 'Midpoint line: the picked end is not accented');
    ok(has(centres(d, WHITE), [x0, y0]), 'Midpoint line: the produced end is not line art');
    const ticks = segs(d).slice(1);
    ok(ticks.length === 2, `Midpoint line: ${ticks.length} ticks, expected 2`);
    const len = (t) => Math.hypot(t[2] - t[0], t[3] - t[1]);
    near(len(ticks[0]), len(ticks[1]), 1e-6, 'Midpoint line: the "equal" ticks are unequal');
    const run = [x1 - x0, y1 - y0];
    for (const [i, t] of ticks.entries()) {
      const v = [t[2] - t[0], t[3] - t[1]];
      near((run[0] * v[0] + run[1] * v[1]) / (Math.hypot(...run) * Math.hypot(...v)), 0, 1e-6,
           `Midpoint line: tick ${i} is not perpendicular to the run`);
      const c = [(t[0] + t[2]) / 2, (t[1] + t[3]) / 2];
      const along = ((c[0] - x0) * run[0] + (c[1] - y0) * run[1]) / (run[0] ** 2 + run[1] ** 2);
      near(along, i === 0 ? 0.25 : 0.75, 1e-6, `Midpoint line: tick ${i} is off its half's centre`);
    }
    const line = segs(R('Line', '2 points'))[0];
    const other = [line[2] - line[0], line[3] - line[1]];
    ok(run[1] * other[1] < 0, 'Midpoint line leans the same way as Line — the pair is one drawing twice');
  }

  // Circle 2-point: the chord is a true DIAMETER (through the centre, length 2r) and the centre
  // itself carries no square, which is the whole difference from centre-diameter.
  {
    const d = R('Circle', '2-point');
    const [, cx, cy, r] = d.match(/<circle cx="([\d.]+)" cy="([\d.]+)" r="([\d.]+)"/).map(Number);
    const [x0, y0, x1, y1] = segs(d)[0];
    near(Math.hypot(x1 - x0, y1 - y0), 2 * r, 5e-3, 'Circle 2-point: the chord is not a diameter');
    near(Math.hypot((x0 + x1) / 2 - cx, (y0 + y1) / 2 - cy), 0, 5e-3,
         'Circle 2-point: the chord does not pass through the centre');
    for (const p of [[x0, y0], [x1, y1]]) {
      near(Math.hypot(p[0] - cx, p[1] - cy), r, 5e-3, 'Circle 2-point: a pick is off the circle');
      ok(has(centres(d, BLUE), p), 'Circle 2-point: a diameter end is not accented');
    }
    ok(!has(centres(d), [cx, cy]), 'Circle 2-point: the centre is marked, which is its sibling');
  }

  // Centre-point arc slot: the accented centre is genuinely equidistant from both cap centres,
  // and the two radii it draws are that same distance — otherwise the fan is decoration.
  {
    const d = R('Slot', 'centre-point arc');
    const radii = segs(d);
    ok(radii.length === 2, `centre-point arc slot: ${radii.length} radii, expected 2`);
    const c = [radii[0][0], radii[0][1]];
    near(radii[1][0], c[0], 1e-6, 'centre-point arc slot: the two radii start from different points');
    near(radii[1][1], c[1], 1e-6, 'centre-point arc slot: the two radii start from different points');
    const r0 = Math.hypot(radii[0][2] - c[0], radii[0][3] - c[1]);
    const r1 = Math.hypot(radii[1][2] - c[0], radii[1][3] - c[1]);
    near(r0, r1, 5e-3, 'centre-point arc slot: the centre is not equidistant from the two caps');
    near(r0, 14, 5e-3, 'centre-point arc slot: the radii are not the family spine radius of 14');
    for (const p of [c, [radii[0][2], radii[0][3]], [radii[1][2], radii[1][3]]]) {
      ok(has(centres(d, BLUE), p), 'centre-point arc slot: a pick is not accented');
    }
    ok(/stroke-dasharray/.test(d), 'centre-point arc slot: the radii are not dashed');
  }

  // Rectangle 3-point: a RECTANGLE, not a lozenge — adjacent edges dot to zero — and no edge is
  // axis-aligned, which is the only thing separating it from 2-point at a glance.
  {
    const d = R('Rectangle', '3-point');
    const m4 = d.match(/M ([\d.-]+) ([\d.-]+) L ([\d.-]+) ([\d.-]+) L ([\d.-]+) ([\d.-]+) L ([\d.-]+) ([\d.-]+)/);
    const n = m4.slice(1).map(Number);
    const P = [[n[0], n[1]], [n[2], n[3]], [n[4], n[5]], [n[6], n[7]]];
    for (let i = 0; i < 4; i++) {
      const a = [P[(i + 1) % 4][0] - P[i][0], P[(i + 1) % 4][1] - P[i][1]];
      const b = [P[(i + 2) % 4][0] - P[(i + 1) % 4][0], P[(i + 2) % 4][1] - P[(i + 1) % 4][1]];
      near((a[0] * b[0] + a[1] * b[1]) / (Math.hypot(...a) * Math.hypot(...b)), 0, 5e-4,
           `Rectangle 3-point: corner ${i} is not square — this is a lozenge`);
    }
    for (let i = 0; i < 4; i++) {
      const j = (i + 1) % 4;
      ok(Math.abs(P[i][0] - P[j][0]) > 1e-6 && Math.abs(P[i][1] - P[j][1]) > 1e-6,
         `Rectangle 3-point: edge ${i} is axis-aligned, which is the 2-point mark`);
    }
    const blue = centres(d, BLUE);
    ok(blue.length === 3, `Rectangle 3-point: ${blue.length} accents, expected 3 picks`);
    ok(P.filter(p => has(blue, p)).length === 3, 'Rectangle 3-point: the accents are not on corners');
  }

  // Centre rectangle: five squares, and the fifth is the true centre of the other four.
  {
    const d = R('Rectangle', 'centre · corner');
    const all = centres(d);
    ok(all.length === 5, `Centre rectangle: ${all.length} squares, expected 5`);
    const b = d.match(/<rect x="([\d.-]+)" y="([\d.-]+)" width="([\d.-]+)" height="([\d.-]+)" stroke="#/);
    const [x, y, w, h] = b.slice(1).map(Number);
    const mid = [x + w / 2, y + h / 2];
    ok(has(all, mid), 'Centre rectangle: no square on the box centre');
    for (const c of [[x, y], [x + w, y], [x, y + h], [x + w, y + h]]) {
      ok(has(all, c), 'Centre rectangle: a corner carries no square');
    }
    const blue = centres(d, BLUE);
    ok(blue.length === 2 && has(blue, mid),
       'Centre rectangle: the accent is not the centre plus one corner');
  }

  // Sketch dimension: witness lines stand off the feature and overrun the dimension line, and the
  // two arrowheads sit on it pointing out. That is the drafting figure, or it is a bracket.
  {
    const d = R('Sketch dimension', 'drive a distance');
    const [feature, wL, wR, dim] = segs(d);
    near(feature[1], feature[3], 1e-9, 'Sketch dimension: the measured feature is not level');
    for (const [i, w] of [wL, wR].entries()) {
      near(w[0], w[2], 1e-9, `Sketch dimension: witness ${i} is not perpendicular to the feature`);
      near(w[0], i === 0 ? feature[0] : feature[2], 1e-9,
           `Sketch dimension: witness ${i} does not rise from its end of the feature`);
      ok(w[1] < feature[1] - 1, `Sketch dimension: witness ${i} touches the feature — it must stand off`);
      ok(w[3] < dim[1], `Sketch dimension: witness ${i} stops short of the dimension line`);
    }
    near(dim[1], dim[3], 1e-9, 'Sketch dimension: the dimension line is not parallel to the feature');
    const heads = [...d.matchAll(/<path d="M ([\d.-]+) ([\d.-]+) L [^"]+" fill="([^"]+)"/g)]
      .map(g => [Number(g[1]), Number(g[2]), g[3]]);
    ok(heads.length === 2, `Sketch dimension: ${heads.length} arrowheads, expected 2`);
    ok(heads.every(h => h[2] === BLUE), 'Sketch dimension: an arrowhead is not the accent');
    for (const [i, h] of heads.entries()) {
      near(h[1], dim[1], 1e-9, `Sketch dimension: arrowhead ${i} is off the dimension line`);
      near(h[0], i === 0 ? wL[0] : wR[0], 1e-9, `Sketch dimension: arrowhead ${i} misses its witness line`);
    }
    ok(centres(d, BLUE).length === 0,
       'Sketch dimension: the measured points are accented — on this mark the accent is the dimension');
    ok(d.split(BLUE).length - 1 === 3,
       'Sketch dimension: the accent is not exactly the dimension line plus its two heads');
  }

  // The three chamfers are one drawing and the ticks are the only thing telling them apart, so
  // the ghost corner has to be there to hang them on and the counts have to say what they claim.
  {
    const bevelOf = (d) => [...d.matchAll(/<path d="M ([\d.-]+) ([\d.-]+) L ([\d.-]+) ([\d.-]+)" stroke="([^"]+)"/g)]
      .filter(g => g[5] === BLUE).map(g => g.slice(1, 5).map(Number))[0];
    const ghostOf = (d) => {
      const g = d.match(/M ([\d.-]+) ([\d.-]+) L ([\d.-]+) ([\d.-]+) L ([\d.-]+) ([\d.-]+)" stroke="[^"]+" stroke-width="[\d.]+" fill="none" stroke-dasharray/);
      return g && g.slice(1).map(Number);
    };
    const stubs = (g) => [Math.hypot(g[2] - g[0], g[3] - g[1]), Math.hypot(g[4] - g[2], g[5] - g[3])];
    const ticks = (d) => segs(d).filter(t => Math.hypot(t[2] - t[0], t[3] - t[1]) <= 6.5 + 1e-9);

    for (const hint of ['equal distance', 'distance and angle', 'two distance']) {
      const d = R('Chamfer', hint);
      const g = ghostOf(d);
      ok(g, `Chamfer ${hint}: no dashed ghost corner — the ticks would measure from nothing`);
      const b = bevelOf(d);
      ok(b, `Chamfer ${hint}: no accented bevel`);
      near(Math.hypot(b[0] - g[0], b[1] - g[1]), 0, 1e-6, `Chamfer ${hint}: bevel misses the first stub end`);
      near(Math.hypot(b[2] - g[4], b[3] - g[5]), 0, 1e-6, `Chamfer ${hint}: bevel misses the second stub end`);
    }

    const eq = stubs(ghostOf(R('Chamfer', 'equal distance')));
    near(eq[0], eq[1], 1e-6, 'Chamfer equal distance: the two stubs are NOT equal');
    const two = stubs(ghostOf(R('Chamfer', 'two distance')));
    ok(Math.abs(two[0] - two[1]) > 2, `Chamfer two distance: stubs ${two} differ too little to read`);
    const da = stubs(ghostOf(R('Chamfer', 'distance and angle')));
    ok(Math.abs(da[0] - da[1]) > 0.5, 'Chamfer distance and angle: its stubs read as equal distance');

    ok(ticks(R('Chamfer', 'equal distance')).length === 2, 'Chamfer equal distance: not one tick per stub');
    ok(ticks(R('Chamfer', 'two distance')).length === 3, 'Chamfer two distance: not two ticks against one');
    const dad = R('Chamfer', 'distance and angle');
    ok(ticks(dad).length === 1, 'Chamfer distance and angle: more than one tick, so it says two distances');
    const arc = [...dad.matchAll(/M ([\d.]+) ([\d.]+) A ([\d.]+) [\d.]+ 0 (\d) (\d) ([\d.]+) ([\d.]+)/g)][0];
    ok(arc, 'Chamfer distance and angle: no angle arc, so nothing in it says "angle"');
    if (arc) {
      const a = arc.slice(1).map(Number);
      const b = bevelOf(dad), corner = [b[2], b[3]];
      near(Math.hypot(a[0] - corner[0], a[1] - corner[1]), a[2], 5e-3,
           'Chamfer distance and angle: the arc is not struck about the bevel far end');
      near(Math.hypot(a[5] - corner[0], a[6] - corner[1]), a[2], 5e-3,
           'Chamfer distance and angle: the arc is not a constant radius about that corner');
    }
  }
}

console.log(fails === 0 ? 'PASS — all checks clean' : `${fails} FAILURES`);
