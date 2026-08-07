# ADR 0047 — A free direction is settled by a gauge, not by damping

- **Status:** Accepted. Closes the instability [ADR 0043](0043-a-snap-lets-go-gradually.md) ranked
  first and could not fix from the authoring side
- **Date:** 2026-08-06

## Context

[ADR 0043](0043-a-snap-lets-go-gradually.md) ends with a defeat. Six approaches had been measured
against one symptom — an unsnapped drag of a curved slot's end swinging by hundreds of times the
cursor step — and none of them touched it:

| heading | gain per unit of cursor | steps out of 200 moving more than 10× the cursor |
| ------- | ----------------------: | -----------------------------------------------: |
| 0°      |                    2.46 |                                                0 |
| **60°** |              **191.13** |                                          **118** |
| 90°     |                    1.70 |                                                0 |
| 120°    |                    1.76 |                                                0 |
| **180°**|             **1317.86** |                                           **21** |

Every attempted cure was a statement about *authoring*: hold the sweep, weigh least motion, project
the null space, price the chord. Four of them were rejected structurally — a weighted penalty row
changes the residual count, the dog leg picks its `JᵀJ` versus `JJᵀ` branch on over- versus
under-determined, and a penalty of `1e-6` broke a `Fix` by 2.4 units. Pricing the sweep at full
weight removed both spikes and opened a bigger one at a heading that had been smooth. 0043
concluded that "the spike attaches to whichever direction is softest where the descent lands", and
named the remaining lever as the cold-start/warm-start trade.

**That conclusion was wrong, and the two experiments that overturned it took ten minutes.**

Two knobs with no geometric meaning were varied, and the answer moved:

- Changing the trust region's **initial radius** from 1.0 to 0.25 or to 4.0 made the 180° spike
  disappear entirely — 1317.86 → 2.46, and all 21 spikes with it.
- **Raising the iteration budget** from 100 to 1000 made 60° *worse*, 191 → 227, and the spike count
  118 → 138.

Neither of those is a statement about the drawing. A drag's answer must not be a function of the
solver's starting radius, and more iterations must not make a converging search land somewhere else.
Together they say the search is not converging to a point — it is landing somewhere on a set,
and which member it lands on is decided by the path.

Instrumenting the solve said where. A curved slot mid-drag runs three passes over 19 parameters,
and the second (24 residuals, the hand at full authority) **exhausted its 100-iteration budget with
the Cholesky factorisation failing on 99 of them**. Every step in that pass came out of the
Levenberg–Marquardt repair — `JᵀJ + λI` — with λ climbing until the matrix went definite.

## The actual cause

`JᵀJ` **squares the condition number.** A sketch's Jacobian routinely carries six or seven digits of
conditioning loss; squared, that is twelve to fourteen, and a `f64` has sixteen. Past that the
normal matrix is not ill-conditioned, it is unfactorisable, and the repair the code falls through to
answers a *different problem* — one perturbed hardest in exactly the directions the constraints
pinned down least.

The free sweep is such a direction. So the mechanism behind the author's complaint was never
authoring at all: the drawing swung because the arithmetic had nothing left to say about the
direction it swung in, and the damping filled the silence with whatever λ happened to reach.

The decomposition's diagonal, measured across neighbouring cursor positions a five-thousandth of a
unit apart, shows the structure plainly:

```
1.0  1.0  0.78 0.70 0.53 0.53 0.43 0.36 0.28   nine real directions
1.9e-5  8.3e-6  1.9e-6                          three weak, but steady between frames
9.3e-9                                          wanders 2.6e-10 … 9.7e-9 frame to frame
2.3e-16 1.4e-16 2.5e-17                         machine dust
```

The wandering entry is the **finite-difference noise floor** showing itself. The Jacobian is taken
by central differences with a step of `6e-6`, whose cancellation error is about `ε/h` — four parts
in `10¹¹` — so nothing below that carries information. Three clear orders of magnitude separate it
from the last real direction.

## Decision

**Every Gauss-Newton step is the minimum-norm least-squares solution of `J h = −r`, computed by
complete orthogonal decomposition on `J` itself. `JᵀJ` is never formed.**

`crates/substrate/src/complete_orthogonal_decomposition.rs` is LAPACK's `xGELSY` written out:
Householder QR with column pivoting, then a second family of Householders annihilating the
trapezoidal remainder from the right. `crates/substrate/src/nonlinear_least_squares.rs` calls it and
nothing else — the normal equations, the Cholesky, the damping loop, the pivot tolerance and the
separate least-norm branch are all deleted.

**Three things follow, and each of them is the point.**

**The conditioning stops being squared.** An orthogonal matrix has condition number one, so working
on `J` costs the same order of arithmetic and loses no digits to the method. The measured cost of
the normal equations was not a slow answer, it was a wrong one.

**Picking the shortest step is a GAUGE CHOICE, and it is now stated.** An under-constrained drawing
has a whole subspace of equally good steps and something has to settle it. This is the move a fluid
solver makes when it pins the constant mode of a pressure field before solving: the free direction
is fixed by a declared rule, and *then* the solve is a function of its input. The shortest step is
also the right rule to declare here for a reason older than the numerics — it leaves every parameter
no relation names nearest where the author put it, which is what [ADR 0043](0043-a-snap-lets-go-gradually.md)'s
own least-norm branch was already reaching for and getting only in the under-determined case.

**The rank tolerance is a measurement, not a tuning.** `JACOBIAN_RANK_TOLERANCE = 1e-8` sits in the
middle of the measured gap above. Sweeping it confirms a plateau: every value from `1e-10` to `1e-6`
gives the same drawing to three decimals, and tightening it to `1e-13` puts the noise back in and
swings the drawing *thousands* of times the cursor step — worse than the damping it replaced,
because nothing is left to mute it. The tolerance is not "how weak a constraint may be". It is where
the finite-difference Jacobian stops being believable.

## Consequences

**The instability is gone, and the directions that were already smooth did not move.**

| heading | before | after |
| ------- | -----: | ----: |
| 0°      |   2.46 |  2.46 |
| **60°** | **191.13** |  **1.52** |
| 90°     |   1.70 |  1.70 |
| 120°    |   1.76 |  1.76 |
| **180°**| **1317.86** |  **2.46** |

Three of the five unchanged to three decimals is what says noise was removed rather than a bias
added. Guarded by `an_unsnapped_walk_is_smooth_in_every_direction`, which replaces
`the_unsnapped_free_sweep_is_still_spent_arbitrarily` — a test written to assert the defect so it
would fail the day someone fixed it. It did.

**The whole rest of the suite passed unchanged.** One failure across the workspace, and it was that
test. For a change to the numerical core under every constraint solve, that is the strongest
available evidence that the old answers were right wherever the arithmetic could reach them, and
arbitrary only where it could not.

**The warm-start question is retired without being answered.** 0043 named the cold-start trade as
the remaining lever and it was the wrong lever: the drag is still cold-started from the pre-drag
drawing every frame, path-independence is still intact, and the smoothness came for free. The
literature says this directly and it is worth recording — Klein and Huang (1983) proved the
pseudo-inverse is non-integrable, so a warm start would have bought smoothness at the cost of a
closed cursor loop leaving the drawing somewhere else, and Baillieul's extended Jacobian buys
repeatability by making the free direction a function of the task instead. **A gauge is that same
move made one level down**, in the linear solve rather than in the constraint set, and it costs no
authoring vocabulary at all.

**The four penalty-row rejections in 0043 keep their conclusion and lose their explanation.** They
failed because extra rows flipped the dog leg's branch choice — and that branch no longer exists.
One code path now answers over-determined, under-determined and rank-deficient alike, so adding a
row can no longer change *which algorithm runs*. Whether a soft preference row is a good idea is
open again; it is no longer structurally forbidden. If one is ever wanted, XPBD's formulation is the
shape to use: carry the multiplier and state a compliance, so the row's strength is a property of
the row rather than of the iteration count.

**Not fixed, and now the largest remaining cost: the second pass does not converge.** 13% of solves
still spend the full 100 iterations, and capping the budget at 30 cuts the drag suite from 10.5s to
3.6s — so **two thirds of the time is spent grinding past iteration 30**. This is pre-existing and
was hidden behind the damping. The cause is that the hand's `Fix` is frequently unreachable, making
that pass a genuinely incompatible least-squares problem on which Gauss-Newton converges only
linearly. The solver has a gradient test, a step test and an absolute residual test, and **no
relative-decrease test** — the standard fourth criterion, and the one that stops exactly this. Left
alone deliberately: adding it changes answers, and this record is about a change that did not.

**The decomposition costs about 1.7× the old step, and the suite runs 6.8s → 10.5s.** Held
column-major so the inner loop walks a contiguous run; the remaining gap is the non-converging pass
above rather than the kernel.
