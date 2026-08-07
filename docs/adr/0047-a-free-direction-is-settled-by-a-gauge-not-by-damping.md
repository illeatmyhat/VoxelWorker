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
still spend the full 100 iterations. This is pre-existing and was hidden behind the damping. The cause is that the hand's `Fix` is frequently unreachable, making
that pass a genuinely incompatible least-squares problem on which Gauss-Newton converges only
linearly. The solver has a gradient test, a step test and an absolute residual test, and **no
relative-decrease test** — the standard fourth criterion, and the one that stops exactly this. Left
alone deliberately: adding it changes answers, and this record is about a change that did not.

**The decomposition costs about 1.1× the old step: the suite runs 6.24s → 6.81s.** Held
column-major so the inner loop walks a contiguous run.

## Amendment, 2026-08-06 — two numbers above were wrong, and the missing criterion is now there

**Correction first.** The two performance figures originally in this record were measured against a
build with `std::env::var` probe lookups compiled into the solve loop, which inflated the wall clock
by roughly half. Measured honestly against the commits themselves, the decomposition costs ~1.1×
rather than the ~1.7× first claimed (6.24s → 6.81s, not 6.8s → 10.5s), and the "10.5s to 3.6s under
a budget cap" figure was contaminated in the same way and has been struck rather than restated. Both
paragraphs above now carry the corrected numbers. **An environment lookup inside a numerical inner
loop is not free, and a probe that changes the thing it measures is worse than no probe** — prefer a
deterministic counter (total iterations, solves hitting the ceiling) to a stopwatch.

**The relative-decrease test is in.** `SolveSettings::improvement_tolerance` stops an accepted step
that improved the sum of squares by less than that share of it, and reports `Stalled` — the honest
outcome, since the search stopped because it had stopped moving and whether the *answer* is one is
read off `residual_norm` as always. Tested only on an accepted step: a rejected step leaves the
objective alone, so counting it as "no improvement" would stop at the first bad guess rather than at
the end of progress, and the collapsing trust radius is already the test for that.

The tolerance is measured, not adopted. Over a drag of a curved slot, 1005 solves:

| tolerance | iterations | hit the ceiling | zoom-invariance error (budget 1e-4) |
| --------- | ---------: | --------------: | ----------------------------------: |
| off       |    349,196 |            2176 |                            2.005e-5 |
| 1e-9      |    296,003 |            1917 |                            1.981e-5 |
| **1e-8**  |**269,624** |        **1015** |                        **2.090e-5** |
| 1e-7      |    212,868 |              45 |                            5.936e-5 |
| 1e-6      |    155,487 |               0 |                 1.989e-4 — **fails** |

**A quarter of the work goes for four percent of the error at 1e-8**, and that is the setting. One
notch looser triples the error; Ceres's own default of 1e-6 breaks
`a_ceiling_in_screen_points_means_the_same_at_every_zoom` outright — which is what a general
optimizer's default looks like on a problem whose answer a person is looking at.

**This is a 23% cut, not the 3× the budget-cap experiment suggested.** The 3× exists only at
tolerances that damage the answer. Recording the gap because the difference between the two is the
entire content of the measurement: a cap on iterations stops a search wherever it happens to be, and
a relative-decrease test stops it where it stopped earning. They cost the same and they are not the
same change.


## Amendment, 2026-08-06 — the title is wrong: it was the truncation, not the gauge

This record claimed the drag stopped swinging because the free direction was settled by a stated
gauge instead of by damping. **That is not what the measurement says**, and the experiment that
would have caught it was never run at the time. Run now, as a 2×2 over the same five-heading walk —
worst gain as a multiple of the cursor step, at 60° and 180°:

| | minimum-norm gauge | basic-solution gauge |
| --- | ---: | ---: |
| on `J`, tolerance `1e-8` | 1.52 / 2.46 | **1.45 / 2.45** |
| on `J`, tolerance `1e-12` | 1556 / 3000 | **1139 / 3346** |

The gauge column makes no difference. The tolerance row makes a factor of a thousand. Forming `JᵀJ`
turns out not to be the deciding factor either: the same minimum-norm answer computed through the
normal equations is smooth at a comparable effective truncation, and only breaks when the truncation
is pushed below the noise floor — the same place `J` itself breaks.

**The correct statement is that a free direction is settled by DISCARDING it.** The Jacobian is
taken by central differences, and below about `1e-8` relative it contains nothing but cancellation
error; the old solver had no rank concept at all, so it damped those directions instead of removing
them, and a damped noise direction is still a noise direction. Rank-revealing truncation removes
them. Which member of the surviving solution set is then chosen is a real decision — minimum norm is
kept because it is unique and pivot-order independent, where a basic solution can flip when a tie
breaks the other way — but it is not the decision that fixed anything.

The complete orthogonal decomposition still earns its place, for a reason one level down from the
one first given: **it is what makes the rank knowable**. Pivoting sorts the directions by size so
"below the noise floor" is a question that can be asked at all. That is the load-bearing property,
not the avoidance of `JᵀJ` and not the shortest-member gauge.

### The metric question, answered and closed

If the answer is the shortest step, shortest in which norm? The parameters are all lengths in the
same voxel units — coordinates and radii, no angles — so the plain Euclidean norm is at least
dimensionally honest, which retires half the objection. For the other half, the textbook answer is
Van der Sluis column equilibration: weight each column by its own norm. Measured at the shipped
tolerance, worst gain at 60° and total collateral travel of the points that were not grabbed:

| weighting | worst gain | collateral travel |
| --------- | ---------: | ----------------: |
| **none** | **1.515** | **3.341** |
| `‖J_j‖^¼` | 1.630 | 3.365 |
| `‖J_j‖^½` | 1.765 | 3.969 |
| `‖J_j‖` | 217.5 | 6.840 |

Monotone, and unweighted wins on both measures — including on collateral travel, which is the
direct measurement of "least motion" and the thing weighting was supposed to improve. **The reason
is the same as the finding above**: equilibration scales every column to the same size, which is
precisely the ordering the rank-revealing pivot needs in order to sort the noise directions last.
Flatten it and the truncation stops finding them. A metric and a truncation cannot both be imposed
through the same pivot; the truncation is worth more.

So the weighted norm is rejected, and this time on its own numbers rather than on numbers taken
while the damping was still in the way — which was the defect in the four penalty-row rejections and
in the "weigh least motion" cure that this record already retracted.

### One thing did change: the tolerance was at the edge of its band, and is now in the middle

Failure is two-sided. Too loose and real directions are truncated away — at `1e-3` the drag stops
responding entirely, and at `1e-5` a workspace test fails. Too tight and the noise comes back.
Measured, the walk is flat from `1e-4` to `1e-8` and the workspace suite is green from `1e-6` to
`1e-8`; the intersection is `1e-6` to `1e-8`, and `1e-8` sat at one end of it.

`JACOBIAN_RANK_TOLERANCE` is now **`1e-7`**, the middle of the measured band, which agrees with the
independent estimate from the decomposition's own diagonal (noise wandering up to `1e-8`, smallest
real direction `1.9e-6`, geometric centre `1.4e-7`). Full workspace suite green. This is the only
behavioural change from the whole investigation, and it buys margin rather than accuracy.

### Extended Jacobian: not built, and argued against

The other half of the original proposal is still unbuilt, now deliberately. Baillieul's construction
buys repeatability under a warm start by making the free direction a function of the task, and the
drag is cold-started from the pre-drag drawing every frame, so path-independence is already
structural. It would cost algorithmic singularities — configurations where the augmented Jacobian is
singular though the original is not — which is trading a solved problem for an unsolved one.
