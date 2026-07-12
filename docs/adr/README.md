# Architecture Decision Records

This directory is **append-only decision history**: each record captures a decision at
the moment it was made — the context, the alternatives weighed, the evidence, and the
ruling. Records are never rewritten to match later reality; when reality moves, the
record's **Status** line is amended (Superseded / Retired / Amended-by) and a new
record carries the new decision.

The **current shape of the system** is not described here. It lives in
[`docs/architecture/`](../architecture/README.md), which is edited freely and kept
timeless. The division of labour:

| Place | Role | Editing rule |
| --- | --- | --- |
| `CONTEXT.md` (repo root) | Terms and their meanings | Prune freely; terms only |
| `docs/adr/` | Decisions and their reasoning | Append-only; amend Status lines only |
| `docs/architecture/` | The living shape of the system | Edit in place; no history, no roadmap |
| `docs/design/` | Dated analysis inputs (sweeps, maps) | Snapshots; supersede by newer files |

When writing a new ADR, describe the **delta** against `docs/architecture/` rather
than restating it, and update the architecture set in the same change that ships the
decision.

Read an ADR for *why* and *what was rejected* — its Status line tells you whether the
*what* still stands.
