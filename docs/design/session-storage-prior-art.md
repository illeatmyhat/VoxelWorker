# Session storage: what other tools do, and where ours diverges

Researched 2026-07-20, after [ADR 0024](../adr/0024-session-state.md) established `session` as a
category but left "whether `session` needs its own file" open. The headline challenges the
owner's stated lean rather than a detail, so it is stated first.

Evidence is labelled **VERIFIED** (source read, official doc quoted, or artifact inspected on
this machine) or **INFERRED**. Several findings below were reproduced against real files in
`%APPDATA%` on the development machine; those are marked *measured here*.

## The finding: the sidecar consensus does not transfer to us, because we are not a folder

The lean going in was "path-keyed sessions in a `sessions/` directory beside the config,
LRU-pruned." The industry evidence splits cleanly — but **not along the axis we were arguing
about.** It splits on *whether the document is a directory or a file*.

| Document shape | Tools | Where session state goes |
| --- | --- | --- |
| **Project is a directory** | Godot 4, Unity ≥2020.1, Visual Studio, Xcode, JetBrains, Sublime | **sidecar inside/beside it** |
| **Document is a single file** | Blender, Rhino, AutoCAD, Photoshop, MagicaVoxel, Vintage Story | **embedded in the document** |
| **Document is a single file** | Krita, VS Code (folder case) | **path-keyed store in app data** |

Every tool that chose a sidecar had a directory to hide it in. **Not one single-file document
format chose a keyed store in app data as its primary answer** — the two that key by path
(Krita's sessions, VS Code) are the ones whose "document" is a workspace concept layered over
files, and Krita's is opt-in.

VoxelWorker's document is a file. So the migration everyone else ran — *away* from keyed app-data
stores, *toward* sidecars — is not a migration we can copy, and the reason it happened is a reason
we should worry about.

### Why the migration happened: pruning, and sidecars get it for free

This is the strongest single argument against the lean, and it is VERIFIED in source.

VS Code's storage cleaner, `src/vs/code/electron-utility/sharedProcess/contrib/storageDataCleaner.ts`,
opens with:

```typescript
if (workspaceStorageFolder.length === NON_EMPTY_WORKSPACE_ID_LENGTH) { return; }
```

That early-returns for **every real workspace**. VS Code prunes *empty-window* storage only.
Folder- and workspace-keyed storage is never deleted — no LRU, no count cap, no age cap, ever.
Confirmed absent from `storageMain.ts` as well. The user-visible consequence is
[microsoft/vscode#142972](https://github.com/microsoft/vscode/issues/142972): ~2 GB accumulated
from workspaces dating to 2019–2021, closed without a cleanup policy, with a third-party
extension filling the gap.

*Measured here:* Godot's `%APPDATA%\Godot\projects.cfg` still lists
`C:/Users/.../Documents/test`, a project directory that no longer exists. Nothing pruned it.

**A sidecar deletes itself.** Delete the project, the state goes with it. Unity `UserSettings/`,
Visual Studio `.vs/`, Godot 4 `.godot/`, Xcode `xcuserdata/` all inherit correct lifecycle for
free. The lean's `sessions/` directory inherits VS Code's unbounded-growth problem *and* has to
solve it by hand.

**If we build the keyed store anyway, the LRU is not a nice-to-have — it is the part the
reference implementation skipped and regretted.**

## Where our design was validated

### Session is a real category, and the four fields are the industry's four fields

VERIFIED by inspecting the live `state.vscdb` for
`workspaceStorage\f5d70f8f373deba127b1ca9c8ba36e65` (*measured here*, 75 keys). The per-workspace
keys are almost exactly ADR 0024's `session` category:

```
workbench.panel.hidden          workbench.sideBar.hidden
workbench.panel.position        workbench.auxiliaryBar.hidden
workbench.zenMode.active        debug.selectedconfigname
workbench.explorer.treeViewState  terminal.integrated.layoutInfo
```

Panel visibility, panel position, an exclusive display mode (`zenMode`), tree fold state, and a
selected debug configuration. That is `stack`, `view_mode`, and the two debug flags, under
different names.

JetBrains stores the same set. *Measured here*, `%APPDATA%\JetBrains\PyCharm2026.1\workspace\3DBBmpr2YcpPNOb00ZLj8gQuEaA.xml`
contains `ToolWindowManager` with per-panel `<window_info id="Terminal" visible="true" weight="0.3298636"/>`,
plus `FileEditorManager`, `ProjectView`, `BookmarksManager`, `FindInProjectRecents`.

**Nobody files these under preferences.** The chosen-versus-left test in ADR 0024 decision 1 is
the test the field actually applies.

### The debug flags belong in the session, and Blender's core devs argue our case for us

ADR 0024 decision 4 ruled the two renderer debug flags session state, on the grounds that a dump
replayed without them reproduces a different picture. That argument has an independent advocate.

On the "Blendit" version-control thread
([devtalk.blender.org/t/blendit.../25992](https://devtalk.blender.org/t/blendit-blender-git-version-control-for-blender/25992)),
the proposer wanted to strip viewport state from `.blend` files. Sybren Stüvel (Blender core)
refused:

> The viewport camera shouldn't be excluded. This can be super important, for example when
> reporting a bug it helps that the viewport is showing the buggy area of the model.

That is ADR 0024 decision 4 and ADR 0022's "a dump must reproduce the scene" law, argued by
someone with no knowledge of this repository. **Validated.**

### Window geometry is a fourth thing, and we were right to suspect it

The briefing's question 5 asked whether window/workspace layout, view state, and recent-files are
three things where we see one. **They are, and the tools disagree about which is which** —
which is itself the finding.

| | window geometry | panel layout | recent files |
| --- | --- | --- | --- |
| VS Code | **global** (`storage.json` → `windowsState`) | per-workspace | global, capped |
| JetBrains | **per-project** (`recentProjects.xml` → `<frame x= y= width= height=>`) | per-project | global, capped |
| Blender | in the document (`wmWindow.posx/sizex`) | in the document | global, capped |

*Measured here:* VS Code's `storage.json` holds one `windowsState.lastActiveWindow.uiState`
`{x:6, y:288, width:2348, height:1068}` for the whole application, while
`windowSplashWorkspaceOverride.layoutInfo.workspaces` holds `sideBarVisible`/`auxiliaryBarVisible`
**per workspace hash**. Two categories, two files, in one product.

JetBrains puts the frame rectangle *per project*, in `recentProjects.xml`.

Blender's newest surfaces moved the opposite way from its own document-embedding tradition:
`DNA_userdef_types.h` defines `struct UserDef_TempWinBounds` with per-editor defaults
(`rctf file = {100.0f, 1160.0f, 350.0f, 950.0f};`) — transient-window geometry in *preferences*,
not the document.

**Recommendation: window geometry is `settings` (it already is in `AppConfig`), panel layout is
`session`, and recent-files is a fifth thing that is neither** — see below.

## Where our design was wrong, or incomplete

### Path-versus-id is a false choice, and both real answers use both

The briefing framed the key as "file path **versus** a stable id written into the document." Both
serious implementations use a **composite**, and — critically — **neither puts the key in the
shared document.**

**VS Code — path salted with directory identity.** `src/vs/platform/workspaces/node/workspaces.ts`:

```typescript
createHash('md5').update(folderUri.fsPath).update(ctime ? String(ctime) : '').digest('hex')
```

The comment `NOTE: DO NOT CHANGE. IDENTIFIERS HAVE TO REMAIN STABLE` appears three times in that
file.

*Measured here, and this is the strongest verification in the report:* I brute-forced the three
real workspace hashes on this machine against candidate inputs and matched **all three**:

| directory | birthtime salt | md5 | matches on-disk folder |
| --- | --- | --- | --- |
| `c:\Users\Kai_Yuu\Documents\terminal` | `1777238384128` | `f5d70f8f…` | yes |
| `c:\Users\Kai_Yuu\Documents\slint-test` | `1765529289711` | `8e96846b…` | yes |
| `…\slint-test\slint_test` | `1765529564666` | `d434b419…` | yes |

Note the lowercased drive letter — `configPathStr.toLowerCase()` on non-Linux.

**JetBrains — a generated id, stored in the file that is gitignored.** From
`platform/configuration-store-impl/src/ProjectStoreImpl.kt`:

```kotlin
var projectWorkspaceId = projectIdManager.id
if (projectWorkspaceId == null) {
  // do not use the project name as part of id, to ensure a project dir rename does not cause data loss
  projectWorkspaceId = ProjectWorkspaceId.generate()
  projectIdManager.id = projectWorkspaceId
}
val productWorkspaceFile = basePath.resolve("workspace/${projectWorkspaceId.value}.xml")
```

The id is a Ksuid. It is persisted by `ProjectIdManager.kt` with
`@State(name = "ProjectId", storages = [Storage(StoragePathMacros.WORKSPACE_FILE)])` — i.e. into
`.idea/workspace.xml`.

*Measured here, the whole chain in one project:*

* `…\ComfyUI\.idea\workspace.xml` contains `<component name="ProjectId" id="3DBBmpr2YcpPNOb00ZLj8gQuEaA" />`
* `…\.idea\.gitignore` — **auto-generated by the IDE** — contains `/workspace.xml` and `/shelf/`
* `%APPDATA%\JetBrains\PyCharm2026.1\options\recentProjects.xml` maps the project path to
  `projectWorkspaceId="3DBBmpr2YcpPNOb00ZLj8gQuEaA"`
* `%APPDATA%\JetBrains\PyCharm2026.1\workspace\3DBBmpr2YcpPNOb00ZLj8gQuEaA.xml` is the 48 KB
  session file

**So the answer to "does anyone write a session key into the shared document" is: JetBrains writes
one into the project directory, into the one file its own tooling guarantees is never shared.**
Two people cloning the same repository each generate their own id and never collide, because the
id was never in the clone.

That is the design to copy, and it dissolves the briefing's dichotomy: *the id is real, but the
document is not where it lives.*

**INFERRED, and worth flagging as the id approach's live hazard:** copying a project *directory*
copies `.idea/workspace.xml` with its `ProjectId` intact, so two projects would share one session
file. A path key handles the copy case correctly for free; the id key handles the rename case
correctly for free. Neither handles both. JetBrains chose the rename side explicitly, per the
comment above.

### The ctime salt does not do what it looks like it does

Worth stating because it is easy to over-read. *Measured here* on NTFS:

| operation | birthtime | path | resulting key |
| --- | --- | --- | --- |
| rename directory | **preserved** | changes | **new key — session orphaned** |
| copy directory | **changes** | changes | new key — copy gets a fresh session |
| delete + recreate at same path | **changes** | same | new key — stale session correctly ignored |

The salt buys exactly one thing: *"same path, different directory"* is distinguishable. It buys
**nothing** for rename, which is the case users actually hit. VS Code's key is path-fragile and
its authors knew it — the third-party cleanup extension exists precisely to sweep the orphans
renames leave behind.

**Our corollary:** a path key means rename loses the session. Silently. If that is acceptable,
say so deliberately rather than discovering it.

### Pruning constants, where anyone has them at all

Only *recent-files* is capped anywhere. VERIFIED constants:

| tool | constant | value | source |
| --- | --- | --- | --- |
| VS Code | `MAX_TOTAL_RECENT_ENTRIES` | **500** | `workspacesHistoryMainService.ts` |
| JetBrains | `ide.max.recent.projects` | **50** | `PlatformExtensions.xml:1733` |
| Blender | `U.recent_files` | **200** (range 0–1000) | `DNA_userdef_types.h`, `rna_userdef.cc` |

JetBrains' eviction is **insertion-order, skipping open projects — not true LRU**:

```kotlin
var toRemove = map.size - limit
if (entry.value.opened) continue
iterator.remove()
```

And JetBrains documents its own gap, verbatim in `ProjectUtil.kt`:

> Note that directory structure used by this function doesn't allow automatic cleaning of all
> caches related to a given project if it was deleted, so consider using [getProjectDataPath]
> instead.

*Measured here:* the per-project cache directory is `comfyui.163ab395`. I reproduced the name
exactly — it is the lowercased directory name, a `.`, and `Integer.toHexString()` of Java's
`String.hashCode()` of `C:/Users/Kai_Yuu/Documents/comfy/ComfyUI` (forward slashes, original
case). Confirmed against `ProjectUtil.kt`:

```kotlin
val locationHash = Integer.toHexString((presentableUrl ?: name).hashCode())
```

**A 32-bit non-cryptographic hash keying a cache directory** — a reminder that these keys are not
required to be strong, only stable.

## The counterexample, examined: Blender, and why it is not the model

The briefing asked for Blender's costs specifically. They are real, they are documented in
Blender's own tracker, and the escape hatch is worse than the disease.

**The mechanism (VERIFIED).** `blenkernel/intern/blendfile.cc`, `setup_app_data()`:

```cpp
const short ui_id_codes[]{ID_WS, ID_SCR};
swap_wm_data_for_blendfile(&reuse_data, mode == LOAD_UI);
```

`ID_WS` (WorkSpace), `ID_SCR` (bScreen), and `wmWindowManager`. `blenkernel/intern/screen.cc`
writes `RegionView3D` for every 3D viewport — `viewquat[4]`, `dist`, `ofs[3]`, `camzoom`. **Every
viewport's orbit pose is in the document.** `DNA_windowmanager_types.h` adds
`short posx, posy, sizex, sizey` per window.

**Load UI defaults to ON (VERIFIED).** `USER_FILENOUI` is `1 << 23` and is *absent* from the
default flag set in `DNA_userdef_types.h`. The RNA property uses
`RNA_def_property_boolean_negative_sdna(..., "flag", USER_FILENOUI)` — the stored bit is the
negation. So the shipped default is *adopt the file author's layout*.

**The costs, each with a tracker citation:**

| cost | evidence |
| --- | --- |
| you inherit a stranger's layout | [#100155](https://projects.blender.org/blender/blender/issues/100155) — unchecking Load UI has no effect if the file is already open; users resort to File ▸ New first |
| the escape hatch is not sticky | [blenderartists 1296853](https://blenderartists.org/t/./1296853) — "the checkbox is not sticky… you have to disable it EVERY TIME"; ignored entirely on drag-and-drop |
| the escape hatch crashes | [#126392](https://projects.blender.org/blender/blender/issues/126392) — "Double clicking a file crashes Blender if load UI is off" |
| window geometry does not survive hardware changes | [#86804](https://projects.blender.org/blender/blender/issues/86804) (DPI mismatch shifts windows), [#36707](https://projects.blender.org/blender/blender/issues/36707) (restores off-screen; "showstopper") |
| the devs refuse to extend it | [#71935](https://projects.blender.org/blender/blender/issues/71935) — declining more window geometry because it must work across "different monitor setups, different DPIs, different OSes… Versioning may also become problematic" |

**And turning it off does not save the read.** There is no parse-time skip. `eBLOReadSkip` is for
the home file. Load-UI-off reads the UI datablocks and *then discards them* —
`wm_file_read_setup_wm_keep_old()`:

```cpp
if (!load_ui) {
  /* … newly read one from file has already been discarded in #setup_app_data. */
  return;
}
```

The document pays the size and parse cost unconditionally. **INFERRED:** that a viewport-only
orbit changes the saved bytes. Structurally it must (`RegionView3D` is written with no exclusion),
but no citation proves byte-level divergence and Blender does not mark the file dirty on
navigation. A `cmp` of two saves would settle it in five minutes.

**A negative result, stated because it was searched for.** The briefing expected to find studio
pipelines that disable Load UI. **No such document exists** that I could verify — not on
studio.blender.org, not in production/TD guides, not on devtalk. The advice is blog- and
community-tier only, and the manual documents the toggle neutrally. **The one primary studio
voice found argues the opposite** (Sybren, quoted above, defending in-file viewport state for
handovers). The anti-embed case should not be overstated: embedding buys something real, and
Blender's staff know what it is.

**What embedding does cost, verified from Blender Studio primary sources.** A `.blend` is a heap
dump — the archived file-format spec states Blender saves "data in memory to disk without any
transformations or translations," with each file-block header carrying the *"old memory address…
where the structure was located when written to disk."* Pointer-bearing binary is unmergeable, and
the studio's own version-control benchmarking post concedes the consequence:

> we only expect linear workflows to work well. So for example branching and merging branches is
> not a workflow we are looking for.

An unanswered comment on that post makes it concrete: `.blend` files cannot be merged, so only one
team member may edit and push a given file at a time.

**Honesty check, against our own interest:** most of that pain is *not* caused by the embedded UI.
The memory-dump format, the embedded pointers, and the default-on `USER_FILECOMPRESS` (present in
the factory `UserDef::flag` set, and the setting Blender Studio tells its artists to disable)
would produce unmergeable files with zero UI in them. Brecht Van Lommel's remedy list — *"diff
against default data values, zero all runtime data, and derandomize pointers"* — is about the
container, not the session state. **The UI-specific costs are narrower than the version-control
horror story suggests**: foreign window geometry on load, foreign workspace layouts, and the
schema problem below. Do not cite unmergeability as an argument against embedding; it is an
argument against Blender's container.

**The cost that does transfer is schema, and it is the one we are most exposed to.** Embedded UI
state is a *versioned schema* that must survive every UI refactor. When it does not, the document
appears broken rather than the UI appearing stale — the failure surfaces as "this old project
won't open properly." Given this repository's standing law that **old configs may break**, session
state in the document would convert a tolerated config reset into a corrupted-document report.
That is the sharpest reason to keep it out.

**Blender's own newer designs do not repeat the pattern** — `UserDef_TempWinBounds` in
preferences, and Blender Projects (`.blender_project/settings.json`, a sidecar). Given a clean
slate, Blender separated.

**Verdict: Blender does not refute ADR 0024. It is a 30-year-old trade with visible scar tissue,
defended for one good reason (bug repro and artist handover) that our `dump` already serves
without contaminating the document.**

## The finding nobody asked for: our own ecosystem embeds session state, badly

VERIFIED against real shipped files via the GitHub code API. **Vintage Story's shape JSON — the
format this project imports — carries an `editor` object as its first key.**

```json
{ "editor": { "allAngles": false, "entityTextureMode": false },
  "textureWidth": 16, "textureHeight": 16, ... }
```

Confirmed identical in four independent mod repositories (`G3rste/vsvillage`,
`FlexibleGames/VintageEngineering`, `glutzer/Fishing`, `Conquest-Reforged/Conquest-VS-Edition`).

And in the fourth, this:

```json
"editor": {
  "collapsedPaths": "Root/angleW2,Root/angleW4,Root/angleW3,Root/angleW,Root/legs/legNE,Root/legs/legNW3,Root/legs/legNW,Root/legs/legNW2",
  "allAngles": false, "entityTextureMode": false }
```

**That is outliner tree fold state — one author's, on one afternoon — committed to a shared,
version-controlled mod asset.** It is `SignalStackState` by another name: one of ADR 0024's four
session fields, inside the document, in the exact ecosystem we target. A GitHub code search for
`collapsedPaths extension:json` returns **1,420 files**.

MagicaVoxel does the same in the format we *export*: the `.vox` extension spec defines an `rCAM`
chunk with `_mode`, `_focus`, `_angle`, `_radius`, `_frustum`, `_fov`
([ephtracy/voxel-model](https://github.com/ephtracy/voxel-model/blob/master/MagicaVoxel-file-format-vox-extension.txt)).
Meanwhile MagicaVoxel's *own* camera slots are global `camera#.txt` files, explicitly
project-independent — so it embeds camera state in exported documents while keeping none per
project. (Camera-slot claim is Fandom-wiki sourced; **weakly verified**.)

**This is the report's strongest support for ADR 0024 decision 3.** The nearest-neighbour
ecosystem does exactly what the ADR forbids, and the artifacts show the predicted cost in the
open.

## Is there a category we are missing?

Yes — **one, possibly two.**

### Recent files is its own artifact everywhere, and we have no category for it

Every tool examined keeps recent-files as a **separate, global, capped** artifact, distinct from
both settings and session:

* Blender — `recent-files.txt`, plain text, one path per line (*measured here*: the file exists
  next to `userpref.blend`, alongside `bookmarks.txt` and `recent-searches.txt`)
* VS Code — `storage.json`, capped at 500
* JetBrains — `recentProjects.xml`, capped at 50, `roamingType = RoamingType.DISABLED`
* Godot — `projects.cfg` + `recent_dirs` (*measured here*)

Note Blender's `recent-searches.txt` and JetBrains' `FindInProjectRecents`: **recency lists
multiply.** They are not preferences (nobody chose them), not session (they outlive any one
workspace), and not document. When VoxelWorker grows a file-open flow this becomes a real fifth
category, and it is the natural home for the path→id index the JetBrains design needs.

### `settings` is really two things: roaming and machine-local

JetBrains marks `recentProjects.xml` `RoamingType.DISABLED` — window rectangles and machine paths
must not follow a user to another machine, while the colour scheme should. ADR 0024's `settings`
is described as "machine-scoped, exactly one set," which is fine today, but window geometry and
theme are not the same kind of preference and a settings-sync feature would expose it. **Not
urgent. Worth a sentence in the ADR so it is not rediscovered.**

### `view` versus `session` survives contact with the evidence

ADR 0024's fourth category held up. Krita draws exactly our line: the `.kra` stores
`selected="true"` per layer and `collapsed` for groups, while zoom, pan and rotation appear
**nowhere** in `kis_kra_tags.h` — they live only in the opt-in Sessions feature
(`~/.local/share/krita/sessions/*.ksn`, `KisView::saveViewState`,
`KisCanvasController::saveCanvasState` → panX, panY, rotation, mirror). Photoshop matches: image
resource **1024** is "the index of target layer," and the PSD spec has **no resource ID for zoom
or scroll at all**. GIMP matches: `PROP_ACTIVE_LAYER` in the XCF, no zoom property.

**Three independent raster editors put selection in the document and the camera outside it.** Our
`view`/`session`/`document` boundary is the field's boundary.

## THE DECIDING QUESTION IS ANSWERED: single file (2026-07-20)

Owner ruling, taken immediately on reading this report. **A VoxelWorker project is a single file,
not a project directory.**

That closes item 1 below by choosing the branch this report warned was the harder one:

* **The sidecar consensus is unavailable to us.** We do not get correct lifecycle for free. There
  is no folder to hide state in and nothing self-deletes when a project is deleted.
* **A keyed store in app data is the only remaining option** for session state that is not
  embedded, which puts us in the Krita / VS Code row of the table above.
* **Therefore the LRU is mandatory, not optional.** It is precisely the part VS Code skipped, and
  microsoft/vscode#142972 is what skipping it looks like at 2 GB. Cap at 50 following JetBrains,
  insertion-order eviction skipping open documents.
* **Key by generated id, path→id mapped in the app-data index** (item 3 below), never in the
  shared document. Copies collide, renames survive — that is the failure we are choosing, and it
  is chosen deliberately.

## What to do with this

1. ~~Decide the document shape first.~~ **Decided above: single file.**
2. **The keyed store is now the only option, so the LRU is mandatory, not optional** — the exact
   part VS Code skipped, and #142972 is what skipping it looks like at 2 GB.
3. **Copy JetBrains' key, not VS Code's.** Generate a stable id; store the path→id map in the
   app-data index beside recent-files; never put the id in the shared document. Accept that copies
   collide and say so, or salt with something (VS Code's `birthtime` trick) and accept that
   renames orphan. **You cannot have both — pick the failure you prefer and record it.**
4. **Cap it at 50, following JetBrains, not 500.** VoxelWorker's session payload is four fields;
   the cost of a generous cap is negligible, but the cost of *no* cap is the failure documented
   above. Insertion-order eviction skipping open documents is the shipped precedent and is simpler
   than true LRU.
5. **Add `recent files` to the classification as its own category** before the file-open flow
   lands, and note the roaming/machine-local split inside `settings`. Both are one-line additions
   now and archaeology later.

The strongest evidence in this report is the Vintage Story `collapsedPaths` artifact: it is our
own ecosystem, it is the exact field ADR 0024 classifies as `session`, and it is sitting in 1,420
public repositories inside documents people share. **The premise that session state must stay out
of the document is not merely validated — it is validated by the failure mode being visible in
the wild, in the format we read.**

Recorded here rather than as an ADR because nothing has landed in code and the deciding question
(document shape) is upstream of this one. It warrants folding into ADR 0024's Open section, or a
successor ADR, when the project open/save flow is designed.
