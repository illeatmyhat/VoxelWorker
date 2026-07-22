//! The dense resolvers' **scope stack**: reconstructing ADR 0017 Decision 3's
//! stack-evaluated depth-first fold ([`sync_grid_scope_stack`]) and folding one
//! CLOSED scope's composed body into its parent ([`fold_closed_scope_into`]). Shared
//! by the runtime chunk resolve and the cfg-gated dense oracle.

use voxel_core::voxel::VoxelGrid;

use super::*;
use crate::scene::*;

/// Sync the dense resolvers' **scope stack** to `target_path` — the stack-evaluated
/// depth-first fold of ADR 0017 Decision 3 (issue #74), reconstructed from each leaf's
/// carried [`ScopeFrame`] path (scopes are contiguous in the depth-first walk, so
/// comparing the open stack against the next leaf's path recovers the exact
/// scope-close / scope-open marker sequence).
///
/// Frames deeper than the common prefix CLOSE (innermost first): the popped scratch
/// grid — the scope's fully composed body so far — folds into its parent (the next
/// stack entry, or `root`) under the SCOPE's own operation via
/// [`fold_closed_scope_into`]. Frames beyond the common prefix OPEN: a fresh scratch
/// grid is pushed, so the scope's leaves compose sealed until it closes. Called once
/// per visited leaf and once with the empty path after the walk (closing everything).
///
/// For a pure-`Union` scene this is provably the identity transformation on the output
/// occupied list: a union close APPENDS the scratch voxels at exactly the walk position
/// the scope closed (before any later sibling stamped), preserving both the element
/// order and the later-wins material resolution of the flat pre-#74 walk — which is why
/// the pure-Union goldens hold byte-identical.
pub(super) fn sync_grid_scope_stack(
    stack: &mut Vec<(ScopeFrame, VoxelGrid)>,
    root: &mut VoxelGrid,
    target_path: &[ScopeFrame],
    accumulator_dimensions: [u32; 3],
) {
    // The longest prefix of open frames the target path keeps open.
    let mut common = 0;
    while common < stack.len()
        && common < target_path.len()
        && stack[common].0 == target_path[common]
    {
        common += 1;
    }
    // Close the scopes deeper than the common prefix, innermost first.
    while stack.len() > common {
        let (frame, closed) = stack.pop().expect("len checked by the loop condition");
        let parent: &mut VoxelGrid = match stack.last_mut() {
            Some((_, scratch)) => scratch,
            None => root,
        };
        fold_closed_scope_into(parent, frame.operation, closed);
    }
    // Open the target path's scopes beyond the common prefix, outermost first.
    for frame in &target_path[common..] {
        stack.push((*frame, VoxelGrid::new(accumulator_dimensions)));
    }
}

/// Fold one CLOSED scope's composed body into its parent accumulator under the scope's
/// own [`CombineOp`] (ADR 0017 Decision 3):
///
/// * `Union` — append the body's voxels. The parent's occupied list is later-wins on
///   overlap (last write persists downstream), and the body's voxels are appended at
///   the walk position the scope closed, so the union close reproduces the flat
///   depth-first later-wins order exactly.
/// * `Subtract` — an occupancy-only mask (ADR 0017 Decision 1): every parent voxel
///   whose integer index coincides with one of the body's occupied cells is REMOVED;
///   surviving voxels keep their material and overlay, and the body's materials never
///   enter the parent.
/// * `Intersect` — the complementary occupancy-only mask (issue #75): the parent KEEPS
///   ONLY the voxels whose index coincides with one of the body's occupied cells;
///   everything else dies, including cells far outside the body's AABB. A scope that
///   closed at the EMPTY body therefore annihilates its parent (`A ∩ ∅ = ∅`), matching
///   the substrate kernel's ∅ identity. Surviving voxels keep their material/overlay.
fn fold_closed_scope_into(parent: &mut VoxelGrid, operation: CombineOp, closed: VoxelGrid) {
    match operation {
        // A scope that folds under Emboss is pre-composed with its siblings into one
        // CompositeProducer, so a composed body never arrives here needing to read the
        // parent's field (`CombineOp::needs_accumulated_field`). Reaching this arm means the
        // scope declined to compose — see the matching arm in the leaf fold.
        CombineOp::Emboss { .. } => {
            eprintln!(
                "scene: skipping an Emboss scope close whose siblings could not be composed \
                 (there is no accumulated field to emboss)"
            );
        }
        CombineOp::Union => parent.occupied.extend(closed.occupied),
        CombineOp::Subtract => {
            let carved: std::collections::HashSet<[i32; 3]> = closed
                .occupied
                .iter()
                .map(|voxel| voxel.local_index)
                .collect();
            parent
                .occupied
                .retain(|voxel| !carved.contains(&voxel.local_index));
        }
        CombineOp::Intersect => {
            let kept: std::collections::HashSet<[i32; 3]> = closed
                .occupied
                .iter()
                .map(|voxel| voxel.local_index)
                .collect();
            parent
                .occupied
                .retain(|voxel| kept.contains(&voxel.local_index));
        }
    }
}
