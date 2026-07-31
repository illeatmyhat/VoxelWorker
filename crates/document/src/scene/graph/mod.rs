//! Node-graph model: the id-keyed arena and root spine, node paths & ids, structural
//! edits (add / remove / group / ungroup / definition / instance), and reference Points.
//! Selection is NOT here — it is workspace state (`ui::panel::Selection`).

use super::*;

mod construct;
mod edits;
mod gizmo;
mod model;
mod navigate;

pub use model::{
    AssemblyDef, CombineOp, DefId, LeafOrigin, Node, NodeBuilder, NodeGrids, NodeId, NodePath,
    Point, PointId, ROOT_NODE_ID,
};
