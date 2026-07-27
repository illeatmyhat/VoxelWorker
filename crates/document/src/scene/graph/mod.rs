//! Node-graph model (ADR 0001 assembly graph, ADR 0003 Phase B stable ids): the
//! id-keyed arena and root spine, node paths & ids, structural edits (add / remove /
//! group / ungroup / definition / instance), and reference Points. Selection is NOT
//! here — ADR 0032 made it workspace state (`ui::panel::Selection`).

use super::*;

mod construct;
mod edits;
mod gizmo;
mod model;
mod navigate;

pub use model::{
    AssemblyDef, CombineOp, DefId, Node, NodeBuilder, NodeGrids, NodeId, NodePath, Point,
    ROOT_NODE_ID,
};
