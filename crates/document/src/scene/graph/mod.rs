//! Node-graph model and selection (ADR 0001 assembly graph, ADR 0003 Phase B
//! stable ids): the id-keyed arena and root spine, node paths & ids, structural
//! edits (add / remove / group / ungroup / definition / instance), reference
//! Points, and the active selection.

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
