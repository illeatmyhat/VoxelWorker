//! Context supplied by the document when an authored value needs evaluation.
//!
//! Density belongs to the document's Scene, not to a sketch or a
//! measurement.  This small value is deliberately passed at an evaluation seam so a retained
//! source can be resolved without caching a second density authority in every consumer.

use std::num::NonZeroU32;

/// Non-persisted facts needed to evaluate authored values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvaluationContext {
    voxels_per_block: NonZeroU32,
}

impl EvaluationContext {
    /// Build the context from the document's already-validated density.
    #[must_use]
    pub const fn new(voxels_per_block: NonZeroU32) -> Self {
        Self { voxels_per_block }
    }

    /// The document density used for this one evaluation.
    #[must_use]
    pub const fn voxels_per_block(self) -> NonZeroU32 {
        self.voxels_per_block
    }
}
