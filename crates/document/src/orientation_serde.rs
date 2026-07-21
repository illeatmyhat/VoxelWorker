//! The serde adapter for [`substrate::spatial::LatticeOrientation`] (ADR 0026).
//!
//! Substrate keeps the type serde-free (the crate's boundary law), so the document owns its
//! on-disk form. The stable codec is the type's **gather form** (`source` permutation + per-axis
//! `sign`), re-validated on load through `from_gather` so a corrupt or hand-edited file can
//! never mint an invalid turn (a reflection or a non-permutation). Used by both the persistent
//! `NodeTransform::orientation` field and the replayable `Intent::PlaceNode` orientation, via
//! `#[serde(with = "crate::orientation_serde")]`.

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use substrate::spatial::LatticeOrientation;

#[derive(Serialize, Deserialize)]
struct Gather {
    source: [u8; 3],
    sign: [i8; 3],
}

pub fn serialize<S: Serializer>(
    orientation: &LatticeOrientation,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let (source, sign) = orientation.to_gather();
    Gather { source, sign }.serialize(serializer)
}

pub fn deserialize<'de, D: Deserializer<'de>>(
    deserializer: D,
) -> Result<LatticeOrientation, D::Error> {
    let gather = Gather::deserialize(deserializer)?;
    LatticeOrientation::from_gather(gather.source, gather.sign)
        .ok_or_else(|| serde::de::Error::custom("not a proper lattice orientation"))
}
