//! Relocated to [`crate::store`] in slice A2b. This module is now a thin
//! re-export shim so existing `chunk_cache::*` call sites keep compiling
//! until later slices migrate them. New code should use `crate::store`.
pub use crate::store::{ChunkCacheKey, ChunkResolveCache, Store};
