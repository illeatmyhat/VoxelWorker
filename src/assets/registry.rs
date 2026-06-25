//! Source registry — runs every known detector and aggregates results (M6).
//!
//! This is the single entry point the app uses for auto-detection: it owns the
//! list of [`SourceDetector`]s (currently just Vintage Story; another game's
//! detector would slot in here) and returns the union of every [`BlockSource`]
//! they find, with no user action.

use super::{vintage_story::VintageStoryDetector, BlockSource, SourceDetector};

/// Run all known detectors and return every source they found.
pub fn detect_all_sources() -> Vec<Box<dyn BlockSource>> {
    let detectors: Vec<Box<dyn SourceDetector>> = vec![Box::new(VintageStoryDetector)];
    let mut sources = Vec::new();
    for detector in detectors {
        sources.extend(detector.detect());
    }
    sources
}
