// Every unclassified field is reported in one build. Retrofitting the derive onto an
// existing struct means meeting this error with a dozen fields at once, and a macro
// that surfaced them one recompile at a time would make that job miserable enough to
// discourage classifying anything.

use snapshot::Snapshot;

#[derive(Snapshot)]
struct PartlyClassified {
    #[snapshot(document)]
    body: u32,
    orbit_target: [f32; 3],
    view_mode: u8,
}

fn main() {}
