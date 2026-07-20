// The second half of the guarantee, and the one the derive cannot give (ADR 0022,
// amendment 2026-07-20). Classifying a field proves somebody DECIDED where it goes; it
// says nothing about whether the capture function took it there. This fixture is the
// missing step: a field that is properly classified, and reaches no artifact anyway,
// because the dump's capture does not mention it.
//
// The mechanism is exhaustive destructuring with no `..` rest pattern, so the error is
// rustc's own E0027 rather than anything this crate emits. That is the right shape for it
// — the guarantee is a property of the capture, not of the macro — but it means the
// message is only pinned here, and a refactor that reached for `..` "just to get it
// building" would silently delete the whole enforcement. This file is what notices.
//
// `src/artifacts.rs` is the production instance of exactly this pattern.

use snapshot::Snapshot;

#[derive(Snapshot)]
struct SessionState {
    #[snapshot(settings)]
    window_size: [u32; 2],
    // Classified, and therefore promised to reach the dump. Whether it does is decided
    // below, not here.
    #[snapshot(view)]
    orbit_target: [f32; 3],
}

/// The superset artifact. Every classified field that is not an escape hatch must arrive.
struct Dump {
    window_size: [u32; 2],
    orbit_target: [f32; 3],
}

impl Dump {
    fn from_state(state: &SessionState) -> Self {
        // No `..`: the pattern must name every field of `SessionState`, and it does not.
        let SessionState { window_size } = state;
        Dump {
            window_size: *window_size,
            orbit_target: [0.0, 0.0, 0.0],
        }
    }
}

fn main() {
    let _ = Dump::from_state(&SessionState {
        window_size: [1280, 800],
        orbit_target: [1.0, 2.0, 3.0],
    });
}
