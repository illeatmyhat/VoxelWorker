// The case the whole scheme exists for: a field is added to a state struct and nobody
// says where it goes. The `.stderr` beside this file is the message a developer meets
// at that moment, and is reviewed as an interface, not as test output.

use snapshot::Snapshot;

#[derive(Snapshot)]
struct WindowState {
    #[snapshot(settings)]
    window_size: [u32; 2],
    // Added later, classified by nobody.
    cursor_is_held: bool,
}

fn main() {}
