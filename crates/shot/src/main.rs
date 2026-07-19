//! `shot` — the headless screenshot harness.
//!
//! Renders the SAME clear colour and the SAME egui panel as the windowed app
//! into an offscreen texture (no window, no surface), reads the pixels back, and
//! writes a PNG. This is the self-verification harness for every later
//! milestone: a milestone is "done" when its `shot` looks right.
//!
//! CLI (fenced as `text`: the `<u32>` / `<path>` placeholders are not HTML, and
//! rustdoc's `-D warnings` doc gate parses unfenced angle brackets as tags):
//!
//! ```text
//!   --out <path>     output PNG path        (default: shots/m1.png)
//!   --width <u32>    capture width          (default: 1280)
//!   --height <u32>   capture height         (default: 800)
//!   --shape <cylinder|tube|sphere|torus|box|debug-clouds>   (default: cylinder)
//!   --size-x <u32> --size-y <u32> --size-z <u32>   size in blocks (default 5/1/5)
//!   --density <u32>  voxels per block       (default: 16)
//!   --wall <u32>     tube wall in blocks    (default: 1)
//!   --proj <perspective|ortho>              (default: perspective)
//!   --material <stone|wood|plain>           (default: stone)
//!   --grid                                  enable the voxel/block grid overlay
//!   --debug-faces                           face-orientation debug render (colour
//!                                            by outward normal + back-face marker)
//!   --theta/--phi/--dist                    orbit overrides (auto-framed dist)
//! ```

mod capture;
mod demos;
mod options;

fn main() {
    let options = options::parse_options();
    pollster::block_on(capture::run_capture(options));
}
