//! The view cube's square, painted as an egui image so it has a tier.

/// The whole of the cube's texture — it is rendered at exactly the size it is drawn at.
const WHOLE_TEXTURE: egui::Rect = egui::Rect {
    min: egui::Pos2 { x: 0.0, y: 0.0 },
    max: egui::Pos2 { x: 1.0, y: 1.0 },
};

/// Paint the cube's rendered square at `rect`, on the chrome tier.
///
/// The cube is drawn on the GPU into its own square and handed here as a texture, rather than
/// composited onto the target by its own render pass. A composited corner is not in egui's z-order
/// AT ALL: it lands under every egui shape in the frame unconditionally, so a sketch mark that
/// reaches the corner paints straight over it and the cube has no tier to defend. As an image it
/// goes through the same door every other instrument does — over the drawing, under a menu, by
/// [`Order`](egui::Order) rather than by which pass ran first.
///
/// `NEAREST` filtering and a WHITE tint at the registration end, an unscaled rect at this end: the
/// image is a straight blit whenever the rect lands on whole pixels, which is what the `shot`
/// goldens compare.
pub fn view_cube_image(ui: &egui::Ui, rect: egui::Rect, cube: egui::TextureId) {
    ui.ctx()
        .layer_painter(super::chrome_layer("view_cube"))
        .image(cube, rect, WHOLE_TEXTURE, egui::Color32::WHITE);
}

#[cfg(test)]
mod tests {
    use crate::theme;

    /// The stand-in for a menu's own panel, found by its rect: an area FADES IN, so the fill it
    /// is asked for is not the fill that comes out.
    const A_MENU_PANEL: egui::Rect = egui::Rect {
        min: egui::Pos2 { x: 220.0, y: 60.0 },
        max: egui::Pos2 { x: 300.0, y: 100.0 },
    };
    /// The cube's texture in the probe frame — nothing else in the frame is a mesh with it.
    const A_CUBE_TEXTURE: egui::TextureId = egui::TextureId::User(1);

    /// Where the three marks came out, as indices into one frame's drained shapes — `None` for
    /// one that did not paint at all, which every caller has to rule out before comparing, since
    /// a missing mark orders below every present one.
    ///
    /// Draw order IS paint order: egui hands the backend a flat list and the backend blends it
    /// front to back, so a larger index is a shape drawn on top. All three go through the doors
    /// the shell uses — a real sketch mark, the real cube image, and a real `Area` raised on
    /// [`MENU_ORDER`](crate::chrome::MENU_ORDER) — because the tier is the whole subject and a
    /// probe that named its own order would be testing itself.
    fn a_frame_of_marquee_cube_and_menu() -> (Option<usize>, Option<usize>, Option<usize>) {
        let viewport = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(400.0, 300.0));
        let context = egui::Context::default();
        let input = egui::RawInput {
            screen_rect: Some(viewport),
            ..Default::default()
        };
        // Twice, and the second one is the frame under test: an `Area` whose size nothing has
        // measured yet spends its first frame being sized, and egui asks for that frame to be
        // thrown away rather than shown.
        let mut a_frame = |context: &egui::Context| {
            egui::CentralPanel::default().show(context, |ui| {
                // A marquee band is a sketch mark, and it reaches into the corner exactly the way
                // a drawing near the top-right of the viewport does.
                crate::chrome::sketch_marquee_band(ui, viewport, viewport.shrink(4.0), true);
                super::view_cube_image(
                    ui,
                    egui::Rect::from_min_size(egui::pos2(240.0, 16.0), egui::vec2(144.0, 144.0)),
                    A_CUBE_TEXTURE,
                );
                egui::Area::new(egui::Id::new("a_menu_over_the_cube"))
                    .order(crate::chrome::MENU_ORDER)
                    .fixed_pos(A_MENU_PANEL.min)
                    .show(&context.clone(), |ui| {
                        ui.painter()
                            .rect_filled(A_MENU_PANEL, 0.0, egui::Color32::RED);
                        ui.allocate_rect(A_MENU_PANEL, egui::Sense::hover());
                    });
            });
        };
        drop(context.run(input.clone(), &mut a_frame));
        let output = context.run(input, &mut a_frame);

        let (mut marquee, mut cube, mut menu) = (None, None, None);
        for (index, clipped) in output.shapes.iter().enumerate() {
            match &clipped.shape {
                egui::Shape::Rect(rect) if rect.fill == theme::MARQUEE_WINDOW_FILL => {
                    marquee = Some(index);
                }
                egui::Shape::Rect(rect) if rect.rect == A_MENU_PANEL => menu = Some(index),
                egui::Shape::Mesh(mesh) if mesh.texture_id == A_CUBE_TEXTURE => cube = Some(index),
                _ => {}
            }
        }
        (marquee, cube, menu)
    }

    /// **The cube covers the drawing.**
    ///
    /// It did not, for as long as the cube was a render pass: a pass that composites before egui
    /// runs is under EVERY egui shape in the frame, and a sketch mark clipped to the whole viewport
    /// reaches the top-right corner freely. No tier was involved, so nothing could be adjusted —
    /// the cube had to become something egui draws before it could be drawn in the right place.
    #[test]
    fn the_view_cube_covers_a_sketch_mark_that_reaches_its_corner() {
        let (marquee, cube, _) = a_frame_of_marquee_cube_and_menu();
        assert!(
            marquee.is_some() && marquee < cube,
            "the drawing paints at {marquee:?} and the cube at {cube:?}, so the mark is on top"
        );
    }

    /// **A menu covers the cube it was opened from.**
    ///
    /// Not by hoping the two land in a useful order: within one tier this application's instruments
    /// are bare layers and its menus are areas, and `GraphicLayers::drain` empties the areas FIRST
    /// — so a menu sharing the chrome tier is guaranteed to come out UNDERNEATH. The menus take the
    /// tier above for that reason.
    #[test]
    fn a_menu_covers_the_view_cube_it_was_opened_from() {
        let (_, cube, menu) = a_frame_of_marquee_cube_and_menu();
        assert!(
            cube.is_some() && cube < menu,
            "the cube paints at {cube:?} and the menu at {menu:?}, so the menu is buried"
        );
    }

    /// **The blit lands on whole pixels at a fractional device ratio too.**
    ///
    /// The cube's square is a number of PHYSICAL pixels and the corner it stands at is decided in
    /// physical pixels; only the last step divides by the ratio, because egui measures in points.
    /// That division is the whole of the exposure. If it left the rect a fraction of a pixel off,
    /// the sampler would be reading between texels for the entire square and the cube would come
    /// out softened — everywhere, evenly, with no edge to notice it by. The `shot` goldens cannot
    /// see this: the headless path is fixed at a ratio of one, where the division is a no-op.
    ///
    /// So the observable is the tessellated geometry rather than any pixel: every vertex the cube's
    /// mesh puts down, taken back into physical pixels, is a whole number. The ratios are the ones
    /// Windows actually offers — 125%, 150%, 175% — plus a corner that is not a multiple of any of
    /// them, since a corner that happened to divide evenly would prove nothing.
    #[test]
    fn the_view_cube_lands_on_whole_pixels_at_a_fractional_device_ratio() {
        // A cube edge and a corner in physical pixels, as `view_cube_corner` answers them.
        let (edge, corner) = (144.0_f32, egui::pos2(1751.0, 67.0));
        for ratio in [1.0_f32, 1.25, 1.5, 1.75, 2.0] {
            let context = egui::Context::default();
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(
                    egui::pos2(0.0, 0.0),
                    egui::vec2(1920.0 / ratio, 1080.0 / ratio),
                )),
                ..Default::default()
            };
            let output = context.run(input, |context| {
                egui::CentralPanel::default().show(context, |ui| {
                    super::view_cube_image(
                        ui,
                        egui::Rect::from_min_size(
                            egui::pos2(corner.x / ratio, corner.y / ratio),
                            egui::vec2(edge / ratio, edge / ratio),
                        ),
                        A_CUBE_TEXTURE,
                    );
                });
            });

            let mut vertices = 0_usize;
            for primitive in context.tessellate(output.shapes, ratio) {
                let egui::epaint::Primitive::Mesh(mesh) = primitive.primitive else {
                    continue;
                };
                if mesh.texture_id != A_CUBE_TEXTURE {
                    continue;
                }
                for vertex in &mesh.vertices {
                    let (across, down) = (vertex.pos.x * ratio, vertex.pos.y * ratio);
                    let off = (across - across.round())
                        .abs()
                        .max((down - down.round()).abs());
                    assert!(
                        off < 0.01,
                        "at {ratio}x the cube puts a corner at {across}, {down} — \
                         {off} of a pixel off the grid, so the whole square samples between texels"
                    );
                    vertices += 1;
                }
            }
            assert!(
                vertices >= 4,
                "at {ratio}x the cube did not tessellate at all"
            );
        }
    }
}
