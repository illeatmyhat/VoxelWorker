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
}
