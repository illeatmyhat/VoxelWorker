//! The reference sheet's content: the token table, the type specimens, the icon catalogue and
//! the widget vocabulary — every one of them read from the shipping source rather than restated.
//!
//! Layout rules are the design's own: zero corner radius, 1 px hairlines, flat fills, monospace
//! throughout, UPPERCASE micro-labels at ~2 px letter-spacing, and exactly one accent.

use egui::{Align, Color32, FontId, Layout, Painter, Pos2, Rect, RichText, Sense, Stroke, Ui, Vec2};
use ui::gizmos::{self, Axis, HandleState};
use ui::icons::large::LargeIcon;
use ui::icons::{Group, Icon};
use ui::signal_theme as tokens;

/// Page ground — a shade below the panel fill, so panels read as surfaces sitting on it.
const PAGE: Color32 = Color32::from_rgb(0x07, 0x08, 0x0a);
/// The sheet's measure: prose wraps here and rows span it, so the reference reads the same
/// whatever the window is dragged to.
const CONTENT_WIDTH: f32 = 980.0;
/// The glyph size the rail uses, and the size every mark must survive.
const RAIL_GLYPH: f32 = 15.0;
/// The glyph size a palette tile or drawer thumbnail uses.
const TILE_GLYPH: f32 = 30.0;

/// The sheet's own state: what the pointer is over, so the catalogue can show a live hover
/// state rather than a printed swatch of one.
#[derive(Default)]
pub struct Sheet {
    hovered: Option<&'static str>,
}

impl Sheet {
    /// Draw the whole reference.
    pub fn show(&mut self, ui: &mut Ui) {
        egui::CentralPanel::default()
            .frame(egui::Frame::new().fill(PAGE).inner_margin(egui::Margin {
                left: 26,
                right: 26,
                top: 22,
                bottom: 26,
            }))
            .show_inside(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    // A measured column. Inside a scroll area the available width is
                    // unbounded, so prose would never wrap and long rows would run off the
                    // right edge — the sheet sets its own measure instead.
                    ui.set_max_width(CONTENT_WIDTH);
                    self.masthead(ui);
                    ui.add_space(18.0);
                    self.rules_strip(ui);
                    ui.add_space(22.0);
                    section(ui, "Palette", "the tokens — meanings live in docs/design/colour-vocabulary.md");
                    self.palette(ui);
                    ui.add_space(22.0);
                    section(ui, "Type", "monospace throughout; UPPERCASE micro-labels at ~2 px tracking");
                    self.type_specimens(ui);
                    ui.add_space(22.0);
                    section(ui, "Widgets", "the app's own egui Style — not a mock of it");
                    self.widgets(ui);
                    ui.add_space(22.0);
                    section(ui, "Icons", "18-unit grid · 1.25 stroke · square caps · no rounding");
                    self.icons(ui);
                    ui.add_space(22.0);
                    section(ui, "Tiles", "the large producer glyphs — a different drawing of the same noun, 26-unit grid at 1.1 stroke");
                    self.tiles(ui);
                    ui.add_space(22.0);
                    section(ui, "Sketch gizmos", "on-canvas manipulators over the 3D plane — screen-space billboards at projected vertices (ADR 0028)");
                    self.sketch_gizmos(ui);
                    ui.add_space(22.0);
                    section(ui, "Sketch cursors", "the pointer's feedback while a sketch tool tracks — four distinct states");
                    self.sketch_cursors(ui);
                    ui.add_space(20.0);
                    self.footer(ui);
                });
            });
    }

    /// Title, and what this window is for.
    fn masthead(&self, ui: &mut Ui) {
        ui.label(
            RichText::new("SIGNAL — DESIGN REFERENCE")
                .font(FontId::monospace(13.0))
                .color(tokens::TEXT_PRIMARY),
        );
        ui.add_space(4.0);
        ui.label(
            RichText::new(
                "Painted by the app's own tokens, glyphs and Style. If a constant changes, this \
                 window changes with it — there is no second copy to drift.",
            )
            .font(FontId::monospace(10.0))
            .color(tokens::TEXT_FAINT),
        );
    }

    /// The invariants, as a row of bordered cells.
    fn rules_strip(&self, ui: &mut Ui) {
        ui.horizontal_wrapped(|ui| {
            for (key, value) in [
                ("GRID", "18 × 18 units"),
                ("STROKE", "1.25 · square · miter"),
                ("RADIUS", "0 — everywhere"),
                ("RAIL", "15 pt"),
                ("TILE", "30 pt"),
                ("ACCENT", "exactly one"),
            ] {
                let (rect, _) = ui.allocate_exact_size(Vec2::new(157.0, 40.0), Sense::hover());
                let painter = ui.painter_at(rect);
                painter.rect_filled(rect, 0.0, tokens::BG);
                painter.rect_stroke(
                    rect,
                    0.0,
                    Stroke::new(1.0_f32, tokens::BORDER),
                    egui::StrokeKind::Inside,
                );
                painter.text(
                    rect.left_top() + Vec2::new(10.0, 8.0),
                    Align2_LEFT_TOP,
                    key,
                    FontId::monospace(8.5),
                    tokens::TEXT_FAINT,
                );
                painter.text(
                    rect.left_top() + Vec2::new(10.0, 21.0),
                    Align2_LEFT_TOP,
                    value,
                    FontId::monospace(10.0),
                    tokens::TEXT_SECONDARY,
                );
            }
        });
    }

    /// Every token, with its hex and the meaning it is permitted to carry.
    fn palette(&self, ui: &mut Ui) {
        let rows: &[(&str, Color32, &str)] = &[
            ("bg", tokens::BG, "panel fill — the instrument surface"),
            ("border", tokens::BORDER, "hairline border, 1 px, outer"),
            ("rule", tokens::RULE, "hairline rule, inner divisions"),
            ("hover-bg", tokens::HOVER_BG, "row hover"),
            ("active-bg", tokens::ACTIVE_BG, "rail button hover"),
            ("text-primary", tokens::TEXT_PRIMARY, "values, names — what is read first"),
            ("text-secondary", tokens::TEXT_SECONDARY, "labels"),
            ("text-muted", tokens::TEXT_MUTED, "idle glyphs, secondary labels"),
            ("text-faint", tokens::TEXT_FAINT, "readouts, counts"),
            ("text-hint", tokens::TEXT_HINT, "hints — the quietest legible step"),
            ("accent", tokens::ACCENT, "ACTIVE · SELECTED · LIVE — and the onion haze. No valence: not 'good', not 'safe'"),
            ("accent-text", tokens::ACCENT_TEXT, "text on an accent fill"),
            ("warn", tokens::WARN, "subtraction and removal, plus genuine warnings · doubles as the X spatial axis"),
            ("axis-y", tokens::AXIS_Y, "Y spatial axis — green; the snap-guide triad is X warn · Y this · Z accent (ADR 0028)"),
            ("sketch-plane-fill", tokens::SKETCH_PLANE_FILL, "sketch working-plane fill — accent at low alpha, so the profile stays primary (ADR 0028)"),
            ("sketch-plane-grid", tokens::SKETCH_PLANE_GRID, "sketch plane fine grid lines — accent, quiet"),
            ("sketch-plane-grid-block", tokens::SKETCH_PLANE_GRID_BLOCK, "sketch plane block grid lines — accent, brighter, reads through the fine grid"),
        ];
        for (name, color, meaning) in rows {
            let (rect, _) = ui.allocate_exact_size(
                Vec2::new(ui.available_width().min(CONTENT_WIDTH), 22.0),
                Sense::hover(),
            );
            let painter = ui.painter_at(rect);
            let swatch = Rect::from_min_size(rect.left_top() + Vec2::new(0.0, 4.0), Vec2::new(34.0, 14.0));
            // A mid-grey backing under every swatch so a TRANSPARENT token (the sketch-plane
            // tints) shows its true weight rather than vanishing into the near-black page;
            // an opaque token covers it completely, so the backing is invisible there.
            painter.rect_filled(swatch, 0.0, Color32::from_gray(0x40));
            painter.rect_filled(swatch, 0.0, *color);
            painter.rect_stroke(
                swatch,
                0.0,
                Stroke::new(1.0_f32, tokens::BORDER),
                egui::StrokeKind::Inside,
            );
            painter.text(
                rect.left_top() + Vec2::new(46.0, 5.0),
                Align2_LEFT_TOP,
                name,
                FontId::monospace(10.0),
                tokens::TEXT_SECONDARY,
            );
            painter.text(
                rect.left_top() + Vec2::new(166.0, 5.0),
                Align2_LEFT_TOP,
                hex_of(*color),
                FontId::monospace(9.5),
                tokens::TEXT_FAINT,
            );
            painter.text(
                rect.left_top() + Vec2::new(246.0, 5.0),
                Align2_LEFT_TOP,
                *meaning,
                FontId::monospace(9.5),
                tokens::TEXT_MUTED,
            );
        }

        ui.add_space(10.0);
        ui.label(
            RichText::new(
                "AXES  X #d9603f · Y #7dba6a · Z #9cb4d8   —   spatial only, never respent \
                 elsewhere in a viewport",
            )
            .font(FontId::monospace(9.5))
            .color(tokens::TEXT_FAINT),
        );
        ui.label(
            RichText::new(
                "SECOND CHANNEL  prefer texture over a second hue: hatch = touches something \
                 not shown here · dashed = uncommitted · dimmed = excluded",
            )
            .font(FontId::monospace(9.5))
            .color(tokens::TEXT_FAINT),
        );
    }

    /// The type ramp, at the sizes the interface actually uses.
    fn type_specimens(&self, ui: &mut Ui) {
        for (size, role) in [
            (11.0_f32, "values, node names"),
            (10.5, "rows"),
            (10.0, "labels, segmented cells"),
            (9.5, "readouts, hints"),
            (9.0, "counts, the quietest step"),
        ] {
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(format!("{size:>4.1}"))
                        .font(FontId::monospace(9.0))
                        .color(tokens::TEXT_HINT),
                );
                ui.add_space(8.0);
                ui.label(
                    RichText::new("3 blocks 8 voxels · CORBEL·A · 128³ · 16 vx/blk")
                        .font(FontId::monospace(size))
                        .color(tokens::TEXT_PRIMARY),
                );
                ui.add_space(10.0);
                ui.label(
                    RichText::new(role)
                        .font(FontId::monospace(9.0))
                        .color(tokens::TEXT_HINT),
                );
            });
        }
        ui.add_space(8.0);
        tokens::section_heading(ui, "Section heading — the painted helper");
    }

    /// Widget specimens, drawn through the shipping Style.
    fn widgets(&self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label(RichText::new("button").font(FontId::monospace(10.0)).color(tokens::TEXT_MUTED));
            ui.add_space(8.0);
            let _ = ui.button("ACCEPT");
            let _ = ui.button("CANCEL");
        });
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label(RichText::new("field").font(FontId::monospace(10.0)).color(tokens::TEXT_MUTED));
            ui.add_space(8.0);
            let mut measurement = String::from("3 blocks 8 voxels");
            ui.add(egui::TextEdit::singleline(&mut measurement).desired_width(190.0));
            ui.label(
                RichText::new("= 56 vx")
                    .font(FontId::monospace(9.5))
                    .color(tokens::TEXT_FAINT),
            );
        });
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label(RichText::new("slider").font(FontId::monospace(10.0)).color(tokens::TEXT_MUTED));
            ui.add_space(8.0);
            let mut layer = 56.0_f32;
            ui.add(egui::Slider::new(&mut layer, 0.0..=128.0).show_value(false));
        });
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.label(RichText::new("selected row").font(FontId::monospace(10.0)).color(tokens::TEXT_MUTED));
            ui.add_space(8.0);
            let (rect, _) = ui.allocate_exact_size(Vec2::new(260.0, 22.0), Sense::hover());
            let painter = ui.painter_at(rect);
            painter.rect_filled(rect, 0.0, tokens::HOVER_BG);
            // Selection is an accent inset bar on the leading edge — never a glow.
            painter.rect_filled(
                Rect::from_min_size(rect.left_top(), Vec2::new(2.0, rect.height())),
                0.0,
                tokens::ACCENT,
            );
            painter.text(
                rect.left_top() + Vec2::new(12.0, 5.0),
                Align2_LEFT_TOP,
                "WALL·A",
                FontId::monospace(10.5),
                tokens::TEXT_PRIMARY,
            );
        });
    }

    /// The icon catalogue: every glyph at both sizes, with its meaning.
    fn icons(&mut self, ui: &mut Ui) {
        let groups = [
            Group::Navigation,
            Group::ViewerModes,
            Group::Combine,
            Group::Fields,
            Group::Producers,
            Group::Structure,
            Group::Tools,
            Group::Sketch,
            Group::Chrome,
        ];
        for group in groups {
            ui.add_space(12.0);
            ui.horizontal(|ui| {
                ui.label(
                    RichText::new(group.title().to_uppercase())
                        .font(FontId::monospace(9.5))
                        .color(tokens::ACCENT),
                );
                ui.add_space(10.0);
                ui.label(
                    RichText::new(group.subtitle())
                        .font(FontId::monospace(9.0))
                        .color(tokens::TEXT_HINT),
                );
            });
            ui.add_space(5.0);
            for icon in Icon::ALL.iter().filter(|i| i.group() == group) {
                self.icon_row(ui, *icon);
            }
        }
    }

    /// One catalogue row: the glyph at 15 pt and 30 pt, its name, and what it means. Hovering
    /// lifts it to the hover colour, so the three states are demonstrated rather than described.
    fn icon_row(&mut self, ui: &mut Ui, icon: Icon) {
        let width = ui.available_width().min(CONTENT_WIDTH);
        let (rect, response) = ui.allocate_exact_size(Vec2::new(width, 40.0), Sense::hover());
        let hovered = response.hovered();
        if hovered {
            self.hovered = Some(icon.name());
        }
        let painter = ui.painter_at(rect);
        if hovered {
            painter.rect_filled(rect, 0.0, tokens::HOVER_BG);
        }
        let color = if hovered {
            tokens::TEXT_HOVER
        } else {
            tokens::TEXT_MUTED
        };

        // 15 pt — the rail size. If a mark fails, it fails here.
        let small = Rect::from_min_size(
            rect.left_top() + Vec2::new(10.0, (40.0 - RAIL_GLYPH) * 0.5),
            Vec2::splat(RAIL_GLYPH),
        );
        icon.draw(&painter, small, color);
        // 30 pt — the tile size.
        let large = Rect::from_min_size(
            rect.left_top() + Vec2::new(40.0, (40.0 - TILE_GLYPH) * 0.5),
            Vec2::splat(TILE_GLYPH),
        );
        icon.draw(&painter, large, color);
        // …and once in the accent, the state a lit rail button wears.
        let lit = Rect::from_min_size(
            rect.left_top() + Vec2::new(84.0, (40.0 - RAIL_GLYPH) * 0.5),
            Vec2::splat(RAIL_GLYPH),
        );
        icon.draw(&painter, lit, tokens::ACCENT);

        painter.text(
            rect.left_top() + Vec2::new(116.0, 8.0),
            Align2_LEFT_TOP,
            icon.name(),
            FontId::monospace(10.0),
            tokens::TEXT_SECONDARY,
        );
        painter.text(
            rect.left_top() + Vec2::new(116.0, 22.0),
            Align2_LEFT_TOP,
            icon.note(),
            FontId::monospace(9.0),
            tokens::TEXT_HINT,
        );
        painter.line_segment(
            [rect.left_bottom(), rect.right_bottom()],
            Stroke::new(1.0_f32, tokens::RULE),
        );
    }

    /// The TILE catalogue: every large producer glyph at tile sizes, with its meaning. The tile
    /// family is a SEPARATE drawing of the same noun (26-unit grid), so it earns its own section
    /// rather than being the rail glyph scaled up.
    fn tiles(&mut self, ui: &mut Ui) {
        for tile in LargeIcon::ALL {
            self.tile_row(ui, *tile);
        }
    }

    /// One tile row: the glyph at 30 pt and 44 pt, once in the accent, its name and meaning.
    fn tile_row(&mut self, ui: &mut Ui, tile: LargeIcon) {
        let width = ui.available_width().min(CONTENT_WIDTH);
        let (rect, response) = ui.allocate_exact_size(Vec2::new(width, 54.0), Sense::hover());
        let hovered = response.hovered();
        if hovered {
            self.hovered = Some(tile.name());
        }
        let painter = ui.painter_at(rect);
        if hovered {
            painter.rect_filled(rect, 0.0, tokens::HOVER_BG);
        }
        let color = if hovered { tokens::TEXT_HOVER } else { tokens::TEXT_MUTED };
        // 30 pt — a shape cell.
        tile.draw(
            &painter,
            Rect::from_min_size(rect.left_top() + Vec2::new(8.0, 12.0), Vec2::splat(30.0)),
            color,
        );
        // 44 pt — a drawer thumbnail.
        tile.draw(
            &painter,
            Rect::from_min_size(rect.left_top() + Vec2::new(46.0, 5.0), Vec2::splat(44.0)),
            color,
        );
        // …and once lit, the selected shape cell.
        tile.draw(
            &painter,
            Rect::from_min_size(rect.left_top() + Vec2::new(100.0, 12.0), Vec2::splat(30.0)),
            tokens::ACCENT,
        );
        painter.text(
            rect.left_top() + Vec2::new(146.0, 11.0),
            Align2_LEFT_TOP,
            tile.name(),
            FontId::monospace(10.0),
            tokens::TEXT_SECONDARY,
        );
        let note = painter.layout(
            tile.note().to_string(),
            FontId::monospace(9.0),
            tokens::TEXT_HINT,
            width - 158.0,
        );
        painter.galley(rect.left_top() + Vec2::new(146.0, 25.0), note, tokens::TEXT_HINT);
        painter.line_segment(
            [rect.left_bottom(), rect.right_bottom()],
            Stroke::new(1.0_f32, tokens::RULE),
        );
    }

    /// The sketch gizmo specimens — each composed from [`ui::gizmos`] on a flat reference plane.
    fn sketch_gizmos(&mut self, ui: &mut Ui) {
        // profile vertex handle — the four states in a row.
        self.specimen_row(
            ui,
            "profile vertex handle",
            "The load-bearing manipulator: a draggable, snapped vertex, billboarded at the projected \
             point. Idle · hover · selected · snapped.",
            |p, s| {
                let y = s.center().y - 4.0;
                let xs = [s.left() + 40.0, s.left() + 80.0, s.left() + 118.0, s.left() + 156.0];
                let states = [
                    (HandleState::Idle, "IDLE"),
                    (HandleState::Hover, "HOVER"),
                    (HandleState::Selected, "SEL"),
                    (HandleState::Snapped, "SNAP"),
                ];
                for (x, (state, tag)) in xs.iter().zip(states) {
                    gizmos::vertex_handle(p, Pos2::new(*x, y), 3.5, state);
                    p.text(
                        Pos2::new(*x, s.bottom() - 16.0),
                        egui::Align2::CENTER_TOP,
                        tag,
                        FontId::monospace(7.5),
                        tokens::TEXT_FAINT,
                    );
                }
            },
        );
        // active / open segment.
        self.specimen_row(
            ui,
            "active / open segment",
            "The real segment from the last committed vertex to the live cursor (hollow diamond). \
             Solid — a segment you're placing, not a rubber band.",
            |p, s| {
                let v1 = Pos2::new(s.left() + 34.0, s.bottom() - 26.0);
                let v2 = Pos2::new(s.left() + 78.0, s.top() + 28.0);
                let cur = Pos2::new(s.right() - 44.0, s.center().y);
                gizmos::segment(p, v1, v2);
                gizmos::open_segment(p, v2, cur, 4.0);
                gizmos::vertex_handle(p, v1, 3.5, HandleState::Idle);
                gizmos::vertex_handle(p, v2, 3.5, HandleState::Idle);
            },
        );
        // close-loop affordance.
        self.specimen_row(
            ui,
            "close-loop affordance",
            "Near the start vertex, it grows an accent ring and the closing run goes dashed — an \
             unmistakable 'click here to close'.",
            |p, s| {
                let start = Pos2::new(s.left() + 44.0, s.bottom() - 28.0);
                let p1 = Pos2::new(s.left() + 44.0, s.top() + 28.0);
                let p2 = Pos2::new(s.right() - 58.0, s.top() + 28.0);
                let cur = Pos2::new(s.right() - 60.0, s.center().y + 6.0);
                gizmos::segment(p, start, p1);
                gizmos::segment(p, p1, p2);
                gizmos::segment(p, p2, cur);
                gizmos::dashed_segment(p, cur, start);
                gizmos::close_loop_ring(p, start, 8.5);
                gizmos::vertex_handle(p, p1, 3.5, HandleState::Idle);
                gizmos::vertex_handle(p, p2, 3.5, HandleState::Idle);
                gizmos::vertex_handle(p, start, 3.5, HandleState::Selected);
                gizmos::diamond(p, cur, 4.0);
            },
        );
        // snap · grid.
        self.specimen_row(
            ui,
            "snap indicator · grid",
            "A vertex engaged the lattice: an accent tick-cross on the locked crossing, the quantum \
             named. Quantization delivers axis-alignment for free (§5).",
            |p, s| {
                let c = Pos2::new(s.center().x - 24.0, s.center().y);
                gizmos::crosshair(p, c, 16.0, tokens::ACCENT, true);
                gizmos::vertex_handle(p, c, 3.5, HandleState::Selected);
                gizmos::label_chip(p, Pos2::new(c.x + 11.0, c.y - 24.0), "voxel", tokens::ACCENT);
            },
        );
        // snap · vertex.
        self.specimen_row(
            ui,
            "snap indicator · vertex",
            "The vertex caught another (coincidence). An accent diamond rings the caught point; the \
             chip names it — the constraint a solver would call coincident.",
            |p, s| {
                let c = Pos2::new(s.center().x - 20.0, s.center().y);
                gizmos::diamond(p, c, 6.5);
                gizmos::vertex_handle(p, c, 3.5, HandleState::Selected);
                gizmos::label_chip(p, Pos2::new(c.x + 13.0, c.y - 24.0), "vertex n2", tokens::ACCENT);
            },
        );
        // snap · axis.
        self.specimen_row(
            ui,
            "snap indicator · axis",
            "The vertex aligned to an in-plane axis: a dashed extension line in the AXIS colour \
             (X warn · Y green) runs through it — axis-lock as a by-product of the lattice.",
            |p, s| {
                let c = Pos2::new(s.center().x + 8.0, s.center().y);
                gizmos::axis_guide(p, Pos2::new(s.left() + 16.0, c.y), Pos2::new(s.right() - 16.0, c.y), Axis::X);
                gizmos::vertex_handle(p, c, 3.5, HandleState::Selected);
                gizmos::label_chip(p, Pos2::new(c.x + 11.0, c.y - 24.0), "x-axis", Axis::X.color());
            },
        );
    }

    /// The sketch cursor specimens — the pointer's four feedback states.
    fn sketch_cursors(&mut self, ui: &mut Ui) {
        // on-plane / place-point (the one kept ghost).
        self.specimen_row(
            ui,
            "on-plane / place-point",
            "Over the plane, before the first point. The ONE ghost: a dashed crosshair + hollow node \
             showing where the snap will drop a vertex.",
            |p, s| {
                let c = Pos2::new(s.center().x, s.center().y - 4.0);
                gizmos::crosshair(p, c, 15.0, tokens::ACCENT, true);
                gizmos::ghost_node(p, c, 3.5);
                p.text(
                    Pos2::new(c.x, c.y + 22.0),
                    egui::Align2::CENTER_TOP,
                    "A POINT LANDS HERE",
                    FontId::monospace(7.5),
                    tokens::TEXT_MUTED,
                );
            },
        );
        // grab-vertex.
        self.specimen_row(
            ui,
            "grab-vertex",
            "Hovering an existing vertex: the handle lights (brighter border) — 'this is draggable', \
             distinct from empty-plane hover.",
            |p, s| {
                let c = s.center();
                gizmos::segment(p, Pos2::new(s.left() + 30.0, s.bottom() - 24.0), c);
                gizmos::segment(p, c, Pos2::new(s.right() - 40.0, s.top() + 26.0));
                gizmos::vertex_handle(p, c, 4.0, HandleState::Hover);
                p.text(
                    Pos2::new(c.x + 12.0, c.y + 10.0),
                    Align2_LEFT_TOP,
                    "GRAB",
                    FontId::monospace(7.5),
                    tokens::TEXT_MUTED,
                );
            },
        );
        // close-loop.
        self.specimen_row(
            ui,
            "close-loop",
            "Near the start vertex with an open polyline: the ring + dashed closing run say 'clicking \
             closes the profile'.",
            |p, s| {
                let start = Pos2::new(s.left() + 50.0, s.bottom() - 26.0);
                let p1 = Pos2::new(s.left() + 50.0, s.top() + 28.0);
                let p2 = Pos2::new(s.right() - 70.0, s.top() + 28.0);
                let cur = Pos2::new(s.center().x + 6.0, s.center().y + 2.0);
                gizmos::segment(p, start, p1);
                gizmos::segment(p, p1, p2);
                gizmos::segment(p, p2, cur);
                gizmos::dashed_segment(p, cur, start);
                gizmos::close_loop_ring(p, start, 9.0);
                gizmos::vertex_handle(p, start, 3.5, HandleState::Selected);
                gizmos::diamond(p, cur, 4.0);
            },
        );
        // snap-engaged.
        self.specimen_row(
            ui,
            "snap-engaged",
            "A candidate snap (grid / vertex / axis) is live: the tick-cross + chip say WHAT it locked \
             to — the vocabulary a single grey dot would destroy.",
            |p, s| {
                let c = Pos2::new(s.center().x - 12.0, s.center().y);
                gizmos::axis_guide(p, Pos2::new(s.left() + 16.0, c.y), Pos2::new(s.right() - 16.0, c.y), Axis::Y);
                gizmos::crosshair(p, c, 14.0, tokens::ACCENT, true);
                gizmos::vertex_handle(p, c, 3.5, HandleState::Selected);
                gizmos::label_chip(p, Pos2::new(c.x + 11.0, c.y - 25.0), "y-axis + voxel", tokens::ACCENT);
            },
        );
    }

    /// One specimen row: a flat plane-grid stage with the gizmo composed on it, then its name and
    /// meaning. The stage is a FLAT reference of the working plane — the sheet has no camera, so it
    /// stands in for the 3D plane the live gizmos billboard over (see [`ui::gizmos`]).
    fn specimen_row(
        &mut self,
        ui: &mut Ui,
        name: &str,
        note: &str,
        draw: impl FnOnce(&Painter, Rect),
    ) {
        let width = ui.available_width().min(CONTENT_WIDTH);
        let (rect, _) = ui.allocate_exact_size(Vec2::new(width, 104.0), Sense::hover());
        let painter = ui.painter_at(rect);
        let stage = Rect::from_min_size(rect.left_top() + Vec2::new(0.0, 6.0), Vec2::new(206.0, 92.0));
        plane_stage(&painter, stage);
        draw(&painter, stage);
        let text_x = 224.0;
        painter.text(
            rect.left_top() + Vec2::new(text_x, 14.0),
            Align2_LEFT_TOP,
            name,
            FontId::monospace(10.5),
            tokens::TEXT_SECONDARY,
        );
        let note_galley = painter.layout(
            note.to_string(),
            FontId::monospace(9.0),
            tokens::TEXT_MUTED,
            width - text_x - 12.0,
        );
        painter.galley(rect.left_top() + Vec2::new(text_x, 32.0), note_galley, tokens::TEXT_MUTED);
        painter.line_segment(
            [rect.left_bottom(), rect.right_bottom()],
            Stroke::new(1.0_f32, tokens::RULE),
        );
    }

    /// What is on screen, and where the words for it live.
    fn footer(&self, ui: &mut Ui) {
        ui.with_layout(Layout::left_to_right(Align::Min), |ui| {
            ui.label(
                RichText::new(format!(
                    "{} glyphs · {} tiles · tokens from crates/ui/src/signal_theme.rs · glyphs \
                     from crates/ui/src/icons/ · gizmos from crates/ui/src/gizmos/",
                    Icon::ALL.len(),
                    LargeIcon::ALL.len()
                ))
                .font(FontId::monospace(9.0))
                .color(tokens::TEXT_HINT),
            );
        });
        ui.label(
            RichText::new(
                "spec: docs/design/viewport-chrome-signal.md · meanings: \
                 docs/design/colour-vocabulary.md",
            )
            .font(FontId::monospace(9.0))
            .color(tokens::TEXT_HINT),
        );
    }
}

/// A section header: an accent title over a hairline, with a faint subtitle.
fn section(ui: &mut Ui, title: &str, subtitle: &str) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(title.to_uppercase())
                .font(FontId::monospace(10.0))
                .color(tokens::ACCENT),
        );
        ui.add_space(10.0);
        ui.label(
            RichText::new(subtitle)
                .font(FontId::monospace(9.0))
                .color(tokens::TEXT_HINT),
        );
    });
    ui.add_space(4.0);
    let (rect, _) = ui.allocate_exact_size(Vec2::new(ui.available_width().min(CONTENT_WIDTH), 1.0), Sense::hover());
    ui.painter_at(rect).line_segment(
        [rect.left_top(), rect.right_top()],
        Stroke::new(1.0_f32, tokens::BORDER),
    );
    ui.add_space(8.0);
}

/// A flat plane-grid stage for a gizmo specimen: the dark viewport ground, the pale accent plane
/// tint, and a fine grid with every third line a block line. A FLAT reference of the working plane
/// — the sheet has no camera, so it stands in for the 3D plane the live gizmos billboard over.
fn plane_stage(painter: &Painter, rect: Rect) {
    painter.rect_filled(rect, 0.0, Color32::from_rgb(0x15, 0x19, 0x1d));
    painter.rect_filled(rect, 0.0, tokens::SKETCH_PLANE_FILL);
    let step = 15.0;
    let (mut i, mut x) = (1, rect.left() + step);
    while x < rect.right() - 0.5 {
        let color = if i % 3 == 0 { tokens::SKETCH_PLANE_GRID_BLOCK } else { tokens::SKETCH_PLANE_GRID };
        painter.line_segment([Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())], Stroke::new(1.0_f32, color));
        x += step;
        i += 1;
    }
    let (mut j, mut y) = (1, rect.top() + step);
    while y < rect.bottom() - 0.5 {
        let color = if j % 3 == 0 { tokens::SKETCH_PLANE_GRID_BLOCK } else { tokens::SKETCH_PLANE_GRID };
        painter.line_segment([Pos2::new(rect.left(), y), Pos2::new(rect.right(), y)], Stroke::new(1.0_f32, color));
        y += step;
        j += 1;
    }
    painter.rect_stroke(rect, 0.0, Stroke::new(1.0_f32, tokens::BORDER), egui::StrokeKind::Inside);
}

/// `#rrggbb` for a token — or `#rrggbbaa` when it carries alpha, so a transparent tint states the
/// weight a stylesheet would need, not just its hue.
fn hex_of(color: Color32) -> String {
    // The UNMULTIPLIED channels — Color32 stores premultiplied alpha, so a low-alpha tint's
    // `.r()/.g()/.b()` are darkened; `to_srgba_unmultiplied` recovers the authored hue.
    let [r, g, b, a] = color.to_srgba_unmultiplied();
    if a == 255 {
        format!("#{r:02x}{g:02x}{b:02x}")
    } else {
        format!("#{r:02x}{g:02x}{b:02x}{a:02x}")
    }
}

/// egui's `Align2::LEFT_TOP`, aliased so the painter calls read as a column of text rather than
/// a column of paths.
#[allow(non_upper_case_globals)]
const Align2_LEFT_TOP: egui::Align2 = egui::Align2::LEFT_TOP;
