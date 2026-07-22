//! The reference sheet's content: the token table, the type specimens, the icon catalogue and
//! the widget vocabulary — every one of them read from the shipping source rather than restated.
//!
//! Layout rules are the design's own: zero corner radius, 1 px hairlines, flat fills, monospace
//! throughout, UPPERCASE micro-labels at ~2 px letter-spacing, and exactly one accent.

use egui::{Align, Color32, FontId, Layout, Rect, RichText, Sense, Stroke, Ui, Vec2};
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
            ("warn", tokens::WARN, "subtraction and removal, plus genuine warnings"),
        ];
        for (name, color, meaning) in rows {
            let (rect, _) = ui.allocate_exact_size(
                Vec2::new(ui.available_width().min(CONTENT_WIDTH), 22.0),
                Sense::hover(),
            );
            let painter = ui.painter_at(rect);
            let swatch = Rect::from_min_size(rect.left_top() + Vec2::new(0.0, 4.0), Vec2::new(34.0, 14.0));
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

    /// What is on screen, and where the words for it live.
    fn footer(&self, ui: &mut Ui) {
        ui.with_layout(Layout::left_to_right(Align::Min), |ui| {
            ui.label(
                RichText::new(format!(
                    "{} glyphs · tokens from crates/ui/src/signal_theme.rs · glyphs from \
                     crates/ui/src/icons/",
                    Icon::ALL.len()
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

/// `#rrggbb` for a token, so the sheet states the value a stylesheet would need.
fn hex_of(color: Color32) -> String {
    format!("#{:02x}{:02x}{:02x}", color.r(), color.g(), color.b())
}

/// egui's `Align2::LEFT_TOP`, aliased so the painter calls read as a column of text rather than
/// a column of paths.
#[allow(non_upper_case_globals)]
const Align2_LEFT_TOP: egui::Align2 = egui::Align2::LEFT_TOP;
