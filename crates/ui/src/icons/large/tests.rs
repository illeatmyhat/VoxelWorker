//! Invariants tying the tile family to the rail family.

use super::*;
use std::collections::BTreeSet;

/// The catalogue cannot silently lose a glyph.
///
/// `ALL` is hand-maintained beside the enum, and nothing about a missing entry fails to
/// compile — the glyph just vanishes from every gallery that iterates the catalogue. This
/// pins it from the OTHER side: whatever [`LargeIcon::for_icon`] can produce is exactly what
/// `ALL` contains, so adding a tile glyph without listing it fails here.
#[test]
fn every_reachable_tile_glyph_is_in_the_catalogue() {
    let reachable: BTreeSet<&str> = Icon::ALL
        .iter()
        .filter_map(|icon| LargeIcon::for_icon(*icon))
        .map(|large| large.name())
        .collect();
    let catalogued: BTreeSet<&str> = LargeIcon::ALL.iter().map(|l| l.name()).collect();
    assert_eq!(
        reachable, catalogued,
        "LargeIcon::ALL and what for_icon can return must agree"
    );
}

/// No duplicate entries — a copy-paste slip when adding a glyph.
#[test]
fn the_catalogue_has_no_duplicates() {
    let unique: BTreeSet<&str> = LargeIcon::ALL.iter().map(|l| l.name()).collect();
    assert_eq!(unique.len(), LargeIcon::ALL.len());
}

/// `for_icon` and `rail` are inverses on the tile set, so the two families cannot drift
/// apart on which noun is which.
#[test]
fn for_icon_and_rail_round_trip() {
    for large in LargeIcon::ALL {
        assert_eq!(
            LargeIcon::for_icon(large.rail()),
            Some(*large),
            "{} must round-trip through its rail twin",
            large.name()
        );
    }
}

/// Only PRODUCERS get tile glyphs. A combine op or a chrome mark asking for one gets `None`
/// and falls back to its rail mark — that is the designed answer, not a gap.
#[test]
fn non_producers_have_no_tile_glyph() {
    for icon in [Icon::Union, Icon::Subtract, Icon::Emboss, Icon::Search, Icon::Part] {
        assert_eq!(LargeIcon::for_icon(icon), None, "{} is not a producer", icon.name());
    }
}

/// A tile glyph answers to the same noun as its rail twin — one name per idea across both
/// families, so a reader looking for "the mark for a revolve" finds one entry, at two sizes.
///
/// `sculpt` is the single deliberate exception: the rail set has no generic sculpt, having
/// split the verb into `sculpt-add` and `carve` so polarity is legible at 15 pt. Pinning the
/// exception as a list means a SECOND divergence has to be argued for, not just added.
#[test]
fn a_tile_glyph_shares_its_rail_twins_name() {
    let divergent: Vec<&str> = LargeIcon::ALL
        .iter()
        .filter(|l| l.name() != l.rail().name())
        .map(|l| l.name())
        .collect();
    assert_eq!(divergent, ["sculpt"]);
}

/// Every glyph paints without panicking, at both a tile size and a thumbnail size.
#[test]
fn every_glyph_paints_at_both_sizes() {
    let context = egui::Context::default();
    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(
            egui::Pos2::ZERO,
            egui::Vec2::new(200.0, 200.0),
        )),
        ..Default::default()
    };
    let _ = context.run_ui(raw_input, |ui| {
        for size in [26.0_f32, 56.0] {
            let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::Vec2::splat(size));
            for large in LargeIcon::ALL {
                large.draw(ui.painter(), rect, egui::Color32::WHITE);
            }
        }
    });
}
