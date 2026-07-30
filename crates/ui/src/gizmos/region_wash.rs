//! `region_wash` — a PICKED sketch region reads as a transparent fill over the face (ADR 0030 §3).
//!
//! The 2D counterpart of the 3D selection wash: what is picked is what will resolve as material, so
//! the face itself carries the signal and an unpicked face carries none. This replaces the centroid
//! badge, whose round mark landed beside an arc's own centre point and read as a second, mystery
//! handle.
//!
//! `egui` fills convex polygons only, so the wash is an [`egui::Mesh`] over a fan from
//! [`substrate::geom2d::triangulate_polygon_with_holes`] — concave outlines and voids alike. The
//! shell passes RESOLVED material pieces, not faces, so no two washes cover the same place and the
//! alpha composites exactly once.

use egui::{Painter, Pos2};

use crate::theme::color_palette;

/// Wash the material bounded by `outer` and outside every one of `holes` (egui points, in order,
/// each implicitly closed).
pub fn region_wash(painter: &Painter, outer: &[Pos2], holes: &[Vec<Pos2>]) {
    let to_polygon = |points: &[Pos2]| -> Vec<[f64; 2]> {
        points.iter().map(|it| [it.x as f64, it.y as f64]).collect()
    };
    let (vertices, fan) = substrate::geom2d::triangulate_polygon_with_holes(
        &to_polygon(outer),
        &holes
            .iter()
            .map(|hole| to_polygon(hole))
            .collect::<Vec<_>>(),
    );
    if fan.is_empty() {
        return;
    }
    let mut mesh = egui::Mesh::default();
    for vertex in &vertices {
        mesh.colored_vertex(
            Pos2::new(vertex[0] as f32, vertex[1] as f32),
            color_palette::SKETCH_REGION_FILL,
        );
    }
    for [a, b, c] in fan {
        mesh.add_triangle(a as u32, b as u32, c as u32);
    }
    painter.add(egui::Shape::mesh(mesh));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The wash of a CONCAVE face covers the face and not the notch — the case that ruled out
    /// `rect_filled`/`convex_polygon` and made this a mesh.
    #[test]
    fn a_concave_region_washes_as_a_triangle_fan() {
        let ell = [
            Pos2::new(0.0, 0.0),
            Pos2::new(40.0, 0.0),
            Pos2::new(40.0, 20.0),
            Pos2::new(20.0, 20.0),
            Pos2::new(20.0, 40.0),
            Pos2::new(0.0, 40.0),
        ];
        let context = egui::Context::default();
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                Pos2::ZERO,
                egui::Vec2::splat(100.0),
            )),
            ..Default::default()
        };
        let output = context.run_ui(raw_input, |ui| {
            region_wash(ui.painter(), &ell, &[]);
            region_wash(ui.painter(), &ell[..2], &[]);
        });
        let meshes: Vec<&egui::Mesh> = output
            .shapes
            .iter()
            .filter_map(|clipped| match &clipped.shape {
                egui::Shape::Mesh(mesh) => Some(&**mesh),
                _ => None,
            })
            .collect();
        assert_eq!(meshes.len(), 1, "a two-point outline washes nothing");
        let mesh = meshes[0];
        assert_eq!(mesh.vertices.len(), ell.len());
        assert_eq!(mesh.indices.len(), 3 * (ell.len() - 2));
        assert!(mesh
            .vertices
            .iter()
            .all(|vertex| vertex.color == color_palette::SKETCH_REGION_FILL));
    }

    /// A void inside a washed piece stays unwashed — the wash mirrors what resolves as material, and
    /// alpha over a hole would claim material that is not there.
    #[test]
    fn a_void_is_left_out_of_the_wash() {
        let outer = [
            Pos2::new(0.0, 0.0),
            Pos2::new(40.0, 0.0),
            Pos2::new(40.0, 40.0),
            Pos2::new(0.0, 40.0),
        ];
        let hole = vec![
            Pos2::new(10.0, 10.0),
            Pos2::new(10.0, 30.0),
            Pos2::new(30.0, 30.0),
            Pos2::new(30.0, 10.0),
        ];
        let context = egui::Context::default();
        let raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                Pos2::ZERO,
                egui::Vec2::splat(100.0),
            )),
            ..Default::default()
        };
        let output = context.run_ui(raw_input, |ui| {
            region_wash(ui.painter(), &outer, std::slice::from_ref(&hole));
        });
        let mesh = output
            .shapes
            .iter()
            .find_map(|clipped| match &clipped.shape {
                egui::Shape::Mesh(mesh) => Some(&**mesh),
                _ => None,
            })
            .expect("a mesh");
        let covered: f32 = mesh
            .indices
            .chunks_exact(3)
            .map(|triple| {
                let corner = |index: u32| mesh.vertices[index as usize].pos;
                let (a, b, c) = (corner(triple[0]), corner(triple[1]), corner(triple[2]));
                ((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)).abs() / 2.0
            })
            .sum();
        assert!(
            (covered - (1600.0 - 400.0)).abs() < 0.01,
            "the ring's area, not the square's: {covered}"
        );
    }
}
