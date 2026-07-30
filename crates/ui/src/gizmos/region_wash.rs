//! `region_wash` — a PICKED sketch region reads as a transparent fill over the face (ADR 0030 §3).
//!
//! The 2D counterpart of the 3D selection wash: what is picked is what will resolve as material, so
//! the face itself carries the signal and an unpicked face carries none. This replaces the centroid
//! badge, whose round mark landed beside an arc's own centre point and read as a second, mystery
//! handle.
//!
//! `egui` fills convex polygons only, so the wash is an [`egui::Mesh`] over an ear-clipped fan —
//! every face of a planar graph is a simple polygon, which is exactly what
//! [`substrate::geom2d::triangulate_simple_polygon`] fans.

use egui::{Painter, Pos2};

use crate::theme::color_palette;

/// Wash the region enclosed by `boundary` (egui points, in order, implicitly closed).
pub fn region_wash(painter: &Painter, boundary: &[Pos2]) {
    let polygon: Vec<[f64; 2]> = boundary
        .iter()
        .map(|point| [point.x as f64, point.y as f64])
        .collect();
    let fan = substrate::geom2d::triangulate_simple_polygon(&polygon);
    if fan.is_empty() {
        return;
    }
    let mut mesh = egui::Mesh::default();
    for &point in boundary {
        mesh.colored_vertex(point, color_palette::SKETCH_REGION_FILL);
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
            region_wash(ui.painter(), &ell);
            region_wash(ui.painter(), &ell[..2]);
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
}
