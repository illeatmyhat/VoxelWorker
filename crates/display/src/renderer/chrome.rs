//! #13 Step 2 — ViewCube chrome overlay (Home/Fit + hover rotate/roll arrows).
//! Screen-space, fixed to the cube rect; the layout fractions mirror
//! `camera::classify_cube_point` EXACTLY so the rendered glyphs sit on the Step-1
//! hit zones. Also owns the CPU rasterisation of the glyph textures.

use super::*;

/// Edge length of each square chrome-glyph texture (Home/Fit badges, rotate/roll
/// arrows). Smaller than the face labels — the glyphs are drawn at modest screen
/// sizes in the margins.
const CHROME_GLYPH_TEXTURE_SIZE: u32 = 64;

/// One screen-space chrome-overlay vertex: NDC position (fixed to the cube rect,
/// it does NOT rotate with the cube), glyph UV, a per-vertex tint (used to
/// brighten a hovered arrow), and the glyph texture-array layer.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct ChromeVertex {
    pub(crate) position: [f32; 2],
    pub(crate) uv: [f32; 2],
    pub(crate) color: [f32; 4],
    pub(crate) layer: u32,
}

/// The chrome-glyph texture-array layers (#13 Step 2), in upload order. The
/// Home/Fit badges are ALWAYS drawn; the arrows are drawn only when the matching
/// zone is hovered.
#[derive(Debug, Clone, Copy)]
enum ChromeGlyph {
    HomeButton,
    FitButton,
    ArrowUp,
    ArrowDown,
    ArrowLeft,
    ArrowRight,
    RollCw,
    RollCcw,
}

impl ChromeGlyph {
    /// Upload/lookup order for the texture array (must match `chrome_glyph_pixels`).
    const ALL: [ChromeGlyph; 8] = [
        ChromeGlyph::HomeButton,
        ChromeGlyph::FitButton,
        ChromeGlyph::ArrowUp,
        ChromeGlyph::ArrowDown,
        ChromeGlyph::ArrowLeft,
        ChromeGlyph::ArrowRight,
        ChromeGlyph::RollCw,
        ChromeGlyph::RollCcw,
    ];

    /// This glyph's index in the texture array.
    fn layer(self) -> u32 {
        self as u32
    }
}

/// Render one chrome glyph into an RGBA8 buffer (`CHROME_GLYPH_TEXTURE_SIZE`
/// square) with a TRANSPARENT background so the glyph floats over the scene; the
/// opaque pixels are white (tinted to parchment/teal by the vertex colour).
fn chrome_glyph_pixels(glyph: ChromeGlyph) -> Vec<u8> {
    let size = CHROME_GLYPH_TEXTURE_SIZE as usize;
    let mut pixels = vec![0u8; size * size * 4]; // transparent
    match glyph {
        ChromeGlyph::HomeButton => draw_home_icon(&mut pixels, size),
        ChromeGlyph::FitButton => draw_fit_icon(&mut pixels, size),
        ChromeGlyph::ArrowUp => draw_triangle_arrow(&mut pixels, size, ArrowFacing::Up),
        ChromeGlyph::ArrowDown => draw_triangle_arrow(&mut pixels, size, ArrowFacing::Down),
        ChromeGlyph::ArrowLeft => draw_triangle_arrow(&mut pixels, size, ArrowFacing::Left),
        ChromeGlyph::ArrowRight => draw_triangle_arrow(&mut pixels, size, ArrowFacing::Right),
        ChromeGlyph::RollCw => draw_roll_arc(&mut pixels, size, true),
        ChromeGlyph::RollCcw => draw_roll_arc(&mut pixels, size, false),
    }
    pixels
}

/// Which way a rotate-arrow triangle points.
#[derive(Clone, Copy)]
enum ArrowFacing {
    Up,
    Down,
    Left,
    Right,
}

/// Draw a clean filled triangular rotate arrow pointing in `facing`, centred.
/// #13 Step 6.3: a crisp equilateral-ish head (apex ~78% across the box, base
/// ~28%..72%) reads as a sharp directional cue at the small gutter size, with
/// anti-aliased edges from `fill_triangle`.
fn draw_triangle_arrow(pixels: &mut [u8], size: usize, facing: ArrowFacing) {
    const INK: [u8; 4] = [0xff, 0xff, 0xff, 0xff];
    let s = size as f32;
    let apex = s * 0.22; // distance of the apex from its edge
    let base = s * 0.74; // the flat base
    let near = s * 0.28; // base extent low
    let far = s * 0.72; // base extent high
    // Three vertices depending on facing (apex first).
    let (ax, ay, bx, by, cx, cy) = match facing {
        ArrowFacing::Up => (s * 0.5, apex, near, base, far, base),
        ArrowFacing::Down => (s * 0.5, base, near, apex, far, apex),
        ArrowFacing::Left => (apex, s * 0.5, base, near, base, far),
        ArrowFacing::Right => (base, s * 0.5, apex, near, apex, far),
    };
    fill_triangle(pixels, size, (ax, ay), (bx, by), (cx, cy), INK);
}

/// Fill a triangle (barycentric scan over its bounding box) onto an RGBA buffer.
/// #13 Step 6.3: edges are anti-aliased by 2×2 supersampling each pixel and writing
/// fractional coverage into the alpha channel, so the small glyphs read as clean
/// shapes instead of jagged stair-steps when scaled to the badge size.
fn fill_triangle(
    pixels: &mut [u8],
    size: usize,
    a: (f32, f32),
    b: (f32, f32),
    c: (f32, f32),
    color: [u8; 4],
) {
    let min_x = a.0.min(b.0).min(c.0).floor().max(0.0) as usize;
    let max_x = (a.0.max(b.0).max(c.0).ceil() as usize).min(size);
    let min_y = a.1.min(b.1).min(c.1).floor().max(0.0) as usize;
    let max_y = (a.1.max(b.1).max(c.1).ceil() as usize).min(size);
    let area = edge(a, b, c);
    if area.abs() < f32::EPSILON {
        return;
    }
    // 2×2 supersample offsets within each pixel.
    const SAMPLES: [(f32, f32); 4] = [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)];
    for y in min_y..max_y {
        for x in min_x..max_x {
            let mut covered = 0u32;
            for (ox, oy) in SAMPLES {
                let p = (x as f32 + ox, y as f32 + oy);
                let w0 = edge(b, c, p) / area;
                let w1 = edge(c, a, p) / area;
                let w2 = edge(a, b, p) / area;
                if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                    covered += 1;
                }
            }
            if covered > 0 {
                blend_pixel(pixels, size, x, y, color, covered as f32 / 4.0);
            }
        }
    }
}

/// Alpha-composite `color` (scaled by `coverage` 0..1) over the existing pixel at
/// `(x, y)`. Used by the anti-aliased glyph rasterisers so overlapping strokes and
/// soft edges accumulate cleanly on the transparent glyph buffer.
fn blend_pixel(pixels: &mut [u8], size: usize, x: usize, y: usize, color: [u8; 4], coverage: f32) {
    if x >= size || y >= size {
        return;
    }
    let index = (y * size + x) * 4;
    let src_a = (color[3] as f32 / 255.0) * coverage.clamp(0.0, 1.0);
    if src_a <= 0.0 {
        return;
    }
    for channel in 0..3 {
        let dst = pixels[index + channel] as f32 / 255.0;
        let src = color[channel] as f32 / 255.0;
        let out = src * src_a + dst * (1.0 - src_a);
        pixels[index + channel] = (out * 255.0).round().clamp(0.0, 255.0) as u8;
    }
    let dst_a = pixels[index + 3] as f32 / 255.0;
    let out_a = src_a + dst_a * (1.0 - src_a);
    pixels[index + 3] = (out_a * 255.0).round().clamp(0.0, 255.0) as u8;
}

/// Signed area of the triangle (a, b, c) — the edge function used for fill tests.
fn edge(a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> f32 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

/// Fill an axis-aligned rectangle (in float coordinates) with anti-aliased edges.
fn fill_rect(pixels: &mut [u8], size: usize, x0: f32, y0: f32, x1: f32, y1: f32, color: [u8; 4]) {
    let min_x = x0.floor().max(0.0) as usize;
    let max_x = (x1.ceil() as usize).min(size);
    let min_y = y0.floor().max(0.0) as usize;
    let max_y = (y1.ceil() as usize).min(size);
    for y in min_y..max_y {
        for x in min_x..max_x {
            // Per-pixel coverage = overlap of the pixel cell with the rect.
            let cover_x = ((x as f32 + 1.0).min(x1) - (x as f32).max(x0)).clamp(0.0, 1.0);
            let cover_y = ((y as f32 + 1.0).min(y1) - (y as f32).max(y0)).clamp(0.0, 1.0);
            let coverage = cover_x * cover_y;
            if coverage > 0.0 {
                blend_pixel(pixels, size, x, y, color, coverage);
            }
        }
    }
}

/// Draw a simple house silhouette (Home button): a triangular roof over a square.
fn draw_home_icon(pixels: &mut [u8], size: usize) {
    const INK: [u8; 4] = [0xff, 0xff, 0xff, 0xff];
    let s = size as f32;
    // Roof triangle (slightly overhanging the body for a cleaner house read).
    fill_triangle(
        pixels,
        size,
        (s * 0.5, s * 0.16),
        (s * 0.14, s * 0.52),
        (s * 0.86, s * 0.52),
        INK,
    );
    // Body square, anti-aliased.
    fill_rect(pixels, size, s * 0.28, s * 0.46, s * 0.72, s * 0.82, INK);
}

/// Draw a "fit to view" icon: four corner brackets (a crop/frame mark). #13 Step
/// 6.3: corner brackets read as "frame the model" and are clearly distinct from
/// the Home house, while staying legible at the small badge size.
fn draw_fit_icon(pixels: &mut [u8], size: usize) {
    const INK: [u8; 4] = [0xff, 0xff, 0xff, 0xff];
    let s = size as f32;
    let lo = s * 0.18;
    let hi = s * 0.82;
    let thick = (s * 0.12).max(2.0);
    let arm = s * 0.26; // length of each bracket arm
    // Four L-shaped corner brackets (each = a horizontal + a vertical bar).
    // Top-left.
    fill_rect(pixels, size, lo, lo, lo + arm, lo + thick, INK);
    fill_rect(pixels, size, lo, lo, lo + thick, lo + arm, INK);
    // Top-right.
    fill_rect(pixels, size, hi - arm, lo, hi, lo + thick, INK);
    fill_rect(pixels, size, hi - thick, lo, hi, lo + arm, INK);
    // Bottom-left.
    fill_rect(pixels, size, lo, hi - thick, lo + arm, hi, INK);
    fill_rect(pixels, size, lo, hi - arm, lo + thick, hi, INK);
    // Bottom-right.
    fill_rect(pixels, size, hi - arm, hi - thick, hi, hi, INK);
    fill_rect(pixels, size, hi - thick, hi - arm, hi, hi, INK);
}

/// Draw a roll arc with an arrowhead (CW or CCW) — a curved 270° stroke with a
/// small triangular head, for the top-right roll buttons.
fn draw_roll_arc(pixels: &mut [u8], size: usize, clockwise: bool) {
    const INK: [u8; 4] = [0xff, 0xff, 0xff, 0xff];
    let s = size as f32;
    let cx = s * 0.5;
    let cy = s * 0.5;
    let radius = s * 0.30;
    let thick = s * 0.09;
    // Stroke a 270° arc (leave a gap so the curl reads).
    let start = if clockwise { 0.6 } else { std::f32::consts::PI - 0.6 };
    let sweep = std::f32::consts::TAU * 0.75;
    let steps = 96;
    for i in 0..=steps {
        let frac = i as f32 / steps as f32;
        let ang = if clockwise {
            start + sweep * frac
        } else {
            start - sweep * frac
        };
        let px = cx + ang.cos() * radius;
        let py = cy + ang.sin() * radius;
        // Stamp a small soft-edged disc for thickness (anti-aliased rim).
        let half = thick * 0.5;
        let r = (half + 1.0) as i32;
        for dy in -r..=r {
            for dx in -r..=r {
                let dist = ((dx * dx + dy * dy) as f32).sqrt();
                let coverage = (half - dist + 0.5).clamp(0.0, 1.0);
                if coverage > 0.0 {
                    let x = px as i32 + dx;
                    let y = py as i32 + dy;
                    if x >= 0 && y >= 0 {
                        blend_pixel(pixels, size, x as usize, y as usize, INK, coverage);
                    }
                }
            }
        }
    }
    // Arrowhead at the arc's END.
    let end_ang = if clockwise { start + sweep } else { start - sweep };
    let hx = cx + end_ang.cos() * radius;
    let hy = cy + end_ang.sin() * radius;
    // Tangent direction at the end (perpendicular to radius, in sweep direction).
    let tang = if clockwise {
        end_ang + std::f32::consts::FRAC_PI_2
    } else {
        end_ang - std::f32::consts::FRAC_PI_2
    };
    let head = s * 0.16;
    let tip = (hx + tang.cos() * head, hy + tang.sin() * head);
    let left = (
        hx + (tang + 2.4).cos() * head * 0.7,
        hy + (tang + 2.4).sin() * head * 0.7,
    );
    let right = (
        hx + (tang - 2.4).cos() * head * 0.7,
        hy + (tang - 2.4).sin() * head * 0.7,
    );
    fill_triangle(pixels, size, tip, left, right, INK);
}

/// The glyph tint for the always-on chrome (parchment, matching the face text).
const CHROME_GLYPH_RGB: [f32; 3] = [0.913, 0.882, 0.819]; // #e9e1d1
/// A hovered arrow is brightened to teal-white so the highlight reads.
const CHROME_HOVER_RGB: [f32; 3] = [0.6, 1.0, 0.9];

/// Build the chrome overlay pipeline (alpha-blended screen-space textured quads)
/// and its glyph-texture bind group.
pub(crate) fn build_chrome_overlay(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    color_format: wgpu::TextureFormat,
) -> (wgpu::RenderPipeline, wgpu::BindGroup) {
    let layer_count = ChromeGlyph::ALL.len() as u32;
    let glyph_size = CHROME_GLYPH_TEXTURE_SIZE;
    let mut pixels = Vec::with_capacity((glyph_size * glyph_size * 4 * layer_count) as usize);
    for glyph in ChromeGlyph::ALL {
        pixels.extend_from_slice(&chrome_glyph_pixels(glyph));
    }
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("view cube chrome textures"),
        size: wgpu::Extent3d {
            width: glyph_size,
            height: glyph_size,
            depth_or_array_layers: layer_count,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * glyph_size),
            rows_per_image: Some(glyph_size),
        },
        wgpu::Extent3d {
            width: glyph_size,
            height: glyph_size,
            depth_or_array_layers: layer_count,
        },
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("view cube chrome sampler"),
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("view cube chrome layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("view cube chrome bind group"),
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("view cube chrome shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/viewcube_chrome.wgsl").into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("view cube chrome pipeline layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });
    let vertex_layout = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<ChromeVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute { offset: 0, shader_location: 0, format: wgpu::VertexFormat::Float32x2 },
            wgpu::VertexAttribute { offset: 8, shader_location: 1, format: wgpu::VertexFormat::Float32x2 },
            wgpu::VertexAttribute { offset: 16, shader_location: 2, format: wgpu::VertexFormat::Float32x4 },
            wgpu::VertexAttribute { offset: 32, shader_location: 3, format: wgpu::VertexFormat::Uint32 },
        ],
    };
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("view cube chrome pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vertex_main"),
            buffers: &[vertex_layout],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fragment_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None, // screen-space quads — don't cull on winding
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        // The view-cube pass binds a depth attachment, so this pipeline must carry
        // a matching depth-stencil state — but with depth TEST and WRITE disabled so
        // the chrome always paints on top of the cube/scene in the corner.
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState { count: 1, mask: !0, alpha_to_coverage_enabled: false },
        multiview_mask: None,
        cache: None,
    });
    (pipeline, bind_group)
}

/// The glyph + rect-fraction centre of the rotate arrow for `dir`. #13 Step 6.8:
/// edge-hugging gutters; #13 Step 6.7: the glyph points the way the cube CONTENT
/// rolls under the 90° step (OPPOSITE the edge it sits on), so it matches the
/// action. Shared by the persistent draw and the hovered-highlight draw so the
/// dim and bright states sit in identical pixels.
fn rotate_arrow_layout(dir: camera::ArrowDir) -> (ChromeGlyph, f32, f32) {
    use camera::ArrowDir;
    match dir {
        // TOP edge gutter v∈[0,.13]; the step pulls the top face down → ArrowDown.
        ArrowDir::Up => (ChromeGlyph::ArrowDown, 0.5, 0.065),
        // BOTTOM edge gutter v∈[.87,1.0]; pushes content up → ArrowUp.
        ArrowDir::Down => (ChromeGlyph::ArrowUp, 0.5, 0.935),
        // LEFT edge gutter u∈[0,.13]; rolls content rightward → ArrowRight.
        ArrowDir::Left => (ChromeGlyph::ArrowRight, 0.065, 0.5),
        // RIGHT edge gutter u∈[.87,1.0]; rolls content leftward → ArrowLeft.
        ArrowDir::Right => (ChromeGlyph::ArrowLeft, 0.935, 0.5),
    }
}

/// Build the per-frame chrome vertices (screen-space, NDC within the cube
/// viewport). `hovered_zone` decides which glyph is brightened. #13 Step 6
/// follow-up: `rotate_arrows_visible` (= the view is face-constrained) draws ALL
/// FOUR rotate arrows PERSISTENTLY in their dim state (Fusion behaviour); the
/// hovered one brightens. When `false` (off-face view) no rotate arrows draw at
/// all. The layout fractions MUST match `classify_cube_point`.
pub(crate) fn build_chrome_vertices(
    hovered_zone: Option<camera::CubeChromeZone>,
    rotate_arrows_visible: bool,
) -> Vec<ChromeVertex> {
    use camera::{ArrowDir, CubeChromeZone, RollDir};

    let mut verts = Vec::new();

    // Helper: is THIS zone the hovered one? Picks the brighter tint.
    let tint = |is_hovered: bool| {
        if is_hovered {
            with_alpha(CHROME_HOVER_RGB, 1.0)
        } else {
            with_alpha(CHROME_GLYPH_RGB, 1.0)
        }
    };

    // --- Always-on: Home / Fit badges (top-left), Step-1 u∈[0,.12]/[.12,.24], v∈[0,.12]. ---
    let badge_y = 0.07;
    let badge_size = 0.12;
    let home_hovered = hovered_zone == Some(CubeChromeZone::HomeButton);
    push_glyph_quad(&mut verts, ChromeGlyph::HomeButton, 0.06, badge_y, badge_size, badge_size, tint(home_hovered));
    let fit_hovered = hovered_zone == Some(CubeChromeZone::FitButton);
    push_glyph_quad(&mut verts, ChromeGlyph::FitButton, 0.18, badge_y, badge_size, badge_size, tint(fit_hovered));

    // --- The 4 rotate arrows: drawn PERSISTENTLY whenever the view is face-
    // constrained (decoupled from hover); the hovered one is brightened. ---
    if rotate_arrows_visible {
        for dir in [ArrowDir::Up, ArrowDir::Down, ArrowDir::Left, ArrowDir::Right] {
            let (glyph, cx, cy) = rotate_arrow_layout(dir);
            let hovered = hovered_zone == Some(CubeChromeZone::RotateArrow(dir));
            push_glyph_quad(&mut verts, glyph, cx, cy, 0.075, 0.075, tint(hovered));
        }
    }

    // --- Hover-only: the 2 roll arrows (top-right). Step-1 u∈[.74,.87]/[.87,1.0], v∈[0,.13]. ---
    if let Some(CubeChromeZone::RollArrow(dir)) = hovered_zone {
        let (glyph, cx) = match dir {
            RollDir::Ccw => (ChromeGlyph::RollCcw, (0.74 + 0.87) / 2.0),
            RollDir::Cw => (ChromeGlyph::RollCw, (0.87 + 1.00) / 2.0),
        };
        push_glyph_quad(&mut verts, glyph, cx, 0.065, 0.11, 0.11, tint(true));
    }

    verts
}

/// Push two triangles for a textured glyph quad. `(cx, cy)` is the centre and
/// `(half_w, half_h)` the half-extents, ALL in rect fractions [0,1] (origin
/// top-left, y down). Converts to NDC (x: f*2-1, y: 1-f*2) for the viewport.
fn push_glyph_quad(
    verts: &mut Vec<ChromeVertex>,
    glyph: ChromeGlyph,
    cx: f32,
    cy: f32,
    half_w: f32,
    half_h: f32,
    color: [f32; 4],
) {
    let to_ndc = |fx: f32, fy: f32| [fx * 2.0 - 1.0, 1.0 - fy * 2.0];
    let layer = glyph.layer();
    // Corners in rect-fraction space (TL, TR, BR, BL) with UV.
    let corners = [
        (cx - half_w, cy - half_h, 0.0, 0.0),
        (cx + half_w, cy - half_h, 1.0, 0.0),
        (cx + half_w, cy + half_h, 1.0, 1.0),
        (cx - half_w, cy + half_h, 0.0, 1.0),
    ];
    let v = |i: usize| {
        let (fx, fy, u, t) = corners[i];
        ChromeVertex { position: to_ndc(fx, fy), uv: [u, t], color, layer }
    };
    // TL,TR,BR  +  TL,BR,BL
    verts.extend_from_slice(&[v(0), v(1), v(2), v(0), v(2), v(3)]);
}
