use super::*;

/// Upload built per-chunk meshes into GPU buffers, one [`CuboidChunkBuffers`] per
/// non-empty chunk (issue #20 S6c-2d).
pub(crate) fn upload_chunk_meshes(
    device: &wgpu::Device,
    chunk_meshes: &[CuboidChunkMesh],
) -> std::collections::HashMap<[i32; 3], CuboidChunkBuffers> {
    let mut buffers = std::collections::HashMap::new();
    for mesh in chunk_meshes {
        if mesh.indices.is_empty() && mesh.indices_overlay.is_empty() {
            continue;
        }
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cuboid chunk vertices"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        // One index buffer = overlay-OFF run then overlay-ON run (ADR 0003 §3c); the two
        // draws slice it by count + offset.
        let mut all_indices = mesh.indices.clone();
        all_indices.extend_from_slice(&mesh.indices_overlay);
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("cuboid chunk indices"),
            contents: bytemuck::cast_slice(&all_indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        buffers.insert(
            mesh.coord,
            CuboidChunkBuffers {
                vertex_buffer,
                index_buffer,
                index_count: mesh.indices.len() as u32,
                index_count_overlay: mesh.indices_overlay.len() as u32,
                aabb: mesh.aabb,
                box_count: mesh.box_count,
            },
        );
    }
    buffers
}

/// Bucket a whole [`VoxelGrid`] into per-chunk sub-grids keyed by integer chunk
/// coord `floor(world_position / chunk_extent)` (issue #20 S6c-2d) — the same key
/// the resolve cache's per-chunk accessor uses (the legacy instanced renderer this
/// key once also matched was removed, part of #20), so the cuboid `new` wrapper's
/// chunk partition matches the resolve cache's. A sub-grid carries only the occupied
/// voxels (its `dimensions` is unused by the apron mesher, which keys off
/// `world_position`).
pub(crate) fn bucket_grid_into_chunk_grids(
    grid: &VoxelGrid,
    voxels_per_block: u32,
) -> Vec<([i32; 3], VoxelGrid)> {
    use std::collections::HashMap;
    let chunk_extent = (voxel_core::core_geom::CHUNK_BLOCKS * voxels_per_block.max(1)) as f32;
    let mut buckets: HashMap<[i32; 3], VoxelGrid> = HashMap::new();
    for voxel in &grid.occupied {
        let position = voxel.world_position();
        let key = [
            (position[0] / chunk_extent).floor() as i32,
            (position[1] / chunk_extent).floor() as i32,
            (position[2] / chunk_extent).floor() as i32,
        ];
        buckets
            .entry(key)
            .or_insert_with(|| VoxelGrid::new([0, 0, 0]))
            .occupied
            .push(*voxel);
    }
    let mut out: Vec<([i32; 3], VoxelGrid)> = buckets.into_iter().collect();
    out.sort_unstable_by_key(|(coord, _)| *coord);
    out
}
