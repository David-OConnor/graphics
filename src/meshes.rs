//! This module generates meshes

use std::{
    f32::consts::TAU,
    fs::File,
    io::{BufReader, Read},
};

use crate::{
    graphics::UP_VEC,
    types::{Mesh, Vertex},
};

use lin_alg2::f32::Vec3;

/// Rotate a 2d vector counter-clockwise a given angle.
fn rotate_vec_2d(vec: [f32; 2], θ: f32) -> [f32; 2] {
    // Self-contained 2d rotation matrix (col-maj)
    let (sin_θ, cos_θ) = θ.sin_cos();
    let mat = [cos_θ, sin_θ, -sin_θ, cos_θ];

    [
        vec[0] * mat[0] + vec[1] * mat[2],
        vec[0] * mat[1] + vec[1] * mat[3],
    ]
}

impl Mesh {
    // /// Create a triangular face, with no volume. Only visible from one side.
    // /// Useful for building a grid surface like terrain, or a surface plot.
    // pub fn new_tri_face(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> Self {
    //     let norm = Vec3::new(0., 0., -1.);

    //     let vertices = vec![
    //         Vertex::new(a, norm),
    //         Vertex::new(b, norm),
    //         Vertex::new(c, norm),
    //     ];

    //     let indices = vec![0, 1, 2];

    //     Self {
    //         vertices,
    //         indices,
    //         material: 0,
    //     }
    // }

    /// Create a grid surface of triangles.
    /// Useful for building a grid surface like terrain, or a surface plot.
    /// `grid`'s outer vec is rows; inner vec is col-associated values within that 
    /// row. The grid is evenly-spaced.
    pub fn new_surface(grid: &Vec<Vec<f32>>, start: f32, step: f32) -> Self {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let mut x = start;
        let mut y = start;
        let mut this_vert_i = 0;

        for (i, row) in grid.iter().enumerate() {
            y = 0.;
            for (j, z) in row.into_iter().enumerate() {
                let norm = Vec3::new(0., 0., -1.);
                vertices.push(Vertex::new([x, y, *z], norm));

                // To understand how we set up the triangles (index buffer),
                // it's best to draw it out.

                // Upper triangle: This exists for every vertex except
                // the bottom and right edges.
                // (grid.length is num_rows)
                if i != grid.len() - 1 && j != row.len() - 1 {
                    indices.append(&mut vec![this_vert_i, this_vert_i + grid.len(), this_vert_i + 1]);
                }

                // Lower triangle: This exists for every vertex except
                // the top and left edges.
                if i != 0 && j != 0 {
                    indices.append(&mut vec![this_vert_i, this_vert_i - grid.len(), this_vert_i - 1]);
                }

                y += step;
                this_vert_i += 1;
            }
            x += step;
        }

        Self {
            vertices,
            indices,
            material: 0,
        }
    }

    /// Create a (normalized cube) sphere mesh. A higher div count results in a smoother sphere.
    /// https://medium.com/@oscarsc/four-ways-to-create-a-mesh-for-a-sphere-d7956b825db4
    /// todo: Temporarily, a uv_sphere while we figure out how to make better ones.
    pub fn new_sphere(radius: f32, num_lats: usize, num_lons: usize) -> Self {
        let mut vertices = Vec::new();
        let mut faces = Vec::new();
        // We use faces to construct indices (triangles)
        let mut indices = Vec::new();

        // In radians
        let lat_size = TAU / (2. * num_lats as f32);
        let lon_size = TAU / num_lons as f32;

        let mut current_i = 0;

        // Bottom vertex and faces
        vertices.push(Vertex::new([0., -radius, 0.], Vec3::new(0., -1., 0.)));
        current_i += 1;

        // Faces connected to the bottom vertex.
        for k in 0..num_lons {
            if k == num_lons - 1 {
                indices.append(&mut vec![0, k + 2 - num_lons, k + 1]);
            } else {
                indices.append(&mut vec![0, k + 2, k + 1]);
            }
        }

        // Don't include the top or bottom (0, TAU/2) angles in lats.
        for i in 1..num_lats {
            let θ = i as f32 * lat_size;

            for j in 0..num_lons {
                let φ = j as f32 * lon_size;

                // https://en.wikipedia.org/wiki/Spherical_coordinate_system
                let x = radius * φ.cos() * θ.sin();
                let y = radius * φ.sin() * θ.sin();
                let z = radius * θ.cos();

                vertices.push(Vertex::new([x, y, z], Vec3::new(x, y, z).to_normalized()));

                if i < num_lats - 1 {
                    // In CCW order
                    if j == num_lons - 1 {
                        faces.push([
                            current_i,
                            current_i + 1 - num_lons,
                            current_i + 1,
                            current_i + num_lons,
                        ]);
                    } else {
                        faces.push([
                            current_i,
                            current_i + 1,
                            current_i + num_lons + 1,
                            current_i + num_lons,
                        ]);
                    }
                }
                current_i += 1;
            }
        }

        // Top vertex and faces
        vertices.push(Vertex::new([0., radius, 0.], Vec3::new(0., 1., 0.)));

        // Faces connected to the bottom vertex.
        let top_ring_start_i = current_i - num_lons;

        // todo: There's a rougue internal triangle on both the top and bottom caps, but it
        // todo does'nt appear to be visible from the outside. Possibly related: The caps look wrong.

        for k in 0..num_lons {
            if k == num_lons - 1 {
                indices.append(&mut vec![current_i, top_ring_start_i + k, top_ring_start_i]);
            } else {
                indices.append(&mut vec![
                    current_i,
                    top_ring_start_i + k,
                    top_ring_start_i + k + 1,
                ]);
            }
        }

        // current_i += 1;

        for f in faces {
            indices.append(&mut vec![f[0], f[1], f[2], f[0], f[2], f[3]]);
        }

        Self {
            vertices,
            indices,
            // vertex_buffer: Vec<usize>,
            // index_buffer: Vec<usize>,
            // num_elements: u32,
            material: 0,
        }
    }

    /// Create a box (rectangular prism) mesh.
    pub fn new_box(len_x: f32, len_y: f32, len_z: f32) -> Self {
        let x = len_x / 2.;
        let y = len_y / 2.;
        let z = len_z / 2.;

        // Aft face
        let abl = [-x, -y, -z];
        let abr = [x, -y, -z];
        let atr = [x, y, -z];
        let atl = [-x, y, -z];

        // Forward face
        let fbl = [-x, -y, z];
        let fbr = [x, -y, z];
        let ftr = [x, y, z];
        let ftl = [-x, y, z];

        // Normal vectors
        let aft = Vec3::new(0., 0., -1.);
        let fwd = Vec3::new(0., 0., 1.);
        let l = Vec3::new(-1., 0., 0.);
        let r = Vec3::new(1., 0., 0.);
        let t = Vec3::new(0., 1., 0.);
        let b = Vec3::new(0., -1., 0.);

        let vertices = vec![
            // Aft
            Vertex::new(abl, aft),
            Vertex::new(abr, aft),
            Vertex::new(atr, aft),
            Vertex::new(atl, aft),
            // Fwd
            Vertex::new(fbl, fwd),
            Vertex::new(ftl, fwd),
            Vertex::new(ftr, fwd),
            Vertex::new(fbr, fwd),
            // Left
            Vertex::new(fbl, l),
            Vertex::new(abl, l),
            Vertex::new(atl, l),
            Vertex::new(ftl, l),
            // Right
            Vertex::new(abr, r),
            Vertex::new(fbr, r),
            Vertex::new(ftr, r),
            Vertex::new(atr, r),
            // Top
            Vertex::new(atl, t),
            Vertex::new(atr, t),
            Vertex::new(ftr, t),
            Vertex::new(ftl, t),
            // Bottom
            Vertex::new(abl, b),
            Vertex::new(fbl, b),
            Vertex::new(fbr, b),
            Vertex::new(abr, b),
        ];

        let faces = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
            [12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23],
        ];

        let mut indices = Vec::new();
        for face in &faces {
            indices.append(&mut vec![
                face[0], face[1], face[2], face[0], face[2], face[3],
            ]);
        }

        Self {
            vertices,
            indices,
            material: 0,
        }
    }

    /// Create a tetrahedron mesh
    pub fn new_tetrahedron(side_len: f32) -> Self {
        let c = side_len / 2.;

        let v_0 = [c, c, c];
        let v_1 = [c, -c, -c];
        let v_2 = [-c, c, -c];
        let v_3 = [-c, -c, c];

        // Note: For tetrahedrons, the normals are the corners of the cube we
        // didn't use for vertices.
        let n_0 = Vec3::new(1., 1., -1.).to_normalized();
        let n_1 = Vec3::new(1., -1., 1.).to_normalized();
        let n_2 = Vec3::new(-1., 1., 1.).to_normalized();
        let n_3 = Vec3::new(-1., -1., -1.).to_normalized();

        let vertices = vec![
            // Face 0
            Vertex::new(v_0, n_0),
            Vertex::new(v_2, n_0),
            Vertex::new(v_1, n_0),
            // Face 1
            Vertex::new(v_0, n_1),
            Vertex::new(v_1, n_1),
            Vertex::new(v_3, n_1),
            // Face 2
            Vertex::new(v_0, n_2),
            Vertex::new(v_3, n_2),
            Vertex::new(v_2, n_2),
            // Face 3
            Vertex::new(v_1, n_3),
            Vertex::new(v_2, n_3),
            Vertex::new(v_3, n_3),
        ];

        // These indices define faces by triangles. (each 3 represent a triangle, starting at index 0.
        // Indices are arranged CCW, from front of face
        // Note that because we're using "hard" lighting on faces, we can't repeat any vertices, since
        // they each have a different normal.
        #[rustfmt::skip]
            // let indices: &[u32] = &[
            let indices = vec![
            0, 1, 2,
            3, 4, 5,
            6, 7, 8,
            9, 10, 11,
        ];

        Self {
            vertices,
            indices,
            // vertex_buffer: Vec<usize>,
            // index_buffer: Vec<usize>,
            // num_elements: u32,
            material: 0,
        }
    }

    /// Create a cylinder
    pub fn new_cylinder(len: f32, radius: f32, num_sides: usize) -> Self {
        let angle_between_vertices = TAU / num_sides as f32;

        let mut circle_vertices = Vec::new();
        for i in 0..num_sides {
            circle_vertices.push(rotate_vec_2d(
                [radius, 0.],
                i as f32 * angle_between_vertices,
            ));
        }

        let half_len = len * 0.5;
        let mod_ = 2 * num_sides;

        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        let mut i_vertex = 0;

        for vert in &circle_vertices {
            // The number of faces is the number of angles - 1.
            // Triangle 1: This top, this bottom, next top.
            indices.append(&mut vec![i_vertex, i_vertex + 1, (i_vertex + 2) % mod_]);
            // Triangle 2: This bottom, next bottom, next top.
            indices.append(&mut vec![
                i_vertex + 1,
                (i_vertex + 3) % mod_,
                (i_vertex + 2) % mod_,
            ]);

            // On edge face, top
            vertices.push(Vertex::new(
                [vert[0], half_len, vert[1]],
                Vec3::new(vert[0], 0., vert[1]).to_normalized(),
            ));
            i_vertex += 1;

            // On edge face, bottom
            vertices.push(Vertex::new(
                [vert[0], -half_len, vert[1]],
                Vec3::new(vert[0], 0., vert[1]).to_normalized(),
            ));
            i_vertex += 1;
        }

        // let mut vertices_top_face = Vec::new();
        // let mut vertices_bottom_face = Vec::new();
        let top_anchor = i_vertex;
        let bottom_anchor = i_vertex + 1;

        for (j, vert) in circle_vertices.iter().enumerate() {
            // We need num_sides - 2 triangles using this anchor-vertex algorithm.
            if j != 0 && j != num_sides - 1 {
                indices.append(&mut vec![top_anchor, i_vertex, i_vertex + 2]);
                // We need CCW triangles for both, so reverse order on the bottom face.
                indices.append(&mut vec![bottom_anchor, i_vertex + 3, i_vertex + 1]);
            }

            // Top face
            vertices.push(Vertex::new([vert[0], half_len, vert[1]], UP_VEC));
            i_vertex += 1;

            // Bottom face
            vertices.push(Vertex::new([vert[0], -half_len, vert[1]], -UP_VEC));
            i_vertex += 1;
        }

        Self {
            vertices,
            indices,
            material: 0,
        }
    }

    /// Load a mesh from a obj file.
    /// [File type description](https://en.wikipedia.org/wiki/Wavefront_.obj_file)
    /// [Example](https://github.com/gfx-rs/wgpu/blob/master/wgpu/examples/skybox/main.rs)
    pub fn from_obj_file(filename: &str) -> Self {
        let f = File::open(filename).unwrap();
        let mut reader = BufReader::new(f);
        let mut file_buf = Vec::new();

        reader.read_to_end(&mut file_buf).unwrap();

        let data = obj::ObjData::load_buf(&file_buf[..]).unwrap();
        let mut vertices = Vec::new();

        for object in data.objects {
            for group in object.groups {
                vertices.clear();

                for poly in group.polys {
                    for end_index in 2..poly.0.len() {
                        for &index in &[0, end_index - 1, end_index] {
                            let obj::IndexTuple(position_id, _texture_id, normal_id) =
                                poly.0[index];

                            let n = data.normal[normal_id.unwrap()];

                            vertices.push(Vertex::new(
                                data.position[position_id],
                                Vec3::new(n[0], n[1], n[2]),
                            ));
                        }
                    }
                }
            }
        }

        // todo: Is this right?
        let indices = (0..vertices.len()).collect();

        Self {
            vertices,
            indices,
            material: 0,
        }
    }
}
