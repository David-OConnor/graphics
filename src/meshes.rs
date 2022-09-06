//! This module generates meshes

use core::f32::consts::TAU;

use crate::{
    init_graphics::{UP_VEC},
    types::{Mesh, Vertex}
};

use lin_alg2::f32::Vec3;

/// Rotate a 2d vector counter-clockwise a given angle.
fn rotate_vec_2d(vec: [f32; 2], angle: f32) -> [f32; 2] {
    [0., 0.] // todo temp
}

impl Mesh {
    /// Create a tetrahedron mesh
    pub fn new_tetrahedron(side_len: f32) -> Self {
        let v_0 = [side_len, side_len, side_len];
        let v_1 = [side_len, -side_len, -side_len];
        let v_2 = [-side_len, side_len, -side_len];
        let v_3 = [-side_len, -side_len, side_len];

        // Note: For tetrahedrons, the normals are the corners of the cube we
        // didn't use for vertices.
        let n_0 = Vec3::new(1., 1., -1.).to_normalized();
        let n_1 = Vec3::new(1., -1., 1.).to_normalized();
        let n_2 = Vec3::new(-1., 1., -1.).to_normalized();
        let n_3 = Vec3::new(-1., -1., -1.).to_normalized();

        let mut vertices = vec![
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

        Mesh {
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
            circle_vertices.push(rotate_vec_2d([1., 0.], i as f32 * angle_between_vertices));
        }

        let half_len = len * 0.5;

        let mut vertices = Vec::new();
        // let mut indices = Vec::new();

        // let mut faces = Vec::new();

        let mut i = 0;

        // // Top center
        // vertices.push(Vertex::new(
        //     [0., half_len, 0.],
        //     UP_VEC,
        // ));
        // i += 1;
        //
        // // Bottom center
        // vertices.push(Vertex::new(
        //     [0., -half_len, 0.],
        //     -UP_VEC
        // ));
        // i += 1;

        // let mut vertices_top = Vec::new();
        // let mut vertices_bottom = Vec::new();

        for vert in circle_vertices {
            // On top face
            vertices.push(Vertex::new(
                [vert[0], half_len, vert[1]],
                UP_VEC,
            ));
            i += 1;

            // On bottom face
            vertices.push(Vertex::new(
                [vert[0], -half_len, vert[1]],
                -UP_VEC,
            ));
            i += 1;

            // On edge face, top
            vertices.push(Vertex::new(
                [vert[0], half_len, vert[1]],
                Vec3::new(vert[0], 0., vert[1]),
            ));
            i += 1;

            // On edge face, bottom
            vertices.push(Vertex::new(
                [vert[0], -half_len, vert[1]],
                Vec3::new(vert[0], 0., vert[1]),
            ));
            i += 1;

            // Top
            // indices.append(vec![0]);
        }

        let indices = Vec::new();

        Mesh {
            vertices,
            indices,
            material: 0,
        }
    }
}
