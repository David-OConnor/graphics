//! This module contains code specific to gaussians. Not all gauss-specific code is here though.

use lin_alg::f32::{Mat4, Quaternion, Vec3};
use wgpu::{VertexAttribute, VertexBufferLayout, VertexFormat, VertexStepMode};
use crate::{RIGHT_VEC, UP_VEC};
use crate::types::{
    F32_SIZE, INSTANCE_SIZE, MAT3_SIZE, MAT4_SIZE, VEC3_SIZE, VEC4_SIZE,
};

pub(crate) const CAM_BASIS_SIZE: usize = 32;

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct CameraBasis {
    pub right: Vec3, pub _pad0: f32,
    pub up:    Vec3, pub _pad1: f32,
}

impl CameraBasis {
    pub fn new(orientation: Quaternion, view: Mat4) -> Self {
        let view_inv = view.inverse().unwrap();
        // todo: QC these. Col 0 and 1?
        let cols = view_inv.to_cols();

        let right = cols.0.xyz();
        let up = cols.1.xyz();

        println!("\n\nRight: {:?} len: {}", cols.0.xyz(), cols.0.xyz().magnitude());
        println!("UP: {:?} len: {}", cols.1.xyz(), cols.1.xyz().magnitude());
        println!("Dot: {:?}", cols.0.xyz().dot(cols.1.xyz()));

        let right = orientation.rotate_vec(RIGHT_VEC);
        let up = orientation.rotate_vec(UP_VEC);

        // let right = RIGHT_VEC;
        // let up = UP_VEC;

        println!("\n\nRight: {:?} len: {}", right, right.magnitude());
        println!("UP: {:?} len: {}", up, up.magnitude());
        println!("Dot: {:?}", up.dot(right));

        Self {
            right,
            _pad0: 0.,
            up,
            _pad1: 0.,
        }
    }

    pub fn to_bytes(&self) -> [u8; CAM_BASIS_SIZE] {
        let mut result = [0; CAM_BASIS_SIZE];

        result[0..12].copy_from_slice(&self.right.to_bytes());
        result[16..28].copy_from_slice(&self.up.to_bytes());

        result
    }
}

/// For the gaussian shader.
pub(crate) struct QuadVertex {
    pos: [f32; 2], // Only the XY of the NDC quad corner
}

impl QuadVertex {
    pub(crate) fn to_bytes(&self) -> [u8; 8] {
        let mut result = [0; 8];
        result[0..4].clone_from_slice(&self.pos[0].to_ne_bytes());
        result[4..8].clone_from_slice(&self.pos[1].to_ne_bytes());
        result
    }
}

// For the Gaussian shader.
pub(crate) const QUAD_VERTEX_LAYOUT: VertexBufferLayout<'static> = VertexBufferLayout {
    array_stride: size_of::<QuadVertex>() as wgpu::BufferAddress,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &[VertexAttribute {
        offset: 0,
        shader_location: 0, // @location(0) in the WGSL
        format: VertexFormat::Float32x2,
    }],
};

pub(crate) const QUAD_VERTICES: &[QuadVertex] = &[
    // triangle 1
    QuadVertex { pos: [-1.0, -1.0] },
    QuadVertex { pos: [1.0, -1.0] },
    QuadVertex { pos: [1.0, 1.0] },
    // triangle 2
    QuadVertex { pos: [-1.0, -1.0] },
    QuadVertex { pos: [1.0, 1.0] },
    QuadVertex { pos: [-1.0, 1.0] },
];

/// This is publicly accessible; set by the user, and stored in Scene.
#[derive(Clone, Copy, Debug)]
pub struct Gaussian {
    pub center: Vec3,
    pub amplitude: f32,
    pub width: f32,
    pub color: [f32; 4],
}

impl Gaussian {
    pub(crate) fn to_instance(&self) -> GaussianInstance {
        GaussianInstance {
            center: self.center.to_arr(),
            amplitude: self.amplitude,
            width: self.width,
            color: self.color,
            _pad: [0.; 3],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct GaussianInstance {
    pub center: [f32; 3],
    pub amplitude: f32,
    pub width: f32,     // σ  (not σ²)
    pub color: [f32; 4],
    pub _pad: [f32; 3], // 16‑B alignment
}

pub(crate) const GAUSS_INST_LAYOUT: VertexBufferLayout<'static> = VertexBufferLayout {
    array_stride: size_of::<GaussianInstance>() as wgpu::BufferAddress, // 32 bytes
    step_mode: VertexStepMode::Instance,
    attributes: &[
        // center.xyz  → @location(1)
        VertexAttribute {
            offset: 0,
            shader_location: 1,
            format: VertexFormat::Float32x3,
        },
        // amplitude    → @location(2)
        VertexAttribute {
            offset: 12,
            shader_location: 2,
            format: VertexFormat::Float32,
        },
        // width        → @location(3)
        VertexAttribute {
            offset: 16,
            shader_location: 3,
            format: VertexFormat::Float32,
        },
        VertexAttribute {
            offset: 20,
            shader_location: 4,
            format: VertexFormat::Float32x4,
        },
    ],
};

impl GaussianInstance {
    pub fn to_bytes(&self) -> [u8; 48] {
        let mut result = [0; 48];
        result[0..4].clone_from_slice(&self.center[0].to_ne_bytes());
        result[4..8].clone_from_slice(&self.center[1].to_ne_bytes());
        result[8..12].clone_from_slice(&self.center[2].to_ne_bytes());
        result[12..16].clone_from_slice(&self.amplitude.to_ne_bytes());
        result[16..20].clone_from_slice(&self.width.to_ne_bytes());
        result[20..24].clone_from_slice(&self.color[0].to_ne_bytes());
        result[24..28].clone_from_slice(&self.color[1].to_ne_bytes());
        result[28..32].clone_from_slice(&self.color[2].to_ne_bytes());
        result[32..36].clone_from_slice(&self.color[3].to_ne_bytes());

        result
    }

    /// Create the vertex buffer memory layout, for our vertexes passed from the
    /// vertex to the fragment shader. Corresponds to `VertexOut` in the shader. Each
    /// item here is for a single vertex. Cannot share locations with `VertexIn`, so
    /// we start locations after `VertexIn`'s last one.
    pub(crate) fn desc<'a>() -> VertexBufferLayout<'a> {
        VertexBufferLayout {
            array_stride: INSTANCE_SIZE as wgpu::BufferAddress,
            // We need to switch from using a step mode of Vertex to Instance
            // This means that our shaders will only change to use the next
            // instance when the shader starts processing a new instance
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // A mat4 takes up 4 vertex slots as it is technically 4 vec4s. We need to define a slot
                // for each vec4. We'll have to reassemble the mat4 in
                // the shader.

                // Model matrix, col 0
                VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: VertexFormat::Float32x4,
                },
                // Model matrix, col 1
                VertexAttribute {
                    offset: (F32_SIZE * 4) as wgpu::BufferAddress,
                    shader_location: 6,
                    format: VertexFormat::Float32x4,
                },
                // Model matrix, col 2
                VertexAttribute {
                    offset: (F32_SIZE * 8) as wgpu::BufferAddress,
                    shader_location: 7,
                    format: VertexFormat::Float32x4,
                },
                // Model matrix, col 3
                VertexAttribute {
                    offset: (F32_SIZE * 12) as wgpu::BufferAddress,
                    shader_location: 8,
                    format: VertexFormat::Float32x4,
                },
                // Normal matrix, col 0
                VertexAttribute {
                    offset: (MAT4_SIZE) as wgpu::BufferAddress,
                    shader_location: 9,
                    format: VertexFormat::Float32x3,
                },
                // Normal matrix, col 1
                VertexAttribute {
                    offset: (MAT4_SIZE + VEC3_SIZE) as wgpu::BufferAddress,
                    shader_location: 10,
                    format: VertexFormat::Float32x3,
                },
                // Normal matrix, col 2
                VertexAttribute {
                    offset: (MAT4_SIZE + VEC3_SIZE * 2) as wgpu::BufferAddress,
                    shader_location: 11,
                    format: VertexFormat::Float32x3,
                },
                // model (and vertex) color
                VertexAttribute {
                    offset: (MAT4_SIZE + MAT3_SIZE) as wgpu::BufferAddress,
                    shader_location: 12,
                    format: VertexFormat::Float32x4,
                },
                // Shinyness
                VertexAttribute {
                    offset: (MAT4_SIZE + MAT3_SIZE + VEC4_SIZE) as wgpu::BufferAddress,
                    shader_location: 13,
                    format: VertexFormat::Float32,
                },
            ],
        }
    }
}
