//! This module contains code specific to gaussians. Not all gauss-specific code is here though.

use lin_alg::f32::{Mat4, Vec3};
use wgpu::{VertexAttribute, VertexBufferLayout, VertexFormat, VertexStepMode};

pub(crate) const CAM_BASIS_SIZE: usize = 32; // Includes padding.

#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub(crate) struct CameraBasis {
    pub right: Vec3,
    pub _pad0: f32,
    pub up: Vec3,
    pub _pad1: f32,
}

impl CameraBasis {
    pub fn new(view: Mat4) -> Self {
        let view_inv = view.inverse().unwrap();
        let cols = view_inv.to_cols();

        let right = cols.0.xyz().to_normalized();
        let up = cols.1.xyz().to_normalized();

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
    step_mode: VertexStepMode::Vertex,
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

#[repr(C, align(16))]
#[derive(Clone, Copy)]
pub(crate) struct GaussianInstance {
    pub center: [f32; 3],
    pub amplitude: f32,
    pub width: f32, // σ  (not σ²)
    pub color: [f32; 4],
    pub _pad: [f32; 3], // 16‑B alignment
}

pub(crate) const GAUSS_INST_LAYOUT: VertexBufferLayout<'static> = VertexBufferLayout {
    array_stride: size_of::<GaussianInstance>() as wgpu::BufferAddress, // 48 bytes
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
        // color, @location(4)
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
}
