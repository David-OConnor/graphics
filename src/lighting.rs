use crate::types::{F32_SIZE, VEC3_UNIFORM_SIZE};

use lin_alg2::f32::Vec3;

// The extra 4 is due to uniform buffers needing ton be a multiple of 16 in size.
// This is for the non-array portion of the lighting uniform.
// The extra 12 is for padding.
pub const LIGHTING_SIZE_FIXED: usize = VEC3_UNIFORM_SIZE + F32_SIZE + 12;

// The extra 8 here for the same reason.
pub const POINT_LIGHT_SIZE: usize = 3 * VEC3_UNIFORM_SIZE + 2 * F32_SIZE + 8;

// Note: These array-to-bytes functions may have broader use than in this lighting module.

// /// Pads to an array4.
// fn array3_to_bytes(a: [f32; 3]) -> [u8; VEC3_UNIFORM_SIZE] {
//     let mut result = [0; VEC3_UNIFORM_SIZE];
//
//     result[0..F32_SIZE].clone_from_slice(&a[0].to_ne_bytes());
//     result[F32_SIZE..2 * F32_SIZE].clone_from_slice(&a[1].to_ne_bytes());
//     result[2 * F32_SIZE..3 * F32_SIZE].clone_from_slice(&a[2].to_ne_bytes());
//     result[3 * F32_SIZE..VEC3_UNIFORM_SIZE].clone_from_slice(&[0; F32_SIZE]);
//
//     result
// }

fn array4_to_bytes(a: [f32; 4]) -> [u8; VEC3_UNIFORM_SIZE] {
    let mut result = [0; VEC3_UNIFORM_SIZE];

    result[0..F32_SIZE].clone_from_slice(&a[0].to_ne_bytes());
    result[F32_SIZE..2 * F32_SIZE].clone_from_slice(&a[1].to_ne_bytes());
    result[2 * F32_SIZE..3 * F32_SIZE].clone_from_slice(&a[2].to_ne_bytes());
    result[3 * F32_SIZE..VEC3_UNIFORM_SIZE].clone_from_slice(&a[3].to_ne_bytes());

    result
}

#[derive(Debug, Clone)]
/// We organize the fields in this order, and serialize them accordingly, to keep the buffer
/// from being too long while adhering to alignment rules.
pub struct Lighting {
    pub ambient_color: [f32; 4],
    pub ambient_intensity: f32,
    pub point_lights: Vec<PointLight>,
}

impl Default for Lighting {
    fn default() -> Self {
        Self {
            ambient_color: [1., 1., 1., 0.5],
            ambient_intensity: 0.15,
            point_lights: vec![PointLight {
                type_: LightType::Omnidirectional,
                position: Vec3::new_zero(),
                diffuse_color: [1., 1., 1., 0.5],
                specular_color: [1., 1., 1., 0.5],
                diffuse_intensity: 1_000_000.,
                specular_intensity: 1_000_000.,
            }],
        }
    }
}

impl Lighting {
    /// We use a vec due to the dynamic size of `point_lights`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut result = Vec::new();

        let mut buf_fixed_size = [0; LIGHTING_SIZE_FIXED];

        buf_fixed_size[0..VEC3_UNIFORM_SIZE].clone_from_slice(&array4_to_bytes(self.ambient_color));

        buf_fixed_size[VEC3_UNIFORM_SIZE..VEC3_UNIFORM_SIZE + F32_SIZE]
            .clone_from_slice(&self.ambient_intensity.to_ne_bytes());

        // We pass size manually, due to trouble getting the array len in the shader.
        buf_fixed_size[VEC3_UNIFORM_SIZE + F32_SIZE..VEC3_UNIFORM_SIZE + F32_SIZE + 4]
            .clone_from_slice(&(self.point_lights.len() as i32).to_le_bytes());

        buf_fixed_size[VEC3_UNIFORM_SIZE + F32_SIZE + 4..LIGHTING_SIZE_FIXED]
            .clone_from_slice(&[0; 8]);

        // buf_fixed_size[VEC3_UNIFORM_SIZE + F32_SIZE..LIGHTING_SIZE_FIXED]
        //     .clone_from_slice(&[0; 12]);

        for byte in buf_fixed_size.into_iter() {
            result.push(byte);
        }

        for light in &self.point_lights {
            for byte in light.to_bytes().into_iter() {
                result.push(byte)
            }
        }

        result
    }
}

#[derive(Debug, Clone)]
pub enum LightType {
    Omnidirectional,
    Directional(Vec3), // direction pointed at // todo: FOV?
    Diffuse,
}

#[derive(Clone, Debug)]
pub struct PointLight {
    // A point light source
    pub type_: LightType,
    pub position: Vec3,
    pub diffuse_color: [f32; 4],
    pub specular_color: [f32; 4],
    pub diffuse_intensity: f32,
    pub specular_intensity: f32,
    // todo: FOV, and direction?
    // shadow_map
}

impl PointLight {
    /// todo: assumes point source for now; ignore type_ field.
    pub fn to_bytes(&self) -> [u8; POINT_LIGHT_SIZE] {
        let mut result = [0; POINT_LIGHT_SIZE];

        // 16 is vec3 size in bytes, including padding.
        result[0..VEC3_UNIFORM_SIZE].clone_from_slice(&self.position.to_bytes_uniform());
        result[VEC3_UNIFORM_SIZE..2 * VEC3_UNIFORM_SIZE]
            .clone_from_slice(&array4_to_bytes(self.diffuse_color));
        result[2 * VEC3_UNIFORM_SIZE..3 * VEC3_UNIFORM_SIZE]
            .clone_from_slice(&array4_to_bytes(self.specular_color));

        result[3 * VEC3_UNIFORM_SIZE..3 * VEC3_UNIFORM_SIZE + F32_SIZE]
            .clone_from_slice(&self.diffuse_intensity.to_ne_bytes());
        result[3 * VEC3_UNIFORM_SIZE + F32_SIZE..3 * VEC3_UNIFORM_SIZE + 2 * F32_SIZE]
            .clone_from_slice(&self.specular_intensity.to_ne_bytes());
        result[3 * VEC3_UNIFORM_SIZE + 2 * F32_SIZE..POINT_LIGHT_SIZE].clone_from_slice(&[0; 8]);

        result
    }
}
