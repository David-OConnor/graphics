//! Code to manage the camera.

use core::f32::consts::TAU;
use std::f32::consts::LN_2;
use lin_alg::f32::{Mat4, Quaternion, Vec3};

use crate::types::{F32_SIZE, MAT4_SIZE, VEC3_UNIFORM_SIZE};

// cam size is only the parts we pass to the shader.
// For each of the 4 matrices in the camera, plus a padded vec3 for position.
pub const CAMERA_SIZE: usize = MAT4_SIZE + 3 * VEC3_UNIFORM_SIZE + 16; // Final 16 is an alignment pad.

#[derive(Clone, Debug)]
pub struct Camera {
    pub fov_y: f32,  // Vertical field of view in radians.
    pub aspect: f32, // width / height.
    pub near: f32,
    pub far: f32,
    /// Position shifts all points prior to the camera transform; this is what
    /// we adjust with move keys.
    pub position: Vec3,
    pub orientation: Quaternion,
    /// We store the projection matrix here since it only changes when we change the camera cfg.
    pub proj_mat: Mat4, // todo: Make provide, and provide a constructor?
    /// ln(2) / half_distance.   0. = disabled
    pub fog_density: f32,
    // todo: Fog color is inoperative at this time.
    pub fog_color: [f32; 3],
}

impl Camera {
    pub fn to_bytes(&self) -> [u8; CAMERA_SIZE] {
        let mut result = [0; CAMERA_SIZE];

        let proj_view = self.proj_mat.clone() * self.view_mat();

        let mut i = 0;

        result[i..i + MAT4_SIZE].clone_from_slice(&proj_view.to_bytes());
        i += MAT4_SIZE;

        result[i..i + VEC3_UNIFORM_SIZE].clone_from_slice(&self.position.to_bytes_uniform());
        i += VEC3_UNIFORM_SIZE;

        result[i..i + F32_SIZE].clone_from_slice(&self.fog_density.to_ne_bytes());
        i += VEC3_UNIFORM_SIZE; // for 16-byte alignment.

        result[i..i + F32_SIZE].clone_from_slice(&self.fog_color[0].to_ne_bytes());
        i += F32_SIZE;
        result[i..i + F32_SIZE].clone_from_slice(&self.fog_color[1].to_ne_bytes());
        i += F32_SIZE;
        result[i..i + F32_SIZE].clone_from_slice(&self.fog_color[2].to_ne_bytes());
        // Pad if we add more fields for 16-byte alignment.

        result
    }

    /// Updates the projection matrix based on the projection parameters.
    /// Run this after updating the parameters.
    pub fn update_proj_mat(&mut self) {
        self.proj_mat = Mat4::new_perspective_lh(self.fov_y, self.aspect, self.near, self.far);
    }

    /// Calculate the view matrix: This is a translation of the negative coordinates of the camera's
    /// position, applied before the camera's rotation.
    pub fn view_mat(&self) -> Mat4 {
        self.orientation.inverse().to_matrix() * Mat4::new_translation(-self.position)
    }

    pub fn view_size(&self, far: bool) -> (f32, f32) {
        // Calculate the projected window width and height, using basic trig.
        let dist = if far { self.far } else { self.near };

        let width = 2. * dist * (self.fov_y * self.aspect / 2.).tan();
        let height = 2. * dist * (self.fov_y / 2.).tan();
        (width, height)
    }

    /// Set fog density so that objects at the specified distance are attenuated to 50% of their
    /// original visibility.
    pub fn set_fog_half_distance(&mut self, half_distance: Option<f32>) {
        self.fog_density = half_distance.map_or(0.0, |d| LN_2 / d.max(1e-6));
    }
}

impl Default for Camera {
    fn default() -> Self {
        let mut result = Self {
            position: Vec3::new(0., 0., 0.),
            orientation: Quaternion::new_identity(),
            fov_y: TAU / 6., // Vertical field of view in radians.
            aspect: 4. / 3., // width / height.
            near: 0.5,
            far: 60.,
            proj_mat: Mat4::new_identity(),
            fog_density: 0.,
            fog_color: [0., 0., 0.],
        };

        result.update_proj_mat();
        result
    }
}
