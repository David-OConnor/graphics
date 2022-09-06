//! Code to manage the camera.

use core::f32::consts::TAU;

use crate::{
    init_graphics::{FWD_VEC, RIGHT_VEC, UP_VEC},
    types::{MAT4_SIZE, VEC3_UNIFORM_SIZE},
};

use lin_alg2::f32::{Mat4, Quaternion, Vec3};

// cam size is only the parts we pass to the shader.
// For each of the 4 matrices in the camera, plus a padded vec3 for position.
pub const CAMERA_SIZE: usize = MAT4_SIZE + VEC3_UNIFORM_SIZE;

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
    pub proj_mat: Mat4,
}

impl Camera {
    pub fn to_bytes(&self) -> [u8; CAMERA_SIZE] {
        let mut result = [0; CAMERA_SIZE];

        let proj_view = self.proj_mat.clone() * self.view_mat();

        result[0..MAT4_SIZE].clone_from_slice(&proj_view.to_bytes());
        result[MAT4_SIZE..CAMERA_SIZE].clone_from_slice(&self.position.to_bytes_uniform());

        result
    }

    pub fn update_proj_mat(&mut self) {
        self.proj_mat = Mat4::new_perspective_lh(self.fov_y, self.aspect, self.near, self.far);
    }

    /// Calculate the view matrix: This is a translation of the negative coordinates of the camera's
    /// position, applied before the camera's rotation.
    pub fn view_mat(&self) -> Mat4 {
        self.orientation.inverse().to_matrix() * Mat4::new_translation(self.position * -1.)
        // self.orientation.to_matrix() * Mat4::new_translation(self.position * -1.)
    }

    pub fn view_size(&self, far: bool) -> (f32, f32) {
        // Calculate the projected window width and height, using basic trig.
        let dist = if far { self.far } else { self.near };

        let width = 2. * dist * (self.fov_y * self.aspect / 2.).tan();
        let height = 2. * dist * (self.fov_y / 2.).tan();
        (width, height)
    }
}

impl Default for Camera {
    fn default() -> Self {
        let mut result = Self {
            position: Vec3::new(0., 0., 0.),
            orientation: Quaternion::new_identity(),
            fov_y: TAU / 5., // Vertical field of view in radians.
            aspect: 4. / 3., // width / height.
            near: 0.5,
            far: 60.,
            proj_mat: Mat4::new_identity(),
        };

        result.update_proj_mat();
        result
    }
}
