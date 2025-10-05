//! Code to manage the camera.

use core::f32::consts::TAU;

use lin_alg::f32::{Mat4, Quaternion, Vec3, Vec4};

use crate::{
    copy_ne,
    types::{F32_SIZE, MAT4_SIZE, VEC3_UNIFORM_SIZE},
};

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
    /// These are in distances from the camera.
    /// E scale within band. 1.0 is a good baseline.
    pub fog_density: f32,
    /// Curve steepness, e.g. 4–8. Higher means more of a near "wall" with heavy far fade.
    pub fog_power: f32,
    /// distance where fog begins
    pub fog_start: f32,
    /// Distance where fog reaches full strength
    pub fog_end: f32,
    pub fog_color: [f32; 3],
}

impl Camera {
    pub fn to_bytes(&self) -> [u8; CAMERA_SIZE] {
        let mut result = [0; CAMERA_SIZE];
        let mut i = 0;

        let proj_view = self.proj_mat.clone() * self.view_mat();

        result[i..i + MAT4_SIZE].clone_from_slice(&proj_view.to_bytes());
        i += MAT4_SIZE;

        result[i..i + VEC3_UNIFORM_SIZE].clone_from_slice(&self.position.to_bytes_uniform());
        i += VEC3_UNIFORM_SIZE;

        copy_ne!(result, self.fog_density, i..i + F32_SIZE);
        i += F32_SIZE;
        copy_ne!(result, self.fog_power, i..i + F32_SIZE);
        i += F32_SIZE;
        copy_ne!(result, self.fog_start, i..i + F32_SIZE);
        i += F32_SIZE;
        copy_ne!(result, self.fog_end, i..i + F32_SIZE);
        i += F32_SIZE;

        copy_ne!(result, self.fog_color[0], i..i + F32_SIZE);
        i += F32_SIZE;
        copy_ne!(result, self.fog_color[1], i..i + F32_SIZE);
        i += F32_SIZE;
        copy_ne!(result, self.fog_color[1], i..i + F32_SIZE);

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

    /// A utility function not used by the engine, but may be called by applications.
    pub fn in_view(&self, point: Vec3) -> bool {
        // todo: QC this
        let pv = self.proj_mat.clone() * self.view_mat();
        let p = pv * Vec4::new(point.x, point.y, point.z, 1.0);
        let w = p.w;

        if w <= 0.0 {
            return false;
        }

        let x = p.x / w;
        let y = p.y / w;
        let z = p.z / w;

        x >= -1.0 && x <= 1.0 && y >= -1.0 && y <= 1.0 && z >= 0.0 && z <= 1.0
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
            fog_density: 1.,
            fog_power: 6.,
            fog_start: 0.,
            fog_end: 0.,
            fog_color: [0., 0., 0.],
        };

        result.update_proj_mat();
        result
    }
}
