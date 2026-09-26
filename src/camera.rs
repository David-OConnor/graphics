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

/// The handedness of the world coordinate system. When viewed with +X pointing right, and +Y
/// pointing up, this determines if +Z points into the screen, or out of it.
///
/// Rendering data authored in one handedness with the other displays its mirror image; no
/// camera rotation can undo this. For example, molecular coordinates (PDB, mmCIF, SDF etc.)
/// are right-handed; displaying them as left-handed turns alpha helices left-handed, and inverts
/// chirality.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Handedness {
    /// +X right and +Y up means +Z points into the screen, away from the viewer. This is the
    /// convention of DirectX and Unity.
    ///
    /// An identity camera orientation looks along +Z, so it shows +X right, and +Y up.
    Left,
    /// +X right and +Y up means +Z points out of the screen, toward the viewer. This is the
    /// convention of OpenGL, most scientific software, and molecular viewers such as PyMOL,
    /// Chimera, and Mol*.
    ///
    /// An identity camera orientation looks along +Z, so it shows +X *left*, and +Y up. To view
    /// with +X right, and +Y up, look along -Z: e.g. rotate the camera a half turn around +Y.
    ///
    /// We implement this by mirroring the screen's X axis. `FWD_VEC` and `UP_VEC` remain camera
    /// forward and up, but `RIGHT_VEC` (camera-local +X) points to the *left* of the screen.
    /// See `Camera::screen_x_sign`.
    #[default]
    Right,
}

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
    /// Strength of edge cueing (silhouette darkening). 0.0 = off, 1.0 = full effect.
    /// Controlled at startup via GraphicsSettings::edge_cueing.
    pub edge_cueing: f32,
    /// World-space expansion along normals used in the depth-aware halo prepass.
    /// 0.0 = disabled. Set from GraphicsSettings::depth_aware_halos.
    pub halo_expansion: f32,
    /// Set this prior to starting the engine; it affects face culling, which we configure
    /// at init. Run `update_proj_mat` after changing it.
    pub handedness: Handedness,
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
        copy_ne!(result, self.fog_color[2], i..i + F32_SIZE);

        // WGSL layout: fog_color: vec3<f32> at 96..108 (12 bytes).
        // After it, edge_cueing at 108, near at 112, far at 116.
        // These fit within CAMERA_SIZE = 128 (previously unused padding).
        copy_ne!(result, self.edge_cueing, 108..112);
        copy_ne!(result, self.near, 112..116);
        copy_ne!(result, self.far, 116..120);
        copy_ne!(result, self.halo_expansion, 120..124);

        result
    }

    /// Updates the projection matrix based on the projection parameters.
    /// Run this after updating the parameters.
    pub fn update_proj_mat(&mut self) {
        let proj = Mat4::new_perspective_lh(self.fov_y, self.aspect, self.near, self.far);

        self.proj_mat = match self.handedness {
            Handedness::Left => proj,
            // Mirroring clip-space X converts the left-handed projection into a right-handed
            // one, while keeping +Z forward, and +Y up in camera space.
            Handedness::Right => Mat4::new_scaler_partial(Vec3::new(-1., 1., 1.)) * proj,
        };
    }

    /// 1 if camera-local +X (`RIGHT_VEC`) points to the right of the screen, and -1 if
    /// it points left, as it does with right-handed coordinates.
    ///
    /// Use this when converting screen-space input (e.g. mouse motion) to camera motion:
    /// Multiply movement along camera-local X, and rotations about camera-local Y (yaw) and
    /// Z (roll) by it. Rotations about camera-local X (pitch) are unaffected.
    pub fn screen_x_sign(&self) -> f32 {
        match self.handedness {
            Handedness::Left => 1.,
            Handedness::Right => -1.,
        }
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

    /// Determines if an object is in view. Also returns NDC coordinates, for use
    /// in some applications.
    pub fn in_view(&self, point: Vec3) -> (bool, (f32, f32, f32)) {
        let pv = self.proj_mat.clone() * self.view_mat();
        let p4 = pv * Vec4::new(point.x, point.y, point.z, 1.0);

        if p4.w <= 0.0 {
            return Default::default();
        }

        let inv_w = 1.0 / p4.w;
        let x = p4.x * inv_w;
        let y = p4.y * inv_w;
        let z = p4.z * inv_w;

        let iv =
            (-1.0..=1.0).contains(&x) && (-1.0..=1.0).contains(&y) && (-1.0..=1.0).contains(&z);

        (iv, (x, y, z))
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
            edge_cueing: 0.,
            halo_expansion: 0.,
            handedness: Default::default(),
        };

        result.update_proj_mat();
        result
    }
}
