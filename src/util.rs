use lin_alg::f32::{Mat4, Vec3, Vec4};

use crate::Camera;

/// Convert a screen position (x, y) to a 3D ray in world space.
///
/// The canonical use case for this is finding the object in 3D space a user is intending to select
/// with the cursor.A follow-up operation, for example, may be to find all objects that this vector
/// passes near, and possibly select the one closest to the camera.
pub fn screen_to_render(screen_pos: (f32, f32), window_dims: (f32, f32), cam: &Camera) -> (Vec3, Vec3) {
    // let view_proj = cam.view_mat() * cam.proj_mat.clone();
    let proj_view = cam.proj_mat.clone() * cam.view_mat();

    let proj_view_inv = match proj_view.inverse() {
        Some(p) => p,
        None => {
            eprintln!("Error inverting the projection matrix.");
            return (Vec3::new_zero(), Vec3::new_zero());
        }
    };

    // Convert screen position (assumed normalized to [0, 1] range)
    let sx = screen_pos.0 / window_dims.0;
    let sy = screen_pos.1 / window_dims.1;

    let clip_x = 2.0 * sx - 1.0;
    let clip_y = 1.0 - 2.0 * sy;  // Flips the Y so 0 is top, 1 is bottom

    let near_clip = Vec4::new(clip_x, clip_y, 0.0, 1.0);
    let far_clip = Vec4::new(clip_x, clip_y, 1.0, 1.0);

    // Un-project them to world space.
    let near_world_h = proj_view_inv.clone() * near_clip;
    let far_world_h = proj_view_inv * far_clip;

    // Perspective divide to go from homogenous -> 3D.
    let near_world = near_world_h.xyz() / near_world_h.w;
    let far_world = far_world_h.xyz() / far_world_h.w;

    (near_world, far_world)
}
