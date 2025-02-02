use lin_alg::f32::{Mat4, Vec3, Vec4};

use crate::Camera;

/// Convert a screen position (x, y) to a 3D ray in world space.
/// `z_limits` determines the near and far points along the ray.
///
/// The canonical use case for this is finding the object in 3D space a user is intending to select
/// with the cursor.A follow-up operation, for example, may be to find all objects that this vector
/// passes near, and possibly select the one closest to the camera.
pub fn screen_to_render(screen_pos: (f32, f32), cam: &Camera) -> (Vec3, Vec3) {
    let view_proj = cam.view_mat() * cam.proj_mat.clone();

    let view_proj_inv = match view_proj.inverse() {
        Some(p) => p,
        None => {
            eprintln!("Error inverting the projection matrix.");
            return (Vec3::new_zero(), Vec3::new_zero());
        }
    };

    // Convert screen position (assumed normalized to [-1, 1] range)
    let (x, y) = screen_pos;

    // Near point in clip space (homogeneous coordinates)
    let near_clip = Vec4::new(x, y, cam.near, 1.0);
    let far_clip = Vec4::new(x, y, cam.far, 1.0);

    // Transform from clip space to world space
    let near_world = view_proj_inv.clone() * near_clip;
    let far_world = view_proj_inv * far_clip;

    // Perspective divide (convert from homogeneous to 3D)
    let near_world = Vec3::new(near_world.x, near_world.y, near_world.z) / near_world.w;
    let far_world = Vec3::new(far_world.x, far_world.y, far_world.z) / far_world.w;

    (near_world, far_world)
}
