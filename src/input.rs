//! Handles keyboard and mouse input, eg for moving the camera.

use egui::Key;
use lin_alg::f32::{Mat3, Quaternion, Vec3};
// todo: remove Winit from this module if you can, and make it agnostic?
use winit::event::{DeviceEvent, ElementState};
use winit::{
    keyboard::{KeyCode, PhysicalKey::Code},
    platform::scancode::PhysicalKeyExtScancode,
};

use crate::{
    camera::Camera,
    graphics::{FWD_VEC, RIGHT_VEC, UP_VEC},
    types::InputSettings,
};

const LEFT_CLICK: u32 = 0;
const RIGHT_CLICK: u32 = 1;

#[derive(Default, Debug)]
pub struct InputsCommanded {
    pub fwd: bool,
    pub back: bool,
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,
    pub roll_ccw: bool,
    pub roll_cw: bool,
    pub mouse_delta_x: f32,
    pub mouse_delta_y: f32,
    pub run: bool,
    pub free_look: bool,
}

impl InputsCommanded {
    /// Return true if there are any inputs.
    pub fn inputs_present(&self) -> bool {
        const EPS: f32 = 0.00001;
        // Note; We don't include `run` or `free_look` here, since it's a modifier.
        self.fwd
            || self.back
            || self.left
            || self.right
            || self.up
            || self.down
            || self.roll_ccw
            || self.roll_cw
            || self.mouse_delta_x.abs() > EPS
            || self.mouse_delta_y.abs() > EPS
    }
}

/// Modifies the commanded inputs in place; triggered by a single input event.
/// dt is in seconds.
/// pub(crate) fn handle_event(event: DeviceEvent, cam: &mut Camera, input_settings: &InputSettings, dt: f32) {
pub(crate) fn add_input_cmd(event: &DeviceEvent, inputs: &mut InputsCommanded) {
    match event {
        DeviceEvent::Key(key) => {
            if key.state == ElementState::Pressed {
                match key.physical_key {
                    Code(key) => match key {
                        KeyCode::KeyW => {
                            inputs.fwd = true;
                        }
                        KeyCode::KeyS => {
                            inputs.back = true;
                        }
                        KeyCode::KeyA => {
                            inputs.left = true;
                        }
                        KeyCode::KeyD => {
                            inputs.right = true;
                        }
                        KeyCode::Space => {
                            inputs.up = true;
                        }
                        KeyCode::KeyC => {
                            inputs.down = true;
                        }
                        KeyCode::KeyQ => {
                            inputs.roll_ccw = true;
                        }
                        KeyCode::KeyE => {
                            inputs.roll_cw = true;
                        }
                        KeyCode::ShiftLeft => {
                            inputs.run = true;
                        }
                        _ => (),
                    },
                    _ => (),
                }
            } else if key.state == ElementState::Released {
                // todo: DRY
                match key.physical_key {
                    Code(key) => match key {
                        KeyCode::KeyW => {
                            inputs.fwd = false;
                        }
                        KeyCode::KeyS => {
                            inputs.back = false;
                        }
                        KeyCode::KeyA => {
                            inputs.left = false;
                        }
                        KeyCode::KeyD => {
                            inputs.right = false;
                        }
                        KeyCode::Space => {
                            inputs.up = false;
                        }
                        KeyCode::KeyC => {
                            inputs.down = false;
                        }
                        KeyCode::KeyQ => {
                            inputs.roll_ccw = false;
                        }
                        KeyCode::KeyE => {
                            inputs.roll_cw = false;
                        }
                        KeyCode::ShiftLeft => {
                            inputs.run = false;
                        }
                        _ => (),
                    },
                    _ => (),
                }
            }
        }
        DeviceEvent::Button { button, state } => {
            if *button == LEFT_CLICK {
                inputs.free_look = match state {
                    ElementState::Pressed => true,
                    ElementState::Released => false,
                }
            }
        }
        DeviceEvent::MouseMotion { delta } => {
            inputs.mouse_delta_x += delta.0 as f32;
            inputs.mouse_delta_y += delta.1 as f32;
        }
        _ => (),
    }
}

/// Adjust the camera orientation and position. Return if there was a change, so we know to update the buffer.
/// For the free (6DOF first-person) camera.
pub fn adjust_camera_free(
    cam: &mut Camera,
    inputs: &InputsCommanded,
    input_settings: &InputSettings,
    dt: f32,
) -> bool {
    let mut move_amt: f32 = input_settings.move_sens * dt;
    let rotate_amt: f32 = input_settings.rotate_sens * dt;
    let mut rotate_key_amt: f32 = input_settings.rotate_key_sens * dt;

    let mut cam_moved = false;
    let mut cam_rotated = false;

    let mut movement_vec = Vec3::new_zero();

    if inputs.run {
        move_amt *= input_settings.run_factor;
        rotate_key_amt *= input_settings.run_factor;
    }

    if inputs.fwd {
        movement_vec.z += move_amt;
        cam_moved = true;
    } else if inputs.back {
        movement_vec.z -= move_amt;
        cam_moved = true;
    }

    if inputs.right {
        movement_vec.x += move_amt;
        cam_moved = true;
    } else if inputs.left {
        movement_vec.x -= move_amt;
        cam_moved = true;
    }

    if inputs.up {
        movement_vec.y += move_amt;
        cam_moved = true;
    } else if inputs.down {
        movement_vec.y -= move_amt;
        cam_moved = true;
    }

    let fwd = cam.orientation.rotate_vec(FWD_VEC);
    // todo: Why do we need to reverse these?
    let up = cam.orientation.rotate_vec(UP_VEC * -1.);
    let right = cam.orientation.rotate_vec(RIGHT_VEC * -1.);

    let mut rotation = Quaternion::new_identity();

    // todo: Why do we need to reverse these?
    if inputs.roll_cw {
        rotation = Quaternion::from_axis_angle(fwd, -rotate_key_amt);
        cam_rotated = true;
    } else if inputs.roll_ccw {
        rotation = Quaternion::from_axis_angle(fwd, rotate_key_amt);
        cam_rotated = true;
    }

    let eps = 0.00001;

    if inputs.free_look && (inputs.mouse_delta_x.abs() > eps || inputs.mouse_delta_y.abs() > eps) {
        // todo: Why do we have the negative signs here?
        rotation = Quaternion::from_axis_angle(up, -inputs.mouse_delta_x * rotate_amt)
            * Quaternion::from_axis_angle(right, -inputs.mouse_delta_y * rotate_amt)
            * rotation;

        cam_rotated = true;
    }

    if cam_moved {
        cam.position += cam.orientation.rotate_vec(movement_vec);
    }

    if cam_rotated {
        cam.orientation = rotation * cam.orientation;
    }

    cam_moved || cam_rotated
}

/// Adjust the camera orientation and position. Return if there was a change, so we know to update the buffer.
pub fn adjust_camera_arc(
    cam: &mut Camera,
    inputs: &InputsCommanded,
    input_settings: &InputSettings,
    center: Vec3,
    dt: f32,
) -> bool {
    // How fast we rotate, derived from your input settings:
    let rotate_amt: f32 = input_settings.rotate_sens * dt;
    let eps = 0.000_01;

    // Track if we actually moved/rotated:
    let mut cam_rotated = false;

    // Vector from `center` to current camera position:
    let mut offset = cam.position - center;

    // Only rotate if "free look" is active and the mouse moved enough:
    if inputs.free_look && (inputs.mouse_delta_x.abs() > eps || inputs.mouse_delta_y.abs() > eps) {
        // Typically, "yaw" around the global up axis:
        let global_up = Vec3::new(0.0, 1.0, 0.0);
        let yaw = Quaternion::from_axis_angle(global_up, -inputs.mouse_delta_x * rotate_amt);

        // For "pitch," use a sideways axis, which is the cross of offset × up:
        // (We normalize to avoid floating accumulation.)
        let right_axis = offset.cross(global_up).to_normalized();
        let pitch = Quaternion::from_axis_angle(right_axis, -inputs.mouse_delta_y * rotate_amt);

        // Combined rotation for orbit:
        let orbit_rotation = yaw * pitch;

        // Apply rotation to the offset-from-center:
        offset = orbit_rotation.rotate_vec(offset);

        // Update the camera's position around the center:
        cam.position = center + offset;

        cam_rotated = true;
    }

    // Reorient the camera so it looks at `center`:
    // 1) Forward is from camera to center.
    let new_forward = (center - cam.position).to_normalized();

    // 2) Use a desired "up" to help compute orientation basis.
    //    Typically we pick a world up, then correct it in case we are near the pole.
    let world_up = Vec3::new(0.0, 1.0, 0.0);

    // 3) Right vector is forward × up
    let new_right = new_forward.cross(world_up).to_normalized();
    // 4) Corrected up is right × forward
    let corrected_up = new_right.cross(new_forward).to_normalized();

    // todo: I don't like this. From chatGPT. We shouldn't use a matrix.
    // Convert these basis vectors into a rotation (orientation) for the camera.
    // Note that some systems prefer forward = -Z; adjust as needed.
    // For example, if your camera's local FWD is -Z, you can invert the forward axis below.

    // let orientation_mat = Mat3::from_cols(new_right, corrected_up, -new_forward);
    // cam.orientation = Quaternion::from_mat3(&orientation_mat);

    // cam.orientation = Quaternion::from_unit_vecs(new_right, corrected_up);
    // cam.orientation = Quaternion::from_unit_vecs(corrected_up, new_right);

    cam.orientation = Quaternion::from_unit_vecs(UP_VEC, corrected_up);

    cam_rotated
}
