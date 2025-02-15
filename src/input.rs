//! Handles keyboard and mouse input, eg for moving the camera.

use egui::Key;
use lin_alg::f32::{Quaternion, Vec3};
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

const MOUSE_0_ID: u32 = 0;
const MOUSE_1_ID: u32 = 1;

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
pub(crate) fn add_input_cmd(event: DeviceEvent, inputs: &mut InputsCommanded) {
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
            if button == MOUSE_0_ID {
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
pub fn adjust_camera(
    cam: &mut Camera,
    inputs: &InputsCommanded,
    input_settings: &InputSettings,
    dt: f32,
) -> bool {
    let mut move_amt: f32 = input_settings.move_sens * dt;
    let rotate_amt: f32 = input_settings.rotate_sens * dt;
    let mut rotate_key_amt: f32 = input_settings.rotate_key_sens * dt;

    // todo: This split is where you can decouple WGPU-specific code from general code.

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
