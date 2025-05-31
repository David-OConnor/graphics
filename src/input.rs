//! Handles keyboard and mouse input, eg for moving the camera.

use lin_alg::f32::{Quaternion, Vec3};
// todo: remove Winit from this module if you can, and make it agnostic?
use winit::event::{DeviceEvent, ElementState, MouseScrollDelta};
use winit::keyboard::{KeyCode, PhysicalKey::Code};

use crate::{
    ScrollBehavior,
    camera::Camera,
    graphics::{FWD_VEC, RIGHT_VEC, UP_VEC},
    types::InputSettings,
};

const EPS_MOUSE: f32 = 0.00001;

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
    pub scroll_up: bool,
    pub scroll_down: bool,
    pub free_look: bool,
    pub panning: bool, // todo: Implement A/R
}

impl InputsCommanded {
    /// Return true if there are any inputs.
    pub fn inputs_present(&self) -> bool {
        // Note; We don't include `run` or `free_look` here, since they're modifiers..
        self.fwd
            || self.back
            || self.left
            || self.right
            || self.up
            || self.down
            || self.roll_ccw
            || self.roll_cw
            || self.mouse_delta_x.abs() > EPS_MOUSE
            || self.mouse_delta_y.abs() > EPS_MOUSE
            || self.scroll_up
            || self.scroll_down
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
            // todo: Experiment?
            #[cfg(target_os = "linux")]
            let left_click = 1;
            #[cfg(not(target_os = "linux"))]
            let left_click = 0;

            // What happened: left click (event 0) triggered behavior of event 1.

            if *button == left_click {
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
        // Move the camera forward and back on scroll.
        DeviceEvent::MouseWheel { delta } => match delta {
            MouseScrollDelta::PixelDelta(_) => (),
            MouseScrollDelta::LineDelta(_x, y) => {
                if *y > 0. {
                    inputs.scroll_down = true;
                } else {
                    inputs.scroll_up = true;
                }
            }
        },
        _ => (),
    }
}

fn handle_scroll(
    cam: &mut Camera,
    inputs: &mut InputsCommanded,
    input_settings: &InputSettings,
    dt: f32,
    movement_vec: &mut Vec3,
    rotation: &mut Quaternion,
    cam_moved: &mut bool,
    cam_rotated: &mut bool,
) {
    if inputs.scroll_down || inputs.scroll_up {
        if let ScrollBehavior::MoveRoll {
            move_amt,
            rotate_amt,
        } = input_settings.scroll_behavior
        {
            if inputs.free_look {
                // Roll if left button down while scrolling
                let fwd = cam.orientation.rotate_vec(FWD_VEC);

                let mut rot_amt = -rotate_amt * dt;
                if inputs.scroll_down {
                    rot_amt *= -1.; // todo: Allow reversed behavior for arc cam?
                }

                *rotation = Quaternion::from_axis_angle(fwd, rot_amt);
                *cam_rotated = true;
            } else {
                // Otherwise, move forward and backward.
                let mut movement = Vec3::new(0., 0., move_amt);
                if inputs.scroll_up {
                    movement *= -1.;
                }
                *movement_vec += movement;

                *cam_moved = true;
            }
        }

        // Immediately send the "release" command; not on a Release event like keys.
        inputs.scroll_down = false;
        inputs.scroll_up = false;
    }
}

/// Used internally for inputs, and externally, e.g. to command an arc rotation.
pub fn arc_rotation(cam: &mut Camera, axis: Vec3, amt: f32, center: Vec3) {
    let rotation = Quaternion::from_axis_angle(axis, amt);

    cam.orientation = (rotation * cam.orientation).to_normalized();

    let dist = (cam.position - center).magnitude();
    // Update position based on the new orientation.
    cam.position = center - cam.orientation.rotate_vec(FWD_VEC) * dist;
}

/// Adjust the camera orientation and position. Return if there was a change, so we know to update the buffer.
/// For the free (6DOF first-person) camera.
pub fn adjust_camera_free(
    cam: &mut Camera,
    inputs: &mut InputsCommanded,
    input_settings: &InputSettings,
    dt: f32,
) -> bool {
    let mut move_amt = input_settings.move_sens * dt;
    let mut rotate_key_amt = input_settings.rotate_key_sens * dt;

    let mut cam_moved = false;
    let mut cam_rotated = false;

    let mut movement_vec = Vec3::new_zero();
    let mut rotation = Quaternion::new_identity();

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

    if inputs.roll_cw {
        let fwd = cam.orientation.rotate_vec(FWD_VEC);
        rotation = Quaternion::from_axis_angle(fwd, -rotate_key_amt);
        cam_rotated = true;
    } else if inputs.roll_ccw {
        let fwd = cam.orientation.rotate_vec(FWD_VEC);
        rotation = Quaternion::from_axis_angle(fwd, rotate_key_amt);
        cam_rotated = true;
    }

    if inputs.free_look
        && (inputs.mouse_delta_x.abs() > EPS_MOUSE || inputs.mouse_delta_y.abs() > EPS_MOUSE)
    {
        let rotate_amt = input_settings.rotate_sens * dt;
        let up = cam.orientation.rotate_vec(-UP_VEC);
        let right = cam.orientation.rotate_vec(-RIGHT_VEC);

        rotation = Quaternion::from_axis_angle(up, -inputs.mouse_delta_x * rotate_amt)
            * Quaternion::from_axis_angle(right, -inputs.mouse_delta_y * rotate_amt)
            * rotation;

        cam_rotated = true;
    }

    handle_scroll(
        cam,
        inputs,
        input_settings,
        dt,
        &mut movement_vec,
        &mut rotation,
        &mut cam_moved,
        &mut cam_rotated,
    );

    // todo: Handle middleclick + drag here too. Move `mol_dock`'s impl here.

    if cam_rotated {
        cam.orientation = (rotation * cam.orientation).to_normalized();
    }

    if cam_moved {
        cam.position += cam.orientation.rotate_vec(movement_vec);
    }

    cam_moved || cam_rotated
}

/// Adjust the camera orientation and position. Return if there was a change, so we know to update the buffer.
/// For the arc (orbital) camera.
pub fn adjust_camera_arc(
    cam: &mut Camera,
    inputs: &mut InputsCommanded,
    input_settings: &InputSettings,
    center: Vec3,
    dt: f32,
) -> bool {
    // How fast we rotate, derived from your input settings:
    // Track if we actually moved/rotated:
    let mut cam_moved = false;
    let mut cam_rotated = false;

    let mut movement_vec = Vec3::new_zero();
    let mut rotation = Quaternion::new_identity();

    // todo: Combine this fn with adjust_free, and accept ControlScheme as a param.

    // Inverse of free
    let rotate_key_amt = -input_settings.rotate_key_sens * dt;

    if inputs.roll_cw {
        let fwd = cam.orientation.rotate_vec(FWD_VEC);
        rotation = Quaternion::from_axis_angle(fwd, -rotate_key_amt);
        cam_rotated = true;
    } else if inputs.roll_ccw {
        let fwd = cam.orientation.rotate_vec(FWD_VEC);
        rotation = Quaternion::from_axis_angle(fwd, rotate_key_amt);
        cam_rotated = true;
    }

    let mut skip_move_vec = false;
    // Only rotate if "free look" is active and the mouse moved enough:
    if inputs.free_look
        && (inputs.mouse_delta_x.abs() > EPS_MOUSE || inputs.mouse_delta_y.abs() > EPS_MOUSE)
    {
        let rotate_amt = input_settings.rotate_sens * dt;
        let up = cam.orientation.rotate_vec(-UP_VEC);
        let right = cam.orientation.rotate_vec(-RIGHT_VEC);

        // Rotation logic: Equivalent to the free camera.
        rotation = Quaternion::from_axis_angle(up, -inputs.mouse_delta_x * rotate_amt)
            * Quaternion::from_axis_angle(right, -inputs.mouse_delta_y * rotate_amt);

        // Distance between cam and center is invariant under this change.
        skip_move_vec = true;

        cam_moved = true;
        cam_rotated = true;
    }

    handle_scroll(
        cam,
        inputs,
        input_settings,
        dt,
        &mut movement_vec,
        &mut rotation,
        &mut cam_moved,
        &mut cam_rotated,
    );

    // todo: Handle middleclick + drag here too. Move `mol_dock`'s impl here.

    // note: We are not using our `arc_roation` fn here due to needing to handle both x and y. Hmm.
    // arc_rotation(cam, axis, amt, center);

    if cam_rotated {
        cam.orientation = (rotation * cam.orientation).to_normalized();
    }

    if cam_moved && !skip_move_vec {
        cam.position += cam.orientation.rotate_vec(movement_vec);
    } else if cam_moved {
        // todo: Bit odd to break this off from the above.
        let dist = (cam.position - center).magnitude();
        // Update position based on the new orientation.
        cam.position = center - cam.orientation.rotate_vec(FWD_VEC) * dist;
    }

    cam_moved || cam_rotated
}
