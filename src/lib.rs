#![allow(mixed_script_confusables)] // Theta in meshes
#![allow(clippy::too_many_arguments)]

//! A 3D rendering engine for rust programs, with GUI integration
//!
//! This library is a framework for building PC applications that have 3D graphics, and a GUI. It uses
//! the [WGPU toolkit](https://wgpu.rs/) with Vulkan backend, and [EGUI](https://docs.rs/egui/latest/egui/).
//! It works on Windows, Linux, and Mac.
//!
//! This is intended as a general-purpose 3D visualization tool.
//! Example use cases including wave-function analysis, n-body simulations, and protein structure viewing.
//! It's also been used to visualize UAS attitude in preflight software. Its goals are to be intuitive and flexible.

#[cfg(feature = "app_utils")]
pub mod app_utils;
mod camera;
mod gauss;
mod graphics;
mod gui;
mod input;
pub mod lighting;
mod meshes;
mod system;
mod texture;
mod types;
mod window;

pub use camera::Camera;
pub use gauss::Gaussian;
pub use graphics::{EntityUpdate, FWD_VEC, RIGHT_VEC, UP_VEC};
pub use input::{InputsCommanded, adjust_camera_free, arc_rotation};
pub use lighting::{LightType, Lighting, PointLight};
pub use system::run;
pub use types::{
    ControlScheme, EngineUpdates, Entity, GraphicsSettings, InputSettings, Mesh, Scene,
    ScrollBehavior, UiLayout, UiSettings, Vertex,
};
// Re-export winit DeviceEvents for use in the API; this prevents the calling
// lib from needing to use winit as a dependency directly.
// todo: the equiv for mouse events too. And in the future, Gamepad events.
pub use winit::{
    self,
    event::{self, DeviceEvent, ElementState, WindowEvent},
};

// A helper macro. Not intended for use outside of this crate.
#[macro_export]
macro_rules! copy_ne {
    ($dest:expr, $src:expr, $range:expr) => {{ $dest[$range].copy_from_slice(&$src.to_ne_bytes()) }};
}
