#![allow(mixed_script_confusables)] // Theta in meshes

#[cfg(feature = "app_utils")]
pub mod app_utils;
mod camera;
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
pub use graphics::{FWD_VEC, RIGHT_VEC, UP_VEC};
pub use input::{InputsCommanded, adjust_camera};
pub use lighting::{LightType, Lighting, PointLight};
pub use system::run;
pub use types::{
    ControlScheme, EngineUpdates, Entity, GraphicsSettings, InputSettings, Mesh, Scene, UiLayout,
    UiSettings, Vertex,
};
// Re-export winit DeviceEvents for use in the API; this prevents the calling
// lib from needing to use winit as a dependency directly.
// todo: the equiv for mouse events too. And in the future, Gamepad events.
pub use winit::{
    self,
    event::{self, DeviceEvent, ElementState, WindowEvent},
};
