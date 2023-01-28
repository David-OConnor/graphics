#![allow(mixed_script_confusables)] // Theta in meshes

mod camera;
mod compute;
mod graphics;
mod gui;
mod input;
pub mod lighting;
mod meshes;
mod system;
mod texture;
mod types;

pub use camera::Camera;
pub use input::InputsCommanded;
pub use lighting::{LightType, Lighting, PointLight};
pub use system::run;
pub use types::{
    ControlScheme, EngineUpdates, Entity, InputSettings, Mesh, Scene, UiLayout, UiSettings,
};

// Re-export winit DeviceEvents for use in the API; this prevents the calling
// lib from needing to use winit as a dependency directly.
// todo: the equiv for mouse events too. And in the future, Gamepad events.
pub use winit::{
    self,
    event::{self, DeviceEvent, ElementState},
};
