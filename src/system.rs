//! This module initiates the window, and graphics hardware.
//! It initializes WGPU settings.

//  Check out this example for winit/egui/wgpu (2024)
// https://github.com/kaphula/winit-egui-wgpu-template/blob/master/src/main.rs

use std::{
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

use image::ImageError;
use wgpu::{
    Adapter, Backends, Device, Features, Instance, InstanceDescriptor, PowerPreference, Queue,
    Surface, SurfaceConfiguration, TextureFormat,
};
use winit::{
    dpi::PhysicalSize,
    event::DeviceEvent,
    event_loop::{ControlFlow, EventLoop},
    window::{Icon, Window},
};

use crate::{
    graphics::GraphicsState,
    gui::GuiState,
    texture::Texture,
    types::{EngineUpdates, InputSettings, Scene, UiLayout, UiSettings},
};

pub const COLOR_FORMAT: TextureFormat = TextureFormat::Bgra8UnormSrgb;
pub const DEPTH_FORMAT: TextureFormat = TextureFormat::Depth32Float;

/// This struct contains state related to the 3D graphics. It is mostly constructed of types
/// that are required by  the WGPU renderer.
pub(crate) struct RenderState {
    pub size: PhysicalSize<u32>,
    pub surface: Surface<'static>, // Sshare the same lifetime as the window, A/R.
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
    pub surface_cfg: SurfaceConfiguration,
}

pub struct State<T: 'static, FRender, FEvent, FGui>
where
    FRender: FnMut(&mut T, &mut Scene, f32) -> EngineUpdates + 'static,
    FEvent: FnMut(&mut T, DeviceEvent, &mut Scene, f32) -> EngineUpdates + 'static,
    FGui: FnMut(&mut T, &egui::Context, &mut Scene) -> EngineUpdates + 'static,
{
    pub instance: Instance,
    /// `render` and `graphics`, and `gui` are only None at init; they require the `Window` event loop
    /// to be run.
    pub render: Option<RenderState>,
    pub graphics: Option<GraphicsState>,
    pub gui: Option<GuiState>,
    pub user_state: T,
    pub render_handler: FRender,
    pub event_handler: FEvent,
    pub gui_handler: FGui,
    pub input_settings: InputSettings,
    pub ui_settings: UiSettings,
    pub scene: Scene,
    pub last_render_time: Instant,
    pub dt: Duration,
}

impl<T: 'static, FRender, FEvent, FGui> State<T, FRender, FEvent, FGui>
where
    FRender: FnMut(&mut T, &mut Scene, f32) -> EngineUpdates + 'static,
    FEvent: FnMut(&mut T, DeviceEvent, &mut Scene, f32) -> EngineUpdates + 'static,
    FGui: FnMut(&mut T, &egui::Context, &mut Scene) -> EngineUpdates + 'static,
{
    /// This constructor sets up the basics required for Winit's events loop. We initialize the important
    /// parts later, once the window has been set up.
    pub(crate) fn new(
        scene: Scene,
        input_settings: InputSettings,
        ui_settings: UiSettings,
        user_state: T,
        render_handler: FRender,
        event_handler: FEvent,
        gui_handler: FGui,
    ) -> Self {
        let last_render_time = Instant::now();
        let dt = Duration::new(0, 0);

        // The instance is a handle to our GPU. Its main purpose is to create Adapters and Surfaces.
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::VULKAN,
            ..Default::default()
        });

        Self {
            instance,
            render: None,
            graphics: None,
            gui: None,
            user_state,
            render_handler,
            event_handler,
            gui_handler,
            input_settings,
            ui_settings,
            scene,
            last_render_time,
            dt,
        }
    }

    /// Initializes the renderer and GUI. We launch this from the Window's event loop.
    pub(crate) fn init(&mut self, window: Window) {
        println!("Initializing graphics and sys...");
        let window = Arc::new(window);

        let size = window.inner_size();

        let surface = self.instance.create_surface(window.clone()).unwrap();

        let (adapter, device, queue) = pollster::block_on(setup_async(&self.instance, &surface));

        // The surface is the part of the window that we draw to. We need it to draw directly to the
        // screen. Our window needs to implement raw-window-handle (opens new window)'s
        // HasRawWindowHandle trait to create a surface.

        // https://docs.rs/wgpu/latest/wgpu/type.SurfaceConfiguration.html
        let surface_cfg = SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            // format: surface.get_supported_formats(&adapter)[0],
            format: COLOR_FORMAT,
            width: size.width,
            height: size.height,
            // https://docs.rs/wgpu/latest/wgpu/enum.PresentMode.html
            // Note that `Fifo` locks FPS to the speed of the monitor.
            present_mode: wgpu::PresentMode::Fifo,
            // todo: Allow config from user.
            // present_mode: wgpu::PresentMode::Immediate,
            // present_mode: wgpu::PresentMode::Mailbox,
            desired_maximum_frame_latency: 2, // Default
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: Vec::new(),
        };

        surface.configure(&device, &surface_cfg);

        let texture_format = surface_cfg.format;

        let render = RenderState {
            size,
            surface,
            adapter,
            device,
            queue,
            surface_cfg,
        };

        let graphics = GraphicsState::new(
            &render.device,
            &render.surface_cfg,
            self.scene.clone(), // todo: Now we have two scene states... not good.
            // input_settings,
            // ui_settings,
            window.clone(),
        );

        self.gui = Some(GuiState::new(window, &render.device, texture_format));

        self.render = Some(render);
        self.graphics = Some(graphics);
    }

    pub(crate) fn resize(&mut self, new_size: PhysicalSize<u32>) {
        println!("Resizing");
        if self.render.is_none() || self.graphics.is_none() {
            return;
        }

        let mut sys = &mut self.render.as_mut().unwrap();
        let mut graphics = &mut self.graphics.as_mut().unwrap();
        let mut gui = &mut self.gui.as_mut().unwrap();

        if new_size.width > 0 && new_size.height > 0 {
            sys.size = new_size;
            sys.surface_cfg.width = new_size.width;
            sys.surface_cfg.height = new_size.height;
            sys.surface.configure(&sys.device, &sys.surface_cfg);

            let (eff_width, eff_height) = match self.ui_settings.layout {
                UiLayout::Left | UiLayout::Right => (
                    sys.surface_cfg.width as f32 - gui.size,
                    sys.surface_cfg.height as f32,
                ),
                _ => (
                    sys.surface_cfg.width as f32,
                    sys.surface_cfg.height as f32 - gui.size,
                ),
            };

            graphics.scene.camera.aspect = eff_width / eff_height;

            graphics.depth_texture =
                Texture::create_depth_texture(&sys.device, &sys.surface_cfg, "Depth texture");

            graphics.scene.camera.update_proj_mat();
        }
    }
}

/// This is the entry point to the renderer. It's called by the application to initialize the event
/// loop.
pub fn run<T: 'static, FRender, FEvent, FGui>(
    user_state: T,
    scene: Scene,
    input_settings: InputSettings,
    ui_settings: UiSettings,
    render_handler: FRender,
    event_handler: FEvent,
    gui_handler: FGui,
) where
    FRender: FnMut(&mut T, &mut Scene, f32) -> EngineUpdates + 'static,
    FEvent: FnMut(&mut T, DeviceEvent, &mut Scene, f32) -> EngineUpdates + 'static,
    FGui: FnMut(&mut T, &egui::Context, &mut Scene) -> EngineUpdates + 'static,
{
    let (_frame_count, _accum_time) = (0, 0.0);

    println!("Settings: {:?}", ui_settings);

    let mut state: State<T, FRender, FEvent, FGui> = State::new(
        scene,
        input_settings,
        ui_settings,
        user_state,
        render_handler,
        event_handler,
        gui_handler,
    );

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    event_loop.run_app(&mut state).expect("Failed to run app");
}

/// Quarantine for the Async part of the API
async fn setup_async(
    instance: &wgpu::Instance,
    surface: &Surface<'static>,
) -> (Adapter, Device, Queue) {
    // The adapter is a handle to our actual graphics card. You can use this to get
    // information about the graphics card such as its name and what backend the
    // adapter uses. We use this to create our Device and Queue.
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            // `Default` prefers low power when on battery, high performance when on mains.
            power_preference: PowerPreference::default(),
            compatible_surface: Some(surface),
            force_fallback_adapter: false,
        })
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                // https://docs.rs/wgpu/latest/wgpu/struct.Features.html
                required_features: Features::empty(),
                // https://docs.rs/wgpu/latest/wgpu/struct.Limits.html
                required_limits: Default::default(),
                memory_hints: Default::default(),
            },
            std::env::var("WGPU_TRACE")
                .ok()
                .as_ref()
                .map(std::path::Path::new),
        )
        .await
        .expect("Unable to find a suitable GPU adapter. :(");

    (adapter, device, queue)
}

/// Process engine updates from render, GUI, or events.
pub(crate) fn process_engine_updates(
    engine_updates: &EngineUpdates,
    g_state: &mut GraphicsState,
    device: &Device,
    queue: &Queue,
) {
    if engine_updates.meshes {
        g_state.setup_vertices_indices(device);
        g_state.setup_entities(device);
    }

    if engine_updates.entities {
        g_state.setup_entities(device);
    }

    if engine_updates.camera {
        // Entities have been updated in the scene; update the buffer.
        g_state.update_camera(queue);
    }

    if engine_updates.lighting {
        // Entities have been updated in the scene; update the buffer.
        g_state.update_lighting(queue);
    }
}
