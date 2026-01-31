//! Handles window initialization and events, using Winit.

use std::{
    path::Path,
    time::{Duration, Instant},
};

use image::ImageError;
use wgpu::TextureViewDescriptor;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::ActiveEventLoop,
    window::{Icon, WindowAttributes, WindowId},
};

use crate::{
    EngineUpdates, Scene, UiLayoutSides, UiLayoutTopBottom, UiSettings,
    system::{State, process_engine_updates},
};

fn load_icon(path: &Path) -> Result<Icon, ImageError> {
    let (icon_rgba, icon_width, icon_height) = {
        let image = image::open(path)?.into_rgba8();
        let (width, height) = image.dimensions();
        let rgba = image.into_raw();
        (rgba, width, height)
    };
    Ok(Icon::from_rgba(icon_rgba, icon_width, icon_height).expect("Failed to open icon"))
}

impl<T, FRender, FEventDev, FEventWin, FGui> State<T, FRender, FEventDev, FEventWin, FGui>
where
    FRender: FnMut(&mut T, &mut Scene, f32) -> EngineUpdates + 'static,
    FEventDev: FnMut(&mut T, DeviceEvent, &mut Scene, bool, f32) -> EngineUpdates + 'static,
    FEventWin: FnMut(&mut T, WindowEvent, &mut Scene, f32) -> EngineUpdates + 'static,
    FGui: FnMut(&mut T, &egui::Context, &mut Scene) -> EngineUpdates + 'static,
{
    fn redraw(&mut self) {
        if self.paused || self.render.is_none() || self.graphics.is_none() {
            return;
        }

        let sys = self.render.as_ref().unwrap();
        let graphics = self.graphics.as_mut().unwrap();

        let now = Instant::now();
        self.dt = now - self.last_render_time;

        // Clamp, e.g. if the loop isn't running. (Maybe when minimized?)
        if self.dt.as_secs() > 1 {
            self.dt = Duration::from_secs(1);
        }

        self.last_render_time = now;

        let dt_secs = self.dt.as_secs() as f32 + self.dt.subsec_micros() as f32 / 1_000_000.;
        let updates_render =
            (self.render_handler)(&mut self.user_state, &mut graphics.scene, dt_secs);

        process_engine_updates(
            &updates_render,
            graphics,
            &self.render.as_ref().unwrap().device,
            &self.render.as_ref().unwrap().queue,
        );

        // Note that the GUI handler can also modify entities, but
        // we do that in the `init_graphics` module.

        // todo: move this into `render`?
        match sys.surface.get_current_texture() {
            Ok(output_frame) => {
                let surface_texture = output_frame
                    .texture
                    .create_view(&TextureViewDescriptor::default());

                let resize_required = graphics.render(
                    self.gui.as_mut().unwrap(),
                    output_frame,
                    &surface_texture,
                    &sys.device,
                    &sys.queue,
                    self.dt,
                    sys.surface_cfg.width,
                    sys.surface_cfg.height,
                    &mut self.ui_settings,
                    &mut self.gui_handler,
                    &mut self.user_state,
                );

                if resize_required {
                    self.resize(sys.size);
                }
            }
            // This occurs when minimized.
            Err(_e) => (),
        }
    }
}

impl<T, FRender, FEventDev, FEventWin, FGui> ApplicationHandler
    for State<T, FRender, FEventDev, FEventWin, FGui>
where
    FRender: FnMut(&mut T, &mut Scene, f32) -> EngineUpdates + 'static,
    FEventDev: FnMut(&mut T, DeviceEvent, &mut Scene, bool, f32) -> EngineUpdates + 'static,
    FEventWin: FnMut(&mut T, WindowEvent, &mut Scene, f32) -> EngineUpdates + 'static,
    FGui: FnMut(&mut T, &egui::Context, &mut Scene) -> EngineUpdates + 'static,
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let icon = match self.ui_settings.icon_path {
            Some(ref p) => load_icon(Path::new(&p)).ok(),
            // No path specified
            None => None,
        };

        let attributes = WindowAttributes::default()
            .with_title(&self.scene.window_title)
            // Physical size vs logical size has implications for pixel-scaled setups,
            // like some high-resolution but small-screen tablets and laptops.
            .with_inner_size(winit::dpi::LogicalSize::new(
                // .with_inner_size(winit::dpi::PhysicalSize::new(
                self.scene.window_size.0,
                self.scene.window_size.1,
            ))
            .with_window_icon(icon);

        let window = event_loop.create_window(attributes).unwrap();

        self.init(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        if self.render.is_none() || self.graphics.is_none() {
            // This may occur prior to init.
            return;
        }

        let graphics = &mut self.graphics.as_mut().unwrap();
        let gui = &mut self.gui.as_mut().unwrap();

        if !gui.mouse_in_gui {
            let dt_secs = self.dt.as_secs() as f32 + self.dt.subsec_micros() as f32 / 1_000_000.;

            let updates_event = (self.event_win_handler)(
                &mut self.user_state,
                event.clone(),
                &mut graphics.scene,
                dt_secs,
            );

            let render = self.render.as_ref().unwrap();
            process_engine_updates(&updates_event, graphics, &render.device, &render.queue);
        }

        let window = &graphics.window;
        let _ = gui.egui_state.on_window_event(window, &event);

        match event {
            WindowEvent::RedrawRequested => {
                self.redraw();

                // todo: Only request the window redraw when required from an event etc. Will be
                // todo much more efficient this way.
                self.graphics.as_ref().unwrap().window.request_redraw();
            }
            WindowEvent::CursorMoved { position, .. } => {
                let in_ui_horizontal = match self.ui_settings.layout_sides {
                    UiLayoutSides::Left => position.x < gui.size.0 as f64,
                    UiLayoutSides::Right => {
                        position.x > window.inner_size().width as f64 - gui.size.0 as f64
                    }
                };

                let in_ui_vertical = match self.ui_settings.layout_top_bottom {
                    UiLayoutTopBottom::Top => position.y < gui.size.1 as f64,
                    UiLayoutTopBottom::Bottom => {
                        position.y > window.inner_size().height as f64 - gui.size.1 as f64
                    }
                };
                let mouse_in_gui = in_ui_horizontal || in_ui_vertical;

                if mouse_in_gui {
                    gui.mouse_in_gui = true;

                    // We reset the inputs, since otherwise a held key that
                    // doesn't get the reset command will continue to execute.
                    self.graphics.as_mut().unwrap().inputs_commanded = Default::default();
                } else {
                    gui.mouse_in_gui = false;
                }
            }
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(physical_size) => {
                self.paused = physical_size.width == 0 || physical_size.height == 0;

                if !self.paused {
                    self.resize(physical_size);
                    self.last_render_time = Instant::now();
                    self.dt = Default::default();
                }

                // Prevents inadvertent mouse-click-activated free-look.
                self.graphics.as_mut().unwrap().inputs_commanded.free_look = false;
            }
            // If the window scale changes, update the renderer size, and camera aspect ratio.
            WindowEvent::ScaleFactorChanged {
                scale_factor: _,
                inner_size_writer: _,
                ..
            } => {
                // Note: This appears to not come up, nor is it required. (Oct 2024)
                println!("Scale factor changed");
            }
            // If the window is being moved, disable mouse inputs, eg so click+drag
            // doesn't cause a drag when moving the window using the mouse.
            WindowEvent::Moved(_) => {
                gui.mouse_in_gui = true;
                // Prevents inadvertent mouse-click-activated free-look after moving the window.
                self.graphics.as_mut().unwrap().inputs_commanded.free_look = false;
            }
            WindowEvent::Occluded(occ) => {
                self.paused = occ;

                // Prevents inadvertent mouse-click-activated free-look after minimizing.
                self.graphics.as_mut().unwrap().inputs_commanded.free_look = false;

                if !self.paused {
                    self.last_render_time = Instant::now();
                    self.dt = Default::default();
                }
            }
            WindowEvent::Focused(focused) => {
                // Eg clicking the tile bar icon.
                self.paused = !focused;

                self.graphics.as_mut().unwrap().inputs_commanded.free_look = false;
                if focused {
                    self.last_render_time = Instant::now();
                    self.dt = Default::default();
                }
            }
            WindowEvent::CursorLeft { device_id: _ } => {
                // When the cursor moves out of the window, stop mouse-looking.
                graphics.inputs_commanded.free_look = false;
                graphics.inputs_commanded.cursor_out_of_window = true;
            }
            WindowEvent::CursorEntered { device_id: _ } => {
                self.paused = false;
                graphics.inputs_commanded.cursor_out_of_window = false;
            }
            // This is required to prevent the application from freezing after dropping a file.
            WindowEvent::HoveredFile(_)
            | WindowEvent::HoveredFileCancelled
            | WindowEvent::DroppedFile(_) => {
                self.graphics.as_ref().unwrap().window.request_redraw();
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        if self.render.is_none() || self.graphics.is_none() {
            return;
        }

        let render = &self.render.as_ref().unwrap();
        let graphics = &mut self.graphics.as_mut().unwrap();
        let gui = &mut self.gui.as_mut().unwrap();

        if !gui.mouse_in_gui {
            let dt_secs = self.dt.as_secs() as f32 + self.dt.subsec_micros() as f32 / 1_000_000.;

            graphics.handle_input(&event, &self.scene.input_settings);
            let inputs_present = graphics.inputs_commanded.inputs_present();

            let updates_event = (self.event_dev_handler)(
                &mut self.user_state,
                event,
                &mut graphics.scene,
                inputs_present,
                dt_secs,
            );

            process_engine_updates(&updates_event, graphics, &render.device, &render.queue);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {}

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {}
}

/// Used in render, the text display pipeline, and may be used by applications, e.g. in mapping
/// 2d to 3d.
pub fn viewport_rect(
    ui_size: (f32, f32), // In EGUI units.
    // These are in physical pixels.
    win_width: u32,
    win_height: u32,
    ui_settings: &UiSettings,
    _pixels_per_pt: f32,
) -> (f32, f32, f32, f32) {
    // Default to full window
    let mut x = 0.0;
    let mut y = 0.0;
    let mut eff_width = win_width as f32;
    let mut eff_height = win_height as f32;

    // Note: This only supports top and left UI; right and bottom is broken.
    if ui_settings.layout_sides == UiLayoutSides::Left {
        x = ui_size.0;
    }
    if ui_settings.layout_top_bottom == UiLayoutTopBottom::Top {
        y = ui_size.1;
    }

    eff_width -= ui_size.0;
    eff_height -= ui_size.1;

    // Safety check to prevent crash if UI takes entire screen
    if eff_width < 1.0 {
        eff_width = 1.0;
    }
    if eff_height < 1.0 {
        eff_height = 1.0;
    }

    (x, y, eff_width, eff_height)
}
