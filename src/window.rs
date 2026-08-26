//! Handles window initialization and events, using Winit.

use std::{
    fs,
    path::Path,
    time::{Duration, Instant},
};

use image::ImageError;
use wgpu::TextureViewDescriptor;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow},
    window::{Icon, WindowAttributes, WindowId},
};

use crate::{
    EngineUpdates, Scene, UiLayoutSides, UiLayoutTopBottom, UiSettings,
    system::{State, UserEvent, process_engine_updates},
};

const IDLE_WAKE_DT_MAX: Duration = Duration::from_nanos(16_666_667);
const CONTINUOUS_DT_MAX: Duration = Duration::from_millis(100);

fn frame_delta(elapsed: Duration, continuous_redraw: bool) -> Duration {
    elapsed.min(if continuous_redraw {
        CONTINUOUS_DT_MAX
    } else {
        IDLE_WAKE_DT_MAX
    })
}

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
    FGui: FnMut(&mut T, &mut egui::Ui, &mut Scene) -> EngineUpdates + 'static,
{
    fn redraw(&mut self) -> bool {
        if self.paused || self.render.is_none() || self.graphics.is_none() {
            return false;
        }
        self.next_repaint = None;

        let sys = self.render.as_ref().unwrap();
        let graphics = self.graphics.as_mut().unwrap();

        let now = Instant::now();
        self.dt = frame_delta(
            now.saturating_duration_since(self.last_render_time),
            self.continuous_redraw,
        );
        self.last_render_time = now;

        let dt_secs = self.dt.as_secs() as f32 + self.dt.subsec_micros() as f32 / 1_000_000.;
        let updates_render =
            (self.render_handler)(&mut self.user_state, &mut graphics.scene, dt_secs);
        let mut request_redraw = updates_render.request_redraw;

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
            wgpu::CurrentSurfaceTexture::Success(output_frame)
            | wgpu::CurrentSurfaceTexture::Suboptimal(output_frame) => {
                let surface_texture = output_frame
                    .texture
                    .create_view(&TextureViewDescriptor::default());

                let (resize_required, gui_request_redraw) = graphics.render(
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
                request_redraw |= gui_request_redraw;

                if resize_required {
                    self.resize(sys.size);
                }

                // Apply any pending MSAA change.
                let pending = self.graphics.as_mut().unwrap().pending_msaa.take();
                if let Some(new_msaa) = pending {
                    let device = &self.render.as_ref().unwrap().device;
                    let graphics = self.graphics.as_mut().unwrap();
                    graphics.msaa_samples = new_msaa;
                    graphics.apply_msaa_change(device);
                    self.graphics_settings.msaa_samples = new_msaa;
                }
            }
            // Timeout, Occluded, Outdated, Lost, or Validation — skip frame.
            _ => (),
        }

        request_redraw
    }

    fn schedule_repaint(&mut self, repaint_at: Instant) {
        if self
            .next_repaint
            .is_none_or(|scheduled| repaint_at < scheduled)
        {
            self.next_repaint = Some(repaint_at);
        }
    }

    fn update_repaint_deadline(&mut self, event_loop: &ActiveEventLoop) {
        if self.paused {
            event_loop.set_control_flow(ControlFlow::Wait);
            return;
        }

        let Some(repaint_at) = self.next_repaint else {
            event_loop.set_control_flow(ControlFlow::Wait);
            return;
        };

        if repaint_at <= Instant::now() {
            self.next_repaint = None;
            if let Some(graphics) = &self.graphics {
                graphics.window.request_redraw();
            }
            event_loop.set_control_flow(ControlFlow::Wait);
        } else {
            event_loop.set_control_flow(ControlFlow::WaitUntil(repaint_at));
        }
    }
}

impl<T, FRender, FEventDev, FEventWin, FGui> ApplicationHandler<UserEvent>
    for State<T, FRender, FEventDev, FEventWin, FGui>
where
    FRender: FnMut(&mut T, &mut Scene, f32) -> EngineUpdates + 'static,
    FEventDev: FnMut(&mut T, DeviceEvent, &mut Scene, bool, f32) -> EngineUpdates + 'static,
    FEventWin: FnMut(&mut T, WindowEvent, &mut Scene, f32) -> EngineUpdates + 'static,
    FGui: FnMut(&mut T, &mut egui::Ui, &mut Scene) -> EngineUpdates + 'static,
{
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let icon = match self.ui_settings.icon_path {
            Some(ref p) => load_icon(Path::new(&p)).ok(),
            // No path specified
            None => None,
        };

        let requested_w = self.scene.window_size.0;
        let requested_h = self.scene.window_size.1;

        // Check if the requested logical size fits on the primary monitor.
        // On HiDPI displays the logical screen size can be smaller than the requested
        // window size (e.g. a Surface tablet at 2× DPI has ~1368×912 logical pixels),
        // so we maximise instead of creating a window that overflows the screen.
        let fits_on_screen = event_loop
            .primary_monitor()
            .map(|m| {
                let scale = m.scale_factor() as f32;
                let logical_w = m.size().width as f32 / scale;
                let logical_h = m.size().height as f32 / scale;
                requested_w <= logical_w && requested_h <= logical_h
            })
            .unwrap_or(true); // If monitor info unavailable, try the requested size

        let base_attributes = WindowAttributes::default()
            .with_title(&self.scene.window_title)
            .with_window_icon(icon);

        let attributes = if fits_on_screen {
            base_attributes.with_inner_size(winit::dpi::LogicalSize::new(requested_w, requested_h))
        } else {
            base_attributes.with_maximized(true)
        };

        let window = event_loop.create_window(attributes).unwrap();

        self.init(window);
        self.graphics.as_ref().unwrap().window.request_redraw();
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

        let redraw_for_event = !matches!(
            event,
            WindowEvent::RedrawRequested | WindowEvent::CloseRequested
        );
        let graphics = &mut self.graphics.as_mut().unwrap();
        let gui = &mut self.gui.as_mut().unwrap();

        if !gui.mouse_in_gui {
            graphics.handle_input_window(&event, &self.scene.input_settings);

            // Handle events processed by the application
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

        // Handle events processed by this engine.
        match event {
            WindowEvent::RedrawRequested => {
                let app_requested_redraw = self.redraw();
                let engine_input_active = self
                    .graphics
                    .as_ref()
                    .is_some_and(|graphics| graphics.inputs_commanded.inputs_present());
                self.continuous_redraw =
                    !self.paused && (app_requested_redraw || engine_input_active);
                if self.continuous_redraw {
                    self.graphics.as_ref().unwrap().window.request_redraw();
                }
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
                    self.continuous_redraw = false;
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
                    self.continuous_redraw = false;
                }
            }
            WindowEvent::Focused(focused) => {
                // Eg clicking the tile bar icon.
                self.paused = !focused;

                self.graphics.as_mut().unwrap().inputs_commanded.free_look = false;
                if focused {
                    self.last_render_time = Instant::now();
                    self.dt = Default::default();
                    self.continuous_redraw = false;
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

        if redraw_for_event && !self.paused {
            self.graphics.as_ref().unwrap().window.request_redraw();
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
            // Handle events processed by this engine.
            graphics.handle_input_device(&event, &self.scene.input_settings);
            let inputs_present = graphics.inputs_commanded.inputs_present();

            // Handle events processed by the application
            let dt_secs = self.dt.as_secs() as f32 + self.dt.subsec_micros() as f32 / 1_000_000.;
            let updates_event = (self.event_dev_handler)(
                &mut self.user_state,
                event,
                &mut graphics.scene,
                inputs_present,
                dt_secs,
            );

            process_engine_updates(&updates_event, graphics, &render.device, &render.queue);
        }

        if !self.paused {
            self.graphics.as_ref().unwrap().window.request_redraw();
        }
    }

    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::RequestRedraw(repaint_at) => self.schedule_repaint(repaint_at),
        }
        self.update_repaint_deadline(event_loop);
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        self.update_repaint_deadline(event_loop);
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        let cache_data = self
            .graphics
            .as_ref()
            .and_then(|graphics| graphics.pipeline_cache.as_ref())
            .and_then(wgpu::PipelineCache::get_data);
        let cache_path = self
            .render
            .as_ref()
            .and_then(|render| render.pipeline_cache_path.as_ref());

        if let (Some(data), Some(path)) = (cache_data, cache_path) {
            let temp_path = path.with_extension("tmp");
            let result = fs::write(&temp_path, data).and_then(|()| {
                if path.exists() {
                    fs::remove_file(path)?;
                }
                fs::rename(temp_path, path)
            });
            if let Err(error) = result {
                eprintln!("Unable to save graphics pipeline cache: {error}");
            }
        }

        // The event loop (and the Wayland display connection) is dropped when `run_app`
        // returns, since it takes `self` by value; so anything holding a display
        // reference must be torn down here, while the connection is still alive:
        // egui-winit's clipboard (smithay-clipboard), the wgpu surface, then the window.
        // Order matters: GPU resources before the device, surface before the window. Not doing this
        // is generally fine, but on Linux/Wayland, will cause a segfault on exit.
        self.gui = None;
        self.render = None;
        self.graphics = None;
    }
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
