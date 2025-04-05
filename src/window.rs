//! Handles window initialization and events, using Winit.

use std::{path::Path, time::Instant};

use image::ImageError;
use wgpu::TextureViewDescriptor;
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, WindowEvent},
    event_loop::ActiveEventLoop,
    window::{Icon, WindowAttributes, WindowId},
};

use crate::{
    EngineUpdates, Scene, UiLayout,
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
        if self.render.is_none() || self.graphics.is_none() {
            return;
        }

        let sys = self.render.as_ref().unwrap();
        let graphics = self.graphics.as_mut().unwrap();

        let now = Instant::now();
        self.dt = now - self.last_render_time;
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

                let layout = self.ui_settings.layout;

                let resize_required = graphics.render(
                    &mut self.gui.as_mut().unwrap(),
                    output_frame,
                    &surface_texture,
                    &sys.device,
                    &sys.queue,
                    self.dt,
                    sys.surface_cfg.width,
                    sys.surface_cfg.height,
                    &mut self.ui_settings,
                    // &self.scene.input_settings,
                    &mut self.gui_handler,
                    &mut self.user_state,
                    layout,
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
            Some(ref p) => {
                match load_icon(Path::new(&p)) {
                    Ok(p_) => Some(p_),
                    // eg can't find the path
                    Err(_) => None,
                }
            }
            // No path specified
            None => None,
        };

        let attributes = WindowAttributes::default()
            .with_title(&self.scene.window_title)
            .with_inner_size(winit::dpi::LogicalSize::new(
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

        //     if let Some(gui) = self.gui.as_mut() {
        //     if let Some(graphics) = self.graphics.as_mut() {
        //         let window = &graphics.window;
        //         let _ = gui.egui_state.on_window_event(window, &event);
        //     }
        // }

        let window = &graphics.window;
        let _ = gui.egui_state.on_window_event(window, &event);

        match event {
            WindowEvent::RedrawRequested => {
                self.redraw();
                self.graphics.as_ref().unwrap().window.request_redraw();
            }
            WindowEvent::CursorMoved { position, .. } => {
                let mouse_in_gui = match self.ui_settings.layout {
                    UiLayout::Left => position.x < gui.size as f64,
                    UiLayout::Right => {
                        position.x > window.inner_size().width as f64 - gui.size as f64
                    }
                    UiLayout::Top => position.y < gui.size as f64,
                    UiLayout::Bottom => {
                        position.y > window.inner_size().height as f64 - gui.size as f64
                    }
                };
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
                self.resize(physical_size);
                // Prevents inadvertent mouse-click-activated free-look.
                self.graphics.as_mut().unwrap().inputs_commanded.free_look = false;
            }
            // If the window scale changes, update the renderer size, and camera aspect ratio.
            WindowEvent::ScaleFactorChanged {
                scale_factor,
                inner_size_writer,
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
            WindowEvent::Occluded(_) => {
                // Prevents inadvertent mouse-click-activated free-look after minimizing.
                self.graphics.as_mut().unwrap().inputs_commanded.free_look = false;
            }
            WindowEvent::Focused(_) => {
                // Eg clicking the tile bar icon.
                self.graphics.as_mut().unwrap().inputs_commanded.free_look = false;
            }
            WindowEvent::CursorLeft { device_id: _ } => {
                // When the cursor moves out of the window, stop mouse-looking.
                graphics.inputs_commanded.free_look = false;
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
