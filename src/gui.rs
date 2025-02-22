//! GUI code for EGUI, to run on the WGPU painter.
//! See [this unofficial example](https://github.com/kaphula/winit-egui-wgpu-template/tree/master/src)
//! https://github.com/rust-windowing/winit/issues/3626

use std::sync::Arc;

use egui::{ClippedPrimitive, Context, FullOutput};
use egui_wgpu::{Renderer, ScreenDescriptor};
use egui_winit;
use wgpu::{self, CommandEncoder, Device, Queue, TextureFormat};
use winit::window::Window;

use crate::{
    UiLayout,
    graphics::GraphicsState,
    system::DEPTH_FORMAT,
    types::{EngineUpdates, Scene},
};

/// State related to the GUI.
pub(crate) struct GuiState {
    pub egui_state: egui_winit::State,
    pub egui_renderer: Renderer,
    /// Used to disable inputs while the mouse is in the GUI section.
    pub mouse_in_gui: bool,
    /// We store this, so we know if we need to perform a resize if it changes.
    pub size: f32,
}

impl GuiState {
    pub fn new(
        window: Arc<Window>,
        device: &Device,
        texture_format: TextureFormat,
        msaa_samples: u32,
    ) -> Self {
        let egui_context = Context::default();
        let egui_state = egui_winit::State::new(
            egui_context,
            egui::viewport::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );

        let egui_renderer = Renderer::new(
            device,
            texture_format,
            Some(DEPTH_FORMAT),
            msaa_samples,
            false, // todo: Dithering?
        );

        Self {
            egui_state,
            egui_renderer,
            mouse_in_gui: false,
            size: 0.,
        }
    }

    /// This function contains code specific to rendering the GUI prior to the render pass.
    pub(crate) fn render_gui_pre_rpass<T>(
        &mut self,
        graphics: &mut GraphicsState,
        user_state: &mut T,
        device: &Device,
        mut gui_handler: impl FnMut(&mut T, &Context, &mut Scene) -> EngineUpdates,
        encoder: &mut CommandEncoder,
        queue: &Queue,
        width: u32,
        height: u32,
        updates_gui: &mut EngineUpdates,
        layout: UiLayout,
    ) -> (FullOutput, Vec<ClippedPrimitive>, ScreenDescriptor, bool) {
        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point: graphics.window.scale_factor() as f32,
        };

        self.egui_state
            .egui_ctx()
            .set_pixels_per_point(screen_descriptor.pixels_per_point);

        let mut resize_required = false;

        let raw_input = self.egui_state.take_egui_input(&graphics.window);
        let full_output = self.egui_state.egui_ctx().run(raw_input, |ctx| {
            *updates_gui = gui_handler(user_state, self.egui_state.egui_ctx(), &mut graphics.scene);

            let mut new_size = match layout {
                UiLayout::Left | UiLayout::Right => ctx.used_size().x,
                _ => ctx.used_size().y,
            };

            // This error doesn't make much sense, but seems to occur when there is no GUI.
            if new_size == f32::NEG_INFINITY {
                new_size = 0.;
            }

            if self.size != new_size {
                resize_required = true;
                self.size = new_size;
            }
        });

        self.egui_state
            .handle_platform_output(&graphics.window, full_output.platform_output.clone()); // todo: Is this clone OK?

        let tris = self.egui_state.egui_ctx().tessellate(
            full_output.shapes.clone(), // todo: Is the clone OK?
            self.egui_state.egui_ctx().pixels_per_point(),
        );

        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(device, queue, *id, image_delta);
        }

        self.egui_renderer
            .update_buffers(device, queue, encoder, &tris, &screen_descriptor);

        (full_output, tris, screen_descriptor, resize_required)
    }
}
