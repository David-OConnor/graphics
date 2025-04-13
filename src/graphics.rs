//! This module contains code specific to the WGPU library.
//! See [Official WGPU examples](https://github.com/gfx-rs/wgpu/tree/master/wgpu/examples)
//! See [Bevy Garphics](https://github.com/bevyengine/bevy/blob/main/crates/bevy_render) for
//! a full graphics engine example that uses Wgpu.
//! https://sotrh.github.io/learn-wgpu/
//!
//! https://github.com/sotrh/learn-wgpu/tree/master/code/intermediate/tutorial12-camera/src
//! https://github.com/gfx-rs/wgpu/tree/master/wgpu/examples/shadow
//!
//! 2022-08-21: https://github.com/gfx-rs/wgpu/blob/master/wgpu/examples/cube/main.rs

use std::{sync::Arc, time::Duration};

use egui::Context;
use lin_alg::f32::Vec3;
use wgpu::{
    self, util::{BufferInitDescriptor, DeviceExt}, BindGroup, BindGroupLayout, BindingType, BlendState, Buffer,
    BufferBindingType, BufferUsages, CommandEncoder, CommandEncoderDescriptor, DepthStencilState,
    Device, FragmentState, Queue, RenderPass, RenderPassDepthStencilAttachment,
    RenderPassDescriptor, RenderPipeline, ShaderStages, StoreOp, SurfaceConfiguration, SurfaceTexture,
    TextureDescriptor, TextureView, VertexBufferLayout,
    VertexState,
};
use winit::{event::DeviceEvent, window::Window};

use crate::{
    gauss::{QUAD_VERTICES},
    gui::GuiState,
    input::{self, InputsCommanded},
    system::{process_engine_updates, DEPTH_FORMAT},
    texture::Texture,
    types::{
        ControlScheme, EngineUpdates, InputSettings, Instance, Scene, UiLayout,
        UiSettings, Vertex,
    },
};
use crate::gauss::{GaussianInstance, GAUSS_INST_LAYOUT, QUAD_VERTEX_LAYOUT};
use crate::types::{INSTANCE_LAYOUT, VERTEX_LAYOUT};

pub const UP_VEC: Vec3 = Vec3 {
    x: 0.,
    y: 1.,
    z: 0.,
};
pub const RIGHT_VEC: Vec3 = Vec3 {
    x: 1.,
    y: 0.,
    z: 0.,
};
pub const FWD_VEC: Vec3 = Vec3 {
    x: 0.,
    y: 0.,
    z: 1.,
};

/// Code related to our specific engine. Buffers, texture data etc.
pub(crate) struct GraphicsState {
    pub vertex_buf: Buffer,
    pub vertex_buf_quad: Buffer, // For gaussians.
    pub index_buf: Buffer,
    instance_buf: Buffer,
    instance_gauss_buf: Buffer,
    pub bind_groups: BindGroupData,
    pub camera_buf: Buffer,
    lighting_buf: Buffer,
    pub pipeline_mesh: RenderPipeline,  // todo: Move to renderer.
    pub pipeline_gauss: RenderPipeline, // todo: Move to renderer.
    pub depth_texture: Texture,
    pub msaa_texture: Option<TextureView>, // MSAA Multisampled texture
    pub inputs_commanded: InputsCommanded,
    // staging_belt: wgpu::util::StagingBelt, // todo: Do we want this? Probably in sys, not here.
    pub scene: Scene,
    mesh_mappings: Vec<(i32, u32, u32)>,
    pub window: Arc<Window>,
}

impl GraphicsState {
    pub(crate) fn new(
        device: &Device,
        surface_cfg: &SurfaceConfiguration,
        mut scene: Scene,
        window: Arc<Window>,
        msaa_samples: u32,
    ) -> Self {
        let vertex_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Vertex buffer"),
            contents: &[], // Populated later.
            usage: BufferUsages::VERTEX,
        });

        let mut quad_bytes = Vec::with_capacity(QUAD_VERTICES.len() * 8);
        for q in QUAD_VERTICES {
            quad_bytes.extend_from_slice(q.to_bytes().as_slice());
        }

        let vertex_buf_quad = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Gauss quadVertex buffer"),
            contents: &quad_bytes,
            usage: BufferUsages::VERTEX,
        });

        let index_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Index buffer"),
            contents: &[], // Populated later.
            usage: BufferUsages::INDEX,
        });

        scene.camera.update_proj_mat();

        let cam_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Camera buffer"),
            contents: &scene.camera.to_bytes(),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let lighting_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Lighting buffer"),
            contents: &scene.lighting.to_bytes(),
            // We use a storage buffer, since our lighting size is unknown by the shader;
            // this is due to the dynamic-sized point light array.
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        });
        //

        let bind_groups = create_bindgroups(device, &cam_buf, &lighting_buf);

        let depth_texture =
            Texture::create_depth_texture(device, surface_cfg, "Depth texture", msaa_samples);

        let shader_mesh = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Graphics shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let pipeline_layout_mesh = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render pipeline layout"),
            bind_group_layouts: &[&bind_groups.layout_cam, &bind_groups.layout_lighting],
            push_constant_ranges: &[],
        });

        let depth_stencil_mesh = Some(DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });

        // todo: You should probably, eventually, make two passes for meshes. One for
        // opaque objects, with no blending (blend_state = None), then a second pass
        // for transparent objects. You would set depth to write only, for opaque objects,
        // and read only, for alpha blending transparent meshes.
        let blend_mesh = Some(BlendState::ALPHA_BLENDING);


        let pipeline_mesh = create_render_pipeline(
            device,
            &pipeline_layout_mesh,
            shader_mesh,
            surface_cfg,
            msaa_samples,
            &[VERTEX_LAYOUT, INSTANCE_LAYOUT],
            depth_stencil_mesh,
            blend_mesh,
        );

        // We initialize instances, the instance buffer and mesh mappings in `setup_entities`.
        // let instances = Vec::new();
        let instance_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Instance buffer"),
            contents: &[], // empty on init
            usage: BufferUsages::VERTEX,
        });

        let shader_gauss = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Graphics shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader_gauss.wgsl").into()),
        });

        let pipeline_layout_gauss =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Gaussian pipeline layout"),
                bind_group_layouts: &[&bind_groups.layout_cam],
                push_constant_ranges: &[],
            });

        // todo: Experiment
        let depth_stencil_gauss = Some(DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });

        let pipeline_gauss = create_render_pipeline(
            device,
            &pipeline_layout_gauss,
            shader_gauss,
            surface_cfg,
            msaa_samples,
            &[QUAD_VERTEX_LAYOUT, GAUSS_INST_LAYOUT],
            depth_stencil_gauss,
            Some(BlendState::ALPHA_BLENDING),
        );

        let instance_gauss_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Gaussian Instance buffer"),
            contents: &[], // empty on init
            usage: BufferUsages::VERTEX,
        });

        // Placeholder value
        let mesh_mappings = Vec::new();

        // todo: Logical (scaling by device?) vs physical pixels
        // let window_size = winit::dpi::LogicalSize::new(scene.window_size.0, scene.window_size.1);
        window.set_title(&scene.window_title);

        let msaa_texture = if msaa_samples > 1 {
            Some(Self::create_msaa_texture(device, surface_cfg, msaa_samples))
        } else {
            None
        };

        let mut result = Self {
            vertex_buf,
            vertex_buf_quad,
            index_buf,
            instance_buf,
            instance_gauss_buf,
            bind_groups,
            camera_buf: cam_buf,
            lighting_buf,
            pipeline_mesh,
            pipeline_gauss,
            depth_texture,
            // staging_belt: wgpu::util::StagingBelt::new(0x100),
            scene,
            inputs_commanded: Default::default(),
            mesh_mappings,
            window,
            msaa_texture,
        };

        result.setup_vertices_indices(device);
        result.setup_entities(device);

        result
    }

    pub(crate) fn create_msaa_texture(
        device: &Device,
        surface_cfg: &SurfaceConfiguration,
        sample_count: u32,
    ) -> TextureView {
        let msaa_texture = device.create_texture(&TextureDescriptor {
            label: Some("Multisampled Texture"),
            size: wgpu::Extent3d {
                width: surface_cfg.width,
                height: surface_cfg.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: surface_cfg.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        msaa_texture.create_view(&wgpu::TextureViewDescriptor::default())
    }

    pub(crate) fn handle_input(&mut self, event: &DeviceEvent, input_settings: &InputSettings) {
        match input_settings.control_scheme {
            ControlScheme::FreeCamera | ControlScheme::Arc { center: _ } => {
                input::add_input_cmd(&event, &mut self.inputs_commanded)
            }
            _ => unimplemented!(),
        }
    }

    /// Updates meshes.
    pub(crate) fn setup_vertices_indices(&mut self, device: &Device) {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for mesh in &self.scene.meshes {
            for vertex in &mesh.vertices {
                vertices.push(vertex)
            }

            for index in &mesh.indices {
                indices.push(index);
            }
        }
        // Convert the vertex and index data to u8 buffers.
        let mut vertex_data = Vec::new();
        for vertex in vertices {
            for byte in vertex.to_bytes() {
                vertex_data.push(byte);
            }
        }

        let mut index_data = Vec::new();
        for index in indices {
            let bytes = index.to_ne_bytes();
            index_data.push(bytes[0]);
            index_data.push(bytes[1]);
            index_data.push(bytes[2]);
            index_data.push(bytes[3]);
        }

        // We can't update using a queue due to buffer size mismatches.
        let vertex_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Vertex buffer"),
            contents: &vertex_data,
            usage: BufferUsages::VERTEX,
        });

        let index_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Index buffer"),
            contents: &index_data,
            usage: BufferUsages::INDEX,
        });

        self.vertex_buf = vertex_buf;
        // Note: Gauss vertex buf is static; we set it up at init, and don't change it.

        self.index_buf = index_buf;
    }

    /// Currently, sets up entities (And the associated instance buf), but doesn't change
    /// meshes, lights, or the camera. The vertex and index buffers aren't changed; only the instances.
    pub(crate) fn setup_entities(&mut self, device: &Device) {
        let mut instances = Vec::new();

        let mut mesh_mappings = Vec::new();

        let mut vertex_start_this_mesh = 0;
        let mut instance_start_this_mesh = 0;

        for (i, mesh) in self.scene.meshes.iter().enumerate() {
            let mut instance_count_this_mesh = 0;
            for entity in self.scene.entities.iter().filter(|e| e.mesh == i) {
                let scale = match entity.scale_partial {
                    Some(s) => s,
                    None => Vec3::new(entity.scale, entity.scale, entity.scale),
                };

                instances.push(Instance {
                    // todo: entity into method?
                    position: entity.position,
                    orientation: entity.orientation,
                    scale,
                    color: Vec3::new(entity.color.0, entity.color.1, entity.color.2),
                    opacity: entity.opacity,
                    shinyness: entity.shinyness,
                });
                instance_count_this_mesh += 1;
            }

            mesh_mappings.push((
                vertex_start_this_mesh,
                instance_start_this_mesh,
                instance_count_this_mesh,
            ));

            vertex_start_this_mesh += mesh.vertices.len() as i32;
            instance_start_this_mesh += instance_count_this_mesh;
        }

        let mut instance_data = Vec::new();
        for instance in &instances {
            for byte in instance.to_bytes() {
                instance_data.push(byte);
            }
        }

        // We can't update using a queue due to buffer size mismatches.
        let instance_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Instance buffer"),
            contents: &instance_data,
            usage: BufferUsages::VERTEX,
        });

        self.instance_buf = instance_buf;

        // todo, once working.
        // let mut instances_gauss = Vec::new();
        // for gauss_instance in self.scene.gaussians {
        //     instances_gauss.push(gauss_instance);/
        // }
        let instances_gauss = vec![GaussianInstance {
            center: [0., 0., 0.],
            amplitude: 1.,
            width: 5.,
            _pad: [0.; 3],
        }];

        let mut instance_gauss_data = Vec::new();
        for instance in &instances_gauss {
            for byte in instance.to_bytes() {
                instance_gauss_data.push(byte);
            }
        }

        let instance_gauss_buf = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Gaussian Instance buffer"),
            contents: &instance_gauss_data,
            usage: BufferUsages::VERTEX,
        });

        self.instance_gauss_buf = instance_gauss_buf;

        self.mesh_mappings = mesh_mappings;
    }

    pub(crate) fn update_camera(&mut self, queue: &Queue) {
        queue.write_buffer(&self.camera_buf, 0, &self.scene.camera.to_bytes());
    }

    pub(crate) fn update_lighting(&mut self, queue: &Queue) {
        queue.write_buffer(&self.lighting_buf, 0, &self.scene.lighting.to_bytes());
    }

    fn setup_render_pass<'a>(
        &mut self,
        ui_size: f32,
        encoder: &'a mut CommandEncoder,
        output_view: &TextureView,
        width: u32,
        height: u32,
        ui_settings: &UiSettings,
    ) -> RenderPass<'a> {
        // Adjust the viewport size for 3D, based on how much size the UI is taking up.
        let (mut x, mut y, mut eff_width, mut eff_height) = match ui_settings.layout {
            UiLayout::Left => (ui_size, 0., width as f32 - ui_size, height as f32),
            UiLayout::Right => (0., 0., width as f32 - ui_size, height as f32),
            UiLayout::Top => (0., ui_size, width as f32, height as f32 - ui_size),
            UiLayout::Bottom => (0., 0., width as f32, height as f32 - ui_size),
        };

        // This has come up during EGUI file_dialog. Causes the render to effectively overlap
        // the UI, instead of being next to it.
        match ui_settings.layout {
            UiLayout::Left | UiLayout::Right => {
                if ui_size > width as f32 {
                    (x, y, eff_width, eff_height) = (0., 0., width as f32, height as f32);
                }
            }
            _ => {
                if ui_size >= height as f32 {
                    (x, y, eff_width, eff_height) = (0., 0., width as f32, height as f32);
                }
            }
        }

        let color_attachment = if let Some(msaa_texture) = &self.msaa_texture {
            // Use MSAA texture as render target, resolve to the swap chain texture
            wgpu::RenderPassColorAttachment {
                view: msaa_texture,
                resolve_target: Some(output_view), // Resolve the multisample texture
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: StoreOp::Discard,
                },
            }
        } else {
            wgpu::RenderPassColorAttachment {
                view: output_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: self.scene.background_color.0 as f64,
                        g: self.scene.background_color.1 as f64,
                        b: self.scene.background_color.2 as f64,
                        a: 1.0,
                    }),
                    store: StoreOp::Store,
                },
            }
        };

        let mut rpass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("Render pass"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
                view: &self.depth_texture.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        rpass.set_viewport(x, y, eff_width, eff_height, 0., 1.);

        rpass.set_pipeline(&self.pipeline_mesh);

        rpass.set_bind_group(0, &self.bind_groups.cam, &[]);
        rpass.set_bind_group(1, &self.bind_groups.lighting, &[]);

        rpass.set_vertex_buffer(0, self.vertex_buf.slice(..));
        // Without this size check, the instance buffer slice fails, and we get a panic.
        if self.instance_buf.size() == 0 {
            return rpass;
        }

        rpass.set_vertex_buffer(1, self.instance_buf.slice(..));
        rpass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint32);

        let mut start_ind = 0;
        for (i, mesh) in self.scene.meshes.iter().enumerate() {
            let (vertex_start_this_mesh, instance_start_this_mesh, instance_count_this_mesh) =
                self.mesh_mappings[i];

            rpass.draw_indexed(
                start_ind..start_ind + mesh.indices.len() as u32,
                vertex_start_this_mesh,
                instance_start_this_mesh..instance_start_this_mesh + instance_count_this_mesh,
            );

            start_ind += mesh.indices.len() as u32;
        }

        // Draw gaussians.
        rpass.set_pipeline(&self.pipeline_gauss);
        rpass.set_vertex_buffer(0, self.vertex_buf_quad.slice(..));
        rpass.set_vertex_buffer(1, self.instance_gauss_buf.slice(..)); // stride = 32 B
        // todo: Impl.
        // rpass.draw(0..6, 0..self.scene.instances_gauss.len() as _); // 6 indices for the quad
        rpass.draw(0..6, 0..1 as _); // 6 indices for the quad

        rpass
    }

    /// The entry point to 3D and GUI rendering.
    /// Note:  `resize_required`, the return, is to handle changes in GUI size.
    pub(crate) fn render<T>(
        &mut self,
        gui: &mut GuiState,
        surface_texture: SurfaceTexture,
        output_texture: &TextureView,
        device: &Device,
        queue: &Queue,
        dt: Duration,
        width: u32,
        height: u32,
        ui_settings: &mut UiSettings,
        gui_handler: impl FnMut(&mut T, &Context, &mut Scene) -> EngineUpdates,
        user_state: &mut T,
        layout: UiLayout,
    ) -> bool {
        // Adjust camera inputs using the in-engine control scheme.
        // Note that camera settings adjusted by the application code are handled in
        // `update_camera`.

        if self.inputs_commanded.inputs_present() {
            let dt_secs = dt.as_secs() as f32 + dt.subsec_micros() as f32 / 1_000_000.;

            let cam_changed = match self.scene.input_settings.control_scheme {
                ControlScheme::FreeCamera => input::adjust_camera_free(
                    &mut self.scene.camera,
                    &mut self.inputs_commanded,
                    &self.scene.input_settings,
                    dt_secs,
                ),
                ControlScheme::Arc { center } => input::adjust_camera_arc(
                    &mut self.scene.camera,
                    &mut self.inputs_commanded,
                    &self.scene.input_settings,
                    center,
                    dt_secs,
                ),
                _ => unimplemented!(),
            };

            if cam_changed {
                self.update_camera(queue);
            }

            // Reset the mouse inputs; keyboard inputs are reset by their release event.
            self.inputs_commanded.mouse_delta_x = 0.;
            self.inputs_commanded.mouse_delta_y = 0.;
        }

        // We create a CommandEncoder to create the actual commands to send to the
        // gpu. Most modern graphics frameworks expect commands to be stored in a command buffer
        // before being sent to the gpu. The encoder builds a command buffer that we can then
        // send to the gpu.
        let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Render encoder"),
        });

        let mut updates_gui = Default::default();

        let (gui_full_output, tris, screen_descriptor, resize_required) = gui.render_gui_pre_rpass(
            self,
            user_state,
            device,
            gui_handler,
            &mut encoder,
            queue,
            width,
            height,
            &mut updates_gui,
            layout,
        );

        // Note: If we process engine updates after setting up the render pass, we will not be
        // able to add meshes at runtime; code run from the `engine_updates.meshes` flag must be
        // done along with a mesh change prior to setting up the render pass, or else we will get
        // an error about an index being out of bounds.
        process_engine_updates(&updates_gui, self, device, queue);

        let rpass = self.setup_render_pass(
            gui.size,
            &mut encoder,
            output_texture,
            width,
            height,
            ui_settings,
        );

        let mut rpass = rpass.forget_lifetime();

        gui.egui_renderer
            .render(&mut rpass, &tris, &screen_descriptor);
        drop(rpass); // Ends the render pass.

        for x in &gui_full_output.textures_delta.free {
            gui.egui_renderer.free_texture(x)
        }

        queue.submit(Some(encoder.finish()));

        surface_texture.present();

        resize_required
    }
}

/// Create a render pipeline. Configurable by parameters to support multiple use cases. E.g., both
/// meshes and gaussians.
fn create_render_pipeline(
    device: &Device,
    layout: &wgpu::PipelineLayout,
    shader: wgpu::ShaderModule,
    config: &SurfaceConfiguration,
    sample_count: u32,
    vertex_buffers: &'static [VertexBufferLayout<'static>],
    depth_stencil: Option<DepthStencilState>,
    blend: Option<BlendState>,
) -> RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render pipeline"),
        layout: Some(layout),

        vertex: VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: vertex_buffers,
        },
        fragment: Some(FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            // This configures with alpha blending. (?)
            targets: &[Some(wgpu::ColorTargetState {
                format: config.format, // Ensure this is a format with alpha (e.g., `wgpu::TextureFormat::Rgba8Unorm`)
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },

        depth_stencil,
        multisample: wgpu::MultisampleState {
            count: sample_count, // Enable MSAA
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        // If the pipeline will be used with a multiview render pass, this
        // indicates how many array layers the attachments will have.
        multiview: None,
        cache: None,
    })
}

pub(crate) struct BindGroupData {
    pub layout_cam: BindGroupLayout,
    pub cam: BindGroup,
    pub layout_lighting: BindGroupLayout,
    pub lighting: BindGroup,
    /// We use this for GUI.
    pub layout_texture: BindGroupLayout,
    // pub texture: BindGroup,
}

fn create_bindgroups(device: &Device, cam_buf: &Buffer, lighting_buf: &Buffer) -> BindGroupData {
    // We only need vertex, not fragment info in the camera uniform.
    let layout_cam = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::VERTEX | ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                // The dynamic field indicates whether this buffer will change size or
                // not. This is useful if we want to store an array of things in our uniforms.
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
        label: Some("Camera bind group layout"),
    });

    let cam = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &layout_cam,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: cam_buf.as_entire_binding(),
        }],
        label: Some("Camera bind group"),
    });

    let layout_lighting = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only: true }, // todo read-only?
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
        label: Some("Lighting bind group layout"),
    });

    let lighting = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &layout_lighting,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: lighting_buf.as_entire_binding(),
        }],
        label: Some("Lighting bind group"),
    });

    // todo: Don't create these (diffuse tex view, sampler every time. Pass as args.
    // We don't need to configure the texture view much, so let's
    // let wgpu define it.
    // let diffuse_bytes = include_bytes!("happy-tree.png");
    // let diffuse_bytes = [];
    // let diffuse_texture = wgpu::texture::Texture::from_bytes(&device, &queue, diffuse_bytes, "happy-tree.png").unwrap();
    //
    // let diffuse_texture_view = diffuse_texture.create_view(&wgpu::TextureViewDescriptor::default());
    // let diffuse_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
    //     address_mode_u: wgpu::AddressMode::ClampToEdge,
    //     address_mode_v: wgpu::AddressMode::ClampToEdge,
    //     address_mode_w: wgpu::AddressMode::ClampToEdge,
    //     mag_filter: wgpu::FilterMode::Linear,
    //     min_filter: wgpu::FilterMode::Nearest,
    //     mipmap_filter: wgpu::FilterMode::Nearest,
    //     ..Default::default()
    // });

    let layout_texture = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("egui_texture_bind_group_layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    multisampled: false,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                // This should match the filterable field of the
                // corresponding Texture entry above.
                ty: BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });

    // let texture = device.create_bind_group(
    //     &wgpu::BindGroupDescriptor {
    //         layout: &layout_texture,
    //         entries: &[
    //             wgpu::BindGroupEntry {
    //                 binding: 0,
    //                 resource: wgpu::BindingResource::TextureView(&diffuse_texture_view),
    //                 // resource: wgpu::BindingResource::TextureView(&[]), // todo?
    //             },
    //             wgpu::BindGroupEntry {
    //                 binding: 1,
    //                 resource: wgpu::BindingResource::Sampler(&diffuse_sampler),
    //             }
    //         ],
    //         label: Some("Texture bind group"),
    //     });

    BindGroupData {
        layout_cam,
        cam,
        layout_lighting,
        lighting,
        layout_texture,
        // texture
    }
}
