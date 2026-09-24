//! Screen-space outlines around the silhouettes of entities that have `Entity::outline` set.
//! See `shader_outline.wgsl` for the method.
//!
//! These resources are created the first time a scene has outlined entities, as many scenes
//! never do.

use wgpu::{
    self, BindGroup, BindGroupLayout, BlendState, Buffer, BufferBindingType, BufferUsages,
    CommandEncoder, Device, FragmentState, PipelineCache, Queue, RenderPass, RenderPipeline,
    ShaderModule, ShaderStages, StoreOp, SurfaceConfiguration, TextureFormat, TextureView,
    VertexState,
};

use crate::{
    texture::Texture,
    types::{INSTANCE_LAYOUT, VERTEX_LAYOUT},
};

/// rgb: Outline color. a: Outline thickness, in logical pixels. 0 where nothing is outlined.
const MASK_FORMAT: TextureFormat = TextureFormat::Rgba16Float;
/// Signed horizontal offset to the nearest covered pixel in the row.
const HORIZ_FORMAT: TextureFormat = TextureFormat::R32Float;

/// Marks "no covered pixel in range" in the horizontal pass output.
/// Must match `NONE_OFFSET` in shader_outline.wgsl.
const NONE_OFFSET: f64 = 10_000.;

/// Caps the search radius, in physical pixels. This bounds the per-pixel cost of very
/// thick outlines.
const MAX_RADIUS: i32 = 64;

/// Size of `OutlineUniforms` in shader_outline.wgsl.
const UNIFORM_SIZE: usize = 16;

pub(crate) struct OutlineState {
    mask_texture: Texture,
    horiz_texture: Texture,
    uniform_buf: Buffer,
    layout_horiz: BindGroupLayout,
    layout_composite: BindGroupLayout,
    bind_group_horiz: BindGroup,
    bind_group_composite: BindGroup,
    /// Draws outlined entities into the mask.
    pipeline_mask: RenderPipeline,
    /// Full-screen: mask -> horizontal offsets.
    pipeline_horiz: RenderPipeline,
    /// Full-screen: mask + horizontal offsets -> outlines, blended onto the scene.
    pipeline_composite: RenderPipeline,
}

impl OutlineState {
    /// `shader_mesh` and `layout_cam` are the main mesh shader, and camera bind group layout; the
    /// mask pass shares the mesh vertex and instance buffers.
    pub(crate) fn new(
        device: &Device,
        surface_cfg: &SurfaceConfiguration,
        shader_mesh: &ShaderModule,
        layout_cam: &BindGroupLayout,
        cache: Option<&PipelineCache>,
    ) -> Self {
        let (mask_texture, horiz_texture) = create_textures(device, surface_cfg);

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Outline uniform buffer"),
            size: UNIFORM_SIZE as wgpu::BufferAddress,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let tex_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: ShaderStages::FRAGMENT,
            // Read with textureLoad, so no sampler, and no need for a filterable format.
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };

        let uniform_entry = wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(UNIFORM_SIZE as u64),
            },
            count: None,
        };

        // Separate layouts, since the horizontal pass renders to the texture the final pass reads.
        let layout_horiz = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Outline horizontal bind group layout"),
            entries: &[tex_entry(0), uniform_entry],
        });

        let layout_composite = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Outline composite bind group layout"),
            entries: &[tex_entry(0), uniform_entry, tex_entry(2)],
        });

        let (bind_group_horiz, bind_group_composite) = create_bind_groups(
            device,
            &layout_horiz,
            &layout_composite,
            &mask_texture.view,
            &horiz_texture.view,
            &uniform_buf,
        );

        let pipeline_mask = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Outline mask pipeline layout"),
                bind_group_layouts: &[Some(layout_cam)],
                immediate_size: 0,
            });

            create_mask_pipeline(device, &layout, shader_mesh, cache)
        };

        let shader_outline = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Outline shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader_outline.wgsl").into()),
        });

        let pipeline_horiz = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Outline horizontal pipeline layout"),
                bind_group_layouts: &[Some(&layout_horiz)],
                immediate_size: 0,
            });

            create_fullscreen_pipeline(
                device,
                &layout,
                &shader_outline,
                "fs_outline_horiz",
                HORIZ_FORMAT,
                None,
                "Outline horizontal pipeline",
                cache,
            )
        };

        let pipeline_composite = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Outline composite pipeline layout"),
                bind_group_layouts: &[Some(&layout_composite)],
                immediate_size: 0,
            });

            create_fullscreen_pipeline(
                device,
                &layout,
                &shader_outline,
                "fs_outline",
                surface_cfg.format,
                Some(BlendState::ALPHA_BLENDING),
                "Outline composite pipeline",
                cache,
            )
        };

        Self {
            mask_texture,
            horiz_texture,
            uniform_buf,
            layout_horiz,
            layout_composite,
            bind_group_horiz,
            bind_group_composite,
            pipeline_mask,
            pipeline_horiz,
            pipeline_composite,
        }
    }

    /// Recreate the window-sized textures, and the bind groups that reference them.
    pub(crate) fn resize(&mut self, device: &Device, surface_cfg: &SurfaceConfiguration) {
        (self.mask_texture, self.horiz_texture) = create_textures(device, surface_cfg);

        (self.bind_group_horiz, self.bind_group_composite) = create_bind_groups(
            device,
            &self.layout_horiz,
            &self.layout_composite,
            &self.mask_texture.view,
            &self.horiz_texture.view,
            &self.uniform_buf,
        );
    }

    /// Draw outlines onto the resolved scene in `output_view`.
    ///
    /// `viewport` is the 3D sub-rect of the window; see `GraphicsState::setup_render_pass`.
    /// `scale` is the window's DPI scale factor, and `max_thickness` the thickest outline
    /// present, in logical pixels. `draw_mask` issues the draw calls for the outlined entities;
    /// the mask pipeline and camera bind group (group 0) are set when it's called.
    pub(crate) fn render(
        &self,
        encoder: &mut CommandEncoder,
        queue: &Queue,
        output_view: &TextureView,
        viewport: (f32, f32, f32, f32),
        scale: f32,
        max_thickness: f32,
        bind_group_cam: &BindGroup,
        draw_mask: impl FnOnce(&mut RenderPass),
    ) {
        let (vp_x, vp_y, vp_width, vp_height) = viewport;

        let radius = ((max_thickness * scale).ceil() as i32 + 1).clamp(1, MAX_RADIUS);

        let mut uniforms = [0; UNIFORM_SIZE];
        uniforms[0..4].copy_from_slice(&scale.to_ne_bytes());
        uniforms[4..8].copy_from_slice(&radius.to_ne_bytes());
        queue.write_buffer(&self.uniform_buf, 0, &uniforms);

        // 1: Silhouettes of the outlined entities.
        {
            let mut pass = begin_pass(
                encoder,
                "Outline mask",
                &self.mask_texture.view,
                wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            );

            pass.set_viewport(vp_x, vp_y, vp_width, vp_height, 0., 1.);
            pass.set_pipeline(&self.pipeline_mask);
            pass.set_bind_group(0, bind_group_cam, &[]);

            draw_mask(&mut pass);
        }

        // 2: Horizontal offsets to the nearest silhouette pixel. Clearing marks the area outside
        // the viewport as having none, for the final pass's taps near its edges.
        {
            let clear = wgpu::Color {
                r: NONE_OFFSET,
                g: 0.,
                b: 0.,
                a: 0.,
            };

            let mut pass = begin_pass(
                encoder,
                "Outline horizontal",
                &self.horiz_texture.view,
                wgpu::LoadOp::Clear(clear),
            );

            pass.set_viewport(vp_x, vp_y, vp_width, vp_height, 0., 1.);
            pass.set_pipeline(&self.pipeline_horiz);
            pass.set_bind_group(0, &self.bind_group_horiz, &[]);
            pass.draw(0..3, 0..1); // full-screen triangle
        }

        // 3: Outlines, blended onto the scene.
        {
            let mut pass = begin_pass(
                encoder,
                "Outline composite",
                output_view,
                wgpu::LoadOp::Load, // preserve the rendered scene
            );

            pass.set_viewport(vp_x, vp_y, vp_width, vp_height, 0., 1.);
            pass.set_pipeline(&self.pipeline_composite);
            pass.set_bind_group(0, &self.bind_group_composite, &[]);
            pass.draw(0..3, 0..1); // full-screen triangle
        }
    }
}

fn create_textures(device: &Device, surface_cfg: &SurfaceConfiguration) -> (Texture, Texture) {
    (
        Texture::create_screen_texture(device, surface_cfg, "Outline mask texture", MASK_FORMAT),
        Texture::create_screen_texture(
            device,
            surface_cfg,
            "Outline horizontal texture",
            HORIZ_FORMAT,
        ),
    )
}

fn create_bind_groups(
    device: &Device,
    layout_horiz: &BindGroupLayout,
    layout_composite: &BindGroupLayout,
    mask_view: &TextureView,
    horiz_view: &TextureView,
    uniform_buf: &Buffer,
) -> (BindGroup, BindGroup) {
    let mask_entry = wgpu::BindGroupEntry {
        binding: 0,
        resource: wgpu::BindingResource::TextureView(mask_view),
    };

    let uniform_entry = wgpu::BindGroupEntry {
        binding: 1,
        resource: uniform_buf.as_entire_binding(),
    };

    let horiz = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Outline horizontal bind group"),
        layout: layout_horiz,
        entries: &[mask_entry.clone(), uniform_entry.clone()],
    });

    let composite = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Outline composite bind group"),
        layout: layout_composite,
        entries: &[
            mask_entry,
            uniform_entry,
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(horiz_view),
            },
        ],
    });

    (horiz, composite)
}

/// A pass with a single, 1-sample color attachment, and no depth.
fn begin_pass<'a>(
    encoder: &'a mut CommandEncoder,
    label: &str,
    view: &TextureView,
    load: wgpu::LoadOp<wgpu::Color>,
) -> RenderPass<'a> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load,
                store: StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

/// Renders outlined entities' silhouettes into the mask. No depth test: the outline follows an
/// entity's full silhouette, including parts hidden behind other geometry.
fn create_mask_pipeline(
    device: &Device,
    layout: &wgpu::PipelineLayout,
    shader_mesh: &ShaderModule,
    cache: Option<&PipelineCache>,
) -> RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Outline mask pipeline"),
        layout: Some(layout),
        vertex: VertexState {
            module: shader_mesh,
            entry_point: Some("vs_outline_mask"),
            compilation_options: Default::default(),
            buffers: &[Some(VERTEX_LAYOUT), Some(INSTANCE_LAYOUT)],
        },
        fragment: Some(FragmentState {
            module: shader_mesh,
            entry_point: Some("fs_outline_mask"),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: MASK_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            // Only coverage matters, so don't depend on meshes' winding.
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview_mask: None,
        cache,
    })
}

fn create_fullscreen_pipeline(
    device: &Device,
    layout: &wgpu::PipelineLayout,
    shader: &ShaderModule,
    fs_entry_point: &str,
    format: TextureFormat,
    blend: Option<BlendState>,
    label: &str,
    cache: Option<&PipelineCache>,
) -> RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: VertexState {
            module: shader,
            entry_point: Some("vs_outline"),
            compilation_options: Default::default(),
            buffers: &[], // positions from vertex_index
        },
        fragment: Some(FragmentState {
            module: shader,
            entry_point: Some(fs_entry_point),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview_mask: None,
        cache,
    })
}
