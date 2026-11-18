//! Projects caster silhouettes onto a horizontal receiver. Each point light gets a
//! screen-space coverage mask, so the receiver can attenuate that light independently.

use std::ops::Range;

use wgpu::{
    BindGroup, BindGroupLayout, Buffer, CommandEncoder, Device, Face, FrontFace, PipelineCache,
    Queue, RenderPass, RenderPipeline, ShaderModule, SurfaceConfiguration, TextureView,
};

use crate::{
    Scene,
    system::DEPTH_FORMAT,
    types::{INSTANCE_LAYOUT, VERTEX_LAYOUT},
};

const MASK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

#[derive(Clone, Debug)]
pub struct FloorShadows {
    /// Y coordinate of the receiving plane in world space.
    pub y: f32,
    /// Dedicated transparent mesh drawn on the plane.
    pub receiver_mesh: usize,
    /// Entity classes drawn into the shadow masks. Rebuild entities when this changes.
    pub caster_classes: Vec<u32>,
    /// Fraction of a covered light's contribution to remove (0 to 1).
    pub strength: f32,
}

struct CasterDraw {
    indices: Range<u32>,
    base_vertex: i32,
    instances: Range<u32>,
}

/// Contiguous ranges in one of the existing opaque or transparent instance buffers.
#[derive(Default)]
pub(crate) struct CasterDraws(Vec<CasterDraw>);

impl CasterDraws {
    pub(crate) fn push(&mut self, indices: Range<u32>, base_vertex: i32, instance: u32) {
        if let Some(last) = self.0.last_mut()
            && last.base_vertex == base_vertex
            && last.indices == indices
            && last.instances.end == instance
        {
            last.instances.end += 1;
        } else {
            self.0.push(CasterDraw {
                indices,
                base_vertex,
                instances: instance..instance + 1,
            });
        }
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn draw(&self, pass: &mut RenderPass, instances: &Buffer) {
        if self.is_empty() {
            return;
        }

        pass.set_vertex_buffer(1, instances.slice(..));
        for draw in &self.0 {
            pass.draw_indexed(
                draw.indices.clone(),
                draw.base_vertex,
                draw.instances.clone(),
            );
        }
    }
}

struct MaskLayer {
    view: TextureView,
    projection: Buffer,
    bind_group: BindGroup,
}

struct ShadowTargets {
    size: wgpu::Extent3d,
    layers: Vec<MaskLayer>,
    receiver_bind_group: BindGroup,
}

pub(crate) struct FloorShadowPass {
    projection_layout: BindGroupLayout,
    receiver_layout: BindGroupLayout,
    mask_pipeline: RenderPipeline,
    // Same back-face / front-face pass order as the normal transparent renderer.
    receiver_pipelines: [RenderPipeline; 2],
    strength: Buffer,
    targets: Option<ShadowTargets>,
}

impl FloorShadowPass {
    pub(crate) fn new(
        device: &Device,
        config: &SurfaceConfiguration,
        samples: u32,
        front_face: FrontFace,
        shader: &ShaderModule,
        camera_layout: &BindGroupLayout,
        lighting_layout: &BindGroupLayout,
        cache: Option<&PipelineCache>,
    ) -> Self {
        let projection_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Floor shadow projection layout"),
            entries: &[uniform_entry(1, wgpu::ShaderStages::VERTEX, 16)],
        });
        let receiver_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Floor shadow receiver layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                uniform_entry(1, wgpu::ShaderStages::FRAGMENT, 4),
            ],
        });

        let mask_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Floor shadow mask pipeline layout"),
            bind_group_layouts: &[Some(camera_layout), Some(&projection_layout)],
            immediate_size: 0,
        });
        let max_blend = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Max,
        };
        let mask_pipeline = create_pipeline(
            device,
            shader,
            &mask_layout,
            "vs_floor_shadow",
            "fs_floor_shadow",
            MASK_FORMAT,
            wgpu::BlendState {
                color: max_blend,
                alpha: max_blend,
            },
            None,
            FrontFace::Ccw,
            None,
            1,
            cache,
        );

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Floor shadow receiver pipeline layout"),
            bind_group_layouts: &[
                Some(camera_layout),
                Some(lighting_layout),
                Some(&receiver_layout),
            ],
            immediate_size: 0,
        });
        let depth = wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: Default::default(),
            bias: Default::default(),
        };
        let receiver_pipelines = [Face::Front, Face::Back].map(|cull| {
            create_pipeline(
                device,
                shader,
                &layout,
                "vs_main",
                "fs_floor",
                config.format,
                wgpu::BlendState::ALPHA_BLENDING,
                Some(depth.clone()),
                front_face,
                Some(cull),
                samples,
                cache,
            )
        });

        Self {
            projection_layout,
            receiver_layout,
            mask_pipeline,
            receiver_pipelines,
            strength: uniform_buffer(device, "Floor shadow strength", 4),
            targets: None,
        }
    }

    fn create_targets(&self, device: &Device, size: wgpu::Extent3d) -> ShadowTargets {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Floor shadow masks"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: MASK_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let array_view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let receiver_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Floor shadow receiver bind group"),
            layout: &self.receiver_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&array_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.strength.as_entire_binding(),
                },
            ],
        });
        let layers = (0..size.depth_or_array_layers)
            .map(|layer| {
                let view = texture.create_view(&wgpu::TextureViewDescriptor {
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                });
                let projection = uniform_buffer(device, "Floor shadow projection", 16);
                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Floor shadow projection bind group"),
                    layout: &self.projection_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 1,
                        resource: projection.as_entire_binding(),
                    }],
                });
                MaskLayer {
                    view,
                    projection,
                    bind_group,
                }
            })
            .collect();

        ShadowTargets {
            size,
            layers,
            receiver_bind_group,
        }
    }

    pub(crate) fn render_masks(
        &mut self,
        device: &Device,
        queue: &Queue,
        encoder: &mut CommandEncoder,
        config: &SurfaceConfiguration,
        viewport: (f32, f32, f32, f32),
        floor: &FloorShadows,
        scene: &Scene,
        camera_bind_group: &BindGroup,
        vertices: &Buffer,
        indices: &Buffer,
        instances: [&Buffer; 2],
        casters: &[CasterDraws; 2],
    ) {
        let size = wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: (scene.lighting.point_lights.len() as u32)
                .min(device.limits().max_texture_array_layers),
        };
        if self
            .targets
            .as_ref()
            .is_none_or(|targets| targets.size != size)
        {
            self.targets = Some(self.create_targets(device, size));
        }

        queue.write_buffer(
            &self.strength,
            0,
            &floor.strength.clamp(0., 1.).to_ne_bytes(),
        );
        let (x, y, width, height) = viewport;

        for (layer, light) in self
            .targets
            .as_ref()
            .unwrap()
            .layers
            .iter()
            .zip(&scene.lighting.point_lights)
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Floor shadow mask"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &layer.view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });

            // Only lights on the visible side of the receiver can illuminate that side.
            let p = light.position;
            if (p.y - floor.y) * (scene.camera.position.y - floor.y) <= 0.
                || (light.diffuse_intensity <= 0. && light.specular_intensity <= 0.)
            {
                continue;
            }

            let mut projection = [0; 16];
            for (bytes, value) in projection.chunks_exact_mut(4).zip([p.x, p.y, p.z, floor.y]) {
                bytes.copy_from_slice(&value.to_ne_bytes());
            }
            queue.write_buffer(&layer.projection, 0, &projection);

            pass.set_viewport(x, y, width, height, 0., 1.);
            pass.set_pipeline(&self.mask_pipeline);
            pass.set_bind_group(0, camera_bind_group, &[]);
            pass.set_bind_group(1, &layer.bind_group, &[]);
            pass.set_vertex_buffer(0, vertices.slice(..));
            pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
            for (draws, buffer) in casters.iter().zip(instances) {
                draws.draw(&mut pass, buffer);
            }
        }
    }

    pub(crate) fn bind_receiver(&self, pass: &mut RenderPass, side: usize) {
        pass.set_pipeline(&self.receiver_pipelines[side]);
        pass.set_bind_group(2, &self.targets.as_ref().unwrap().receiver_bind_group, &[]);
    }

    pub(crate) fn release_targets(&mut self) {
        self.targets = None;
    }
}

fn uniform_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    size: u64,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: wgpu::BufferSize::new(size),
        },
        count: None,
    }
}

fn uniform_buffer(device: &Device, label: &str, size: u64) -> Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn create_pipeline(
    device: &Device,
    module: &ShaderModule,
    layout: &wgpu::PipelineLayout,
    vertex: &str,
    fragment: &str,
    format: wgpu::TextureFormat,
    blend: wgpu::BlendState,
    depth_stencil: Option<wgpu::DepthStencilState>,
    front_face: FrontFace,
    cull_mode: Option<Face>,
    samples: u32,
    cache: Option<&PipelineCache>,
) -> RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(fragment),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some(vertex),
            compilation_options: Default::default(),
            buffers: &[Some(VERTEX_LAYOUT), Some(INSTANCE_LAYOUT)],
        },
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some(fragment),
            compilation_options: Default::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(blend),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            front_face,
            cull_mode,
            ..Default::default()
        },
        depth_stencil,
        multisample: wgpu::MultisampleState {
            count: samples,
            ..Default::default()
        },
        multiview_mask: None,
        cache,
    })
}
