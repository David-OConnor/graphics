//! https://sotrh.github.io/learn-wgpu/beginner/tutorial9-models/#rendering-a-mesh

use crate::{camera::Camera, lighting::Lighting};

use lin_alg2::f32::{Mat4, Quaternion, Vec3};

// These sizes are in bytes. We do this, since that's the data format expected by the shader.
pub const F32_SIZE: usize = 4;

pub const VEC3_SIZE: usize = 3 * F32_SIZE;
pub const VEC3_UNIFORM_SIZE: usize = 4 * F32_SIZE;
pub const MAT4_SIZE: usize = 16 * F32_SIZE;
pub const MAT3_SIZE: usize = 9 * F32_SIZE;

pub const VERTEX_SIZE: usize = 14 * F32_SIZE;
// Note that position, orientation, and scale are combined into a single 4x4 transformation
// matrix. Note that unlike uniforms, we don't need alignment padding, and can use Vec3 directly.
pub const INSTANCE_SIZE: usize = MAT4_SIZE + MAT3_SIZE + VEC3_SIZE + F32_SIZE;

#[derive(Clone, Copy, Debug)]
/// Example attributes: https://github.com/bevyengine/bevy/blob/main/crates/bevy_render/src/mesh/mesh/mod.rs#L56
/// // todo: Vec3 vs arrays?
pub struct Vertex {
    /// Where the vertex is located in space
    pub position: [f32; 3],
    /// AKA UV mapping. https://en.wikipedia.org/wiki/UV_mapping
    pub tex_coords: [f32; 2],
    /// The direction the vertex normal is facing in
    pub normal: Vec3,
    /// "Tangent and Binormal vectors are vectors that are perpendicular to each other
    /// and the normal vector which essentially describe the direction of the u,v texture
    /// coordinates with respect to the surface that you are trying to render. Typically
    /// they can be used alongside normal maps which allow you to create sub surface
    /// lighting detail to your model(bumpiness)."
    /// This is used to orient normal maps; corresponds to the +X texture direction.
    pub tangent: [f32; 3],
    /// A bitangent vector is the result of the Cross Product between Vertex Normal and Vertex
    /// Tangent which is a unit vector perpendicular to both vectors at a given point.
    /// This is used to orient normal maps; corresponds to the +Y texture direction.
    pub bitangent: [f32; 3],
}

impl Vertex {
    /// Initialize position; change the others after init.
    pub fn new(position: [f32; 3], normal: Vec3) -> Self {
        Self {
            position,
            tex_coords: [0., 0.],
            normal,
            tangent: [0., 0., 0.],
            bitangent: [0., 0., 0.],
        }
    }

    pub fn to_bytes(&self) -> [u8; VERTEX_SIZE] {
        let mut result = [0; VERTEX_SIZE];

        result[0..4].clone_from_slice(&self.position[0].to_ne_bytes());
        result[4..8].clone_from_slice(&self.position[1].to_ne_bytes());
        result[8..12].clone_from_slice(&self.position[2].to_ne_bytes());
        result[12..16].clone_from_slice(&self.tex_coords[0].to_ne_bytes());
        result[16..20].clone_from_slice(&self.tex_coords[1].to_ne_bytes());

        result[20..32].clone_from_slice(&self.normal.to_bytes_vertex());

        result[32..36].clone_from_slice(&self.tangent[0].to_ne_bytes());
        result[36..40].clone_from_slice(&self.tangent[1].to_ne_bytes());
        result[40..44].clone_from_slice(&self.tangent[2].to_ne_bytes());
        result[44..48].clone_from_slice(&self.bitangent[0].to_ne_bytes());
        result[48..52].clone_from_slice(&self.bitangent[1].to_ne_bytes());
        result[52..56].clone_from_slice(&self.bitangent[2].to_ne_bytes());

        result
    }

    /// Create the vertex buffer memory layout, for our vertexes passed from CPU
    /// to the vertex shader. Corresponds to `VertexIn` in the shader. Each
    /// item here is for a single vertex.
    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: VERTEX_SIZE as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // Vertex position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Texture coordinates
                wgpu::VertexAttribute {
                    offset: VEC3_SIZE as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // Normal vector
                wgpu::VertexAttribute {
                    offset: (2 * F32_SIZE + VEC3_SIZE) as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Tangent (Used to align textures)
                wgpu::VertexAttribute {
                    offset: (2 * F32_SIZE + 2 * VEC3_SIZE) as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Bitangent (Used to align textures)
                wgpu::VertexAttribute {
                    offset: (2 * F32_SIZE + 3 * VEC3_SIZE) as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

/// Instances allow the GPU to render the same object multiple times.
/// "Instancing allows us to draw the same object multiple times with different properties
/// (position, orientation, size, color, etc.). "
/// todo: Relationship between this and entity?
pub struct Instance {
    pub position: Vec3,
    pub orientation: Quaternion,
    pub scale: f32,
    pub color: Vec3,
    pub shinyness: f32,
}

impl Instance {
    /// Create the vertex buffer memory layout, for our vertexes passed from the
    /// vertex to the fragment shader. Corresponds to `VertexOut` in the shader. Each
    /// item here is for a single vertex. Cannot share locations with `VertexIn`, so
    /// we start locations after `VertexIn`'s last one.
    pub fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        wgpu::VertexBufferLayout {
            array_stride: INSTANCE_SIZE as wgpu::BufferAddress,
            // We need to switch from using a step mode of Vertex to Instance
            // This means that our shaders will only change to use the next
            // instance when the shader starts processing a new instance
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // A mat4 takes up 4 vertex slots as it is technically 4 vec4s. We need to define a slot
                // for each vec4. We'll have to reassemble the mat4 in
                // the shader.

                // Model matrix, col 0
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // Model matrix, col 1
                wgpu::VertexAttribute {
                    offset: (F32_SIZE * 4) as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // Model matrix, col 2
                wgpu::VertexAttribute {
                    offset: (F32_SIZE * 8) as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // Model matrix, col 3
                wgpu::VertexAttribute {
                    offset: (F32_SIZE * 12) as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // Normal matrix, col 0
                wgpu::VertexAttribute {
                    offset: (MAT4_SIZE) as wgpu::BufferAddress,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Normal matrix, col 1
                wgpu::VertexAttribute {
                    offset: (MAT4_SIZE + VEC3_SIZE) as wgpu::BufferAddress,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Normal matrix, col 2
                wgpu::VertexAttribute {
                    offset: (MAT4_SIZE + VEC3_SIZE * 2) as wgpu::BufferAddress,
                    shader_location: 11,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // model (and vertex) color
                wgpu::VertexAttribute {
                    offset: (MAT4_SIZE + MAT3_SIZE) as wgpu::BufferAddress,
                    shader_location: 12,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // Shinyness
                wgpu::VertexAttribute {
                    offset: (MAT4_SIZE + MAT3_SIZE + VEC3_SIZE) as wgpu::BufferAddress,
                    shader_location: 13,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }

    /// Converts to a model matrix
    pub fn to_bytes(&self) -> [u8; INSTANCE_SIZE] {
        let mut result = [0; INSTANCE_SIZE];

        let model_mat = Mat4::new_translation(self.position)
            * self.orientation.to_matrix()
            * Mat4::new_scaler(self.scale);

        let normal_mat = self.orientation.to_matrix3();

        result[0..MAT4_SIZE].clone_from_slice(&model_mat.to_bytes());

        result[MAT4_SIZE..MAT4_SIZE + MAT3_SIZE].clone_from_slice(&normal_mat.to_bytes());

        // todo: fn to convert Vec3 to byte array?
        let mut color_buf = [0; VEC3_SIZE];
        color_buf[0..F32_SIZE].clone_from_slice(&self.color.x.to_ne_bytes());
        color_buf[F32_SIZE..2 * F32_SIZE].clone_from_slice(&self.color.y.to_ne_bytes());
        color_buf[2 * F32_SIZE..3 * F32_SIZE].clone_from_slice(&self.color.z.to_ne_bytes());

        result[MAT4_SIZE + MAT3_SIZE..INSTANCE_SIZE - F32_SIZE].clone_from_slice(&color_buf);
        // todo
        // result[MAT4_SIZE + MAT3_SIZE..INSTANCE_SIZE - F32_SIZE]
        //     // .clone_from_slice(&self.color.to_bytes_uniform());
        //     .clone_from_slice(&self.color.to_bytes());

        result[INSTANCE_SIZE - F32_SIZE..INSTANCE_SIZE]
            .clone_from_slice(&self.shinyness.to_ne_bytes());

        result
    }
}

#[derive(Clone, Debug)]
pub struct Mesh {
    // pub name: String,
    // pub vertex_buffer: wgpu::Buffer,
    // pub index_buffer: wgpu::Buffer,
    // pub vertex_buffer: Vec<usize>,
    // pub index_buffer: Vec<usize>,
    // pub num_elements: u32,
    pub vertices: Vec<Vertex>,
    /// These indices are relative to 0 for this mesh. When adding to a global index
    /// buffer, we offset them by previous meshes' vertex counts.
    pub indices: Vec<usize>,
    pub material: usize,
}

/// Represents an entity in the world. This is not fundamental to the WGPU system.
#[derive(Clone, Debug)]
pub struct Entity {
    /// Index of the mesh this entity references. (or perhaps its index?)
    pub mesh: usize,
    /// Position in the world, relative to world origin
    pub position: Vec3,
    /// Rotation, relative to up.
    pub orientation: Quaternion,
    pub scale: f32, // 1.0 is original.
    pub color: (f32, f32, f32),
    pub shinyness: f32, // 0 to 1.
}

impl Entity {
    pub fn new(
        mesh: usize,
        position: Vec3,
        orientation: Quaternion,
        scale: f32,
        color: (f32, f32, f32),
        shinyness: f32,
    ) -> Self {
        Self {
            mesh,
            position,
            orientation,
            scale,
            color,
            shinyness,
        }
    }
}

#[derive(Clone, Copy, Debug)]
/// Default controls. Provides easy defaults. For maximum flexibility, choose `None`,
/// and implement controls in the `event_handler` function.
pub enum ControlScheme {
    /// No controls; provide all controls in application code.
    None,
    /// Keyboard controls for movement along 3 axis, and rotation around the Z axis. Mouse
    /// for rotation around the X and Y axes. Shift to multiply speed of keyboard controls.
    FreeCamera,
    /// FPS-style camera. Ie, no Z-axis roll, no up/down movement, and can't look up past TAU/4.
    /// todo: Unimplemented
    Fps,
    /// The mouse rotates the camera around a fixed point.
    /// todo: inner Vec of the point?
    /// todo: Unimplemented
    Arc,
}

impl Default for ControlScheme {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Clone, Debug)]
pub struct Scene {
    pub meshes: Vec<Mesh>,
    pub entities: Vec<Entity>,
    pub camera: Camera,
    pub lighting: Lighting,
    pub background_color: (f32, f32, f32),
    pub window_title: String,
    pub window_size: (f32, f32),
}

impl Default for Scene {
    fn default() -> Self {
        Self {
            meshes: Vec::new(),
            entities: Vec::new(),
            camera: Default::default(),
            lighting: Default::default(),
            // todo: Consider a separate window struct.
            background_color: (0.7, 0.7, 0.7),
            window_title: "(Window title here)".to_owned(),
            window_size: (900., 600.),
        }
    }
}

#[derive(Clone, Debug)]
/// These sensitivities are in units (position), or radians (orientation) per second.
pub struct InputSettings {
    pub move_sens: f32,
    pub rotate_sens: f32,
    pub rotate_key_sens: f32,
    /// How much the move speed is multiplied when holding the run key.
    pub run_factor: f32,
    pub initial_controls: ControlScheme,
}

impl Default for InputSettings {
    fn default() -> Self {
        Self {
            initial_controls: Default::default(),
            move_sens: 1.5,
            rotate_sens: 0.45,
            rotate_key_sens: 1.0,
            run_factor: 5.,
        }
    }
}

#[derive(Clone, Debug)]
/// GUI settings
pub struct UiSettings {
    /// Used, as a quick+dirty approach, to disable events when the mouse is in the GUI section.
    pub width: f64,
    pub icon_path: Option<String>,
}

impl Default for UiSettings {
    fn default() -> Self {
        Self {
            width: 0.,
            icon_path: None,
        }
    }
}

/// This struct is exposed in the API, and passed by callers to indicate in the render,
/// event, GUI etc update functions, if the engine should update various things.
#[derive(Default)]
pub struct EngineUpdates {
    pub entities: bool,
    pub camera: bool,
    pub lighting: bool,
}
