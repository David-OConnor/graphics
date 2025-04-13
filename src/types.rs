//! https://sotrh.github.io/learn-wgpu/beginner/tutorial9-models/#rendering-a-mesh

#[cfg(feature = "app_utils")]
use bincode::{Decode, Encode};
use lin_alg::f32::{Mat4, Quaternion, Vec3, Vec4};
use wgpu::{VertexAttribute, VertexBufferLayout, VertexFormat};

use crate::{camera::Camera, lighting::Lighting};

// These sizes are in bytes. We do this, since that's the data format expected by the shader.
pub const F32_SIZE: usize = 4;

pub const VEC3_SIZE: usize = 3 * F32_SIZE;
pub const VEC4_SIZE: usize = 4 * F32_SIZE;
pub const VEC3_UNIFORM_SIZE: usize = 4 * F32_SIZE;
pub const MAT4_SIZE: usize = 16 * F32_SIZE;
pub const MAT3_SIZE: usize = 9 * F32_SIZE;

pub const VERTEX_SIZE: usize = 14 * F32_SIZE;
// Note that position, orientation, and scale are combined into a single 4x4 transformation
// matrix. Note that unlike uniforms, we don't need alignment padding, and can use Vec3 directly.
pub const INSTANCE_SIZE: usize = MAT4_SIZE + MAT3_SIZE + VEC4_SIZE + F32_SIZE;

// Create the vertex buffer memory layout, for our vertexes passed from CPU
// to the vertex shader. Corresponds to `VertexIn` in the shader. Each
// item here is for a single vertex.
pub(crate) const VERTEX_LAYOUT: VertexBufferLayout<'static> = VertexBufferLayout {
    array_stride: VERTEX_SIZE as wgpu::BufferAddress,
    step_mode: wgpu::VertexStepMode::Vertex,
    attributes: &[
        // Vertex position
        VertexAttribute {
            offset: 0,
            shader_location: 0,
            format: VertexFormat::Float32x3,
        },
        // Texture coordinates
        VertexAttribute {
            offset: VEC3_SIZE as wgpu::BufferAddress,
            shader_location: 1,
            format: VertexFormat::Float32x2,
        },
        // Normal vector
        VertexAttribute {
            offset: (2 * F32_SIZE + VEC3_SIZE) as wgpu::BufferAddress,
            shader_location: 2,
            format: VertexFormat::Float32x3,
        },
        // Tangent (Used to align textures)
        VertexAttribute {
            offset: (2 * F32_SIZE + 2 * VEC3_SIZE) as wgpu::BufferAddress,
            shader_location: 3,
            format: VertexFormat::Float32x3,
        },
        // Bitangent (Used to align textures)
        VertexAttribute {
            offset: (2 * F32_SIZE + 3 * VEC3_SIZE) as wgpu::BufferAddress,
            shader_location: 4,
            format: VertexFormat::Float32x3,
        },
    ],
};

// Create the vertex buffer memory layout, for our vertexes passed from the
// vertex to the fragment shader. Corresponds to `VertexOut` in the shader. Each
// item here is for a single vertex. Cannot share locations with `VertexIn`, so
// we start locations after `VertexIn`'s last one.
pub(crate) const INSTANCE_LAYOUT: VertexBufferLayout<'static> = VertexBufferLayout {
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
        VertexAttribute {
            offset: 0,
            shader_location: 5,
            format: VertexFormat::Float32x4,
        },
        // Model matrix, col 1
        VertexAttribute {
            offset: (F32_SIZE * 4) as wgpu::BufferAddress,
            shader_location: 6,
            format: VertexFormat::Float32x4,
        },
        // Model matrix, col 2
        VertexAttribute {
            offset: (F32_SIZE * 8) as wgpu::BufferAddress,
            shader_location: 7,
            format: VertexFormat::Float32x4,
        },
        // Model matrix, col 3
        VertexAttribute {
            offset: (F32_SIZE * 12) as wgpu::BufferAddress,
            shader_location: 8,
            format: VertexFormat::Float32x4,
        },
        // Normal matrix, col 0
        VertexAttribute {
            offset: (MAT4_SIZE) as wgpu::BufferAddress,
            shader_location: 9,
            format: VertexFormat::Float32x3,
        },
        // Normal matrix, col 1
        VertexAttribute {
            offset: (MAT4_SIZE + VEC3_SIZE) as wgpu::BufferAddress,
            shader_location: 10,
            format: VertexFormat::Float32x3,
        },
        // Normal matrix, col 2
        VertexAttribute {
            offset: (MAT4_SIZE + VEC3_SIZE * 2) as wgpu::BufferAddress,
            shader_location: 11,
            format: VertexFormat::Float32x3,
        },
        // model (and vertex) color
        VertexAttribute {
            offset: (MAT4_SIZE + MAT3_SIZE) as wgpu::BufferAddress,
            shader_location: 12,
            format: VertexFormat::Float32x4,
        },
        // Shinyness
        VertexAttribute {
            offset: (MAT4_SIZE + MAT3_SIZE + VEC4_SIZE) as wgpu::BufferAddress,
            shader_location: 13,
            format: VertexFormat::Float32,
        },
    ],
};

#[derive(Clone, Copy, Debug)]
/// A general mesh. This should be sufficiently versatile to use for a number of purposes.
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
    pub tangent: Vec3,
    /// A bitangent vector is the result of the Cross Product between Vertex Normal and Vertex
    /// Tangent which is a unit vector perpendicular to both vectors at a given point.
    /// This is used to orient normal maps; corresponds to the +Y texture direction.
    pub bitangent: Vec3,
}

impl Vertex {
    /// Initialize position; change the others after init.
    pub fn new(position: [f32; 3], normal: Vec3) -> Self {
        Self {
            position,
            tex_coords: [0., 0.],
            normal,
            tangent: Vec3::new_zero(),
            bitangent: Vec3::new_zero(),
        }
    }

    pub fn to_bytes(&self) -> [u8; VERTEX_SIZE] {
        let mut result = [0; VERTEX_SIZE];

        result[0..4].clone_from_slice(&self.position[0].to_ne_bytes());
        result[4..8].clone_from_slice(&self.position[1].to_ne_bytes());
        result[8..12].clone_from_slice(&self.position[2].to_ne_bytes());
        result[12..16].clone_from_slice(&self.tex_coords[0].to_ne_bytes());
        result[16..20].clone_from_slice(&self.tex_coords[1].to_ne_bytes());

        result[20..32].clone_from_slice(&self.normal.to_bytes());
        result[32..44].clone_from_slice(&self.tangent.to_bytes());
        result[44..56].clone_from_slice(&self.bitangent.to_bytes());

        result
    }
}

/// Instances allow the GPU to render the same object multiple times.
/// "Instancing allows us to draw the same object multiple times with different properties
/// (position, orientation, size, color, etc.). "
/// todo: Relationship between this and entity?
pub struct Instance {
    pub position: Vec3,
    pub orientation: Quaternion,
    pub scale: Vec3,
    pub color: Vec3,
    pub opacity: f32,
    pub shinyness: f32,
}

impl Instance {
    /// Converts to a model matrix to a byte array, for passing to the GPU.
    pub fn to_bytes(&self) -> [u8; INSTANCE_SIZE] {
        let mut result = [0; INSTANCE_SIZE];

        let model_mat = Mat4::new_translation(self.position)
            * self.orientation.to_matrix()
            * Mat4::new_scaler_partial(self.scale);

        let normal_mat = self.orientation.to_matrix3();

        result[0..MAT4_SIZE].clone_from_slice(&model_mat.to_bytes());

        result[MAT4_SIZE..MAT4_SIZE + MAT3_SIZE].clone_from_slice(&normal_mat.to_bytes());

        // todo: fn to convert Vec3 to byte array?
        let mut color_buf = [0; VEC4_SIZE];
        color_buf[0..F32_SIZE].clone_from_slice(&self.color.x.to_ne_bytes());
        color_buf[F32_SIZE..2 * F32_SIZE].clone_from_slice(&self.color.y.to_ne_bytes());
        color_buf[2 * F32_SIZE..3 * F32_SIZE].clone_from_slice(&self.color.z.to_ne_bytes());
        color_buf[3 * F32_SIZE..4 * F32_SIZE].clone_from_slice(&self.opacity.to_ne_bytes());

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
    pub vertices: Vec<Vertex>,
    /// These indices are relative to 0 for this mesh. When adding to a global index
    /// buffer, we offset them by previous meshes' vertex counts.
    pub indices: Vec<usize>,
    pub material: usize,
}

/// Represents an entity in the world. This is not fundamental to the WGPU system.
#[derive(Clone, Debug)]
pub struct Entity {
    pub id: usize,
    /// Index of the mesh this entity references. (or perhaps its index?)
    pub mesh: usize,
    /// Position in the world, relative to world origin
    pub position: Vec3,
    /// Rotation, relative to up.
    pub orientation: Quaternion,
    pub scale: f32, // 1.0 is original.
    /// Scale by  axis. If `Some`, overrides scale.
    /// Not set in the constructor; set after manually.
    pub scale_partial: Option<Vec3>,
    pub color: (f32, f32, f32),
    pub opacity: f32,
    pub shinyness: f32, // 0 to 1.
}

impl Default for Entity {
    fn default() -> Self {
        Self {
            id: 0,
            mesh: 0,
            position: Vec3::new_zero(),
            orientation: Quaternion::new_identity(),
            scale: 1.,
            scale_partial: None,
            color: (1., 1., 1.),
            opacity: 1.,
            shinyness: 0.,
        }
    }
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
            id: 0, // todo: Determine how you'll handle this.
            mesh,
            position,
            orientation,
            scale,
            scale_partial: None,
            color,
            opacity: 1.,
            shinyness,
        }
    }
}

#[cfg_attr(feature = "app_utils", derive(Encode, Decode))]
#[derive(Clone, Copy, PartialEq, Debug, Default)]
/// Default controls. Provides easy defaults. For maximum flexibility, choose `None`,
/// and implement controls in the `event_handler` function.
pub enum ControlScheme {
    /// No controls; provide all controls in application code.
    None,
    #[default]
    /// Keyboard controls for movement along 3 axis, and rotation around the Z axis. Mouse
    /// for rotation around the X and Y axes. Shift to multiply speed of keyboard controls.
    FreeCamera,
    /// FPS-style camera. Ie, no Z-axis roll, no up/down movement, and can't look up past TAU/4.
    /// todo: Unimplemented
    Fps,
    /// The mouse rotates the camera around a fixed point.
    Arc { center: Vec3 },
}

#[derive(Clone, Debug)]
pub struct Scene {
    pub meshes: Vec<Mesh>,
    pub entities: Vec<Entity>,
    pub camera: Camera,
    pub lighting: Lighting,
    pub input_settings: InputSettings,
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
            input_settings: Default::default(),
            // todo: Consider a separate window struct.
            background_color: (0.7, 0.7, 0.7),
            window_title: "(Window title here)".to_owned(),
            window_size: (900., 600.),
        }
    }
}

impl Scene {
    /// Convert a screen position (x, y) to a 3D ray in world space.
    ///
    /// The canonical use case for this is finding the object in 3D space a user is intending to select
    /// with the cursor.A follow-up operation, for example, may be to find all objects that this vector
    /// passes near, and possibly select the one closest to the camera.
    pub fn screen_to_render(&self, screen_pos: (f32, f32)) -> (Vec3, Vec3) {
        let proj_view = self.camera.proj_mat.clone() * self.camera.view_mat();

        let proj_view_inv = match proj_view.inverse() {
            Some(p) => p,
            None => {
                eprintln!("Error inverting the projection matrix.");
                return (Vec3::new_zero(), Vec3::new_zero());
            }
        };

        // Convert screen position (assumed normalized to [0, 1] range)
        let sx = screen_pos.0 / self.window_size.0;
        let sy = screen_pos.1 / self.window_size.1;

        let clip_x = 2.0 * sx - 1.0;
        let clip_y = 1.0 - 2.0 * sy; // Flips the Y so 0 is top, 1 is bottom

        let near_clip = Vec4::new(clip_x, clip_y, 0.0, 1.0);
        let far_clip = Vec4::new(clip_x, clip_y, 1.0, 1.0);

        // Un-project them to world space.
        let near_world_h = proj_view_inv.clone() * near_clip;
        let far_world_h = proj_view_inv * far_clip;

        // Perspective divide to go from homogenous -> 3D.
        let near_world = near_world_h.xyz() / near_world_h.w;
        let far_world = far_world_h.xyz() / far_world_h.w;

        (near_world, far_world)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub enum ScrollBehavior {
    #[default]
    None,
    /// Move forward and back, relative to the camera when scrolling.
    /// When the left mouse button is held, this behavior changes to camera rolling.
    MoveRoll { move_amt: f32, rotate_amt: f32 },
}

#[derive(Clone, Debug)]
/// These sensitivities are in units (position), or radians (orientation) per second.
pub struct InputSettings {
    pub move_sens: f32,
    pub rotate_sens: f32,
    pub rotate_key_sens: f32,
    /// How much the move speed is multiplied when holding the run key.
    pub run_factor: f32,
    pub control_scheme: ControlScheme,
    /// Move forward and backwards with the scroll wheel; largely independent from
    /// control scheme. For now
    pub scroll_behavior: ScrollBehavior,
    pub middle_click_pan: bool,
}

impl Default for InputSettings {
    fn default() -> Self {
        Self {
            control_scheme: Default::default(),
            move_sens: 1.5,
            rotate_sens: 0.45,
            rotate_key_sens: 1.0,
            run_factor: 5.,
            scroll_behavior: Default::default(),
            middle_click_pan: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GraphicsSettings {
    pub msaa_samples: u32,
}

impl Default for GraphicsSettings {
    fn default() -> Self {
        Self { msaa_samples: 4 }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum UiLayout {
    Left,
    Right,
    Top,
    Bottom,
}

#[derive(Clone, Debug)]
/// GUI settings
pub struct UiSettings {
    pub layout: UiLayout,
    pub icon_path: Option<String>,
}

impl Default for UiSettings {
    fn default() -> Self {
        Self {
            layout: UiLayout::Left,
            icon_path: None,
        }
    }
}

/// This struct is exposed in the API, and passed by callers to indicate in the render,
/// event, GUI etc update functions, if the engine should update various things. When you change
/// the relevant part of the scene, the callbacks (event, etc) should set the corresponding
/// flag in this struct.
///
/// This process is required so certain internal structures like camera buffers, lighting buffers
/// etc are only computed and changed when necessary.
#[derive(Default)]
pub struct EngineUpdates {
    pub meshes: bool,
    pub entities: bool,
    pub camera: bool,
    pub lighting: bool,
}
