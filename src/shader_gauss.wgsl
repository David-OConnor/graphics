struct Camera {
    proj_view : mat4x4<f32>,
    position  : vec4<f32>,
};

// Can't directly calculate the required inverse on CPU.
struct CameraBasis {
    right : vec3<f32>,  _pad0 : f32,
    up    : vec3<f32>,  _pad1 : f32,
};

@group(0) @binding(0)
var<uniform> camera: Camera;
@group(0) @binding(1)
var<uniform> cameraBasis: CameraBasis;


// Vertex input (per-vertex and per-instance):
struct VertexInput {
    @location(0) pos: vec2<f32>,         // Corner position (–1 to +1)
    @location(1) center: vec3<f32>,       // Instance center position (world-space)
    @location(2) amplitude: f32,          // Instance peak brightness
    @location(3) width: f32,             // Gaussian width (std. dev.)
    @location(4) color: vec4<f32>,        // Instance RGBA color
};

// Vertex output (to fragment):
struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) local_offset: vec2<f32>, // Offset from center in world (along right/up plane)
    @location(1) inst_color: vec4<f32>,   // Pass through instance color
    @location(2) inst_amplitude: f32,     // Pass through amplitude
    @location(3) inst_width: f32,         // Pass through width
};

// Fragment input matches VertexOutput
struct FragmentInput {
    @location(0) local_offset: vec2<f32>,
    @location(1) inst_color: vec4<f32>,
    @location(2) inst_amplitude: f32,
    @location(3) inst_width: f32,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    // Compute world-space offset for this vertex using camera basis (billboarding)
    let right = cameraBasis.right;
    let up    = cameraBasis.up;

    // Treat `width` as the quad's half-size (so quad spans 2*width in world units)

     // This thresh affects saturation, and when the gauss stops drawing. 3. is a good default.
     // Higher values will draw fainter areas.
    let cutoff_thresh = 3.5;
    let offset_world = input.pos.x * right * input.width * cutoff_thresh +
                       input.pos.y * up    * input.width * cutoff_thresh;

    // World-space position of this vertex (billboard oriented toward camera)
    let world_pos = input.center + offset_world;
    // Project to clip space
    let clip_pos = camera.proj_view * vec4<f32>(world_pos, 1.0);

    var output: VertexOutput;

    output.clip_position = clip_pos;
    // Pass the 2D offset in world-plane coordinates to fragment (for distance calc)
//    output.local_offset = input.pos * input.width;
    output.local_offset = input.pos * input.width * cutoff_thresh;
    // Pass through instance attributes needed in fragment
    output.inst_color = input.color;
    output.inst_amplitude = input.amplitude;
    output.inst_width = input.width;

    return output;
}

@fragment
fn fs_main(input: FragmentInput) -> @location(0) vec4<f32> {
    // Compute squared distance from center in world space (r^2 = x^2 + y^2)
    let offset = input.local_offset;
    let r_sq = offset.x * offset.x + offset.y * offset.y;

    // Gaussian radial falloff: exp(-r^2 / (2σ^2)), σ = inst_width
    let sigma = input.inst_width;

    // intensity = amplitude * exp(-(r^2) / (2 * sigma^2))
    let intensity = input.inst_amplitude * exp(-r_sq / (2.0 * sigma * sigma));

    // Modulate base color by intensity (premultiplied alpha)
    var color = input.inst_color * intensity;

    return color;
}