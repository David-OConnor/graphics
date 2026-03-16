// Full-screen contour overlay pass.
// Reads the 1-sample depth texture written by the contour prepass, detects
// depth discontinuities between neighbouring pixels, and alpha-blends dark
// contour lines on top of the already-rendered scene.

struct ContourUniforms {
    // Minimum linear-depth difference that registers as an edge.
    depth_threshold: f32,
    // Strength for depth-revealing lines (opacity scales with depth jump).
    depth_revealing: f32,
    // Strength for intersection-revealing lines (binary threshold).
    intersection_revealing: f32,
    // Camera near/far for linearising the perspective depth buffer.
    near: f32,
    far: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var<uniform> cu: ContourUniforms;

struct VOut {
    @builtin(position) pos: vec4<f32>,
}

// Full-screen triangle — no vertex buffer needed.
@vertex
fn vs_contour(@builtin(vertex_index) vi: u32) -> VOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1., -1.),
        vec2<f32>( 3., -1.),
        vec2<f32>(-1.,  3.),
    );
    return VOut(vec4<f32>(positions[vi], 0., 1.));
}

fn load_depth(px: vec2<i32>, dims: vec2<i32>) -> f32 {
    let c = clamp(px, vec2<i32>(0), dims - 1);
    return textureLoad(depth_tex, c, 0);
}

// Convert non-linear perspective depth (0..1) to linear view-space distance.
fn linearize(d: f32) -> f32 {
    return cu.near * cu.far / (cu.far - d * (cu.far - cu.near));
}

@fragment
fn fs_contour(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let px   = vec2<i32>(i32(frag_pos.x), i32(frag_pos.y));
    let dims = vec2<i32>(textureDimensions(depth_tex));

    let d  = linearize(load_depth(px,                        dims));
    let dr = linearize(load_depth(px + vec2<i32>( 1,  0),   dims));
    let dl = linearize(load_depth(px + vec2<i32>(-1,  0),   dims));
    let du = linearize(load_depth(px + vec2<i32>( 0,  1),   dims));
    let dd = linearize(load_depth(px + vec2<i32>( 0, -1),   dims));

    let diff = max(max(abs(d - dr), abs(d - dl)),
                   max(abs(d - du), abs(d - dd)));

    // depth_revealing: opacity grows linearly with the size of the depth jump.
    let depth_alpha = saturate(diff / cu.depth_threshold) * cu.depth_revealing;

    // intersection_revealing: binary once the threshold is crossed.
    let isect_alpha = step(cu.depth_threshold, diff) * cu.intersection_revealing;

    let alpha = saturate(depth_alpha + isect_alpha);
    return vec4<f32>(0., 0., 0., alpha);
}
