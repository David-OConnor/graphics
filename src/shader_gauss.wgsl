// ────────────────────────────────────────────────────────────────────
// Gaussian impostor pass ‑ complete WGSL
// ────────────────────────────────────────────────────────────────────
struct Camera {
    proj_view : mat4x4<f32>,
    position  : vec4<f32>,
    proj      : mat4x4<f32>,
    view      : mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> camera : Camera;

// Can't directly calculate the required inverse on CPU.
struct CameraBasis {
    right : vec3<f32>,  _pad0 : f32,
    up    : vec3<f32>,  _pad1 : f32,
};
@group(0) @binding(1) var<uniform> cam_basis : CameraBasis;

// ── vertex input ───────────────────────────────────────────────────
struct QuadVertex {
    @location(0) pos : vec2<f32>,   // −1 … 1
};

// ── varyings ───────────────────────────────────────────────────────
struct VSOut {
    @builtin(position) clip_pos : vec4<f32>,
    @location(0) world_pos      : vec3<f32>,
    //todo: What is interpolate?
    @interpolate(flat) @location(1) center : vec3<f32>,
    @interpolate(flat) @location(2) amp    : f32,
    @interpolate(flat) @location(3) width  : f32,
    @interpolate(flat) @location(4) color  : vec4<f32>,
};

// ── vertex shader ──────────────────────────────────────────────────
@vertex
fn vs_main(
        v                  : QuadVertex,
        @location(1) instance_center : vec3<f32>,
        @location(2) instance_amp    : f32,
        @location(3) instance_width  : f32,
        @location(4) instance_color  : vec4<f32>,
    ) -> VSOut
{
    // world‑space camera basis comes straight from the uniform
    let view_right = cam_basis.right;
    let view_up    = cam_basis.up;

    let radius = 3.0 * instance_width;

    let world_pos =
          instance_center
        + v.pos.x * radius * view_right
        + v.pos.y * radius * view_up;

    var out : VSOut;
    out.clip_pos  = camera.proj_view * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.center    = instance_center;
    out.amp       = instance_amp;
    out.width     = instance_width;
    out.color     = instance_color;
    return out;
}

// ── fragment shader ────────────────────────────────────────────────
@fragment
fn fs_main(in : VSOut) -> @location(0) vec4<f32> {
    // r² in world space
    let r2 = dot(in.world_pos - in.center, in.world_pos - in.center);

    // ρ(r) = A·exp(−r² / 2σ²)
//    let density = in.amp * exp( -r2 / (2.0 * in.width * in.width) );
    let density = in.amp * exp( -r2 * in.width );

    // Soft‑cut very faint fragments to save overdraw
    if (density < 0.005) { discard; }

    let color = in.color.xyz;          // white cloud
    let alpha  = clamp(density, 0.0, in.color[3]);          // or map however you like

    // premultiplied‑alpha output (works with ALPHA_BLENDING)
    return vec4<f32>(color * alpha, alpha);
}
