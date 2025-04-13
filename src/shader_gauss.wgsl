// ────────────────────────────────────────────────────────────────────
// Gaussian impostor pass ‑ complete WGSL
// ────────────────────────────────────────────────────────────────────
struct Camera {
    proj_view : mat4x4<f32>,
    position  : vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera : Camera;

// ── vertex input ───────────────────────────────────────────────────
struct QuadVertex {
    @location(0) pos : vec2<f32>,   // −1 … 1
};

// ── varyings ───────────────────────────────────────────────────────
struct VSOut {
    @builtin(position) clip_pos : vec4<f32>,
    @location(0) world_pos      : vec3<f32>,
    @interpolate(flat) @location(1) center : vec3<f32>,
    @interpolate(flat) @location(2) amp    : f32,
    @interpolate(flat) @location(3) width  : f32,
};

// ── vertex shader ──────────────────────────────────────────────────
@vertex
fn vs_main(
        v                  : QuadVertex,
        // per‑instance data
        @location(1) instance_center : vec3<f32>,
        @location(2) instance_amp    : f32,
        @location(3) instance_width  : f32
    ) -> VSOut
{
    // NOTE: camera.proj_view is *projection × view*.
    // To get camera‑space basis vectors you should really pass the view
    // (or its inverse) separately.  For this demo we assume the view
    // matrix is rigid‑body only and extract its first two columns.
    let view_right = camera.proj_view[0].xyz;   // column 0
    let view_up    = camera.proj_view[1].xyz;   // column 1

    let radius = 3.0 * instance_width;          // 3 σ ≈ 99.7 %

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
    return out;
}

// ── fragment shader ────────────────────────────────────────────────
@fragment
fn fs_main(in : VSOut) -> @location(0) vec4<f32> {
    // r² in world space
    let r2 = dot(in.world_pos - in.center, in.world_pos - in.center);

    // ρ(r) = A·exp(−r² / 2σ²)
    let density = in.amp * exp( -r2 / (2.0 * in.width * in.width) );

    // Soft‑cut very faint fragments to save overdraw
    if (density < 0.005) { discard; }

    let colour = vec3<f32>(1.0, 1.0, 1.0);          // white cloud
    let alpha  = clamp(density, 0.0, 1.0);          // or map however you like

    // premultiplied‑alpha output (works with ALPHA_BLENDING)
    return vec4<f32>(colour * alpha, alpha);
}
