// SSAO blur + composite.
//
// `shader_ssao.wgsl` rotates its sample kernel by one of 16 angles chosen from a 4×4
// screen tile, so the raw AO buffer is noisy by construction — that noise is the
// price of trading a fixed kernel's banding for something a blur can integrate away.
// This pass runs a 4×4 box blur, which is exactly one tile period, so every output
// pixel averages all 16 rotations, then composites the result onto the scene with a
// multiplicative blend.
//
// The blur is depth-aware: taps whose linear depth differs from the centre by more
// than the SSAO radius are dropped, so occlusion never bleeds across a silhouette or
// out of geometry into the sky.

struct SsaoUniforms {
    proj_view:     mat4x4<f32>,
    proj_view_inv: mat4x4<f32>,
    cam_pos:       vec4<f32>,
    near:          f32,
    far:           f32,
    radius:        f32,
    bias:          f32,
    strength:      f32,
    _pad0:         f32,
    _pad1:         f32,
    _pad2:         f32,
}

@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var<uniform> su: SsaoUniforms;
@group(0) @binding(2) var ao_tex: texture_2d<f32>;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

@vertex
fn vs_ssao_blur(@builtin(vertex_index) vi: u32) -> VOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1., -1.),
        vec2<f32>( 3., -1.),
        vec2<f32>(-1.,  3.),
    );
    let p = positions[vi];
    return VOut(vec4<f32>(p, 0., 1.), p);
}

// See the matching comment in shader_ssao.wgsl: the AO and depth textures are
// window-sized but only the viewport sub-rect was rendered, so taps have to stay
// inside it. Returns (min_x, min_y, max_x, max_y) in window pixels, max exclusive.
fn viewport_bounds(pos: vec2<f32>, ndc: vec2<f32>, ndc_per_px: vec2<f32>) -> vec4<f32> {
    let a = pos + (vec2<f32>(-1.) - ndc) / ndc_per_px;
    let b = pos + (vec2<f32>( 1.) - ndc) / ndc_per_px;
    return vec4<f32>(min(a, b), max(a, b));
}

fn clamp_px(px: vec2<i32>, dims: vec2<i32>, vp: vec4<f32>) -> vec2<i32> {
    let lo = max(vec2<i32>(vp.xy), vec2<i32>(0));
    let hi = min(vec2<i32>(vp.zw) - 1, dims - 1);
    return clamp(px, lo, hi);
}

fn linearize(d: f32) -> f32 {
    return su.near * su.far / (su.far - d * (su.far - su.near));
}

// 4×4 box: offsets -2..1 on each axis cover one full noise tile.
// LO = -1 and HI = 0 produces a less smooth/more pixelated effect.
const TAP_LO: i32 = -2;
const TAP_HI: i32 =  1;

@fragment
fn fs_ssao_blur(input: VOut) -> @location(0) vec4<f32> {
    let ndc_per_px = vec2<f32>(dpdx(input.ndc.x), dpdy(input.ndc.y));
    let dims = vec2<i32>(textureDimensions(ao_tex));
    let vp   = viewport_bounds(input.pos.xy, input.ndc, ndc_per_px);
    let px   = vec2<i32>(input.pos.xy);

    let linear0 = linearize(textureLoad(depth_tex, clamp_px(px, dims, vp), 0));

    // Two surfaces further apart than the sample radius can't occlude each other, so
    // that is also the right cutoff for deciding whether their AO should be mixed. The
    // relative term keeps oblique or distant geometry — where one pixel of screen space
    // covers a lot of depth — from rejecting its own neighbours and staying noisy.
    let depth_cutoff = max(su.radius, 0.05 * linear0);

    var sum    = 0.0;
    var weight = 0.0;

    for (var y = TAP_LO; y <= TAP_HI; y++) {
        for (var x = TAP_LO; x <= TAP_HI; x++) {
            let p = clamp_px(px + vec2<i32>(x, y), dims, vp);
            let d = linearize(textureLoad(depth_tex, p, 0));
            // 1 when the tap sits on the same surface as the centre, 0 otherwise.
            let w = select(0.0, 1.0, abs(d - linear0) <= depth_cutoff);
            sum    += textureLoad(ao_tex, p, 0).r * w;
            weight += w;
        }
    }

    // The centre tap always passes its own test, so `weight` is never 0.
    let ao = sum / weight;
    return vec4<f32>(ao, ao, ao, 1.0);
}
