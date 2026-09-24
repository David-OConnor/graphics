// Screen-space outlines, drawn around the silhouettes of outlined entities.
//
// Input: The mask texture from the outline mask pass (`fs_outline_mask` in shader.wgsl). Covered
// pixels hold the outline color in rgb, and its thickness, in logical pixels, in alpha. Uncovered
// pixels are 0.
//
// We find the nearest covered pixel to each uncovered one in two separable passes, so the cost per
// pixel is linear in the outline thickness instead of quadratic:
// 1: `fs_outline_horiz` finds, for each pixel, the horizontal offset to the nearest covered pixel
// in its row.
// 2: `fs_outline` combines those row offsets with vertical offsets to get the Euclidean distance
// to the nearest covered pixel, and alpha-blends the outline onto the scene.

struct OutlineUniforms {
    // Physical pixels per logical pixel.
    scale: f32,
    // How far to search for covered pixels, in physical pixels. Covers the thickest outline present.
    radius: i32,
    _pad0: f32,
    _pad1: f32,
}

// Written by the horizontal pass when there is no covered pixel within `radius` in the row.
// Must match `NONE_OFFSET` in outline.rs, which clears the texture to it.
const NONE_OFFSET: f32 = 10000.;

@group(0) @binding(0) var mask_tex: texture_2d<f32>;
@group(0) @binding(1) var<uniform> ou: OutlineUniforms;
// Output of the horizontal pass. Only bound for the final pass.
@group(0) @binding(2) var horiz_tex: texture_2d<f32>;

struct VOut {
    @builtin(position) pos: vec4<f32>,
}

// Full-screen triangle — no vertex buffer needed.
@vertex
fn vs_outline(@builtin(vertex_index) vi: u32) -> VOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1., -1.),
        vec2<f32>( 3., -1.),
        vec2<f32>(-1.,  3.),
    );
    return VOut(vec4<f32>(positions[vi], 0., 1.));
}

fn in_bounds(px: vec2<i32>, dims: vec2<i32>) -> bool {
    return all(px >= vec2<i32>(0)) && all(px < dims);
}

fn is_covered(px: vec2<i32>) -> bool {
    return textureLoad(mask_tex, px, 0).a > 0.;
}

@fragment
fn fs_outline_horiz(v: VOut) -> @location(0) vec4<f32> {
    let px = vec2<i32>(v.pos.xy);
    let dims = vec2<i32>(textureDimensions(mask_tex));

    if is_covered(px) {
        return vec4<f32>(0., 0., 0., 0.);
    }

    // Search outward, so the first hit is the nearest.
    for (var d = 1; d <= ou.radius; d++) {
        let left = px - vec2<i32>(d, 0);
        if in_bounds(left, dims) && is_covered(left) {
            return vec4<f32>(f32(-d), 0., 0., 0.);
        }

        let right = px + vec2<i32>(d, 0);
        if in_bounds(right, dims) && is_covered(right) {
            return vec4<f32>(f32(d), 0., 0., 0.);
        }
    }

    return vec4<f32>(NONE_OFFSET, 0., 0., 0.);
}

@fragment
fn fs_outline(v: VOut) -> @location(0) vec4<f32> {
    let px = vec2<i32>(v.pos.xy);
    let dims = vec2<i32>(textureDimensions(mask_tex));

    // The outline surrounds the silhouette; it doesn't cover it.
    if is_covered(px) {
        discard;
    }

    var color = vec3<f32>(0.);
    var alpha = 0.;

    for (var dy = -ou.radius; dy <= ou.radius; dy++) {
        let row = px + vec2<i32>(0, dy);
        if !in_bounds(row, dims) {
            continue;
        }

        // The nearest covered pixel in this row.
        let dx = textureLoad(horiz_tex, row, 0).r;
        if abs(dx) > f32(ou.radius) {
            continue;
        }

        let mask = textureLoad(mask_tex, row + vec2<i32>(i32(dx), 0), 0);
        let dist = length(vec2<f32>(dx, f32(dy)));

        // Full coverage out to the thickness, then a 1-pixel falloff to anti-alias the outer edge.
        let a = clamp(mask.a * ou.scale + 1. - dist, 0., 1.);
        if a > alpha {
            alpha = a;
            color = mask.rgb;
        }
    }

    if alpha <= 0. {
        discard;
    }

    return vec4<f32>(color, alpha);
}
