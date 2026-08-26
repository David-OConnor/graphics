// Screen-space ambient occlusion (SSAO) — occlusion estimation pass.
//
// Reads the 1-sample depth buffer written by the geometry prepass, reconstructs
// world-space positions and normals per-pixel, then samples a fixed 16-point
// hemisphere kernel to estimate how much of the hemisphere is occluded by nearby
// geometry.
//
// The kernel is rotated by one of 16 angles picked from a 4×4 screen-space tile.
// That trades banding for high-frequency noise, which `shader_ssao_blur.wgsl` then
// removes exactly: a 4×4 box blur spans one whole tile, so every blurred pixel
// averages all 16 rotations (16 rotations × 16 samples = 256 effective samples).
// The blur is what makes the result smooth rather than speckled, so this pass
// renders to an offscreen AO buffer and must not be composited onto the scene
// directly.
//
// Output: R8 texture where 1.0 = unoccluded, 0.0 = fully occluded.

struct SsaoUniforms {
    // Combined projection * view matrix, for projecting hemisphere samples back to screen.
    proj_view:     mat4x4<f32>,
    // Inverse of proj_view, for reconstructing world-space position from depth + NDC.
    proj_view_inv: mat4x4<f32>,
    // World-space camera position (to orient normals toward the viewer).
    cam_pos:       vec4<f32>,
    near:          f32,
    far:           f32,
    // World-space hemisphere sample radius.
    radius:        f32,
    // Self-occlusion bias, as a fraction of the linear view distance to the pixel.
    // Relative rather than absolute because depth precision degrades with distance.
    bias:          f32,
    // Output strength multiplier: higher → darker crevices.
    strength:      f32,
    _pad0:         f32,
    _pad1:         f32,
    _pad2:         f32,
}

@group(0) @binding(0) var depth_tex: texture_depth_2d;
@group(0) @binding(1) var<uniform> su: SsaoUniforms;

struct VOut {
    @builtin(position) pos: vec4<f32>, // Full-screen triangle: no vertex buffer needed.
    @location(0) ndc: vec2<f32>,       // True rasterizer-injected NDC space
}

@vertex
fn vs_ssao(@builtin(vertex_index) vi: u32) -> VOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1., -1.),
        vec2<f32>( 3., -1.),
        vec2<f32>(-1.,  3.),
    );
    let p = positions[vi];
    return VOut(vec4<f32>(p, 0., 1.), p);
}

// The 3D viewport is a sub-rect of the window when the GUI takes a side or top
// panel, and the depth texture is window-sized — so pixels outside the viewport
// hold the prepass clear value, not geometry. Recover the viewport rect from the
// rasterizer: the full-screen triangle spans NDC [-1, 1], so the NDC-per-pixel
// scale locates both edges. Returns (min_x, min_y, max_x, max_y) in window pixels,
// max exclusive.
fn viewport_bounds(pos: vec2<f32>, ndc: vec2<f32>, ndc_per_px: vec2<f32>) -> vec4<f32> {
    let a = pos + (vec2<f32>(-1.) - ndc) / ndc_per_px;
    let b = pos + (vec2<f32>( 1.) - ndc) / ndc_per_px;
    // ndc_per_px.y is negative (NDC +y is up, window +y is down), so sort the pair.
    return vec4<f32>(min(a, b), max(a, b));
}

// Clamp to the viewport rect (and to the texture, defensively), so edge taps never
// read the untouched region under the GUI panel.
fn load_depth(px: vec2<i32>, dims: vec2<i32>, vp: vec4<f32>) -> f32 {
    let lo = max(vec2<i32>(vp.xy), vec2<i32>(0));
    let hi = min(vec2<i32>(vp.zw) - 1, dims - 1);
    return textureLoad(depth_tex, clamp(px, lo, hi), 0);
}

// Non-linear perspective depth (0..1) → linear view-space distance.
fn linearize(d: f32) -> f32 {
    return su.near * su.far / (su.far - d * (su.far - su.near));
}

// Reconstruct world-space position directly from proper NDC space.
fn world_from_depth(ndc_xy: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(ndc_xy, depth, 1.0);
    let world_h = su.proj_view_inv * ndc;
    return world_h.xyz / world_h.w;
}

// Rotate 2-D vector by angle (radians).
fn rot2(v: vec2<f32>, a: f32) -> vec2<f32> {
    let s = sin(a);
    let c = cos(a);
    return vec2<f32>(c * v.x - s * v.y, s * v.x + c * v.y);
}

@fragment
fn fs_ssao(input: VOut) -> @location(0) vec4<f32> {
    // Differentiate NDC with respect to window pixels. Constant over the triangle,
    // but derivatives are only defined in uniform control flow, so take them before
    // the sky early-out below.
    let ndc_per_px = vec2<f32>(dpdx(input.ndc.x), dpdy(input.ndc.y));

    let dims = vec2<i32>(textureDimensions(depth_tex));
    let vp   = viewport_bounds(input.pos.xy, input.ndc, ndc_per_px);
    let px   = vec2<i32>(input.pos.xy);

    let depth0 = load_depth(px, dims, vp);

    // Sky / background: nothing to occlude, return unoccluded white.
    if depth0 >= 1.0 { return vec4<f32>(1., 1., 1., 1.); }

    let pos0 = world_from_depth(input.ndc, depth0);

    // ── Normal reconstruction from neighbouring depth samples ────────────────
    // Pick the neighbour that is closest in depth on each axis (avoids artefacts
    // at depth discontinuities such as silhouettes).
    let dx = vec2<i32>(1, 0);
    let dy = vec2<i32>(0, 1);

    let depth_r = load_depth(px + dx, dims, vp);
    let depth_l = load_depth(px - dx, dims, vp);
    let depth_u = load_depth(px + dy, dims, vp);
    let depth_d = load_depth(px - dy, dims, vp);

    let use_right = abs(depth_r - depth0) < abs(depth_l - depth0);
    let use_down  = abs(depth_u - depth0) < abs(depth_d - depth0);

    let h_off = vec2<f32>(ndc_per_px.x, 0.);
    let v_off = vec2<f32>(0., ndc_per_px.y);

    let h_ndc   = select(input.ndc - h_off, input.ndc + h_off, use_right);
    let h_depth = select(depth_l, depth_r, use_right);
    let h_sign  = select(-1.0, 1.0, use_right);

    let v_ndc   = select(input.ndc - v_off, input.ndc + v_off, use_down);
    let v_depth = select(depth_d, depth_u, use_down);
    let v_sign  = select(-1.0, 1.0, use_down);

    let tang  = (world_from_depth(h_ndc, h_depth) - pos0) * h_sign;
    let btng  = (world_from_depth(v_ndc, v_depth) - pos0) * v_sign;
    var normal = normalize(cross(tang, btng));

    // Ensure normal faces the camera.
    if dot(normal, pos0 - su.cam_pos.xyz) > 0.0 { normal = -normal; }

    // ── Orthonormal TBN frame for rotating the hemisphere ────────────────────
    var tangent: vec3<f32>;
    if abs(normal.y) < 0.9 {
        tangent = normalize(cross(normal, vec3<f32>(0., 1., 0.)));
    } else {
        tangent = normalize(cross(normal, vec3<f32>(1., 0., 0.)));
    }
    let bitangent = cross(normal, tangent);

    // Kernel rotation angle, keyed to a 4×4 screen tile so the blur pass can average
    // all 16 of them back out. The golden-ratio stride keeps neighbouring pixels far
    // apart in angle both horizontally (index +1) and vertically (index +4), which is
    // what stops the tile itself from showing up as a pattern.
    let tile      = px & vec2<i32>(3, 3);
    let tile_idx  = f32(tile.y * 4 + tile.x);
    let rot_angle = fract(tile_idx * 0.6180339887) * 6.283185;

    let linear0 = linearize(depth0);

    // ── Fixed 16-point hemisphere kernel (z > 0 → away from surface) ─────────
    // Samples are intentionally non-uniform in distance (closer ones carry more
    // weight for small crevices; farther ones catch larger occluders).
    var kernel = array<vec3<f32>, 16>(
        vec3<f32>( 0.53813,  0.18508,  0.19261),
        vec3<f32>( 0.13488, -0.87838,  0.40077),
        vec3<f32>( 0.35758, -0.38415,  0.28973),
        vec3<f32>(-0.22072,  0.12715,  0.11035),
        vec3<f32>(-0.26987,  0.53448,  0.32635),
        vec3<f32>(-0.07967,  0.04402,  0.26865),
        vec3<f32>(-0.09560, -0.30776,  0.24669),
        vec3<f32>( 0.16201,  0.11422,  0.29748),
        vec3<f32>(-0.38296,  0.56558,  0.46939),
        vec3<f32>(-0.10660, -0.64196,  0.44051),
        vec3<f32>( 0.01006,  0.09939,  0.18907),
        vec3<f32>( 0.09448,  0.60592,  0.58505),
        vec3<f32>( 0.55916,  0.67533,  0.08070),
        vec3<f32>(-0.18608,  0.16559,  0.07534),
        vec3<f32>( 0.14520, -0.39572,  0.08600),
        vec3<f32>(-0.32639,  0.26070,  0.09231),
    );

    // Accelerating sample distance: i/16 lerped into i²/16² keeps short-range
    // samples dense (better for tight crevices) without losing coverage.
    var scale_sq = array<f32, 16>(
        0.0039, 0.0156, 0.0352, 0.0625,
        0.0977, 0.1406, 0.1914, 0.2500,
        0.3164, 0.3906, 0.4727, 0.5625,
        0.6602, 0.7656, 0.8789, 1.0000,
    );

    // Self-occlusion bias in linear view-space distance. A constant bias in raw
    // depth-buffer units is scale-dependent — the same delta spans a hair's width up
    // close and metres far away — which is what produced speckled acne on curved
    // surfaces. Scaling with distance tracks how depth precision actually degrades.
    let bias_linear = su.bias * linear0;

    var occlusion = 0.0;
    let num_samples = 16;

    for (var i = 0; i < num_samples; i++) {
        var s = kernel[i];

        // Rotate the in-plane (xy) components by the per-pixel random angle.
        let rxy = rot2(s.xy, rot_angle);
        s = vec3<f32>(rxy, s.z);

        // Transform from TBN to world space.
        let world_dir = tangent * s.x + bitangent * s.y + normal * s.z;

        // Accelerated radial scale: samples cluster near the surface.
        let scale = mix(0.1, 1.0, scale_sq[i]);
        let sample_world = pos0 + world_dir * su.radius * scale;

        // Project sample point into clip space.
        let clip = su.proj_view * vec4<f32>(sample_world, 1.0);
        if clip.w <= 0.0 { continue; }
        let ndc_xy        = clip.xy / clip.w;
        let sample_depth  = clip.z  / clip.w;   // depth in [0,1] for LH proj

        // Discard out-of-viewport and behind-near-plane samples.
        if any(ndc_xy < vec2<f32>(-1.0)) || any(ndc_xy > vec2<f32>(1.0)) { continue; }
        if sample_depth < 0.0 || sample_depth > 1.0 { continue; }

        // Find exactly which physical window pixel this projected coordinate maps to
        let delta_ndc = ndc_xy - input.ndc;
        let sample_px = vec2<i32>(input.pos.xy + delta_ndc / ndc_per_px);

        // Fetch geometry depth at the projected location.
        let geom_depth = load_depth(sample_px, dims, vp);

        // Range check: suppress contributions from surfaces far away in depth
        // (avoids halos at depth discontinuities).
        let linear_geom   = linearize(geom_depth);
        let linear_sample = linearize(sample_depth);
        let range_check = smoothstep(0.0, 1.0, su.radius / abs(linear0 - linear_geom));

        // Occluded when geometry is closer to the camera than the sample point.
        if linear_geom < linear_sample - bias_linear {
            occlusion += range_check;
        }
    }

    occlusion /= f32(num_samples);

    // ao = 1 (no occlusion) → multiply scene by 1 (no change).
    // ao = 0 (full occlusion) → multiply scene by 0 (black).
    let ao = 1.0 - clamp(occlusion * su.strength, 0.0, 1.0);
    return vec4<f32>(ao, ao, ao, 1.0);
}
