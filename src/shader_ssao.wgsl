// Screen-space ambient occlusion (SSAO).
//
// Reads the 1-sample depth buffer written by the geometry prepass, reconstructs
// world-space positions and normals per-pixel, then samples a fixed 16-point
// hemisphere kernel (with per-pixel random rotation to reduce banding) to
// estimate how much of the hemisphere is occluded by nearby geometry.
//
// Output: grey vec4 where 1.0 = unoccluded, 0.0 = fully occluded.
// Applied to the scene via a multiplicative blend: scene_color * ao.

struct SsaoUniforms {
    // Combined projection * view matrix, for projecting hemisphere samples back to screen.
    proj_view:     mat4x4<f32>,
    // Inverse of proj_view, for reconstructing world-space position from depth + UV.
    proj_view_inv: mat4x4<f32>,
    // World-space camera position (to orient normals toward the viewer).
    cam_pos:       vec4<f32>,
    near:          f32,
    far:           f32,
    // World-space hemisphere sample radius.
    radius:        f32,
    // Small depth bias to prevent surfaces from self-occluding.
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
    @builtin(position) pos: vec4<f32>,
}

// Full-screen triangle: no vertex buffer needed.
@vertex
fn vs_ssao(@builtin(vertex_index) vi: u32) -> VOut {
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

// Non-linear perspective depth (0..1) → linear view-space distance.
fn linearize(d: f32) -> f32 {
    return su.near * su.far / (su.far - d * (su.far - su.near));
}

// Reconstruct world-space position from a screen UV (0..1) and raw depth value.
fn world_from_depth(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    // wgpu NDC: x ∈ [-1,1] (left→right), y ∈ [-1,1] (bottom→top), depth ∈ [0,1].
    // Screen UV: (0,0) = top-left, so y must be flipped.
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world_h = su.proj_view_inv * ndc;
    return world_h.xyz / world_h.w;
}

// Simple hash for pseudo-random per-pixel rotation (reduces banding).
fn hash21(p: vec2<f32>) -> f32 {
    return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

// Rotate 2-D vector by angle (radians).
fn rot2(v: vec2<f32>, a: f32) -> vec2<f32> {
    let s = sin(a);
    let c = cos(a);
    return vec2<f32>(c * v.x - s * v.y, s * v.x + c * v.y);
}

@fragment
fn fs_ssao(@builtin(position) frag_pos: vec4<f32>) -> @location(0) vec4<f32> {
    let dims  = vec2<i32>(textureDimensions(depth_tex));
    let fdims = vec2<f32>(f32(dims.x), f32(dims.y));
    let px    = vec2<i32>(i32(frag_pos.x), i32(frag_pos.y));
    let uv    = frag_pos.xy / fdims;

    let depth0 = load_depth(px, dims);
    // Sky / background: nothing to occlude, return unoccluded white.
    if depth0 >= 1.0 { return vec4<f32>(1., 1., 1., 1.); }

    let pos0 = world_from_depth(uv, depth0);

    // ── Normal reconstruction from neighbouring depth samples ────────────────
    // Pick the neighbour that is closest in depth on each axis (avoids artefacts
    // at depth discontinuities such as silhouettes).
    let dx = vec2<i32>(1, 0);
    let dy = vec2<i32>(0, 1);

    let depth_r = load_depth(px + dx, dims);
    let depth_l = load_depth(px - dx, dims);
    let depth_u = load_depth(px + dy, dims);
    let depth_d = load_depth(px - dy, dims);

    let use_right = abs(depth_r - depth0) < abs(depth_l - depth0);
    let use_down  = abs(depth_u - depth0) < abs(depth_d - depth0);

    let h_uv    = select((frag_pos.xy + vec2(-1., 0.)) / fdims,
                         (frag_pos.xy + vec2( 1., 0.)) / fdims, use_right);
    let h_depth = select(depth_l, depth_r, use_right);
    let h_sign  = select(-1.0, 1.0, use_right);

    let v_uv    = select((frag_pos.xy + vec2(0., -1.)) / fdims,
                         (frag_pos.xy + vec2( 0., 1.)) / fdims, use_down);
    let v_depth = select(depth_d, depth_u, use_down);
    let v_sign  = select(-1.0, 1.0, use_down);

    let tang  = (world_from_depth(h_uv, h_depth) - pos0) * h_sign;
    let btng  = (world_from_depth(v_uv, v_depth) - pos0) * v_sign;
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

    // Per-pixel random rotation angle to break up the fixed kernel pattern.
    let rot_angle = hash21(frag_pos.xy) * 6.283185;

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

        // Convert NDC to screen UV (y flipped: NDC +1 = top = UV 0).
        let sample_uv = ndc_xy * vec2<f32>(0.5, -0.5) + 0.5;

        // Discard out-of-screen and behind-near-plane samples.
        if any(sample_uv < vec2<f32>(0.0)) || any(sample_uv > vec2<f32>(1.0)) { continue; }
        if sample_depth < 0.0 || sample_depth > 1.0 { continue; }

        // Fetch geometry depth at the projected location.
        let sample_px  = vec2<i32>(i32(sample_uv.x * fdims.x), i32(sample_uv.y * fdims.y));
        let geom_depth = load_depth(sample_px, dims);

        // Range check: suppress contributions from surfaces far away in depth
        // (avoids halos at depth discontinuities).
        let linear_geom   = linearize(geom_depth);
        let linear_sample = linearize(sample_depth);
        let range_check = smoothstep(0.0, 1.0, su.radius / abs(linear0 - linear_geom));

        // Occluded when geometry is closer to the camera than the sample point.
        if geom_depth < sample_depth - su.bias {
            occlusion += range_check;
        }
    }

    occlusion /= f32(num_samples);

    // ao = 1 (no occlusion) → multiply scene by 1 (no change).
    // ao = 0 (full occlusion) → multiply scene by 0 (black).
    let ao = 1.0 - clamp(occlusion * su.strength, 0.0, 1.0);
    return vec4<f32>(ao, ao, ao, 1.0);
}
