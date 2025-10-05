// Reference: https://www.w3.org/TR/WGSL

struct Camera {
    proj_view: mat4x4<f32>,
    position: vec4<f32>,
    fog_density: f32,
    fog_power: f32,
    fog_start: f32,
    fog_end: f32,
    fog_color: vec3<f32>,
}

struct PointLight {
    position: vec4<f32>,
    diffuse_color: vec4<f32>,
    specular_color: vec4<f32>,
    diffuse_intensity: f32,
    specular_intensity: f32,
    directional: u32, // Boolean
    direction: vec3<f32>,
    fov: f32,
}

// Note: Don't use vec3 in uniforms due to alignment issues.
struct Lighting {
    ambient_color: vec4<f32>,
    ambient_intensity: f32,
    lights_len: i32,
    point_lights: array<PointLight>
}

@group(0) @binding(0)
var<uniform> camera: Camera;

// todo: Experimenting with fadeing objects beyond the front.
// todo front-most depth texture (R32Float copy of your depth) + sampler
@group(0) @binding(1)
var depthTexFront: texture_depth_2d;
@group(0) @binding(2)
var depthSmp: sampler;

@group(1) @binding(0)
// We use a storage buffer, since our lighting size is unknown by the shader;
// this is due to the dynamic-sized point light array.
var<storage, read> lighting: Lighting;


struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>, // unused
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) bitangent: vec3<f32>,
}

// These are matrix columns; we can't pass matrices directly for vertex attributes.
struct InstanceIn {
    @location(5) model_matrix_0: vec4<f32>,
    @location(6) model_matrix_1: vec4<f32>,
    @location(7) model_matrix_2: vec4<f32>,
    @location(8) model_matrix_3: vec4<f32>,
    @location(9) normal_matrix_0: vec3<f32>,
    @location(10) normal_matrix_1: vec3<f32>,
    @location(11) normal_matrix_2: vec3<f32>,
    @location(12) color: vec4<f32>, // Len 4; includes alpha.
    @location(13) shinyness: f32,
}

struct VertexOut {
    @builtin(position) clip_posit: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) shinyness: f32,
    @location(4) world_posit: vec3<f32>, // todo: Experimenting
    @location(5) screen_uv: vec2<f32>,   // todo: New, for fade in back
//        @location(1) tangent_position: vec3<f32>,
//        @location(2) tangent_light_position: vec3<f32>,
//        @location(3) tangent_view_position: vec3<f32>,
}

fn saturate(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }

fn fog_weight_band(distance_to_cam: f32) -> f32 {
    // Map distance into 0..1 between start and end
    let span = max(1e-4, camera.fog_end - camera.fog_start);
    let t = saturate((distance_to_cam - camera.fog_start) / span);

    // Make far fade much harsher with a power curve
    let shaped = pow(t, camera.fog_power);
    return 1.0 - exp(-camera.fog_density * shaped);
}

@vertex
fn vs_main(
    vertex_in: VertexIn,
    instance: InstanceIn,
) -> VertexOut {
    // The model matrix includes translation, rotation, and scale.
    var model_mat = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );

    // The normal matrix includes rotation only.
    var normal_mat = mat3x3<f32>(
        instance.normal_matrix_0,
        instance.normal_matrix_1,
        instance.normal_matrix_2,
    );

    // We use the tangent matrix, and tangent out values for normal mapping.
    // This is currently unimplemented.
    var world_normal = normalize(normal_mat * vertex_in.normal);
    var world_tangent = normalize(normal_mat * vertex_in.tangent);
    var world_bitangent = normalize(normal_mat * vertex_in.bitangent);

// Construct the tangent matrix
    var tangent_mat = transpose(mat3x3<f32>(
        world_tangent,
        world_bitangent,
        world_normal,
    ));

    // Pad the model position with 1., for use with the 4x4 transform mats.
    var world_posit = model_mat * vec4<f32>(vertex_in.position, 1.0);

    var result: VertexOut;

    result.clip_posit = camera.proj_view * world_posit;

    // todo: Part of our fade-background-objects attempt
    // Screen UV from clip-space
    let ndc = result.clip_posit.xyz / result.clip_posit.w;
    result.screen_uv = ndc.xy * 0.5 + 0.5;

//    result.tangent_position = tangent_mat * world_posit.xyz;
//    result.tangent_view_position = tangent_mat * camera.position.xyz;
//    result.tangent_light_position = tangent_matrix * light.position;
    result.normal = world_normal;

    result.color = instance.color;
    result.shinyness = instance.shinyness;
    result.world_posit = world_posit.xyz;

    return result;
}

// Unused
//fn fxaa(uv: vec2<f32>) -> vec3<f32> {
//    let color = textureSample(scene_texture, scene_sampler, uv).rgb;
//    let color_left = textureSample(scene_texture, scene_sampler, uv + vec2<f32>(-1.0, 0.0) / resolution).rgb;
//    let color_right = textureSample(scene_texture, scene_sampler, uv + vec2<f32>(1.0, 0.0) / resolution).rgb;
//    return (color + color_left + color_right) / 3.0; // Simplified averaging
//}

/// Fragment shader, which is mostly lighting calculations.
@fragment
fn fs_main(
    vertex: VertexOut,
    @builtin(front_facing) front: bool,
) -> @location(0) vec4<f32> {
    // Always renormalise after interpolation
    var normal = normalize(vertex.normal);

    // If it’s a transparent surface's back face, flip the normal so it still points *out* of the surface
    if (!front) {
        normal = -normal;
    }

    // Ambient lighting
    // todo: Don't multiply ambient for every fragment; do it on the CPU.
    var ambient = lighting.ambient_color * lighting.ambient_intensity;

    var view_diff = camera.position.xyz - vertex.world_posit.xyz;
    var view_dir = normalize(view_diff);

    // todo: Emmissive term?

//    let tangent_normal = object_normal.xyz * 2.0 - 1.0;

    // Initialize diffuse and specular components
    // These values include color and intensity
    var diffuse = vec4<f32>(0., 0., 0., 0.);
    var specular = vec4<f32>(0., 0., 0., 0.);

    for (var i=0; i < lighting.lights_len; i++) {
        var light = lighting.point_lights[i];

        // Direction from light to the vertex; we use this to calculate attentiation,
        // and diffuse-lighting cosine loss.

        var light_to_vert_diff =  vertex.world_posit.xyz - light.position.xyz;

        var light_to_vert_dir = normalize(light_to_vert_diff);

        let k1 = 0.09; // Linear attenuation term
        let k2 = 0.032; // Quadratic attenuation term
        var dist_attenuation = 1.0 / (1.0 + k1 * length(light_to_vert_diff) + k2 * pow(length(light_to_vert_diff), 2.0));

        // Diffuse lighting. This is essentially cosine los.
        var diffuse_attenuation = max(dot(normal, -light_to_vert_dir), 0.);

        // For directional lights, don't attenuate further if the vertex is inside the light's
        // FOV. If outside, gradually attentuate to 0.
        if light.directional != 0u {
            let light_dir = normalize(light.direction); // Ideally handled upstream.
            // todo: This likely not handle opposite direction correctly.(?)
            // e.g. the vector from the camera to the fragment.
            let angle_diff = acos(dot(-light_dir, normalize(vertex.world_posit.xyz - camera.position.xyz)));
            if (angle_diff > light.fov/2.) {
                diffuse_attenuation = 0.0;
            }
        }

        diffuse += light.diffuse_color * diffuse_attenuation * light.diffuse_intensity * dist_attenuation;

        // Specular lighting.
        var specular_this_light = vec4<f32>(0., 0., 0., 0.);

        if (diffuse_attenuation > 0.0) {
            var half_dir = normalize(view_dir - light_to_vert_dir);

            // Fresnel Effect: Adjust specular based on view angle
            var fresnel = pow(1.0 - dot(view_dir, normal), 5.0);
            var specular_coeff = pow(max(dot(normal, half_dir), 0.), vertex.shinyness);
            specular += fresnel * light.specular_color * specular_coeff * light.specular_intensity * dist_attenuation;
        }
    }

    // Modulated combine
    let base   = vertex.color.rgb;       // Albedo / base colour coming from the mesh
    let litRGB = (ambient.rgb + diffuse.rgb) * base   // Lambert terms tinted
               + specular.rgb;                        // Specular left un-tinted

    var result = vec4<f32>(litRGB, vertex.color.a);


    // todo: Part of our fading objects behind the front.
    // Depth-proximity fade against front-most depth (all in 0..1 depth)
    {
        // Current fragment depth in 0..1 from clip z/w
        let z01_frag = (vertex.clip_posit.z / vertex.clip_posit.w) * 0.5 + 0.5;
        let z01_front = textureSample(depthTexFront, depthSmp, vertex.screen_uv);

        // Start/End thresholds in depth units (0..1). Tune as desired.
        let t0 = 0.002;  // start fading when ~0.2% behind front
        let t1 = 0.030;  // fully faded by ~3% behind front

        let delta = max(0.0, z01_frag - z01_front);
        let fade  = clamp((delta - t0) / max(1e-6, (t1 - t0)), 0.0, 1.0);

        let a = (1.0 - fade);
        result = vec4<f32>(result.rgb * a, result.a * a);
    }

    // Apply the fog; attentuate pixels that meet the fog criteria.
    if (camera.fog_end > camera.fog_start) {
        let view_dist = length(view_diff);
        let w = clamp(fog_weight_band(view_dist), 0.0, 1.0);
        let fogged = mix(result.rgb, camera.fog_color, w);
        result = vec4<f32>(fogged, result.a);
    }

    result = vec4<f32>(result.rgb, result.a);

    return result;
}