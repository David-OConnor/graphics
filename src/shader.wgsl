// Reference: https://www.w3.org/TR/WGSL

struct Camera {
    proj_view: mat4x4<f32>,
    position: vec4<f32>,
    fog_density: f32,
    fog_power: f32,
    fog_start: f32,
    fog_end: f32,
    fog_color: vec3<f32>,
    // Edge cueing: 0.0 = off, 1.0 = full strength. Darkens silhouette edges.
    edge_cueing: f32,
    near: f32,
    far: f32,
    // Depth-aware halo prepass: > 0 means inflate vertices along normals by this amount.
    halo_expansion: f32,
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

// Note: Don't us vec3 in uniforms due to alignment issues.
struct Lighting {
    ambient_color: vec4<f32>,
    ambient_intensity: f32,
    lights_len: i32,
    point_lights: array<PointLight>
}

@group(0) @binding(0)
var<uniform> camera: Camera;

@group(1) @binding(0)
// We use a storage buffer, since our lighting size is unknown by the shader;
// this is due to the dynamic-sized point light array.
var<storage, read> lighting: Lighting;

// The shadow mask pass binds its projection settings in place of the lighting buffer.
// xyz is the point light; w is the world-space height of the floor.
@group(1) @binding(1)
var<uniform> shadow_projection: vec4<f32>;

@group(2) @binding(0)
var floor_shadow_masks: texture_2d_array<f32>;

@group(2) @binding(1)
var<uniform> floor_shadow_strength: f32;


struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>, // unused
    @location(2) normal: vec3<f32>,
    @location(3) tangent: vec3<f32>,
    @location(4) bitangent: vec3<f32>,
    @location(5) color: vec4<f32>, // from Unorm8x4 -> converted to 0..1
}

// These are matrix columns; we can't pass matrices directly for vertex attributes.
struct InstanceIn {
    @location(6) model_matrix_0: vec4<f32>,
    @location(7) model_matrix_1: vec4<f32>,
    @location(8) model_matrix_2: vec4<f32>,
    @location(9) model_matrix_3: vec4<f32>,

    @location(10) normal_matrix_0: vec3<f32>,
    @location(11) normal_matrix_1: vec3<f32>,
    @location(12) normal_matrix_2: vec3<f32>,

    @location(13) color: vec4<f32>, // entity color + opacity.
    @location(14) shinyness: f32,
}

struct VertexOut {
    @builtin(position) clip_posit: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
    @location(3) shinyness: f32,
    @location(4) world_posit: vec3<f32>,

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

fn world_position(vertex_in: VertexIn, instance: InstanceIn) -> vec4<f32> {
    // The model matrix includes translation, rotation, and scale.
    let model_mat = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );

    return model_mat * vec4<f32>(vertex_in.position, 1.0);
}

fn vertex_color(vertex_in: VertexIn, instance: InstanceIn) -> vec4<f32> {
    return select(vertex_in.color, instance.color, vertex_in.color.a == 0.0);
}

@vertex
fn vs_main(
    vertex_in: VertexIn,
    instance: InstanceIn,
) -> VertexOut {
    // The normal matrix includes rotation only.
    var normal_mat = mat3x3<f32>(
        instance.normal_matrix_0,
        instance.normal_matrix_1,
        instance.normal_matrix_2,
    );

    // We use the tangent matrix, and tangent out values for normal mapping.
    // This is currently unimplemented.
    var world_normal = normalize(normal_mat * vertex_in.normal);
//    var world_tangent = normalize(normal_mat * vertex_in.tangent);
//    var world_bitangent = normalize(normal_mat * vertex_in.bitangent);

    // Construct the tangent matrix
//    var tangent_mat = transpose(mat3x3<f32>(
//        world_tangent,
//        world_bitangent,
//        world_normal,
//    ));

    var world_posit = world_position(vertex_in, instance);

    // Depth-aware halo prepass: push vertices outward along world normals.
    // Combined with front-face culling and depth-only writes, this causes background
    // objects near a foreground silhouette to fail the depth test, producing a halo ring.
    if camera.halo_expansion > 0.0 {
        world_posit = world_posit + vec4<f32>(world_normal * camera.halo_expansion, 0.0);
    }

    var result: VertexOut;

    result.clip_posit = camera.proj_view * world_posit;

//    result.tangent_position = tangent_mat * world_posit.xyz;
//    result.tangent_view_position = tangent_mat * camera.position.xyz;
//    result.tangent_light_position = tangent_matrix * light.position;
    result.normal = world_normal;

    result.color = vertex_color(vertex_in, instance);

    result.shinyness = instance.shinyness;
    result.world_posit = world_posit.xyz;

    return result;
}

struct OutlineMaskOut {
    @builtin(position) clip_posit: vec4<f32>,
    // rgb: Outline color. a: Outline thickness, in logical pixels.
    @location(0) @interpolate(flat) outline: vec4<f32>,
}

// Outline mask pass: Writes each outlined entity's silhouette into the mask texture read by
// shader_outline.wgsl. The instance color and opacity slots carry the outline color and
// thickness; see `Instance::new_outline`.
@vertex
fn vs_outline_mask(
    vertex_in: VertexIn,
    instance: InstanceIn,
) -> OutlineMaskOut {
    let model_mat = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );

    var result: OutlineMaskOut;

    result.clip_posit = camera.proj_view * model_mat * vec4<f32>(vertex_in.position, 1.0);
    result.outline = instance.color;

    return result;
}

@fragment
fn fs_outline_mask(vertex: OutlineMaskOut) -> @location(0) vec4<f32> {
    return vertex.outline;
}

struct ShadowMaskOut {
    @builtin(position) clip_posit: vec4<f32>,
    @location(0) height_fraction: f32,
    @location(1) opacity: f32,
}

@vertex
fn vs_floor_shadow(vertex_in: VertexIn, instance: InstanceIn) -> ShadowMaskOut {
    let world = world_position(vertex_in, instance).xyz;
    let light = shadow_projection.xyz;
    let floor_y = shadow_projection.w;
    let t = (world.y - floor_y) / (light.y - floor_y);
    let w = 1.0 - t;

    // Intersect the ray from the light through this vertex with the floor. Keep the
    // result homogeneous so the rasterizer can clip silhouettes that cross the light.
    let on_floor = vec4<f32>(world.x - t * light.x, w * floor_y, world.z - t * light.z, w);
    return ShadowMaskOut(camera.proj_view * on_floor, t, vertex_color(vertex_in, instance).a);
}

@fragment
fn fs_floor_shadow(vertex: ShadowMaskOut) -> @location(0) f32 {
    // A caster has to lie between the light and the receiving plane.
    if vertex.height_fraction < 0.0 || vertex.height_fraction >= 1.0 {
        discard;
    }
    // Max blending unions silhouettes without darkening overlaps between atoms and bonds.
    return saturate(vertex.opacity);
}

// Unused
//fn fxaa(uv: vec2<f32>) -> vec3<f32> {
//    let color = textureSample(scene_texture, scene_sampler, uv).rgb;
//    let color_left = textureSample(scene_texture, scene_sampler, uv + vec2<f32>(-1.0, 0.0) / resolution).rgb;
//    let color_right = textureSample(scene_texture, scene_sampler, uv + vec2<f32>(1.0, 0.0) / resolution).rgb;
//    return (color + color_left + color_right) / 3.0; // Simplified averaging
//}

struct LightContribution {
    diffuse: vec3<f32>,
    specular: vec3<f32>,
}

fn surface_normal(vertex: VertexOut, front: bool) -> vec3<f32> {
    let normal = normalize(vertex.normal);
    // Transparent meshes are drawn on both sides.
    return select(-normal, normal, front);
}

fn shade_light(
    light: PointLight,
    position: vec3<f32>,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    shinyness: f32,
) -> LightContribution {
    let light_diff = position - light.position.xyz;
    let light_dir = normalize(light_diff);
    let dist = length(light_diff);
    let dist_attenuation = 1.0 / (1.0 + 0.09 * dist + 0.032 * pow(dist, 2.0));
    var diffuse_attenuation = max(dot(normal, -light_dir), 0.0);

    if light.directional != 0u {
        let angle_diff = acos(dot(-normalize(light.direction), normalize(position - camera.position.xyz)));
        if angle_diff > light.fov / 2.0 {
            diffuse_attenuation = 0.0;
        }
    }

    let diffuse = light.diffuse_color.rgb * diffuse_attenuation * light.diffuse_intensity * dist_attenuation;
    var specular = vec3<f32>(0.0);
    if diffuse_attenuation > 0.0 {
        let half_dir = normalize(view_dir - light_dir);
        let fresnel = pow(1.0 - dot(view_dir, normal), 5.0);
        let specular_coeff = pow(max(dot(normal, half_dir), 0.0), shinyness);
        specular = fresnel * light.specular_color.rgb * specular_coeff * light.specular_intensity * dist_attenuation;
    }
    return LightContribution(diffuse, specular);
}

@fragment
fn fs_main(vertex: VertexOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let normal = surface_normal(vertex, front);
    let view_diff = camera.position.xyz - vertex.world_posit;
    let view_dir = normalize(view_diff);
    var light: LightContribution;
    for (var i = 0; i < lighting.lights_len; i++) {
        let direct = shade_light(lighting.point_lights[i], vertex.world_posit, normal, view_dir, vertex.shinyness);
        light.diffuse += direct.diffuse;
        light.specular += direct.specular;
    }
    return finish_surface(vertex, normal, view_diff, view_dir, light);
}

@fragment
fn fs_floor(vertex: VertexOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let normal = surface_normal(vertex, front);
    let view_diff = camera.position.xyz - vertex.world_posit;
    let view_dir = normalize(view_diff);
    let pixel = vec2<i32>(vertex.clip_posit.xy);
    let layers = i32(textureNumLayers(floor_shadow_masks));
    var light: LightContribution;
    for (var i = 0; i < lighting.lights_len; i++) {
        let direct = shade_light(lighting.point_lights[i], vertex.world_posit, normal, view_dir, vertex.shinyness);
        var visible = 1.0;
        if i < layers {
            let coverage = textureLoad(floor_shadow_masks, pixel, i, 0).r;
            visible -= floor_shadow_strength * coverage;
        }
        light.diffuse += direct.diffuse * visible;
        light.specular += direct.specular * visible;
    }
    return finish_surface(vertex, normal, view_diff, view_dir, light);
}

fn finish_surface(
    vertex: VertexOut,
    normal: vec3<f32>,
    view_diff: vec3<f32>,
    view_dir: vec3<f32>,
    light: LightContribution,
) -> vec4<f32> {
    // Modulated combine
    let ambient = lighting.ambient_color.rgb * lighting.ambient_intensity;
    let base   = vertex.color.rgb;       // Albedo / base colour coming from the mesh
    let litRGB = (ambient + light.diffuse) * base   // Lambert terms tinted
               + light.specular;                        // Specular left un-tinted

    var result = vec4<f32>(litRGB, vertex.color.a);

    // Apply the fog; attentuate pixels that meet the fog criteria.
    if (camera.fog_end > camera.fog_start) {
        let view_dist = length(view_diff);
        let w = clamp(fog_weight_band(view_dist), 0.0, 1.0);
        let fogged = mix(result.rgb, camera.fog_color, w);
        result = vec4<f32>(fogged, result.a);
    }

    // Edge cueing: darken silhouette edges where the surface normal is nearly
    // perpendicular to the view direction (Tarini et al. 2006).
    if camera.edge_cueing > 0.0 {
        let n_dot_v = abs(dot(normal, view_dir));
        let edge_factor = pow(1.0 - n_dot_v, 3.0);
        let darkening = 1.0 - camera.edge_cueing * 0.85 * edge_factor;
        result = vec4<f32>(result.rgb * darkening, result.a);
    }

    return result;
}
