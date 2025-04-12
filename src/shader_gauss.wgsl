// https://chatgpt.com/c/67f9bde5-e700-8007-8211-e42e8190048a

struct Camera {
    proj_view : mat4x4<f32>,
    position  : vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera : Camera;

// Per‑vertex quad (−1..1, −1..1)
struct QuadVertex {
    @location(0) pos : vec2<f32>,
}

struct Gaussian {
    center    : vec3<f32>,
    amplitude : f32,
    width     : f32,
};

@vertex
fn gaussian_vs(v : QuadVertex,
               @location(1) instance_center   : vec3<f32>,
               @location(2) instance_amp      : f32,
               @location(3) instance_width    : f32) 
    ->  @builtin(position) vec4<f32> {

    // Camera basis vectors in world space
    let view_right = vec3<f32>(camera.proj_view[0].xyz); // first column of view matrix⁻¹
    let view_up    = vec3<f32>(camera.proj_view[1].xyz); // second column

    // Size: pick 3σ as visual radius
    let radius = 3.0 * instance_width;

    // World‑space position of this corner
    let world_pos =
          instance_center
        + v.pos.x * radius * view_right
        + v.pos.y * radius * view_up;

    return camera.proj_view * vec4<f32>(world_pos, 1.0);
}

@fragment
fn gaussian_fs(@location(0) frag_pos : vec2<f32>,
               @location(1) center    : vec3<f32>,
               @location(2) amp       : f32,
               @location(3) width     : f32) -> @location(0) vec4<f32> {

    // Reconstruct world space of this fragment (passed via varying in real code)
    let world = /* interpolate world_pos from VS */;

    let r2 = distanceSquared(world, center);
    let density = amp * exp( -r2 / (2.0 * width * width) );

    // Map density to colour – here simple white fog
    let colour = vec3<f32>(1.0, 1.0, 1.0);

    // Premultiplied‑alpha output
    return vec4<f32>(colour * density, density);
}
