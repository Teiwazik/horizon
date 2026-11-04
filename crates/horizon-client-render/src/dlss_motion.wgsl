// DLSS's motion vectors (`gfx/dlss.rs`), at the world's render size: for each pixel, where its
// point was last frame minus where it is now, in render pixels, without this frame's jitter.
// The camera's motion comes from the depth buffer; a mob, an item or a ridden boat moves by
// itself (frame generation's motion: xy the UV offset to last frame, z 1 where set).
@group(0) @binding(0) var depth: texture_depth_2d;
@group(0) @binding(1) var own_motion: texture_2d<f32>;
struct DlssMotion {
    // This frame's unjittered clip space to the previous frame's.
    reproject: mat4x4<f32>,
    // xy: this frame's jitter in clip space.
    jitter: vec4<f32>,
};
@group(0) @binding(2) var<uniform> motion: DlssMotion;

@vertex fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4(p * 2.0 - 1.0, 0.0, 1.0);
}

@fragment fn fs_dlss_motion(@builtin(position) at: vec4<f32>) -> @location(0) vec4<f32> {
    let size = vec2<f32>(textureDimensions(depth));
    let p = vec2<i32>(at.xy);
    let own_size = vec2<i32>(textureDimensions(own_motion));
    if own_size.x > 1 {
        let own = textureLoad(own_motion, clamp(p, vec2(0), own_size - 1), 0);
        if own.z > 0.5 {
            return vec4(own.xy * size, 0.0, 0.0);
        }
    }
    let uv = at.xy / size;
    // The pixel's point, without the jitter that moved it.
    let ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0) - motion.jitter.xy;
    let now = vec2(0.5, -0.5) * ndc + 0.5;
    let previous = motion.reproject * vec4(ndc, textureLoad(depth, p, 0), 1.0);
    let then = vec2(0.5, -0.5) * previous.xy / previous.w + 0.5;
    return vec4((then - now) * size, 0.0, 0.0);
}
