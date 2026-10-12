// Tone-maps accumulated dust onto a transparent canvas (premultiplied alpha),
// so the HTML behind and in front of the canvas stays visible.

struct Composite {
  tint: vec3f,
  exposure: f32,
  contrast: f32,
  grain: f32,
  time: f32,
  aspect: f32,

  lightPos: vec2f,
  lightRadius: f32,
  atmosphere: f32,
  floorY: f32,
  floorGlow: f32,
  floorGlowHeight: f32,
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> comp: Composite;
@group(0) @binding(3) var smoke: texture_2d<f32>; // low-res smoke layer, upsampled by the filter

fn hash21(p: vec2f) -> f32 {
  var q = fract(p * vec2f(123.34, 456.21));
  q += dot(q, q + 45.32);
  return fract(q.x * q.y);
}

@fragment
fn fs_main(@location(0) uv: vec2f, @builtin(position) frag: vec4f) -> @location(0) vec4f {
  let acc = textureSampleLevel(src, samp, uv, 0.0) + textureSampleLevel(smoke, samp, uv, 0.0);
  let world = vec2f((uv.x * 2.0 - 1.0) * comp.aspect, 1.0 - uv.y * 2.0);

  // Faint airborne haze lit by the backlight, plus a low layer of light raking across the floor.
  let toLight = world - comp.lightPos;
  let lit = exp(-dot(toLight, toLight) / (comp.lightRadius * comp.lightRadius));
  let fy = (world.y - comp.floorY) / comp.floorGlowHeight;
  let floorBand = exp(-fy * fy) * exp(-toLight.x * toLight.x / (comp.lightRadius * comp.lightRadius * 2.0));
  let ambient = lit * comp.atmosphere + floorBand * comp.floorGlow;

  // Beer-Lambert style: thin dust is translucent, thick dust and lit dust saturate to white.
  let radiance = max(acc.rgb, vec3f(0.0)) + ambient;
  var color = 1.0 - exp(-radiance * comp.exposure);
  color = pow(color, vec3f(comp.contrast));

  let g = hash21(floor(frag.xy) + fract(comp.time) * 97.0) - 0.5;
  color = clamp(color * (1.0 + g * comp.grain), vec3f(0.0), vec3f(1.0)) * comp.tint;

  // Opaque where dust is dense; lit-but-thin areas stay additive over the page.
  let cover = 1.0 - exp(-max(acc.a, 0.0) * comp.exposure);
  let alpha = clamp(max(cover, max(color.r, max(color.g, color.b))), 0.0, 1.0);
  return vec4f(color, alpha);
}
