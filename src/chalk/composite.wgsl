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

  // Built-up floor layer, lit like resting grains in render.wgsl.
  floorBase: f32,      // floor height at depth 0; < -50 when the floor is off
  floorDepth: f32,
  depositSize: vec2u,
  depositScale: f32,   // deposit coverage -> optical density per output pixel
  depthShrink: f32,
  depthFade: f32,
  ambientLight: f32,
  lightIntensity: f32,
  floorShade: f32,
  floorTexture: f32,
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> comp: Composite;
@group(0) @binding(3) var smoke: texture_2d<f32>; // low-res smoke layer, upsampled by the filter
@group(0) @binding(4) var<storage, read> deposit: array<u32>; // written by sim.wgsl

const DEPOSIT_UNIT = 65536.0; // keep in sync with sim.wgsl

fn depositCell(c: vec2i) -> vec2f {
  let size = vec2i(comp.depositSize);
  let q = clamp(c, vec2i(0), size - 1);
  let i = u32(q.y * size.x + q.x) * 2u;
  return vec2f(f32(deposit[i]), f32(deposit[i + 1u])) / DEPOSIT_UNIT;
}

// (density, lit density) of the floor layer at this point, bilinearly filtered.
fn floorLayer(world: vec2f) -> vec2f {
  let v = (world.y - comp.floorBase) / comp.floorDepth;
  if (comp.floorBase < -50.0 || v < 0.0 || v > 1.0) { return vec2f(0.0); }
  let u = world.x / comp.aspect * 0.5 + 0.5;
  let g = vec2f(u, v) * vec2f(comp.depositSize) - 0.5;
  let i = vec2i(floor(g));
  let f = fract(g);
  let d = mix(
    mix(depositCell(i), depositCell(i + vec2i(1, 0)), f.x),
    mix(depositCell(i + vec2i(0, 1)), depositCell(i + vec2i(1, 1)), f.x),
    f.y,
  );
  // Same depth cues as the grains that made it: far dust is smaller and fainter.
  let shrink = 1.0 - comp.depthShrink * v;
  return d * comp.depositScale * shrink * shrink * mix(1.0, 1.0 - comp.depthFade, v);
}

fn hash21(p: vec2f) -> f32 {
  var q = fract(p * vec2f(123.34, 456.21));
  q += dot(q, q + 45.32);
  return fract(q.x * q.y);
}

@fragment
fn fs_main(@location(0) uv: vec2f, @builtin(position) frag: vec4f) -> @location(0) vec4f {
  var acc = textureSampleLevel(src, samp, uv, 0.0) + textureSampleLevel(smoke, samp, uv, 0.0);
  let world = vec2f((uv.x * 2.0 - 1.0) * comp.aspect, 1.0 - uv.y * 2.0);

  // Faint airborne haze lit by the backlight, plus a low layer of light raking across the floor.
  let toLight = world - comp.lightPos;
  let lit = exp(-dot(toLight, toLight) / (comp.lightRadius * comp.lightRadius));

  // Chalk that has settled for good, broken up by a static powdery grain.
  let speck = mix(1.0, hash21(floor(frag.xy) * 0.731) * 2.0, comp.floorTexture);
  let layer = floorLayer(world) * speck;
  let floorLight = (comp.ambientLight + comp.lightIntensity * lit) * comp.floorShade;
  acc += vec4f(vec3f(layer.y * floorLight), layer.x);
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
