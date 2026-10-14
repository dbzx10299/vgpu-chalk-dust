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
  floorBase: f32,      // screen height of the floor's front edge; < -50 when the floor is off
  floorDepth: f32,     // screen height of the floor band, front edge to depth 1
  horizon: f32,        // camera, as in sim.wgsl
  perspective: f32,
  depositSize: vec2u,
  depositScale: f32,   // deposit coverage -> optical density per output pixel
  depthFade: f32,
  ambientLight: f32,
  lightIntensity: f32,
  floorShade: f32,
  floorTexture: f32,   // share of the floor dust drawn as specks rather than a smooth film
  pxPerUnit: f32,      // output pixels per world unit
  speckPx: f32,        // speck grid cell on the floor's front edge, output pixels
  floorFlatten: f32,
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

// On-screen scale of the floor at this screen height (1 at the front edge), or 0 off the floor.
// Inverts the projection in sim.wgsl for points on the floor plane.
fn floorScale(y: f32) -> f32 {
  if (comp.floorBase < -50.0 || y < comp.floorBase || y > comp.floorBase + comp.floorDepth) { return 0.0; }
  return (comp.horizon - y) / (comp.horizon - comp.floorBase);
}

// (density, lit density) of the floor layer at this screen x and depth, bilinearly filtered.
fn floorLayer(x: f32, depth: f32) -> vec2f {
  let u = x / comp.aspect * 0.5 + 0.5;
  let g = vec2f(u, depth) * vec2f(comp.depositSize) - 0.5;
  let i = vec2i(floor(g));
  let f = fract(g);
  let d = mix(
    mix(depositCell(i), depositCell(i + vec2i(1, 0)), f.x),
    mix(depositCell(i + vec2i(0, 1)), depositCell(i + vec2i(1, 1)), f.x),
    f.y,
  );
  // Far grains and the floor cells they land in shrink alike, so only the depth haze is left to apply.
  return d * comp.depositScale * mix(1.0, 1.0 - comp.depthFade, depth);
}

fn hash21(p: vec2f) -> f32 {
  var q = fract(p * vec2f(123.34, 456.21));
  q += dot(q, q + 45.32);
  return fract(q.x * q.y);
}

fn vnoise2(p: vec2f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(hash21(i), hash21(i + vec2f(1, 0)), u.x),
    mix(hash21(i + vec2f(0, 1)), hash21(i + vec2f(1, 1)), u.x),
    u.y,
  );
}

const SPECK_DENSITY = 1.5; // optical density of a single speck
const SPECK_AREA = 0.28;   // mean share of its cell a speck covers
const FILM_MAX = 0.3;      // the smooth film never thickens past a faint haze

// Settled chalk as specks: a grid laid on the floor plane with one speck per cell, each switched on once
// the dust around it is thick enough (`fill`). Coverage of this pixel, 0..1.
fn specks(p: vec2f, cell: f32, fill: f32, scale: f32, salt: f32) -> f32 {
  let g = p / cell;
  let id = floor(g) + salt;
  let r = mix(0.16, 0.42, hash21(id + 0.37));
  let c = r + (1.0 - 2.0 * r) * vec2f(hash21(id + 1.91), hash21(id + 5.13));
  let px = cell * scale * comp.floorFlatten; // on-screen size of a cell along its short, foreshortened axis
  let shape = 1.0 - smoothstep(r - 0.5 / px, r + 0.5 / px, length(fract(g) - c));
  let on = step(hash21(id), fill) * shape;
  // Far specks shrink below a pixel and would shimmer: fade them to their average coverage instead.
  return mix(fill * SPECK_AREA, on, smoothstep(0.6, 1.5, 2.0 * r * px));
}

@fragment
fn fs_main(@location(0) uv: vec2f, @builtin(position) frag: vec4f) -> @location(0) vec4f {
  var acc = textureSampleLevel(src, samp, uv, 0.0) + textureSampleLevel(smoke, samp, uv, 0.0);
  let world = vec2f((uv.x * 2.0 - 1.0) * comp.aspect, 1.0 - uv.y * 2.0);

  // Faint airborne haze lit by the backlight, plus a low layer of light raking across the floor.
  let toLight = world - comp.lightPos;
  let lit = exp(-dot(toLight, toLight) / (comp.lightRadius * comp.lightRadius));

  // Chalk that has settled for good: a thin dusting of specks over a faint film, never an opaque coat.
  let s = floorScale(world.y);
  if (s > 0.0) {
    let depth = (1.0 / s - 1.0) / max(comp.perspective, 1e-3);
    let layer = floorLayer(world.x, depth);
    // Floor-plane coordinates, in output px at the front edge, so the pattern stays put on the floor
    // as the page scrolls and its specks shrink and foreshorten with distance.
    let fp = vec2f(world.x / s, -(comp.horizon - comp.floorBase) * log(s) / comp.floorFlatten) * comp.pxPerUnit;
    // Dust settles unevenly: patchy where it's thin, filling in as it builds up.
    let patchy = mix(0.4, 1.6, vnoise2(fp / (comp.speckPx * 14.0)));
    let fill = 1.0 - exp(-layer.x * patchy / (SPECK_AREA * SPECK_DENSITY));
    let cover = specks(fp, comp.speckPx, fill, s, 0.0) + 0.6 * specks(fp, comp.speckPx * 2.3, fill * fill, s, 71.0);
    let film = FILM_MAX * (1.0 - exp(-layer.x / FILM_MAX));
    let dust = mix(film, cover * SPECK_DENSITY, comp.floorTexture);
    let bright = layer.y / max(layer.x, 1e-6);
    let floorLight = (comp.ambientLight + comp.lightIntensity * lit) * comp.floorShade;
    acc += vec4f(vec3f(dust * bright * floorLight), dust);
  }
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
