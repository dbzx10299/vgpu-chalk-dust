// Instanced soft quads. Accumulates dust additively into an HDR target:
// rgb = lit radiance, a = optical density. The composite pass tone-maps it.
// Drawn twice: grains into a sharp target, smoke into a low-res one (`view.layer` picks which).

struct Particle {
  pos: vec2f,
  vel: vec2f,
  life: f32,
  maxLife: f32,
  size: f32,
  kind: f32,
  depth: f32,
  seed: f32,
  burst: u32,
  bright: f32,
}

struct View {
  resolution: vec2f,
  aspect: f32,
  time: f32,
  sizeScale: f32,
  grainOpacity: f32,
  cloudOpacity: f32,
  cloudGrowth: f32,
  depthFade: f32,
  depthShrink: f32,
  wispiness: f32,
  speckle: f32,
  speckleScale: f32,

  lightPos: vec2f,
  lightRadius: f32,
  lightIntensity: f32,
  ambientLight: f32,
  hazeScatter: f32,

  shutter: f32,
  focusDepth: f32,
  dofAmount: f32,
  layer: f32, // 0 = grains, 1 = smoke
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<uniform> view: View;
@group(0) @binding(2) var noiseTex: texture_2d<f32>;
@group(0) @binding(3) var noiseSamp: sampler;

struct VsOut {
  @builtin(position) position: vec4f,
  @location(0) uv: vec2f,
  @location(1) alpha: f32,
  @location(2) radiance: f32,
  @location(3) @interpolate(flat) kind: f32,
  @location(4) @interpolate(flat) seed: f32,
  @location(5) @interpolate(flat) soft: f32,
  @location(6) @interpolate(flat) chunk: vec2f, // x = lumpiness, y = tumble angle
}

@vertex
fn vs_main(@builtin(vertex_index) v: u32, @builtin(instance_index) i: u32) -> VsOut {
  var corners = array<vec2f, 6>(
    vec2f(-1, -1), vec2f(1, -1), vec2f(1, 1),
    vec2f(-1, -1), vec2f(1, 1), vec2f(-1, 1),
  );
  let p = particles[i];
  var out: VsOut;

  let isCloud = p.kind > 0.5;
  if (p.life >= p.maxLife || isCloud != (view.layer > 0.5)) {
    out.position = vec4f(2.0, 2.0, 2.0, 1.0); // outside clip space: culled
    return out;
  }

  let t = clamp(p.life / p.maxLife, 0.0, 1.0);
  let fadeIn = smoothstep(0.0, select(0.03, 0.12, isCloud), t);
  let fadeOut = 1.0 - smoothstep(select(0.55, 0.35, isCloud), 1.0, t);

  let growth = select(1.0, 1.0 + view.cloudGrowth * t, isCloud);
  var sizePx = p.size * view.sizeScale * growth * mix(1.0, 1.0 - view.depthShrink, p.depth);

  var alpha = fadeIn * fadeOut * mix(1.0, 1.0 - view.depthFade, p.depth);

  // Backlight: dust glows near the light and falls off into darkness. Haze scatters more than grains.
  let toLight = p.pos - view.lightPos;
  let lit = exp(-dot(toLight, toLight) / (view.lightRadius * view.lightRadius));
  var light = view.ambientLight + view.lightIntensity * lit;

  let c = corners[v];
  var offset: vec2f;
  var soft = 0.0;
  var chunk = vec2f(0.0);

  if (isCloud) {
    alpha *= view.cloudOpacity / (growth * growth); // expanding puffs thin out
    light *= view.hazeScatter;
    // Rotate cloud quads per particle so noise patterns never line up.
    let a = p.seed * 6.2831853 + p.life * (p.seed - 0.5) * 0.4;
    let rot = mat2x2f(cos(a), sin(a), -sin(a), cos(a));
    offset = rot * c * sizePx;
  } else {
    alpha *= view.grainOpacity;
    // Larger crumbs get an irregular, slowly tumbling silhouette instead of a perfect disc.
    chunk = vec2f(smoothstep(3.0, 6.0, p.size), p.seed * 6.2831853 + p.life * (p.seed - 0.5) * 4.0);

    // Depth of field: grains off the focal plane spread into soft discs with the same total energy.
    let blur = abs(p.depth - view.focusDepth) * view.dofAmount * view.sizeScale;
    let focused = sizePx;
    sizePx = sqrt(sizePx * sizePx + blur * blur);
    alpha *= (focused * focused) / (sizePx * sizePx);
    soft = clamp(blur / sizePx, 0.0, 1.0);

    // Sub-pixel grains: keep a stable 1.5px footprint and fade by coverage instead of shimmering.
    let minPx = 1.5;
    if (sizePx < minPx) {
      alpha *= (sizePx * sizePx) / (minPx * minPx);
      sizePx = minPx;
    }

    // Motion blur: stretch along the screen velocity, thinning so a streak carries the same light.
    let velPx = p.vel * view.resolution.y * 0.5;
    let speed = length(velPx);
    let dir = select(vec2f(1.0, 0.0), velPx / max(speed, 1e-4), speed > 1e-3);
    let streak = speed * view.shutter;
    alpha *= sizePx / (sizePx + streak);
    offset = dir * c.x * (sizePx + streak) + vec2f(-dir.y, dir.x) * c.y * sizePx;
  }

  let center = vec2f(p.pos.x / view.aspect, p.pos.y);
  out.position = vec4f(center + offset / view.resolution, 0.0, 1.0);
  out.uv = c;
  out.alpha = alpha;
  out.radiance = p.bright * light;
  out.kind = p.kind;
  out.seed = p.seed;
  out.soft = soft;
  out.chunk = chunk;
  return out;
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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4f {
  let r2 = dot(in.uv, in.uv);
  if (r2 > 1.0) { discard; }

  var d: f32;
  if (in.kind > 0.5) {
    // Soft gaussian body broken up into wisps, plus fine speckle that travels with the puff
    // so it reads as powder rather than smooth smoke.
    let body = exp(-r2 * 3.5) * (1.0 - r2);
    let off = vec2f(in.seed * 37.0, in.seed * 91.0);
    let n = textureSampleLevel(noiseTex, noiseSamp, in.uv * 0.425 + off + view.time * 0.004, 0.0).r;
    let wisps = mix(1.0, smoothstep(0.3, 0.85, n) * 1.8, view.wispiness);
    let s = textureSampleLevel(noiseTex, noiseSamp, in.uv * view.speckleScale * 0.031 + off * 1.7, 0.0).g;
    let speck = mix(1.0, s * s * 2.6, view.speckle);
    d = body * wisps * speck;
  } else {
    // In-focus grains are crisp points; defocused ones become flat bokeh discs.
    let edge = mix(0.15, 0.5, in.soft);
    var r = sqrt(r2);
    if (in.chunk.x > 0.0) {
      let ca = cos(in.chunk.y);
      let sa = sin(in.chunk.y);
      let ruv = vec2f(ca * in.uv.x - sa * in.uv.y, sa * in.uv.x + ca * in.uv.y);
      let n = vnoise2(ruv * 2.2 + in.seed * 51.0);
      r = mix(r, r / (0.55 + 0.6 * n), in.chunk.x);
    }
    d = 1.0 - smoothstep(edge, 1.0, r);
  }

  let a = d * in.alpha;
  return vec4f(vec3f(a * in.radiance), a);
}
