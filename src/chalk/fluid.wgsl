// The air the chalk floats in: Stable Fluids (Stam 1999) on a grid of square cells covering the viewport.
// Velocity is in world units/s; x in [-aspect, aspect], y in [-1, 1]. Every frame runs, one dispatch each:
//   advect (+ cursor and clap forces) -> vorticity -> confine -> divergence -> jacobi x N -> project
// Cells below the floor are solid; the other edges are open, so air can leave the screen.

struct Fluid {
  size: vec2u,
  cell: f32,          // cell width in world units
  dt: f32,
  aspect: f32,
  floorY: f32,
  damping: f32,       // momentum the air loses per second
  vorticity: f32,     // confinement strength: restores the swirls numerical diffusion smears out
  pressureDecay: f32, // warm start for the pressure solve, fraction kept from last frame

  mouse: vec2f,
  mousePrev: vec2f,
  mouseVel: vec2f,
  mouseRadius: f32,
  mouseStrength: f32,
  mouseDrag: f32,

  burstPos: vec2f,
  burstAxis: vec2f,
  burstAir: f32,      // jet speed; non-zero only on the frame of a clap
  burstAirRadius: f32,
}

@group(0) @binding(0) var<uniform> fluid: Fluid;
@group(0) @binding(1) var<storage, read> velSrc: array<vec2f>;
@group(0) @binding(2) var<storage, read_write> velDst: array<vec2f>;
@group(0) @binding(3) var<storage, read_write> curlField: array<f32>;
@group(0) @binding(4) var<storage, read_write> divField: array<f32>;
@group(0) @binding(5) var<storage, read> pressureSrc: array<f32>;
@group(0) @binding(6) var<storage, read_write> pressureDst: array<f32>;

fn idx(c: vec2i) -> u32 {
  return u32(c.y) * fluid.size.x + u32(c.x);
}

fn inside(c: vec2i) -> bool {
  return all(c >= vec2i(0)) && all(c < vec2i(fluid.size));
}

fn clampCell(c: vec2i) -> vec2i {
  return clamp(c, vec2i(0), vec2i(fluid.size) - 1);
}

fn center(c: vec2i) -> vec2f {
  return vec2f(-fluid.aspect, -1.0) + (vec2f(c) + 0.5) * fluid.cell;
}

fn solid(c: vec2i) -> bool {
  return center(c).y < fluid.floorY;
}

// Neighbour velocity: the floor is at rest, and past an open edge the air keeps flowing as it does here.
fn velAt(c: vec2i, here: vec2f) -> vec2f {
  if (solid(c)) { return vec2f(0.0); }
  if (!inside(c)) { return here; }
  return velSrc[idx(c)];
}

// Neighbour pressure: no flow through the floor, and the room beyond an open edge is at rest pressure.
fn pressureAt(c: vec2i, here: f32) -> f32 {
  if (solid(c)) { return here; }
  if (!inside(c)) { return 0.0; }
  return pressureSrc[idx(c)];
}

fn sampleVel(p: vec2f) -> vec2f {
  let g = clamp((p - vec2f(-fluid.aspect, -1.0)) / fluid.cell - 0.5, vec2f(0.0), vec2f(fluid.size) - 1.0);
  let i0 = vec2i(floor(g));
  let i1 = clampCell(i0 + 1);
  let f = g - vec2f(i0);
  let a = velSrc[idx(i0)];
  let b = velSrc[idx(vec2i(i1.x, i0.y))];
  let c = velSrc[idx(vec2i(i0.x, i1.y))];
  let d = velSrc[idx(i1)];
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

@compute @workgroup_size(8, 8)
fn advect(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= fluid.size)) { return; }
  let c = vec2i(id.xy);
  let i = idx(c);
  if (solid(c)) {
    velDst[i] = vec2f(0.0);
    return;
  }
  let x = center(c);
  let dt = fluid.dt;

  // Semi-Lagrangian: the air arriving here is whatever sat upstream a step ago (midpoint backtrace).
  let mid = sampleVel(x - 0.5 * dt * velSrc[i]);
  var v = sampleVel(x - dt * mid) * exp(-fluid.damping * dt);

  // The cursor drags the air it sweeps through this frame toward its own velocity. The pressure
  // projection turns that blob of momentum into a pushed front and a pair of trailing vortices.
  let seg = fluid.mouse - fluid.mousePrev;
  let t = clamp(dot(x - fluid.mousePrev, seg) / max(dot(seg, seg), 1e-8), 0.0, 1.0);
  let d = x - (fluid.mousePrev + seg * t);
  let falloff = exp(-dot(d, d) / (fluid.mouseRadius * fluid.mouseRadius));
  let moving = smoothstep(0.0, 0.2, length(fluid.mouseVel)); // a resting cursor doesn't brake the air
  let pull = falloff * moving * (1.0 - exp(-fluid.mouseDrag * dt));
  v += (fluid.mouseVel * fluid.mouseStrength - v) * pull;

  // A clap squeezes air out between the palms: two opposite jets along the clap axis.
  if (fluid.burstAir > 0.0) {
    let b = x - fluid.burstPos;
    let r = fluid.burstAirRadius;
    let side = clamp(dot(b, fluid.burstAxis) / (0.3 * r), -1.0, 1.0);
    v += fluid.burstAxis * side * fluid.burstAir * exp(-dot(b, b) / (r * r));
  }

  velDst[i] = v;
}

@compute @workgroup_size(8, 8)
fn vorticity(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= fluid.size)) { return; }
  let c = vec2i(id.xy);
  let v = velSrc[idx(c)];
  let l = velAt(c - vec2i(1, 0), v);
  let r = velAt(c + vec2i(1, 0), v);
  let b = velAt(c - vec2i(0, 1), v);
  let t = velAt(c + vec2i(0, 1), v);
  curlField[idx(c)] = ((r.y - l.y) - (t.x - b.x)) / (2.0 * fluid.cell);
}

// Vorticity confinement (Fedkiw et al. 2001): push along N x w, which spins up existing eddies.
@compute @workgroup_size(8, 8)
fn confine(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= fluid.size)) { return; }
  let c = vec2i(id.xy);
  let i = idx(c);
  if (solid(c)) {
    velDst[i] = vec2f(0.0);
    return;
  }
  let wl = abs(curlField[idx(clampCell(c - vec2i(1, 0)))]);
  let wr = abs(curlField[idx(clampCell(c + vec2i(1, 0)))]);
  let wb = abs(curlField[idx(clampCell(c - vec2i(0, 1)))]);
  let wt = abs(curlField[idx(clampCell(c + vec2i(0, 1)))]);
  var n = vec2f(wr - wl, wt - wb);
  n /= length(n) + 1e-5;
  let w = curlField[i];
  let f = fluid.vorticity * fluid.cell * w * vec2f(n.y, -n.x);
  velDst[i] = velSrc[i] + f * fluid.dt;
}

@compute @workgroup_size(8, 8)
fn divergence(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= fluid.size)) { return; }
  let c = vec2i(id.xy);
  let i = idx(c);
  if (solid(c)) {
    divField[i] = 0.0;
    pressureDst[i] = 0.0;
    return;
  }
  let v = velSrc[i];
  let l = velAt(c - vec2i(1, 0), v);
  let r = velAt(c + vec2i(1, 0), v);
  let b = velAt(c - vec2i(0, 1), v);
  let t = velAt(c + vec2i(0, 1), v);
  divField[i] = ((r.x - l.x) + (t.y - b.y)) / (2.0 * fluid.cell);
  pressureDst[i] *= fluid.pressureDecay;
}

// One Jacobi sweep of the pressure Poisson equation, lap(p) = div(v). dt / density is folded into p.
@compute @workgroup_size(8, 8)
fn jacobi(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= fluid.size)) { return; }
  let c = vec2i(id.xy);
  let i = idx(c);
  if (solid(c)) {
    pressureDst[i] = 0.0;
    return;
  }
  let p = pressureSrc[i];
  let sum = pressureAt(c - vec2i(1, 0), p) + pressureAt(c + vec2i(1, 0), p)
          + pressureAt(c - vec2i(0, 1), p) + pressureAt(c + vec2i(0, 1), p);
  pressureDst[i] = (sum - fluid.cell * fluid.cell * divField[i]) * 0.25;
}

// Subtract the pressure gradient, leaving the divergence-free part: air is incompressible.
@compute @workgroup_size(8, 8)
fn project(@builtin(global_invocation_id) id: vec3u) {
  if (any(id.xy >= fluid.size)) { return; }
  let c = vec2i(id.xy);
  let i = idx(c);
  if (solid(c)) {
    velDst[i] = vec2f(0.0);
    return;
  }
  let p = pressureSrc[i];
  let grad = vec2f(
    pressureAt(c + vec2i(1, 0), p) - pressureAt(c - vec2i(1, 0), p),
    pressureAt(c + vec2i(0, 1), p) - pressureAt(c - vec2i(0, 1), p),
  ) / (2.0 * fluid.cell);
  velDst[i] -= grad;
}
