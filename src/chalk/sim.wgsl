// Chalk dust simulation. One invocation per particle, updated in place.
// World space: y in [-1, 1] (up), x in [-aspect, aspect].

struct Particle {
  pos: vec2f,
  vel: vec2f,
  life: f32,     // age in seconds
  maxLife: f32,  // dead when life >= maxLife
  size: f32,     // base size in px
  kind: f32,     // 0 = grain, 1 = cloud
  depth: f32,    // 0 near .. 1 far
  seed: f32,
  burst: u32,    // last burst id this particle has seen
  bright: f32,
  breakAt: f32,  // > 0 while bonded into a falling clump; the clump shatters at this age
  grounded: f32, // 1 while resting on the floor
}

struct Sim {
  time: f32,
  dt: f32,
  aspect: f32,
  count: u32,
  frame: u32,

  gravity: f32,
  grainDrag: f32,
  cloudDrag: f32,
  cloudLift: f32,
  subgrid: f32,      // small eddies the fluid grid is too coarse to resolve, scaled by local air speed
  stillAir: f32,
  diffusion: f32,
  noiseScale: f32,
  noiseSpeed: f32,
  heavySize: f32,

  cloudRatio: f32,
  lifeMin: f32,
  lifeMax: f32,
  grainSizeMin: f32,
  grainSizeMax: f32,
  cloudSizeMin: f32,
  cloudSizeMax: f32,
  chunkChance: f32,
  chunkSizeMin: f32,
  chunkSizeMax: f32,

  emitRate: f32,
  emitSpeed: f32,
  emitter: vec2f,
  emitterSize: vec2f,

  fluidSize: vec2u,
  fluidCell: f32,

  burstPos: vec2f,
  burstId: u32,
  burstFraction: f32,
  burstSpeed: f32,
  burstRadius: f32,
  clapAngle: f32,
  jetSpread: f32,
  clapPuff: f32,
  burstLift: f32,

  clumpChance: f32,
  clumpRadius: f32,
  clumpTilt: f32,
  clumpBreakMin: f32,
  clumpBreakMax: f32,
  clumpGravity: f32,
  clumpDrag: f32,
  shatterSpeed: f32,

  glintChance: f32,
  glintBoost: f32,

  floorY: f32,
  floorDepth: f32,
  floorFriction: f32,
  landingDamping: f32,
  liftThreshold: f32,
  settledAging: f32,
  scroll: f32,       // how far the page (and with it the floor) moved up the screen this step
  depositSize: vec2u,
}

// Consecutive particles form a group; a burst selects whole groups, and some groups fly as one clump.
const CLUMP_SIZE = 16u;

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> sim: Sim;
@group(0) @binding(2) var<storage, read> air: array<vec2f>; // fluid.wgsl velocity grid
// Chalk that has built up on the floor, drawn by composite.wgsl. Cells span x across the screen and y
// across floor depth; each holds (coverage, lit coverage) in DEPOSIT_UNIT fixed point.
@group(0) @binding(3) var<storage, read_write> deposit: array<atomic<u32>>;

const DEPOSIT_UNIT = 65536.0; // keep in sync with composite.wgsl

fn airAt(p: vec2f) -> vec2f {
  let size = vec2i(sim.fluidSize);
  let g = clamp((p - vec2f(-sim.aspect, -1.0)) / sim.fluidCell - 0.5, vec2f(0.0), vec2f(size - 1));
  let i0 = vec2i(floor(g));
  let i1 = min(i0 + 1, size - 1);
  let f = g - vec2f(i0);
  let a = air[i0.y * size.x + i0.x];
  let b = air[i0.y * size.x + i1.x];
  let c = air[i1.y * size.x + i0.x];
  let d = air[i1.y * size.x + i1.x];
  return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}

fn rnd(state: ptr<function, u32>) -> f32 {
  *state = pcg(*state);
  return f32(*state) * (1.0 / 4294967296.0);
}

fn hash31(p: vec3f) -> f32 {
  var q = fract(p * 0.1031);
  q += dot(q, q.zyx + 31.32);
  return fract((q.x + q.y) * q.z);
}

fn vnoise(p: vec3f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
  let a = mix(hash31(i), hash31(i + vec3f(1, 0, 0)), u.x);
  let b = mix(hash31(i + vec3f(0, 1, 0)), hash31(i + vec3f(1, 1, 0)), u.x);
  let c = mix(hash31(i + vec3f(0, 0, 1)), hash31(i + vec3f(1, 0, 1)), u.x);
  let d = mix(hash31(i + vec3f(0, 1, 1)), hash31(i + vec3f(1, 1, 1)), u.x);
  return mix(mix(a, b, u.y), mix(c, d, u.y), u.z);
}

fn potential(p: vec2f, t: f32) -> f32 {
  let q = vec3f(p, t);
  return vnoise(q) + 0.5 * vnoise(q * 2.03 + 17.1);
}

// Divergence-free flow: rotated gradient of a scalar noise field. Swirls without sinks or sources.
fn curl(p: vec2f, t: f32) -> vec2f {
  let e = 0.05;
  let dx = potential(p + vec2f(e, 0), t) - potential(p - vec2f(e, 0), t);
  let dy = potential(p + vec2f(0, e), t) - potential(p - vec2f(0, e), t);
  return vec2f(dy, -dx) / (2.0 * e);
}

// Roughly normal with unit variance; cheap enough for spawn-time use.
fn gauss(st: ptr<function, u32>) -> f32 {
  return (rnd(st) + rnd(st) + rnd(st) + rnd(st) - 2.0) * 1.732;
}

fn spawn(p: ptr<function, Particle>, st: ptr<function, u32>, isBurst: bool) {
  let isCloud = rnd(st) < sim.cloudRatio;
  (*p).kind = select(0.0, 1.0, isCloud);
  (*p).depth = rnd(st);
  (*p).seed = rnd(st);
  (*p).bright = 0.55 + 0.45 * rnd(st);
  if (!isCloud && rnd(st) < sim.glintChance) {
    (*p).bright *= sim.glintBoost; // a few chalk crystals catch the light
  }
  (*p).life = 0.0;
  (*p).breakAt = 0.0;
  (*p).grounded = 0.0;
  (*p).maxLife = mix(sim.lifeMin, sim.lifeMax, rnd(st)) * select(1.0, 1.4, isCloud);

  let s = rnd(st);
  if (isCloud) {
    (*p).size = mix(sim.cloudSizeMin, sim.cloudSizeMax, s);
  } else {
    // Mostly fine powder, plus the occasional crumb of block chalk.
    (*p).size = mix(sim.grainSizeMin, sim.grainSizeMax, s * s);
    if (rnd(st) < sim.chunkChance) {
      (*p).size = mix(sim.chunkSizeMin, sim.chunkSizeMax, rnd(st));
    }
  }

  if (isBurst) {
    // A clap squeezes chalk out of the gap between the palms: two opposite jets along the clap axis,
    // plus a puff that spills out in every direction. Smoke is split between the two.
    let dir = clapDir(st, select(sim.clapPuff, 0.35, isCloud));
    let r = sqrt(rnd(st)) * sim.burstRadius;
    let ra = rnd(st) * 6.2831853;
    (*p).pos = sim.burstPos + vec2f(cos(ra), sin(ra)) * r;
    // Exponential speed distribution: most dust puffs gently, a few grains shoot far.
    let speed = sim.burstSpeed * -log(max(rnd(st), 1e-4)) * select(1.0, 0.9, isCloud);
    (*p).vel = dir * speed + vec2f(0.0, sim.burstSpeed * sim.burstLift);
  } else {
    let o = vec2f(rnd(st), rnd(st)) * 2.0 - 1.0;
    (*p).pos = sim.emitter + o * sim.emitterSize;
    let up = sim.emitSpeed * (0.3 + rnd(st));
    (*p).vel = vec2f((rnd(st) - 0.5) * sim.emitSpeed, up) * select(1.0, 0.6, isCloud);
  }
}

// Direction a clap throws chalk: mostly along two opposite jets, some in every direction.
fn clapDir(st: ptr<function, u32>, puff: f32) -> vec2f {
  var a: f32;
  if (rnd(st) < puff) {
    a = rnd(st) * 6.2831853;
  } else {
    let side = select(0.0, 3.1415927, rnd(st) < 0.5);
    a = sim.clapAngle + side + gauss(st) * sim.jetSpread;
  }
  return vec2f(cos(a), sin(a));
}

// A clump of block chalk: every member draws its trajectory from the shared group RNG `gst`,
// so the group moves as one lump. Only the member's offset, look and break time come from `st`.
// Returns false for members beyond this clump's random size; they spawn as loose powder instead.
fn spawnClump(p: ptr<function, Particle>, gst: ptr<function, u32>, st: ptr<function, u32>, member: u32) -> bool {
  // Thrown sideways along the jets and tilted down: heavy lumps don't float up like the powder.
  let right = rnd(gst) < 0.5;
  let spread = gauss(gst) * sim.jetSpread * 0.5;
  let a = select(sim.clapAngle + 3.1415927 - sim.clumpTilt, sim.clapAngle + sim.clumpTilt, right) + spread;
  let dir = vec2f(cos(a), sin(a));
  let speed = sim.burstSpeed * (0.5 + 0.9 * rnd(gst));
  let ra = rnd(gst) * 6.2831853;
  let center = sim.burstPos + vec2f(cos(ra), sin(ra)) * sqrt(rnd(gst)) * sim.burstRadius;

  // Random lumps: member count, size, elongation and orientation all vary per clump.
  let members = u32(mix(5.0, f32(CLUMP_SIZE), rnd(gst)));
  let s = rnd(gst);
  let radius = sim.clumpRadius * (0.35 + 1.65 * s * s);
  let stretch = mix(0.45, 1.0, rnd(gst));
  let rot = rnd(gst) * 6.2831853;
  let depth = rnd(gst);
  let groupBreak = mix(sim.clumpBreakMin, sim.clumpBreakMax, rnd(gst));
  if (member >= members) { return false; }

  let g = clamp(vec2f(gauss(st), gauss(st)), vec2f(-2.0), vec2f(2.0));
  let local = vec2f(g.x, g.y * stretch) * radius * 0.5;
  let o = vec2f(cos(rot) * local.x - sin(rot) * local.y, sin(rot) * local.x + cos(rot) * local.y);

  (*p).kind = 0.0;
  (*p).grounded = 0.0;
  (*p).depth = depth;
  (*p).vel = dir * speed + vec2f(0.0, sim.burstSpeed * sim.burstLift);
  (*p).pos = center + o;
  (*p).seed = rnd(st);
  (*p).bright = 0.6 + 0.4 * rnd(st);
  (*p).size = mix(sim.grainSizeMin, sim.grainSizeMax * 1.3, rnd(st));
  (*p).life = 0.0;
  (*p).maxLife = mix(sim.lifeMin, sim.lifeMax, rnd(st));

  // Gradual crumbling: outer grains shed first, the core holds together longest.
  let edge = clamp(length(g) * 0.5, 0.0, 1.0);
  (*p).breakAt = max(groupBreak * mix(1.0, 0.12, edge) * (0.75 + 0.5 * rnd(st)), 0.02);
  return true;
}

// The clump crumbles: each fragment gets its own kick and from now on falls as loose powder.
fn shatter(p: ptr<function, Particle>, st: ptr<function, u32>, onImpact: bool) {
  let a = rnd(st) * 6.2831853;
  var kick = vec2f(cos(a), sin(a)) * sim.shatterSpeed * (0.2 + rnd(st));
  if (onImpact) {
    // Hitting the floor splashes fragments sideways and a little up.
    kick = vec2f(kick.x * 1.5, abs(kick.y) * 0.6);
    (*p).vel = vec2f((*p).vel.x * 0.3, 0.0);
  }
  (*p).vel += kick;
  (*p).breakAt = 0.0;
}

// Matches the grain fade-out in render.wgsl.
fn grainFade(life: f32, maxLife: f32) -> f32 {
  return 1.0 - smoothstep(0.55, 1.0, life / maxLife);
}

// Hands `share` of a resting grain's visible dust over to the floor layer, so as the grain fades
// out (or is recycled by a burst) the floor keeps exactly what it showed and fills up over time.
fn depositGrain(p: Particle, share: f32, st: ptr<function, u32>) {
  if (p.kind > 0.5 || share <= 0.0) { return; }
  let u = p.pos.x / sim.aspect * 0.5 + 0.5;
  if (u < 0.0 || u >= 1.0) { return; }
  let cell = vec2u(vec2f(u, clamp(p.depth, 0.0, 0.999)) * vec2f(sim.depositSize));
  let i = (cell.y * sim.depositSize.x + cell.x) * 2u;
  // Stochastic rounding, so tiny per-frame shares still add up on average instead of truncating away.
  let coverage = share * p.size * p.size * DEPOSIT_UNIT;
  atomicAdd(&deposit[i], u32(coverage + rnd(st)));
  atomicAdd(&deposit[i + 1u], u32(coverage * p.bright + rnd(st)));
}

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3u) {
  let i = id.x;
  if (i >= sim.count) { return; }

  var p = particles[i];
  var st = pcg(i ^ pcg(sim.frame * 9781u + 1u));
  let dt = sim.dt;

  if (p.burst != sim.burstId) {
    p.burst = sim.burstId;
    var gst = pcg(((i / CLUMP_SIZE) * 2654435761u) ^ pcg(sim.burstId * 97u + 13u));
    if (rnd(&gst) < sim.burstFraction) {
      if (p.grounded > 0.5 && p.life < p.maxLife) {
        depositGrain(p, grainFade(p.life, p.maxLife), &st); // its dust stays on the floor
      }
      let isClump = rnd(&gst) < sim.clumpChance;
      if (!isClump || !spawnClump(&p, &gst, &st, i % CLUMP_SIZE)) {
        spawn(&p, &st, true);
      }
    }
  }

  if (p.life >= p.maxLife) {
    if (rnd(&st) < 1.0 - exp(-sim.emitRate * dt)) {
      spawn(&p, &st, false);
    } else {
      particles[i] = p;
      return;
    }
  }

  let isCloud = p.kind > 0.5;

  let flow = airAt(p.pos);

  // Far particles rest higher, so settled dust forms a receding ground plane instead of a line.
  let floorY = sim.floorY + p.depth * sim.floorDepth;

  // The dust lives in the page, not on the screen: it scrolls along with the hero and its floor.
  // Below the floor is the rest of the page, so only dust falling onto it from above lands.
  if (p.grounded > 0.5) {
    p.pos.y = floorY;
  } else {
    p.pos.y += sim.scroll;
  }
  let above = p.pos.y >= floorY;

  if (p.breakAt > 0.0) {
    if (p.life >= p.breakAt || length(flow - p.vel) > sim.liftThreshold) {
      shatter(&p, &st, false); // crumbles mid-fall, or a gust (e.g. the cursor) knocks it apart
    } else {
      // Bonded: a dense lump ignores the air, so every member follows the identical trajectory.
      p.vel.y -= sim.gravity * sim.clumpGravity * dt;
      p.vel *= exp(-sim.clumpDrag * dt);
      p.pos += p.vel * dt;
      p.life += dt;
      if (above && p.pos.y < floorY) {
        p.pos.y = floorY + 0.002; // just above the floor, so the fragments start airborne
        shatter(&p, &st, true);
      }
      particles[i] = p;
      return;
    }
  }

  if (p.grounded > 0.5) {
    // Resting dust stays put: no air, no gravity, and the cursor doesn't disturb it.
    p.vel = vec2f(p.vel.x * exp(-sim.floorFriction * dt), 0.0);
    p.pos += p.vel * dt;
    p.pos.y = floorY;
    let fadeBefore = grainFade(p.life, p.maxLife);
    p.life += dt * sim.settledAging; // settled chalk lingers before it fades into the floor layer
    depositGrain(p, fadeBefore - grainFade(p.life, p.maxLife), &st);
  } else {
    // The air is the fluid grid, stirred by claps and the cursor. Where it moves fast it is also turbulent
    // below the grid's resolution, so add curl noise there; the room air itself is otherwise still.
    let eddy = sim.subgrid * length(flow) + sim.stillAir;
    let air = flow + curl(p.pos * sim.noiseScale + p.depth * 3.0, sim.time * sim.noiseSpeed) * eddy;

    // Stokes-like settling: bigger pieces fall faster and shrug off the air; smoke barely sinks.
    let heavy = clamp(p.size / sim.heavySize, 0.0, 1.0);
    let g = select(sim.gravity * mix(0.35, 2.5, heavy), sim.gravity * (1.0 - sim.cloudLift), isCloud);
    let coupling = select(sim.grainDrag * mix(1.6, 0.2, heavy), sim.cloudDrag, isCloud);
    p.vel.y -= g * dt;
    if (isCloud) {
      // Slow Brownian spreading so lingering smoke diffuses in place instead of drifting.
      p.vel += vec2f(gauss(&st), gauss(&st)) * sim.diffusion * sqrt(dt);
    }
    p.vel = air + (p.vel - air) * exp(-coupling * dt);
    p.pos += p.vel * dt;

    // Landing: the impact kills vertical motion and most of the horizontal, so dust stays where it falls.
    if (above && p.pos.y < floorY) {
      p.pos.y = floorY;
      p.vel = vec2f(p.vel.x * sim.landingDamping, 0.0);
      p.grounded = 1.0;
    }
    p.life += dt;
  }

  let margin = 0.4;
  if (abs(p.pos.x) > sim.aspect + margin || p.pos.y > 1.0 + margin || p.pos.y < -1.0 - margin) {
    if (p.grounded > 0.5) {
      depositGrain(p, grainFade(p.life, p.maxLife), &st); // scrolled off with the floor: its dust stays on it
    }
    p.life = p.maxLife;
  }

  particles[i] = p;
}
