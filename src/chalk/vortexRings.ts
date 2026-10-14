// Vortex rings shed by the cursor. A hand moving through air leaves a wake of ring vortices behind it:
// each pushes air through its middle along the hand's path, follows the hand on its own induced velocity
// and rolls the dust around its core. The rings are tracked here as rigid circles, a few dozen at most;
// the dust samples their 3D Biot–Savart velocity on the GPU (`ringAir` in sim.wgsl).
//
// Coordinates are the sim's world space: x and y at depth-0 scale, z = depth * depthSpan.

export const MAX_RINGS = 32
export const RING_FLOATS = 12 // keep in sync with `Ring` in sim.wgsl: center + radius, axis + circulation, core + pad

type Vec3 = [number, number, number]

interface Ring {
  c: Vec3 // center
  n: Vec3 // unit axis: the direction air flows through the ring and the ring travels
  r: number
  r0: number
  gamma: number // circulation
  gamma0: number
  core0: number
  age: number
}

export interface RingParams {
  radius: number // radius of a newly shed ring
  strength: number // air speed through a new ring's middle, as a fraction of the hand speed
  spacing: number // hand travel between rings, in ring radii
  minSpeed: number // slower hands shed nothing
  core: number // core radius as a fraction of the ring radius
  decay: number // circulation lost per second
  viscosity: number // core growth by diffusion
}

const SEGMENTS = 8 // per ring for ring-ring induction; the GPU uses more for the dust

const add = (a: Vec3, b: Vec3, s = 1): Vec3 => [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s]
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
]
const normalize = (a: Vec3): Vec3 => {
  const l = Math.hypot(a[0], a[1], a[2]) || 1
  return [a[0] / l, a[1] / l, a[2] / l]
}

/** Two unit vectors spanning the ring's plane, with e1 x e2 = n. Same construction as sim.wgsl. */
function basis(n: Vec3): [Vec3, Vec3] {
  const helper: Vec3 = Math.abs(n[2]) < 0.9 ? [0, 0, 1] : [1, 0, 0]
  const e1 = normalize(cross(n, helper))
  return [e1, cross(n, e1)]
}

function coreOf(ring: Ring, p: RingParams): number {
  // Stretching thins the core (its volume is conserved), viscosity slowly fattens it.
  return Math.sqrt((ring.core0 * ring.core0 * ring.r0) / ring.r + 4 * p.viscosity * ring.age)
}

/** The floor's image of a ring: mirrored across the floor with the same circulation, so no air crosses it. */
function mirror(ring: Ring, floorY: number): Ring {
  return { ...ring, c: [ring.c[0], 2 * floorY - ring.c[1], ring.c[2]], n: [ring.n[0], -ring.n[1], ring.n[2]] }
}

/** Velocity a ring induces at `x`: Biot–Savart over the ring as SEGMENTS smoothed vortex elements. */
function induced(x: Vec3, ring: Ring, core: number): Vec3 {
  const [e1, e2] = basis(ring.n)
  let u: Vec3 = [0, 0, 0]
  for (let k = 0; k < SEGMENTS; k++) {
    const a = (k / SEGMENTS) * Math.PI * 2
    const rho = add(add([0, 0, 0], e1, Math.cos(a)), e2, Math.sin(a))
    const t = add(add([0, 0, 0], e1, -Math.sin(a)), e2, Math.cos(a))
    const d = add(x, add(ring.c, rho, ring.r), -1)
    const q = dot(d, d) + core * core
    u = add(u, cross(t, d), 1 / (q * Math.sqrt(q)))
  }
  // Element strength Γ·dl, dl = r·2π/N, over the 4π of Biot–Savart.
  const s = (ring.gamma * ring.r) / (2 * SEGMENTS)
  return [u[0] * s, u[1] * s, u[2] * s]
}

export class VortexRings {
  private rings: Ring[] = []
  private readonly data = new Float32Array(new ArrayBuffer(MAX_RINGS * RING_FLOATS * 4))
  private lastHand: Vec3 | null = null
  private travelled = 0

  get count(): number {
    return this.rings.length
  }

  clear(): void {
    this.rings = []
  }

  /** The page scrolled: the rings move with it, like the dust and the floor. */
  scroll(dy: number): void {
    for (const ring of this.rings) ring.c[1] += dy
  }

  /** Follows the hand (null when the cursor is gone) and sheds a ring every `spacing` radii it travels. */
  shed(hand: Vec3 | null, vel: Vec3, p: RingParams): void {
    if (!hand) {
      this.lastHand = null
      return
    }
    if (this.lastHand) this.travelled += Math.hypot(...add(hand, this.lastHand, -1))
    this.lastHand = hand
    const speed = Math.hypot(...vel)
    if (p.strength <= 0 || speed < p.minSpeed || this.travelled < p.radius * p.spacing) return
    this.travelled = 0

    const n = normalize(vel)
    // Air through a ring's middle moves at Γ / 2r, so pick Γ for the requested share of the hand's speed.
    const gamma = 2 * p.radius * p.strength * speed
    const ring: Ring = {
      c: add(hand, n, -0.5 * p.radius), // just behind the hand, in its wake
      n,
      r: p.radius,
      r0: p.radius,
      gamma,
      gamma0: gamma,
      core0: p.radius * p.core,
      age: 0,
    }
    if (this.rings.length >= MAX_RINGS) {
      // Recycle the faintest ring.
      let weakest = 0
      for (let i = 1; i < this.rings.length; i++) if (this.rings[i].gamma < this.rings[weakest].gamma) weakest = i
      this.rings.splice(weakest, 1)
    }
    this.rings.push(ring)
  }

  /**
   * Advances the rings: each travels on its own induced velocity and is carried, stretched or squeezed by
   * the other rings and by the floor's images, which makes trailing rings slip through leading ones and
   * rings that hit the floor spread out over it. `floorY` < -50 means there is no floor.
   */
  step(dt: number, floorY: number, p: RingParams): void {
    if (dt <= 0 || this.rings.length === 0) return
    const hasFloor = floorY > -50
    const cores = this.rings.map((ring) => coreOf(ring, p))
    const sources: [Ring, number][] = this.rings.map((ring, i) => [ring, cores[i]])
    if (hasFloor) for (let i = 0; i < this.rings.length; i++) sources.push([mirror(this.rings[i], floorY), cores[i]])

    const moves = this.rings.map((ring, i) => {
      // Kelvin's speed for a thin-cored ring.
      const self = (ring.gamma / (4 * Math.PI * ring.r)) * (Math.log((8 * ring.r) / cores[i]) - 0.25)
      let drift: Vec3 = add([0, 0, 0], ring.n, self)
      let grow = 0
      const [e1, e2] = basis(ring.n)
      for (let k = 0; k < SEGMENTS; k++) {
        const a = ((k + 0.5) / SEGMENTS) * Math.PI * 2
        const rho = add(add([0, 0, 0], e1, Math.cos(a)), e2, Math.sin(a))
        const x = add(ring.c, rho, ring.r)
        let u: Vec3 = [0, 0, 0]
        for (const [src, core] of sources) {
          if (src === ring) continue
          u = add(u, induced(x, src, core))
        }
        drift = add(drift, u, 1 / SEGMENTS)
        grow += dot(u, rho) / SEGMENTS
      }
      return { drift, grow }
    })

    const fade = Math.exp(-p.decay * dt)
    this.rings.forEach((ring, i) => {
      ring.c = add(ring.c, moves[i].drift, dt)
      ring.r = Math.min(Math.max(ring.r + moves[i].grow * dt, ring.r0 * 0.3), ring.r0 * 4)
      if (hasFloor) ring.c[1] = Math.max(ring.c[1], floorY + cores[i])
      ring.gamma *= fade
      ring.age += dt
    })
    this.rings = this.rings.filter((ring) => ring.gamma > ring.gamma0 * 0.03)
  }

  /** Ring data for the GPU, `count` rings long. */
  pack(p: RingParams): Float32Array<ArrayBuffer> {
    this.rings.forEach((ring, i) => {
      this.data.set([...ring.c, ring.r, ...ring.n, ring.gamma, coreOf(ring, p), 0, 0, 0], i * RING_FLOATS)
    })
    return this.data
  }
}
