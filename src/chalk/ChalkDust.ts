import {
  clock,
  compute,
  draw,
  effect,
  frameLoop,
  init,
  sampler,
  storage,
  surface,
  target,
} from 'vgpu'
import type { FrameLoopHandle, Gpu, Surface, Target } from 'vgpu'
import compositeWgsl from './composite.wgsl?raw'
import fluidWgsl from './fluid.wgsl?raw'
import noiseWgsl from './noise.wgsl?raw'
import renderWgsl from './render.wgsl?raw'
import simWgsl from './sim.wgsl?raw'
import { MAX_RINGS, RING_FLOATS, type RingParams, VortexRings } from './vortexRings'

export const MAX_PARTICLES = 250_000
const PARTICLE_BYTES = 64 // keep in sync with `Particle` in sim.wgsl / render.wgsl (60 B, padded to 8-byte alignment)
const WORKGROUP = 64
const MAX_FLUID_CELLS = 512 * 256
const FLUID_WORKGROUP = 8
const DEPOSIT_SIZE: [number, number] = [1024, 48] // floor layer cells: across the screen x floor depth

export const defaultParams = {
  count: 120_000,
  paused: false,
  timeScale: 0.4, // slow motion

  // Physics (world units: 1 = half the viewport height)
  gravity: 0.35,
  grainDrag: 2,
  cloudDrag: 2.5,
  cloudLift: 0.9,
  subgrid: 0.4,
  stillAir: 0,
  diffusion: 0.04,
  noiseScale: 3,
  noiseSpeed: 0.3,
  heavySize: 8,

  // Air: Stable Fluids grid the chalk is carried by
  fluidRes: 128, // cells across the viewport height
  airDamping: 0.6,
  vorticity: 12,
  pressureIterations: 24,
  pressureDecay: 0.8,

  // Population
  cloudRatio: 0.03, // few but large puffs: smoke cost is fill rate, not count
  lifeMin: 4,
  lifeMax: 10,
  grainSizeMin: 0.5,
  grainSizeMax: 3.2,
  cloudSizeMin: 90,
  cloudSizeMax: 360,
  chunkChance: 0.012,
  chunkSizeMin: 6,
  chunkSizeMax: 14,

  // Ambient emitter
  emitRate: 0,
  emitSpeed: 0.15,
  emitterX: 0,
  emitterY: -0.55,
  emitterWidth: 1,
  emitterHeight: 0.05,

  // Interaction
  mouseRadius: 0.1,
  mouseStrength: 0.6, // the grid's share of the cursor wake; the vortex rings carry the rest
  mouseDrag: 25,

  // Vortex rings: the wake a hand leaves moving through the powder, shed along the cursor's path
  ringStrength: 0.8, // air speed through a new ring, as a share of the hand speed; 0 = off
  ringRadius: 0.07,
  ringSpacing: 1.2, // hand travel between rings, in ring radii
  ringMinSpeed: 0.15,
  ringCore: 0.3, // core radius / ring radius
  ringDecay: 0.7,
  ringViscosity: 0.0005,
  handDepth: 0.5, // how deep in the dust the cursor's hand moves, 0 near .. 1 far
  depthSpan: 0.6, // world-space depth of the dust volume, for the rings' 3D air

  // Bursts
  burstFraction: 0.9,
  burstSpeed: 2,
  burstRadius: 0.08,
  clapAngle: 0, // degrees; 0 = jets left and right
  jetSpread: 0.22,
  clapPuff: 0.3,
  burstLift: 0,
  burstAir: 0.6,
  burstAirRadius: 0.12,

  // Clumps: groups of grains that fly as one lump, then crumble into falling powder
  clumpChance: 0.45,
  clumpRadius: 0.045,
  clumpTilt: -15, // degrees; negative throws clumps downward
  clumpBreakMin: 0.6,
  clumpBreakMax: 2,
  clumpGravity: 4,
  clumpDrag: 1.2,
  shatterSpeed: 0.1,

  // Lighting (positions in world units, x as a fraction of the half-width)
  lightX: 0.35,
  lightY: 0.25,
  lightRadius: 0.9,
  lightIntensity: 2.2,
  ambientLight: 0.18,
  hazeScatter: 1.6,
  atmosphere: 0.05,
  floorGlow: 0.12,
  floorGlowHeight: 0.06,
  glintChance: 0.04,
  glintBoost: 3,

  // Floor
  floor: true,
  floorY: -0.78,
  floorDepth: 0.22,
  perspective: 1, // depth 1 is twice as far from the camera as depth 0, so it looks half the size
  floorFriction: 12,
  landingDamping: 0.2,
  liftThreshold: 1.5,
  settledAging: 0.35,
  floorBuildup: 1,
  floorTexture: 0.85,
  floorFlatten: 0.45,
  floorShade: 0.6,

  // Rendering
  renderScale: 0.75,
  smokeScale: 0.25, // smoke is soft, so it renders at a fraction of the resolution
  sizeScale: 1,
  grainOpacity: 0.8,
  cloudOpacity: 0.035,
  cloudGrowth: 3,
  depthFade: 0.6,
  wispiness: 0.85,
  speckle: 0.6,
  speckleScale: 9,
  shutter: 0.04,
  maxStreak: 10,
  focusDepth: 0.5,
  dofAmount: 4,

  // Composite
  tint: '#f4f1ea',
  exposure: 1.4,
  contrast: 1.15,
  grain: 0.08,
}

export type ChalkParams = typeof defaultParams

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.replace('#', ''), 16)
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255]
}

export class ChalkDust {
  readonly params: ChalkParams = { ...defaultParams }

  private gpu!: Gpu
  private output!: Surface
  private accum!: Target // grains
  private smoke!: Target // low-res smoke layer
  private loop?: FrameLoopHandle
  private sim!: ReturnType<typeof compute>
  private fluidKernels!: ReturnType<typeof compute>[]
  private advect!: ReturnType<typeof compute>
  private vorticity!: ReturnType<typeof compute>
  private confine!: ReturnType<typeof compute>
  private divergence!: ReturnType<typeof compute>
  private jacobi!: [ReturnType<typeof compute>, ReturnType<typeof compute>]
  private project!: ReturnType<typeof compute>
  private airVel!: ReturnType<typeof storage>
  private pressure!: ReturnType<typeof storage>
  private fluidSize: [number, number] = [0, 0]
  private deposit!: ReturnType<typeof storage>
  private ringBuf!: ReturnType<typeof storage>
  private readonly rings = new VortexRings()
  private grains!: ReturnType<typeof draw>
  private smokeDraw!: ReturnType<typeof draw>
  private composite!: ReturnType<typeof effect>
  private linear!: ReturnType<typeof sampler>

  private frameIndex = 0
  private burstId = 0
  private burstPos: [number, number] = [0, 0]
  private burstFraction = 0
  private burstSpeed = 0
  private burstAirPending = 0

  private pointer: [number, number] | null = null
  private lastPointer: [number, number] | null = null
  private prevPointer: [number, number] | null = null
  private mouseVel: [number, number] = [0, 0]

  // The floor sits a fixed height above this element's bottom edge, so it scrolls with the page.
  private floorAnchor: Element | null = null
  private lastAnchorBottom: number | null = null
  private pendingScroll = 0

  private readonly canvas: HTMLCanvasElement

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas
  }

  async start(): Promise<void> {
    this.gpu = await init()
    this.output = surface(this.gpu, this.canvas, { dpr: [1, 2], alphaMode: 'premultiplied' })
    this.accum = target(this.gpu, {
      size: this.layerSize(this.params.renderScale),
      format: 'rgba16float',
      clearColor: [0, 0, 0, 0],
    })
    this.smoke = target(this.gpu, {
      size: this.layerSize(this.params.smokeScale),
      format: 'rgba16float',
      clearColor: [0, 0, 0, 0],
    })

    // Bake the tiling noise once; smoke samples it instead of computing noise per pixel.
    const noise = target(this.gpu, { size: [256, 256], format: 'rgba8unorm' })
    effect(this.gpu, noiseWgsl, { label: 'chalk-noise' }).draw(noise)
    const repeat = sampler(this.gpu, {
      minFilter: 'linear',
      magFilter: 'linear',
      addressModeU: 'repeat',
      addressModeV: 'repeat',
    })

    // All particles start dead (life = maxLife = 0); the emitter and bursts bring them to life.
    const state = storage(this.gpu, MAX_PARTICLES * PARTICLE_BYTES, 'read-write')
    state.write(new Float32Array((MAX_PARTICLES * PARTICLE_BYTES) / 4))

    this.createFluid()
    // (coverage, lit coverage) per floor cell, accumulated by the sim and drawn by the composite.
    this.deposit = storage(this.gpu, DEPOSIT_SIZE[0] * DEPOSIT_SIZE[1] * 8, 'read-write')
    this.clearFloor()
    this.ringBuf = storage(this.gpu, MAX_RINGS * RING_FLOATS * 4, 'read')
    this.sim = compute(this.gpu, simWgsl, {
      label: 'chalk-sim',
      set: { particles: state, air: this.airVel, deposit: this.deposit, rings: this.ringBuf },
    })
    const layerDraw = (label: string) =>
      draw(this.gpu, {
        shader: renderWgsl,
        label,
        vertices: 6,
        blend: { color: { src: 'one', dst: 'one' } },
        set: { particles: state, noiseTex: noise, noiseSamp: repeat },
      })
    this.grains = layerDraw('chalk-grains')
    this.smokeDraw = layerDraw('chalk-smoke')
    this.linear = sampler(this.gpu, { minFilter: 'linear', magFilter: 'linear' })
    this.composite = effect(this.gpu, compositeWgsl, { label: 'chalk-composite' })
    this.bindLayers()

    // Surfaces only expose a texture inside a frame, so pre-warm the composite by signature.
    await Promise.all([
      this.grains.compile(this.accum),
      this.smokeDraw.compile(this.smoke),
      this.composite.compile({ colors: [navigator.gpu.getPreferredCanvasFormat()] }),
    ])

    window.addEventListener('pointermove', this.onPointerMove, { passive: true })
    window.addEventListener('pointerdown', this.onPointerDown, { passive: true })
    document.addEventListener('pointerleave', this.onPointerLeave)

    const time = clock(this.gpu)
    let simTime = 0
    this.loop = frameLoop(this.gpu, (frame) => {
      const p = this.params
      const dt = Math.min(time.deltaTime, 1 / 30) * p.timeScale
      simTime += dt
      this.syncAccumSize()
      this.updateInteraction(dt)

      const aspect = this.output.size[0] / this.output.size[1]
      const count = Math.min(Math.floor(p.count), MAX_PARTICLES)
      const floorLine = this.floorLineY()
      const floorY = p.floor ? floorLine : -100
      // Camera: the floor band spans floorDepth on screen from depth 0 to depth 1 and recedes toward the
      // horizon, which follows the page like the floor does.
      const perspective = Math.max(p.perspective, 0.05)
      const floorBand = Math.max(p.floorDepth, 1e-3)
      const horizon = floorLine + (floorBand * (1 + perspective)) / perspective

      if (!p.paused) {
        this.stepFluid(dt, aspect, floorY)
        this.stepRings(dt, floorY, horizon, perspective)
        this.sim.set({
          sim: {
            time: simTime,
            dt,
            aspect,
            count,
            frame: this.frameIndex++,
            gravity: p.gravity,
            grainDrag: p.grainDrag,
            cloudDrag: p.cloudDrag,
            cloudLift: p.cloudLift,
            subgrid: p.subgrid,
            stillAir: p.stillAir,
            diffusion: p.diffusion,
            heavySize: p.heavySize,
            noiseScale: p.noiseScale,
            noiseSpeed: p.noiseSpeed,
            cloudRatio: p.cloudRatio,
            lifeMin: p.lifeMin,
            lifeMax: Math.max(p.lifeMax, p.lifeMin),
            grainSizeMin: p.grainSizeMin,
            grainSizeMax: p.grainSizeMax,
            cloudSizeMin: p.cloudSizeMin,
            cloudSizeMax: p.cloudSizeMax,
            chunkChance: p.chunkChance,
            chunkSizeMin: p.chunkSizeMin,
            chunkSizeMax: p.chunkSizeMax,
            emitRate: p.emitRate,
            emitSpeed: p.emitSpeed,
            emitter: [p.emitterX * aspect, p.emitterY],
            emitterSize: [p.emitterWidth * aspect, p.emitterHeight],
            fluidSize: this.fluidSize,
            fluidCell: 2 / this.fluidSize[1],
            burstPos: this.burstPos,
            burstId: this.burstId,
            burstFraction: this.burstFraction,
            burstSpeed: this.burstSpeed,
            burstRadius: p.burstRadius,
            clapAngle: (p.clapAngle * Math.PI) / 180,
            jetSpread: p.jetSpread,
            clapPuff: p.clapPuff,
            burstLift: p.burstLift,
            clumpChance: p.clumpChance,
            clumpRadius: p.clumpRadius,
            clumpTilt: (p.clumpTilt * Math.PI) / 180,
            clumpBreakMin: p.clumpBreakMin,
            clumpBreakMax: Math.max(p.clumpBreakMax, p.clumpBreakMin),
            clumpGravity: p.clumpGravity,
            clumpDrag: p.clumpDrag,
            shatterSpeed: p.shatterSpeed,
            glintChance: p.glintChance,
            glintBoost: p.glintBoost,
            floorY,
            horizon,
            perspective,
            floorFriction: p.floorFriction,
            landingDamping: p.landingDamping,
            liftThreshold: p.liftThreshold,
            settledAging: p.settledAging,
            scroll: this.pendingScroll,
            depositSize: DEPOSIT_SIZE,
            ringCount: this.rings.count,
            depthSpan: Math.max(p.depthSpan, 1e-3),
          },
        })
        this.pendingScroll = 0
        // Dispatch submits on its own, ahead of this frame's render passes.
        this.sim.dispatch(Math.ceil(count / WORKGROUP))
      }

      const dpr = this.output.dpr
      // A grain's coverage is ~0.45π r² output px², flattened on the floor; spread over one deposit cell.
      // Both shrink with depth alike, so measure the cell at the front edge, where a depth row spans
      // floorDepth * (1 + perspective) / rows of screen height.
      const [outW, outH] = this.output.size
      const cellPx = (outW / DEPOSIT_SIZE[0]) * ((floorBand * (1 + perspective) * 0.5 * outH) / DEPOSIT_SIZE[1])
      const grainPx = p.sizeScale * dpr
      const depositScale =
        (p.floorBuildup * 0.45 * Math.PI * grainPx * grainPx * p.floorFlatten * p.grainOpacity) / Math.max(cellPx, 1e-6)
      const lightPos: [number, number] = [p.lightX * aspect, p.lightY]
      const view = {
        aspect,
        time: simTime,
        grainOpacity: p.grainOpacity,
        cloudOpacity: p.cloudOpacity,
        cloudGrowth: p.cloudGrowth,
        depthFade: p.depthFade,
        horizon,
        perspective,
        wispiness: p.wispiness,
        speckle: p.speckle,
        speckleScale: p.speckleScale,
        lightPos,
        lightRadius: p.lightRadius,
        lightIntensity: p.lightIntensity,
        ambientLight: p.ambientLight,
        hazeScatter: p.hazeScatter,
        shutter: p.shutter * p.timeScale, // streak length follows on-screen speed
        focusDepth: p.focusDepth,
        dofAmount: p.dofAmount,
        maxStreak: p.maxStreak,
        floorFlatten: p.floorFlatten,
        floorShade: p.floorShade,
      }
      this.grains.set({
        view: { ...view, layer: 0, resolution: this.accum.size, sizeScale: p.sizeScale * dpr * p.renderScale },
      })
      this.smokeDraw.set({
        view: { ...view, layer: 1, resolution: this.smoke.size, sizeScale: p.sizeScale * dpr * p.smokeScale },
      })
      this.composite.set({
        comp: {
          tint: hexToRgb(p.tint),
          exposure: p.exposure,
          contrast: p.contrast,
          grain: p.grain,
          time: simTime,
          aspect,
          lightPos,
          lightRadius: p.lightRadius,
          atmosphere: p.atmosphere,
          floorY: floorY + p.floorDepth * 0.5,
          floorGlow: p.floor ? p.floorGlow : 0,
          floorGlowHeight: p.floorGlowHeight,
          floorBase: floorY,
          floorDepth: floorBand,
          horizon,
          perspective,
          depositSize: DEPOSIT_SIZE,
          depositScale,
          depthFade: p.depthFade,
          ambientLight: p.ambientLight,
          lightIntensity: p.lightIntensity,
          floorShade: p.floorShade,
          floorTexture: p.floorTexture,
          pxPerUnit: outH * 0.5,
          speckPx: 2.4 * grainPx,
          floorFlatten: p.floorFlatten,
        },
      })

      frame.pass({ target: this.accum, clear: [0, 0, 0, 0] }, (pass) => {
        pass.draw(this.grains, { instances: count })
      })
      frame.pass({ target: this.smoke, clear: [0, 0, 0, 0] }, (pass) => {
        pass.draw(this.smokeDraw, { instances: count })
      })
      frame.pass({ target: this.output, clear: [0, 0, 0, 0] }, (pass) => {
        pass.draw(this.composite)
      })
    })
  }

  /** Puff of chalk at a viewport position (CSS px). `strength` scales speed and particle share. */
  burstAt(clientX: number, clientY: number, strength = 1): void {
    this.burstPos = this.toWorld(clientX, clientY)
    this.burstFraction = Math.min(this.params.burstFraction * strength, 1)
    this.burstSpeed = this.params.burstSpeed * Math.sqrt(strength)
    this.burstId = (this.burstId + 1) >>> 0
    this.burstAirPending = this.burstSpeed
  }

  /** Burst from the centre of an element, e.g. behind a headline or product shot. */
  burstAtElement(el: Element, strength = 1): void {
    const r = el.getBoundingClientRect()
    this.burstAt(r.left + r.width / 2, r.top + r.height / 2, strength)
  }

  /**
   * Ties the floor to an element, e.g. the hero section: it sits `floorY` above the element's bottom
   * edge and the dust scrolls along with the page. Pass null for a floor fixed to the viewport.
   */
  anchorFloor(el: Element | null): void {
    this.floorAnchor = el
    this.lastAnchorBottom = null
  }

  /** Sweeps away all the chalk that has built up on the floor. */
  clearFloor(): void {
    this.deposit.write(new Uint32Array(DEPOSIT_SIZE[0] * DEPOSIT_SIZE[1] * 2))
  }

  dispose(): void {
    this.loop?.stop()
    window.removeEventListener('pointermove', this.onPointerMove)
    window.removeEventListener('pointerdown', this.onPointerDown)
    document.removeEventListener('pointerleave', this.onPointerLeave)
    this.gpu?.dispose()
  }

  private createFluid(): void {
    const field = (bytesPerCell: number) => {
      const buf = storage(this.gpu, MAX_FLUID_CELLS * bytesPerCell, 'read-write')
      buf.write(new Float32Array((MAX_FLUID_CELLS * bytesPerCell) / 4))
      return buf
    }
    const velA = field(8)
    const velB = field(8)
    const curl = field(4)
    const div = field(4)
    const pA = field(4)
    const pB = field(4)
    this.airVel = velA
    this.pressure = pA

    const kernel = (entry: string, set: Record<string, unknown>) =>
      compute(this.gpu, fluidWgsl, { label: `chalk-fluid-${entry}`, entry, set })
    this.advect = kernel('advect', { velSrc: velA, velDst: velB })
    this.vorticity = kernel('vorticity', { velSrc: velB, curlField: curl })
    this.confine = kernel('confine', { velSrc: velB, curlField: curl, velDst: velA })
    this.divergence = kernel('divergence', { velSrc: velA, divField: div, pressureDst: pA })
    this.jacobi = [
      kernel('jacobi', { pressureSrc: pA, pressureDst: pB, divField: div }),
      kernel('jacobi', { pressureSrc: pB, pressureDst: pA, divField: div }),
    ]
    this.project = kernel('project', { pressureSrc: pA, velDst: velA })
    this.fluidKernels = [this.advect, this.vorticity, this.confine, this.divergence, ...this.jacobi, this.project]
  }

  /**
   * Screen height of the floor's front edge this frame, whether or not the floor is on. With an anchor,
   * `floorY` is measured from the anchor's bottom edge instead of the bottom of the viewport, and the
   * distance the anchor moved since the last frame is queued up as a scroll shift for the sim.
   */
  private floorLineY(): number {
    let anchorBottom = -1
    if (this.floorAnchor) {
      anchorBottom = this.toWorld(0, this.floorAnchor.getBoundingClientRect().bottom)[1]
      if (this.lastAnchorBottom !== null) this.pendingScroll += anchorBottom - this.lastAnchorBottom
      this.lastAnchorBottom = anchorBottom
    }
    return this.params.floorY + anchorBottom + 1
  }

  /** One Stable Fluids step: the air the particles are about to sample. */
  private stepFluid(dt: number, aspect: number, floorY: number): void {
    const p = this.params
    let rows = Math.max(8, Math.round(p.fluidRes))
    let cols = Math.ceil(aspect * rows)
    if (cols * rows > MAX_FLUID_CELLS) {
      rows = Math.floor(Math.sqrt(MAX_FLUID_CELLS / aspect))
      cols = Math.min(Math.ceil(aspect * rows), Math.floor(MAX_FLUID_CELLS / rows))
    }
    if (cols !== this.fluidSize[0] || rows !== this.fluidSize[1]) {
      // The cell layout changed, so the old field is meaningless: start from still air.
      this.fluidSize = [cols, rows]
      this.airVel.write(new Float32Array(MAX_FLUID_CELLS * 2))
      this.pressure.write(new Float32Array(MAX_FLUID_CELLS))
    }

    const clap = (p.clapAngle * Math.PI) / 180
    const fluid = {
      size: this.fluidSize,
      cell: 2 / rows,
      dt,
      aspect,
      floorY,
      damping: p.airDamping,
      vorticity: p.vorticity,
      pressureDecay: p.pressureDecay,
      mouse: this.pointer ?? [1e4, 1e4],
      mousePrev: this.pointer ? (this.prevPointer ?? this.pointer) : [1e4, 1e4],
      mouseVel: this.mouseVel,
      mouseRadius: Math.max(p.mouseRadius, 1e-3),
      mouseStrength: p.mouseStrength,
      mouseDrag: p.mouseDrag,
      burstPos: this.burstPos,
      burstAxis: [Math.cos(clap), Math.sin(clap)],
      burstAir: this.burstAirPending * p.burstAir,
      burstAirRadius: Math.max(p.burstAirRadius, 1e-3),
    }
    this.burstAirPending = 0
    for (const k of this.fluidKernels) k.set({ fluid })

    const gx = Math.ceil(cols / FLUID_WORKGROUP)
    const gy = Math.ceil(rows / FLUID_WORKGROUP)
    this.advect.dispatch(gx, gy)
    this.vorticity.dispatch(gx, gy)
    this.confine.dispatch(gx, gy)
    this.divergence.dispatch(gx, gy)
    // Even sweep count, so the solved pressure ends up back in the buffer `project` reads.
    const sweeps = Math.max(2, Math.round(p.pressureIterations / 2) * 2)
    for (let i = 0; i < sweeps; i++) this.jacobi[i % 2].dispatch(gx, gy)
    this.project.dispatch(gx, gy)
  }

  /**
   * The cursor is a hand moving through the dust at `handDepth`: it sheds vortex rings along its path,
   * which then move on their own. Runs before the sim reads the rings, and before `pendingScroll` is spent.
   */
  private stepRings(dt: number, floorY: number, horizon: number, perspective: number): void {
    const p = this.params
    const rp: RingParams = {
      radius: Math.max(p.ringRadius, 1e-3),
      strength: p.ringStrength,
      spacing: Math.max(p.ringSpacing, 0.1),
      minSpeed: p.ringMinSpeed,
      core: Math.max(p.ringCore, 0.02),
      decay: p.ringDecay,
      viscosity: p.ringViscosity,
    }
    this.rings.scroll(this.pendingScroll)
    if (this.pointer) {
      // From the screen to world space at the hand's depth, as `unproject` in sim.wgsl.
      const s = 1 / (1 + perspective * p.handDepth)
      const [x, y] = this.pointer
      const hand: [number, number, number] = [x / s, horizon + (y - horizon) / s, p.handDepth * p.depthSpan]
      this.rings.shed(hand, [this.mouseVel[0] / s, this.mouseVel[1] / s, 0], rp)
    } else {
      this.rings.shed(null, [0, 0, 0], rp)
    }
    this.rings.step(dt, floorY, rp)
    this.ringBuf.write(this.rings.pack(rp))
  }

  private layerSize(s: number): [number, number] {
    return [
      Math.max(1, Math.round(this.output.size[0] * s)),
      Math.max(1, Math.round(this.output.size[1] * s)),
    ]
  }

  private syncAccumSize(): void {
    let changed = false
    for (const [layer, scale] of [
      [this.accum, this.params.renderScale],
      [this.smoke, this.params.smokeScale],
    ] as const) {
      const [w, h] = this.layerSize(scale)
      if (layer.size[0] === w && layer.size[1] === h) continue
      layer.resize([w, h])
      changed = true
    }
    if (changed) this.bindLayers()
  }

  private bindLayers(): void {
    this.composite.set({ src: this.accum, smoke: this.smoke, samp: this.linear, deposit: this.deposit })
  }

  private toWorld(clientX: number, clientY: number): [number, number] {
    const rect = this.canvas.getBoundingClientRect()
    const aspect = rect.width / rect.height
    const x = ((clientX - rect.left) / rect.width) * 2 - 1
    const y = 1 - ((clientY - rect.top) / rect.height) * 2
    return [x * aspect, y]
  }

  private updateInteraction(dt: number): void {
    if (dt <= 0) return
    // Cursor velocity in world units/s, smoothed so single jittery events don't spike the wake.
    let vx = 0
    let vy = 0
    if (this.pointer && this.lastPointer) {
      vx = (this.pointer[0] - this.lastPointer[0]) / dt
      vy = (this.pointer[1] - this.lastPointer[1]) / dt
      const speed = Math.hypot(vx, vy)
      const max = 6
      if (speed > max) {
        vx *= max / speed
        vy *= max / speed
      }
    }
    this.prevPointer = this.lastPointer ?? this.pointer
    this.lastPointer = this.pointer
    const k = 1 - Math.exp(-dt * 20)
    this.mouseVel = [
      this.mouseVel[0] + (vx - this.mouseVel[0]) * k,
      this.mouseVel[1] + (vy - this.mouseVel[1]) * k,
    ]
  }

  private onPointerMove = (e: PointerEvent) => {
    this.pointer = this.toWorld(e.clientX, e.clientY)
  }

  private onPointerDown = (e: PointerEvent) => {
    if ((e.target as Element | null)?.closest('.lil-gui')) return
    this.burstAt(e.clientX, e.clientY, 0.5)
  }

  private onPointerLeave = () => {
    this.pointer = null
    this.lastPointer = null
  }
}
