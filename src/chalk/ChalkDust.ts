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
import noiseWgsl from './noise.wgsl?raw'
import renderWgsl from './render.wgsl?raw'
import simWgsl from './sim.wgsl?raw'

export const MAX_PARTICLES = 250_000
const PARTICLE_BYTES = 56 // keep in sync with `Particle` in sim.wgsl / render.wgsl
const WORKGROUP = 64

export const defaultParams = {
  count: 120_000,
  paused: false,
  timeScale: 0.4, // slow motion

  // Physics (world units: 1 = half the viewport height)
  gravity: 0.35,
  grainDrag: 2,
  cloudDrag: 2.5,
  cloudLift: 0.9,
  turbulence: 0.8,
  eddyDecay: 0.8,
  stillAir: 0,
  diffusion: 0.04,
  noiseScale: 3,
  noiseSpeed: 0.3,
  heavySize: 8,

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
  mouseRadius: 0.25,
  mouseStrength: 1.2,
  mouseSwirl: 0.6,

  // Bursts
  burstFraction: 0.9,
  burstSpeed: 2,
  burstRadius: 0.08,
  clapAngle: 0, // degrees; 0 = jets left and right
  jetSpread: 0.22,
  clapPuff: 0.3,
  burstLift: 0,

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
  floorFriction: 12,
  landingDamping: 0.2,
  liftThreshold: 0.6,
  settledAging: 0.35,
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
  depthShrink: 0.5,
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
  private grains!: ReturnType<typeof draw>
  private smokeDraw!: ReturnType<typeof draw>
  private composite!: ReturnType<typeof effect>
  private linear!: ReturnType<typeof sampler>

  private frameIndex = 0
  private burstId = 0
  private burstPos: [number, number] = [0, 0]
  private burstFraction = 0
  private burstSpeed = 0

  private pointer: [number, number] | null = null
  private lastPointer: [number, number] | null = null
  private mouseVel: [number, number] = [0, 0]

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

    this.sim = compute(this.gpu, simWgsl, { label: 'chalk-sim', set: { particles: state } })
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

      if (!p.paused) {
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
            turbulence: p.turbulence,
            eddyDecay: Math.max(p.eddyDecay, 0.01),
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
            mouse: this.pointer ?? [1e4, 1e4],
            mouseVel: this.mouseVel,
            mouseRadius: p.mouseRadius,
            mouseStrength: p.mouseStrength,
            mouseSwirl: p.mouseSwirl,
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
            floorY: p.floor ? p.floorY : -100,
            floorDepth: p.floorDepth,
            floorFriction: p.floorFriction,
            landingDamping: p.landingDamping,
            liftThreshold: p.liftThreshold,
            settledAging: p.settledAging,
          },
        })
        // Dispatch submits on its own, ahead of this frame's render passes.
        this.sim.dispatch(Math.ceil(count / WORKGROUP))
      }

      const dpr = this.output.dpr
      const lightPos: [number, number] = [p.lightX * aspect, p.lightY]
      const view = {
        aspect,
        time: simTime,
        grainOpacity: p.grainOpacity,
        cloudOpacity: p.cloudOpacity,
        cloudGrowth: p.cloudGrowth,
        depthFade: p.depthFade,
        depthShrink: p.depthShrink,
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
          floorY: p.floorY + p.floorDepth * 0.5,
          floorGlow: p.floor ? p.floorGlow : 0,
          floorGlowHeight: p.floorGlowHeight,
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
  }

  /** Burst from the centre of an element, e.g. behind a headline or product shot. */
  burstAtElement(el: Element, strength = 1): void {
    const r = el.getBoundingClientRect()
    this.burstAt(r.left + r.width / 2, r.top + r.height / 2, strength)
  }

  dispose(): void {
    this.loop?.stop()
    window.removeEventListener('pointermove', this.onPointerMove)
    window.removeEventListener('pointerdown', this.onPointerDown)
    document.removeEventListener('pointerleave', this.onPointerLeave)
    this.gpu?.dispose()
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
    this.composite.set({ src: this.accum, smoke: this.smoke, samp: this.linear })
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
