import GUI from 'lil-gui'
import { type ChalkDust, defaultParams, MAX_PARTICLES } from './ChalkDust'

export function createChalkGui(dust: ChalkDust, onBurst: () => void): GUI {
  const p = dust.params
  const gui = new GUI({ title: 'Chalk' })

  gui.add(p, 'count', 1000, MAX_PARTICLES, 1000).name('particles')
  gui.add(p, 'paused')
  gui.add(p, 'timeScale', 0, 2, 0.01).name('time scale (slow-mo)')
  gui.add({ burst: onBurst }, 'burst').name('burst at headline')

  const physics = gui.addFolder('Physics')
  physics.add(p, 'gravity', 0, 1.5, 0.01)
  physics.add(p, 'grainDrag', 0.1, 10, 0.1).name('grain drag')
  physics.add(p, 'cloudDrag', 0.1, 10, 0.1).name('cloud drag')
  physics.add(p, 'cloudLift', 0, 1.2, 0.01).name('cloud lift')
  physics.add(p, 'heavySize', 1, 20, 0.1).name('heavy size px')
  physics.add(p, 'subgrid', 0, 2, 0.01).name('small eddies')
  physics.add(p, 'stillAir', 0, 0.3, 0.001).name('room air motion')
  physics.add(p, 'diffusion', 0, 0.3, 0.001).name('smoke diffusion')
  physics.add(p, 'noiseScale', 0.1, 8, 0.01).name('eddy scale')
  physics.add(p, 'noiseSpeed', 0, 1, 0.01).name('eddy speed')

  const air = gui.addFolder('Air (fluid)')
  air.add(p, 'fluidRes', 16, 256, 1).name('grid rows')
  air.add(p, 'airDamping', 0, 5, 0.01).name('damping')
  air.add(p, 'vorticity', 0, 60, 0.1).name('vorticity')
  air.add(p, 'pressureIterations', 2, 80, 2).name('pressure iterations')
  air.add(p, 'pressureDecay', 0, 1, 0.01).name('pressure warm start')

  const population = gui.addFolder('Population')
  population.add(p, 'cloudRatio', 0, 0.5, 0.01).name('cloud ratio')
  population.add(p, 'lifeMin', 0.2, 10, 0.1).name('life min (s)')
  population.add(p, 'lifeMax', 0.2, 20, 0.1).name('life max (s)')
  population.add(p, 'grainSizeMin', 0.2, 5, 0.1).name('grain min px')
  population.add(p, 'grainSizeMax', 0.5, 12, 0.1).name('grain max px')
  population.add(p, 'cloudSizeMin', 5, 300, 1).name('cloud min px')
  population.add(p, 'cloudSizeMax', 10, 500, 1).name('cloud max px')
  population.add(p, 'chunkChance', 0, 0.2, 0.001).name('chunk chance')
  population.add(p, 'chunkSizeMin', 2, 20, 0.1).name('chunk min px')
  population.add(p, 'chunkSizeMax', 2, 30, 0.1).name('chunk max px')

  const emitter = gui.addFolder('Ambient emitter')
  emitter.add(p, 'emitRate', 0, 5, 0.01).name('respawn rate')
  emitter.add(p, 'emitSpeed', 0, 1.5, 0.01).name('speed')
  emitter.add(p, 'emitterX', -1, 1, 0.01).name('x')
  emitter.add(p, 'emitterY', -1.2, 1, 0.01).name('y')
  emitter.add(p, 'emitterWidth', 0, 1.5, 0.01).name('width')
  emitter.add(p, 'emitterHeight', 0, 1, 0.01).name('height')

  const interaction = gui.addFolder('Interaction')
  interaction.add(p, 'mouseRadius', 0.02, 1, 0.01).name('cursor radius')
  interaction.add(p, 'mouseStrength', 0, 4, 0.01).name('cursor push')
  interaction.add(p, 'mouseDrag', 0, 100, 0.1).name('cursor grip')

  const burst = gui.addFolder('Burst')
  burst.add(p, 'burstFraction', 0, 1, 0.01).name('particle share')
  burst.add(p, 'burstSpeed', 0, 4, 0.01).name('speed')
  burst.add(p, 'burstRadius', 0, 0.5, 0.01).name('radius')
  burst.add(p, 'clapAngle', -90, 90, 1).name('clap jet angle °')
  burst.add(p, 'jetSpread', 0, 1.5, 0.01).name('jet spread')
  burst.add(p, 'clapPuff', 0, 1, 0.01).name('omni puff share')
  burst.add(p, 'burstLift', -0.5, 1, 0.01).name('upward lift')
  burst.add(p, 'burstAir', 0, 3, 0.01).name('air jet')
  burst.add(p, 'burstAirRadius', 0.02, 0.5, 0.01).name('air jet radius')

  const clumps = gui.addFolder('Clumps')
  clumps.add(p, 'clumpChance', 0, 1, 0.01).name('clump share')
  clumps.add(p, 'clumpRadius', 0.002, 0.1, 0.001).name('clump radius')
  clumps.add(p, 'clumpTilt', -60, 30, 1).name('launch tilt °')
  clumps.add(p, 'clumpBreakMin', 0, 3, 0.01).name('core holds min (s)')
  clumps.add(p, 'clumpBreakMax', 0, 5, 0.01).name('core holds max (s)')
  clumps.add(p, 'clumpGravity', 0, 6, 0.01).name('clump weight')
  clumps.add(p, 'clumpDrag', 0, 3, 0.01).name('clump drag')
  clumps.add(p, 'shatterSpeed', 0, 1, 0.01).name('shatter speed')
  clumps.add(p, 'liftThreshold', 0, 5, 0.01).name('cursor break threshold')

  const light = gui.addFolder('Lighting')
  light.add(p, 'lightX', -1.2, 1.2, 0.01).name('light x')
  light.add(p, 'lightY', -1.2, 1.2, 0.01).name('light y')
  light.add(p, 'lightRadius', 0.1, 3, 0.01).name('light radius')
  light.add(p, 'lightIntensity', 0, 8, 0.01).name('light intensity')
  light.add(p, 'ambientLight', 0, 1.5, 0.01).name('ambient')
  light.add(p, 'hazeScatter', 0, 4, 0.01).name('haze scatter')
  light.add(p, 'atmosphere', 0, 0.5, 0.001).name('air glow')
  light.add(p, 'floorGlow', 0, 1, 0.01).name('floor glow')
  light.add(p, 'floorGlowHeight', 0.01, 0.5, 0.01).name('floor glow height')
  light.add(p, 'glintChance', 0, 0.5, 0.01).name('glint chance')
  light.add(p, 'glintBoost', 1, 8, 0.1).name('glint boost')

  const floor = gui.addFolder('Floor')
  floor.add(p, 'floor').name('enabled')
  floor.add(p, 'floorY', -1, 0.5, 0.01).name('height')
  floor.add(p, 'floorDepth', 0, 1, 0.01).name('depth')
  floor.add(p, 'perspective', 0.05, 4, 0.01).name('perspective')
  floor.add(p, 'floorFriction', 0, 40, 0.1).name('friction')
  floor.add(p, 'landingDamping', 0, 1, 0.01).name('landing slide')
  floor.add(p, 'floorFlatten', 0.1, 1, 0.01).name('flatten')
  floor.add(p, 'floorShade', 0, 1.5, 0.01).name('shade')
  floor.add(p, 'settledAging', 0, 1, 0.01).name('settled aging')
  floor.add(p, 'floorBuildup', 0, 4, 0.01).name('buildup')
  floor.add(p, 'floorTexture', 0, 1, 0.01).name('dust specks')
  floor.add({ sweep: () => dust.clearFloor() }, 'sweep').name('sweep floor')

  const look = gui.addFolder('Look')
  look.add(p, 'renderScale', 0.25, 1, 0.05).name('render scale')
  look.add(p, 'smokeScale', 0.1, 1, 0.01).name('smoke render scale')
  look.add(p, 'sizeScale', 0.2, 3, 0.01).name('size scale')
  look.add(p, 'grainOpacity', 0, 2, 0.01).name('grain opacity')
  look.add(p, 'cloudOpacity', 0, 0.5, 0.001).name('cloud opacity')
  look.add(p, 'cloudGrowth', 0, 5, 0.01).name('cloud growth')
  look.add(p, 'wispiness', 0, 1, 0.01)
  look.add(p, 'speckle', 0, 1, 0.01).name('powder speckle')
  look.add(p, 'speckleScale', 1, 30, 0.1).name('speckle scale')
  look.add(p, 'depthFade', 0, 1, 0.01).name('depth fade')
  look.add(p, 'shutter', 0, 0.1, 0.001).name('motion blur (s)')
  look.add(p, 'maxStreak', 0, 60, 0.5).name('max streak px')
  look.add(p, 'focusDepth', 0, 1, 0.01).name('focus depth')
  look.add(p, 'dofAmount', 0, 40, 0.1).name('depth of field px')

  const tone = gui.addFolder('Tone')
  tone.addColor(p, 'tint')
  tone.add(p, 'exposure', 0.1, 6, 0.01)
  tone.add(p, 'contrast', 0.3, 3, 0.01)
  tone.add(p, 'grain', 0, 1, 0.01).name('film grain')

  for (const folder of [physics, air, population, emitter, interaction, burst, clumps, light, floor, look, tone]) {
    folder.close()
  }

  gui.add({ copy: () => navigator.clipboard.writeText(JSON.stringify(p, null, 2)) }, 'copy').name('copy settings JSON')
  gui.add(
    {
      reset: () => {
        Object.assign(p, defaultParams)
        gui.controllersRecursive().forEach((c) => c.updateDisplay())
      },
    },
    'reset',
  ).name('reset')

  return gui
}
