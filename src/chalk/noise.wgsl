// Bakes a seamlessly tiling noise texture once at startup, so smoke fragments sample it
// instead of evaluating value noise per pixel. r = wisp fbm, g = fine powder speckle.

fn hash21(p: vec2f) -> f32 {
  var q = fract(p * vec2f(123.34, 456.21));
  q += dot(q, q + 45.32);
  return fract(q.x * q.y);
}

// Value noise whose lattice wraps every `period` cells, so the tile repeats without seams.
fn pnoise(p: vec2f, period: f32) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  let a = hash21(i % period);
  let b = hash21((i + vec2f(1, 0)) % period);
  let c = hash21((i + vec2f(0, 1)) % period);
  let d = hash21((i + vec2f(1, 1)) % period);
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

@fragment
fn fs_main(@location(0) uv: vec2f) -> @location(0) vec4f {
  let wisps = pnoise(uv * 4.0, 4.0) * 0.55 + pnoise(uv * 8.0 + 3.0, 8.0) * 0.3 + pnoise(uv * 16.0 + 7.0, 16.0) * 0.15;
  let speck = pnoise(uv * 32.0, 32.0);
  return vec4f(wisps, speck, 0.0, 1.0);
}
