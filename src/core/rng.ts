/**
 * 确定性随机数与噪声原语。
 * 合成数据要求「按不同光照/角度/湿度」可复现地生成，全部走种子 PRNG，
 * 禁止 Math.random()，保证同一头牛的纹路真值在任意环境下一致。
 */

/** 32 位字符串哈希（FNV-1a 变体），用于把牛 ID 变成种子。 */
export function hashStringToUint32(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/** mulberry32：小巧、快速、统计特性够用的种子 PRNG，返回 [0,1)。 */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function next() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 从一个种子派生出独立的子流。 */
export function splitRng(rng: () => number, salt: number): () => number {
  const seed = (Math.imul((rng() * 0xffffffff) >>> 0, 2654435761) ^ salt) >>> 0;
  return mulberry32(seed);
}

/** 标准正态采样（Box-Muller）。 */
export function gaussian(rng: () => number): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/** 一维平滑值噪声，周期 [0,1)，插值到任意 x。 */
export function makeValueNoise1d(rng: () => number, length: number): Float32Array {
  const table = new Float32Array(length);
  for (let i = 0; i < length; i++) table[i] = rng() * 2 - 1;
  return table;
}

export function sampleNoise1d(table: Float32Array, x: number): number {
  const n = table.length;
  const p = ((x % 1) + 1) % 1 * n;
  const i = Math.floor(p);
  const f = p - i;
  const a = table[i % n];
  const b = table[(i + 1) % n];
  const t = f * f * (3 - 2 * f);
  return a * (1 - t) + b * t;
}

/** 二维值噪声场（双线性），坐标归一化到 [0,1]^2。 */
export class ValueNoise2d {
  readonly size: number;
  private readonly grid: Float32Array;

  constructor(rng: () => number, size = 32) {
    this.size = size;
    this.grid = new Float32Array(size * size);
    for (let i = 0; i < this.grid.length; i++) this.grid[i] = rng() * 2 - 1;
  }

  sample(x: number, y: number): number {
    const n = this.size;
    const px = ((x % 1) + 1) % 1 * n;
    const py = ((y % 1) + 1) % 1 * n;
    const ix = Math.floor(px);
    const iy = Math.floor(py);
    const fx = px - ix;
    const fy = py - iy;
    const tx = fx * fx * (3 - 2 * fx);
    const ty = fy * fy * (3 - 2 * fy);
    const i0 = iy % n;
    const i1 = (iy + 1) % n;
    const j0 = ix % n;
    const j1 = (ix + 1) % n;
    const v00 = this.grid[i0 * n + j0];
    const v10 = this.grid[i0 * n + j1];
    const v01 = this.grid[i1 * n + j0];
    const v11 = this.grid[i1 * n + j1];
    const a = v00 * (1 - tx) + v10 * tx;
    const b = v01 * (1 - tx) + v11 * tx;
    return a * (1 - ty) + b * ty;
  }

  /** 多频叠加，返回大致落在 [-1,1] 的分形噪声。 */
  fbm(x: number, y: number, octaves = 4): number {
    let amp = 0.55;
    let freq = 1;
    let sum = 0;
    let norm = 0;
    for (let o = 0; o < octaves; o++) {
      sum += amp * this.sample(x * freq, y * freq);
      norm += amp;
      amp *= 0.5;
      freq *= 2;
    }
    return sum / norm;
  }
}
