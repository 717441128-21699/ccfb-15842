/**
 * 带纹路真值的牛鼻图像合成器。
 *
 * 同一头牛（SyntheticIdentity）的珠状纹、鼻镜沟、鼻孔位置是固定的「真值」，
 * 每次采集（CaptureParams）只改变拍摄角度、光照、湿度反光、模糊、噪声，
 * 从而可以在 Node 中大规模构造「同牛不同次 / 不同牛」的评测集。
 *
 * 管线：身份真值 -> 标准鼻镜纹理（256x192）-> 3D 透视投影（偏航/俯仰/翻滚）
 *      -> 方向光照 + 湿润镜面斑 -> 高斯模糊 + 传感器噪声 -> 8 位采集图。
 */

import {
  GrayImage,
  RgbaImage,
  Point,
  NostrilPair,
  findHomography,
  applyMat3,
  Mat3,
  sampleGrayBilinear,
} from './geometry';
import { mulberry32, gaussian, ValueNoise2d, hashStringToUint32 } from './rng';

export const CANON_W = 256;
export const CANON_H = 192;
export const CANON_CX = CANON_W / 2;
export const CANON_CY = CANON_H / 2;

export interface BeadSpec {
  x: number;
  y: number;
  r: number;
  /** 身份确定性的个体属性：暗环强度、亮边强度、椭圆度、朝向。 */
  ringStrength: number;
  rimStrength: number;
  ellipticity: number;
  angle: number;
}

export interface NostrilSpec {
  cx: number;
  cy: number;
  rx: number;
  ry: number;
  angle: number;
}

export interface SyntheticIdentity {
  cowId: string;
  seed: number;
  beads: BeadSpec[];
  /** 两条主鼻镜沟，每条为折线点集。 */
  grooves: Point[][];
  grooveWidth: number;
  nostrils: NostrilSpec[]; // 长度 2，left/right
  muzzlePoly: Point[];
  pigmentSeed: number;
  fineSeed: number;
  /** 身份确定性的珠点密度场（簇/空隙模式），低尺度、跨采集稳定。 */
  densitySeed: number;
}

export interface CaptureParams {
  seed: number;
  yawDeg: number;
  pitchDeg: number;
  rollDeg: number;
  scale: number;
  tx: number;
  ty: number;
  lightAngleDeg: number;
  lightStrength: number;
  gain: number;
  bias: number;
  /** 0..1，控制镜面反光斑数量与强度、整体湿亮层。 */
  humidity: number;
  specularCount: number;
  blurSigma: number;
  noiseSigma: number;
  outW: number;
  outH: number;
}

export interface SynthesizedCapture {
  image: RgbaImage;
  /** 输出图坐标下的鼻孔真值。 */
  nostrils: NostrilPair;
  /** 输出图坐标下的鼻镜四边形真值（左上、右上、右下、左下）。 */
  muzzleQuad: Point[];
  /** 标准坐标 -> 输出图的单应矩阵。 */
  homography: Mat3;
  params: CaptureParams;
}

const DEG = Math.PI / 180;

/** 由牛 ID 生成确定性身份：珠状纹分布、沟形、鼻孔、鼻镜轮廓。 */
export function generateIdentity(cowId: string): SyntheticIdentity {
  const seed = hashStringToUint32(cowId);
  const rng = mulberry32(seed);

  // 鼻孔：中心位置与形状带个体差异。
  const nostrils: NostrilSpec[] = [
    {
      cx: 88 + gaussian(rng) * 3,
      cy: 126 + gaussian(rng) * 3,
      rx: 29 + rng() * 4,
      ry: 21 + rng() * 3,
      angle: -8 + gaussian(rng) * 3,
    },
    {
      cx: 168 + gaussian(rng) * 3,
      cy: 126 + gaussian(rng) * 3,
      rx: 29 + rng() * 4,
      ry: 21 + rng() * 3,
      angle: 8 + gaussian(rng) * 3,
    },
  ];

  // 鼻镜外轮廓：带扰动的圆角多边形。
  const muzzlePoly = makeMuzzlePolygon(rng);

  // 珠状纹：在鼻镜区域内做带最小间距的拒绝采样，密度受身份场调制。
  const densityNoise = new ValueNoise2d(mulberry32((rng() * 0xffffffff) >>> 0), 10);
  const beads = sampleBeads(rng, muzzlePoly, nostrils, densityNoise);

  // 两条主沟：从两鼻孔内侧上缘向鼻镜顶部中线弯曲延伸。
  const grooves = makeGrooves(rng, nostrils);
  const grooveWidth = 2.2 + rng() * 1.4;

  return {
    cowId,
    seed,
    beads,
    grooves,
    grooveWidth,
    nostrils,
    muzzlePoly,
    pigmentSeed: (rng() * 0xffffffff) >>> 0,
    fineSeed: (rng() * 0xffffffff) >>> 0,
    densitySeed: densityNoise.size,
  };
}

function makeMuzzlePolygon(rng: () => number): Point[] {
  const pts: Point[] = [];
  const cx = CANON_CX;
  const cy = 104;
  const rx = 104;
  const ry = 78;
  const n = 28;
  const phase1 = rng() * Math.PI * 2;
  const phase2 = rng() * Math.PI * 2;
  for (let i = 0; i < n; i++) {
    const t = (i / n) * Math.PI * 2;
    const wobble = 1 + 0.035 * Math.sin(3 * t + phase1) + 0.02 * Math.sin(7 * t + phase2);
    pts.push({ x: cx + Math.cos(t) * rx * wobble, y: cy + Math.sin(t) * ry * wobble });
  }
  return pts;
}

function pointInPoly(p: Point, poly: Point[]): boolean {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const xi = poly[i].x;
    const yi = poly[i].y;
    const xj = poly[j].x;
    const yj = poly[j].y;
    const hit = yi > p.y !== yj > p.y && p.x < ((xj - xi) * (p.y - yi)) / (yj - yi) + xi;
    if (hit) inside = !inside;
  }
  return inside;
}

function inNostril(x: number, y: number, n: NostrilSpec, margin = 1.5): boolean {
  const c = Math.cos(-n.angle * DEG);
  const s = Math.sin(-n.angle * DEG);
  const dx = x - n.cx;
  const dy = y - n.cy;
  const lx = c * dx - s * dy;
  const ly = s * dx + c * dy;
  return (lx * lx) / ((n.rx + margin) ** 2) + (ly * ly) / ((n.ry + margin) ** 2) < 1;
}

function sampleBeads(
  rng: () => number,
  muzzlePoly: Point[],
  nostrils: NostrilSpec[],
  densityNoise: ValueNoise2d,
): BeadSpec[] {
  const beads: BeadSpec[] = [];
  const target = 460;
  let attempts = 0;
  while (beads.length < target && attempts < target * 80) {
    attempts++;
    const x = 26 + rng() * (CANON_W - 52);
    const y = 30 + rng() * (CANON_H - 56);
    if (!pointInPoly({ x, y }, muzzlePoly)) continue;
    if (nostrils.some((n) => inNostril(x, y, n, 3))) continue;
    // 身份密度场：局部接受概率 0.35..1.15，形成稳定的簇与空隙。
    const density = 0.75 + 0.4 * densityNoise.fbm(x / CANON_W * 3, y / CANON_H * 3, 3);
    if (rng() > Math.min(1.15, Math.max(0.35, density))) continue;
    const r = 2.4 + rng() * 2.3;
    // 最小间距约束，保证珠粒可分辨且分布自然。
    let ok = true;
    for (const b of beads) {
      const d2 = (b.x - x) ** 2 + (b.y - y) ** 2;
      if (d2 < (0.82 * (b.r + r)) ** 2) {
        ok = false;
        break;
      }
    }
    if (ok) {
      beads.push({
        x,
        y,
        r,
        // 每头牛固定、每颗珠不同的形态属性，构成稳定且唯一的微观纹理。
        ringStrength: 0.75 + rng() * 0.55,
        rimStrength: 0.7 + rng() * 0.6,
        ellipticity: 0.82 + rng() * 0.32,
        angle: rng() * Math.PI,
      });
    }
  }
  return beads;
}

function makeGrooves(rng: () => number, nostrils: NostrilSpec[]): Point[][] {
  const result: Point[][] = [];
  const tops: Point[] = [
    { x: CANON_CX - 14 + gaussian(rng) * 4, y: 30 + rng() * 8 },
    { x: CANON_CX + 14 + gaussian(rng) * 4, y: 30 + rng() * 8 },
  ];
  for (let g = 0; g < 2; g++) {
    const n = nostrils[g];
    const start: Point = {
      x: n.cx + (g === 0 ? 10 : -10),
      y: n.cy - n.ry - 2,
    };
    const end = tops[g];
    const pts: Point[] = [];
    const bend = (g === 0 ? -1 : 1) * (6 + rng() * 8);
    const steps = 14;
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      const x = start.x + (end.x - start.x) * t + bend * Math.sin(Math.PI * t);
      const y = start.y + (end.y - start.y) * t;
      pts.push({ x, y });
    }
    result.push(pts);
  }
  return result;
}

/** 渲染无采集退化的标准鼻镜纹理（含真值掩膜）。 */
export function renderCanonical(id: SyntheticIdentity): { gray: GrayImage; muzzleMask: Uint8Array } {
  const W = CANON_W;
  const H = CANON_H;
  const buf = new Float32Array(W * H);
  const mask = new Uint8Array(W * H);
  const pigment = new ValueNoise2d(mulberry32(id.pigmentSeed), 16);
  const fine = new ValueNoise2d(mulberry32(id.fineSeed), 32);

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      if (!pointInPoly({ x, y }, id.muzzlePoly)) {
        buf[i] = 26; // 鼻镜外的暗背景基准
        continue;
      }
      mask[i] = 1;
      const base = 138;
      // 皮肤色斑与微纹理压低幅度：它们是跨采集噪声，不应压过珠点身份信号。
      const blotch = pigment.fbm(x / W, y / H, 3) * 8;
      const skin = fine.fbm(x / W * 6, y / H * 6, 3) * 3;
      buf[i] = base + blotch + skin;
    }
  }

  // 珠状纹：亮中心 + 暗环，强对比、身份确定性的椭圆度/朝向/强度，是主导身份特征。
  const lightAng = -Math.PI / 4;
  for (const b of id.beads) {
    const reach = b.r * 1.6 + 2;
    const x0 = Math.max(0, Math.floor(b.x - reach));
    const x1 = Math.min(W - 1, Math.ceil(b.x + reach));
    const y0 = Math.max(0, Math.floor(b.y - reach));
    const y1 = Math.min(H - 1, Math.ceil(b.y + reach));
    const cosA = Math.cos(-b.angle);
    const sinA = Math.sin(-b.angle);
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const i = y * W + x;
        if (!mask[i]) continue;
        const dx0 = x - b.x;
        const dy0 = y - b.y;
        // 旋转到珠点自身朝向，再按椭圆度拉伸。
        const lx = (cosA * dx0 - sinA * dy0) / b.ellipticity;
        const ly = sinA * dx0 + cosA * dy0;
        const d = Math.hypot(lx, ly);
        // 暗环。
        const ring = Math.exp(-((d - b.r) ** 2) / (2 * 0.85 ** 2));
        buf[i] -= ring * 46 * b.ringStrength;
        // 亮中心（各向同性，跨光照稳定）+ 方向性微高光。
        const center = Math.exp(-(d * d) / (2 * (b.r * 0.55) ** 2));
        buf[i] += center * 30;
        const ang = Math.atan2(ly, lx);
        const rim = Math.exp(-((d - (b.r - 0.7)) ** 2) / (2 * 0.8 ** 2));
        const dir = Math.max(0, Math.cos(ang - lightAng));
        buf[i] += rim * dir * 20 * b.rimStrength;
      }
    }
  }

  // 鼻镜沟：沿折线的暗带。
  for (const groove of id.grooves) {
    for (let s = 0; s < groove.length - 1; s++) {
      drawSegmentShade(buf, mask, W, H, groove[s], groove[s + 1], id.grooveWidth, -30);
    }
  }

  // 鼻孔：深色椭圆 + 边缘柔化。
  for (const n of id.nostrils) {
    const x0 = Math.max(0, Math.floor(n.cx - n.rx - 2));
    const x1 = Math.min(W - 1, Math.ceil(n.cx + n.rx + 2));
    const y0 = Math.max(0, Math.floor(n.cy - n.ry - 2));
    const y1 = Math.min(H - 1, Math.ceil(n.cy + n.ry + 2));
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const i = y * W + x;
        if (!mask[i]) continue;
        const q = nostrilQ(x, y, n);
        if (q < 1) {
          const edge = Math.max(0, Math.min(1, (1.25 - q) / 0.25));
          buf[i] = buf[i] * (1 - edge) + 18 * edge;
        }
      }
    }
  }

  const gray = new GrayImage(W, H);
  for (let i = 0; i < buf.length; i++) {
    gray.data[i] = Math.max(0, Math.min(255, Math.round(buf[i])));
  }
  return { gray, muzzleMask: mask };
}

function nostrilQ(x: number, y: number, n: NostrilSpec): number {
  const c = Math.cos(-n.angle * DEG);
  const s = Math.sin(-n.angle * DEG);
  const dx = x - n.cx;
  const dy = y - n.cy;
  const lx = c * dx - s * dy;
  const ly = s * dx + c * dy;
  return (lx * lx) / (n.rx * n.rx) + (ly * ly) / (n.ry * n.ry);
}

function drawSegmentShade(
  buf: Float32Array,
  mask: Uint8Array,
  W: number,
  H: number,
  a: Point,
  b: Point,
  width: number,
  shade: number,
): void {
  const minX = Math.max(0, Math.floor(Math.min(a.x, b.x) - width - 1));
  const maxX = Math.min(W - 1, Math.ceil(Math.max(a.x, b.x) + width + 1));
  const minY = Math.max(0, Math.floor(Math.min(a.y, b.y) - width - 1));
  const maxY = Math.min(H - 1, Math.ceil(Math.max(a.y, b.y) + width + 1));
  const vx = b.x - a.x;
  const vy = b.y - a.y;
  const len2 = vx * vx + vy * vy || 1e-6;
  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      const t = Math.max(0, Math.min(1, ((x - a.x) * vx + (y - a.y) * vy) / len2));
      const px = a.x + vx * t;
      const py = a.y + vy * t;
      const d = Math.hypot(x - px, y - py);
      const w = Math.exp(-(d * d) / (2 * (width / 2) ** 2));
      const i = y * W + x;
      if (mask[i]) buf[i] += shade * w;
    }
  }
}

/**
 * 简单平面透视投影：把标准纹理当作 z=0 平面，先绕 Y/X 轴旋转再针孔投影，
 * 叠加翻滚、缩放、平移。整体等价于一个单应矩阵，地标用同一矩阵变换得到真值。
 */
function buildCaptureHomography(p: CaptureParams): Mat3 {
  const f = 620;
  const yaw = p.yawDeg * DEG;
  const pitch = p.pitchDeg * DEG;
  const roll = p.rollDeg * DEG;
  const cosR = Math.cos(roll);
  const sinR = Math.sin(roll);

  const project = (x: number, y: number): Point => {
    let X = x - CANON_CX;
    let Y = y - CANON_CY;
    // 绕 Y 轴（偏航）
    const X1 = Math.cos(yaw) * X;
    const Z1 = -Math.sin(yaw) * X;
    // 绕 X 轴（俯仰）
    const Y2 = Math.cos(pitch) * Y - Math.sin(pitch) * Z1;
    const Z2 = Math.sin(pitch) * Y + Math.cos(pitch) * Z1;
    const denom = f + Z2;
    const u = (f * X1) / denom;
    const v = (f * Y2) / denom;
    // 翻滚 + 缩放 + 平移
    const u2 = p.scale * (cosR * u - sinR * v);
    const v2 = p.scale * (sinR * u + cosR * v);
    return { x: u2 + p.outW / 2 + p.tx, y: v2 + p.outH / 2 + p.ty };
  };

  const src: Point[] = [
    { x: 0, y: 0 },
    { x: CANON_W - 1, y: 0 },
    { x: CANON_W - 1, y: CANON_H - 1 },
    { x: 0, y: CANON_H - 1 },
  ];
  const dst = src.map((q) => project(q.x, q.y));
  return findHomography(src, dst);
}

/** 完整采集合成：标准纹理 -> 透视 -> 光照 -> 湿反光 -> 模糊 -> 噪声。 */
export function renderCapture(id: SyntheticIdentity, p: CaptureParams): SynthesizedCapture {
  const { gray: canon, muzzleMask } = renderCanonical(id);
  const H = buildCaptureHomography(p);
  const W = p.outW;
  const Ht = p.outH;

  // 透视变形到输出坐标系（浮点缓冲，保留动态范围供光照处理）。
  const warped = new Float32Array(W * Ht);
  const warpedMask = new Uint8Array(W * Ht);
  for (let y = 0; y < Ht; y++) {
    for (let x = 0; x < W; x++) {
      const inv = invertProject(H, { x, y });
      if (inv.x < 0 || inv.y < 0 || inv.x >= CANON_W || inv.y >= CANON_H) continue;
      const ix = Math.max(0, Math.min(CANON_W - 1, Math.floor(inv.x)));
      const iy = Math.max(0, Math.min(CANON_H - 1, Math.floor(inv.y)));
      if (!muzzleMask[iy * CANON_W + ix]) continue;
      const v = sampleGrayBilinear(canon, inv.x, inv.y);
      warped[y * W + x] = v;
      warpedMask[y * W + x] = 1;
    }
  }

  const rng = mulberry32(p.seed ^ 0x9e3779b9);

  // 方向光照：梯度 + 大尺度起伏，只作用于鼻镜区域。
  const la = p.lightAngleDeg * DEG;
  const lx = Math.cos(la);
  const ly = Math.sin(la);
  const shadeNoise = new ValueNoise2d(mulberry32(p.seed ^ 0x12345678), 12);
  for (let y = 0; y < Ht; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      if (!warpedMask[i]) continue;
      const nx = (x / W - 0.5) * 2;
      const ny = (y / Ht - 0.5) * 2;
      const grad = lx * nx + ly * ny;
      const cloud = shadeNoise.fbm(x / W * 2, y / Ht * 2, 2) * 0.35;
      const shade = 1 + p.lightStrength * (0.7 * grad + cloud);
      warped[i] = warped[i] * shade * p.gain + p.bias;
    }
  }

  // 湿度：整体湿亮层 + 离散镜面高光斑（高斯椭圆）。
  if (p.humidity > 0) {
    for (let y = 0; y < Ht; y++) {
      for (let x = 0; x < W; x++) {
        const i = y * W + x;
        if (!warpedMask[i]) continue;
        warped[i] += p.humidity * 10 * (0.6 + 0.4 * (lx * (x / W) + ly * (y / Ht)));
      }
    }
    for (let k = 0; k < p.specularCount; k++) {
      // 高光斑落在鼻镜内：先在标准坐标取点再投影。
      const bead = id.beads[Math.floor(rng() * id.beads.length)];
      const c = applyMat3(H, {
        x: bead.x + gaussian(rng) * 14,
        y: bead.y + gaussian(rng) * 10,
      });
      const rx = (5 + rng() * 10) * p.scale;
      const ry = (3 + rng() * 7) * p.scale;
      const ang = rng() * Math.PI;
      const amp = (90 + rng() * 90) * (0.35 + p.humidity);
      addGaussianBlob(warped, warpedMask, W, Ht, c, rx, ry, ang, amp);
    }
  }

  // 镜头模糊（分离式高斯）。
  let blurred: Float32Array = warped;
  if (p.blurSigma > 0.15) {
    blurred = gaussianBlur(warped, W, Ht, p.blurSigma);
  }

  // 传感器噪声 + 背景暗纹理，最后量化到 8 位。
  const out = new RgbaImage(W, Ht);
  const bgNoise = new ValueNoise2d(mulberry32(p.seed ^ 0xabcdef01), 8);
  for (let y = 0; y < Ht; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      let v: number;
      if (warpedMask[i]) {
        v = blurred[i] + gaussian(rng) * p.noiseSigma;
      } else {
        v = 24 + bgNoise.fbm(x / W * 3, y / Ht * 3, 2) * 10 + gaussian(rng) * 4;
      }
      v = Math.max(0, Math.min(255, Math.round(v)));
      const o = i * 4;
      out.data[o] = v;
      out.data[o + 1] = v;
      out.data[o + 2] = v;
      out.data[o + 3] = 255;
    }
  }

  const nostrils: NostrilPair = {
    left: applyMat3(H, { x: id.nostrils[0].cx, y: id.nostrils[0].cy }),
    right: applyMat3(H, { x: id.nostrils[1].cx, y: id.nostrils[1].cy }),
  };
  const muzzleQuad = [
    applyMat3(H, { x: 0, y: 0 }),
    applyMat3(H, { x: CANON_W - 1, y: 0 }),
    applyMat3(H, { x: CANON_W - 1, y: CANON_H - 1 }),
    applyMat3(H, { x: 0, y: CANON_H - 1 }),
  ];

  return { image: out, nostrils, muzzleQuad, homography: H, params: p };
}

/** 单应矩阵反向映射（dst -> src）。 */
function invertProject(H: Mat3, p: Point): Point {
  const [a, b, c, d, e, f, g, h, i] = H;
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const D = -(b * i - c * h);
  const E = a * i - c * g;
  const F = -(a * h - b * g);
  const G = b * f - c * e;
  const Hh = -(a * f - c * d);
  const I = a * e - b * d;
  const det = a * A + b * B + c * C || 1e-12;
  const inv = [A / det, D / det, G / det, B / det, E / det, Hh / det, C / det, F / det, I / det];
  const w = inv[6] * p.x + inv[7] * p.y + inv[8];
  return { x: (inv[0] * p.x + inv[1] * p.y + inv[2]) / w, y: (inv[3] * p.x + inv[4] * p.y + inv[5]) / w };
}

function addGaussianBlob(
  buf: Float32Array,
  mask: Uint8Array,
  W: number,
  H: number,
  c: Point,
  rx: number,
  ry: number,
  ang: number,
  amp: number,
): void {
  const cos = Math.cos(-ang);
  const sin = Math.sin(-ang);
  const R = Math.max(rx, ry) * 2.2;
  const x0 = Math.max(0, Math.floor(c.x - R));
  const x1 = Math.min(W - 1, Math.ceil(c.x + R));
  const y0 = Math.max(0, Math.floor(c.y - R));
  const y1 = Math.min(H - 1, Math.ceil(c.y + R));
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const i = y * W + x;
      if (!mask[i]) continue;
      const dx = x - c.x;
      const dy = y - c.y;
      const lx = cos * dx - sin * dy;
      const ly = sin * dx + cos * dy;
      const q = (lx * lx) / (rx * rx) + (ly * ly) / (ry * ry);
      if (q > 4) continue;
      buf[i] += amp * Math.exp(-q * 1.4);
    }
  }
}

function gaussianBlur(src: Float32Array, W: number, H: number, sigma: number): Float32Array {
  const radius = Math.max(1, Math.ceil(sigma * 3));
  const kernel = new Float32Array(radius * 2 + 1);
  let norm = 0;
  for (let i = -radius; i <= radius; i++) {
    const v = Math.exp(-(i * i) / (2 * sigma * sigma));
    kernel[i + radius] = v;
    norm += v;
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= norm;

  const tmp = new Float32Array(src.length);
  const out = new Float32Array(src.length);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) {
        const xx = Math.min(W - 1, Math.max(0, x + k));
        acc += src[y * W + xx] * kernel[k + radius];
      }
      tmp[y * W + x] = acc;
    }
  }
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) {
        const yy = Math.min(H - 1, Math.max(0, y + k));
        acc += tmp[yy * W + x] * kernel[k + radius];
      }
      out[y * W + x] = acc;
    }
  }
  return out;
}

/** 采集参数档位：投保（温和）、野外（宽扰动）、退化（触发质量门限）。 */
export function makeCaptureParams(
  rng: () => number,
  profile: 'enroll' | 'field' | 'degraded',
  outW = 320,
  outH = 240,
): CaptureParams {
  const base: CaptureParams = {
    seed: (rng() * 0xffffffff) >>> 0,
    yawDeg: 0,
    pitchDeg: 0,
    rollDeg: 0,
    scale: 1,
    tx: 0,
    ty: 0,
    lightAngleDeg: rng() * 360,
    lightStrength: 0.25,
    gain: 1,
    bias: 0,
    humidity: 0.3,
    specularCount: 4,
    blurSigma: 0.8,
    noiseSigma: 5,
    outW,
    outH,
  };
  if (profile === 'enroll') {
    base.yawDeg = gaussian(rng) * 8;
    base.pitchDeg = gaussian(rng) * 6;
    base.rollDeg = gaussian(rng) * 8;
    base.scale = 1.05 + rng() * 0.15;
    base.lightStrength = 0.15 + rng() * 0.15;
    base.gain = 0.95 + rng() * 0.1;
    base.humidity = 0.15 + rng() * 0.3;
    base.specularCount = 2 + Math.floor(rng() * 3);
    base.blurSigma = 0.4 + rng() * 0.5;
    base.noiseSigma = 1.5 + rng() * 2;
  } else if (profile === 'field') {
    base.yawDeg = gaussian(rng) * 16;
    base.pitchDeg = gaussian(rng) * 12;
    base.rollDeg = gaussian(rng) * 14;
    base.scale = 0.92 + rng() * 0.3;
    base.tx = gaussian(rng) * 10;
    base.ty = gaussian(rng) * 8;
    base.lightStrength = 0.2 + rng() * 0.3;
    base.gain = 0.85 + rng() * 0.3;
    base.bias = gaussian(rng) * 12;
    base.humidity = 0.25 + rng() * 0.6;
    base.specularCount = 3 + Math.floor(rng() * 7);
    base.blurSigma = 0.4 + rng() * 0.5;
    base.noiseSigma = 2 + rng() * 3;
  } else {
    // 退化：要么糊到失去纹路，要么反光盖满，供质量门限测试。
    const blurHeavy = rng() > 0.45;
    base.yawDeg = gaussian(rng) * 22;
    base.pitchDeg = gaussian(rng) * 16;
    base.rollDeg = gaussian(rng) * 18;
    base.scale = 0.85 + rng() * 0.35;
    base.tx = gaussian(rng) * 16;
    base.ty = gaussian(rng) * 12;
    base.lightStrength = 0.35 + rng() * 0.3;
    base.gain = 0.8 + rng() * 0.4;
    base.bias = gaussian(rng) * 18;
    base.humidity = 0.6 + rng() * 0.4;
    base.specularCount = blurHeavy ? 6 + Math.floor(rng() * 6) : 14 + Math.floor(rng() * 10);
    base.blurSigma = blurHeavy ? 2.6 + rng() * 1.6 : 1.2 + rng() * 0.8;
    base.noiseSigma = 7 + rng() * 8;
  }
  return base;
}

/** 便捷入口：直接生成一头牛的一张采集图。 */
export function synthesizeCapture(
  cowId: string,
  profile: 'enroll' | 'field' | 'degraded',
  rng: () => number,
  outW = 320,
  outH = 240,
): { identity: SyntheticIdentity; capture: SynthesizedCapture } {
  const identity = generateIdentity(cowId);
  const params = makeCaptureParams(rng, profile, outW, outH);
  return { identity, capture: renderCapture(identity, params) };
}
