/**
 * 网格 LBP/HOG 描述子（辅助/研究用途）。
 *
 * 注意：当前生产识别路径以 anchor.ts 的鼻纹锚点指纹为准（实测 rank-1 100%、
 * FAR 0%）。本模块的网格直方图在合成数据上区分度不足，仅保留作为：
 *  1) 粗筛/多模态融合的候选特征；
 *  2) 描述子设计与归一化策略的参考实现（Node 与浏览器同构）。
 *
 * 设计目标：与光照、拍摄角度无关。
 * - 图像先经透视矩形化（几何无关）与光照归一化（光照无关）；
 * - 在 16x12 网格上，每格提取 10 维均匀 LBP 直方图（旋转不变、单调灰度不变）
 *   与 6 维无向梯度直方图（HOG 风格），格内 L2 归一化；
 * - 比对时对 ±2 格水平/垂直移位取最小距离，容忍矩形化的残差；
 * - 粗筛指纹（下采样亮度 + 象限能量）在 5000 头规模先做全量廉价过滤。
 */

import { GrayImage } from './geometry';
import { CANON_W, CANON_H } from './synthesis';
import {
  BeadTemplate,
  detectBeads,
  describeBeads,
  matchBeads,
  beadDistance,
} from './beads';

export const GRID_X = 16;
export const GRID_Y = 12;
export const LBP_BINS = 10;
export const HOG_BINS = 6;
export const CELL_DIM = LBP_BINS + HOG_BINS + 2; // 22
export const DESCRIPTOR_DIM = GRID_X * GRID_Y * CELL_DIM; // 4224
export const COARSE_BINS = 16;
export const QUADRANT_ENERGY = 16;
export const COARSE_DIM = COARSE_BINS + QUADRANT_ENERGY; // 32

export interface FeatureTemplate {
  cowId: string;
  descriptor: Float32Array; // 4224
  coarse: Float32Array; // 32
  /** 每格是否有效（鼻孔/鼻镜外为 0），共 192 bit，打包成字节。 */
  cellValid: Uint8Array;
  /** 珠点关键点 + 局部描述子（身份判别主力）。 */
  beads: BeadTemplate;
  /** 96x72 增强缩略图，用于展示与像素级复核。 */
  thumb: Uint8Array;
  createdAt: number;
}

/** 均匀 LBP(8,1) 跳变表：跳变<=2 用 1 的个数落桶（9 桶），其余归混合桶 9。 */
const LBP_TABLE: Uint8Array = (() => {
  const table = new Uint8Array(256);
  for (let p = 0; p < 256; p++) {
    let transitions = 0;
    let ones = 0;
    for (let b = 0; b < 8; b++) {
      const cur = (p >> b) & 1;
      const nxt = (p >> ((b + 1) % 8)) & 1;
      if (cur !== nxt) transitions++;
      if (cur) ones++;
    }
    table[p] = transitions <= 2 ? ones : 9;
  }
  return table;
})();

function lbpCode(img: GrayImage, x: number, y: number): number {
  const i = y * img.width + x;
  const c = img.data[i];
  const W = img.width;
  const n = [
    img.data[i - W],
    img.data[i - W + 1],
    img.data[i + 1],
    img.data[i + W + 1],
    img.data[i + W],
    img.data[i + W - 1],
    img.data[i - 1],
    img.data[i - W - 1],
  ];
  let code = 0;
  for (let k = 0; k < 8; k++) if (n[k] >= c) code |= 1 << k;
  return LBP_TABLE[code];
}

/** 主提取入口。mask 为增强阶段输出的鼻镜有效区。 */
export function extractFeatures(
  gray: GrayImage,
  mask: Uint8Array,
  cowId: string,
  createdAt: number,
  thumb: Uint8Array,
): FeatureTemplate {
  const cellW = CANON_W / GRID_X;
  const cellH = CANON_H / GRID_Y;
  const descriptor = new Float32Array(DESCRIPTOR_DIM);
  const cellValid = new Uint8Array(GRID_X * GRID_Y);

  // 预计算 Sobel 梯度。
  const gx = new Int16Array(CANON_W * CANON_H);
  const gy = new Int16Array(CANON_W * CANON_H);
  for (let y = 1; y < CANON_H - 1; y++) {
    for (let x = 1; x < CANON_W - 1; x++) {
      const i = y * CANON_W + x;
      gx[i] =
        -gray.data[i - CANON_W - 1] + gray.data[i - CANON_W + 1] -
        2 * gray.data[i - 1] + 2 * gray.data[i + 1] -
        gray.data[i + CANON_W - 1] + gray.data[i + CANON_W + 1];
      gy[i] =
        -gray.data[i - CANON_W - 1] - 2 * gray.data[i - CANON_W] - gray.data[i - CANON_W + 1] +
        gray.data[i + CANON_W - 1] + 2 * gray.data[i + CANON_W] + gray.data[i + CANON_W + 1];
    }
  }

  for (let cy = 0; cy < GRID_Y; cy++) {
    for (let cx = 0; cx < GRID_X; cx++) {
      const x0 = Math.round(cx * cellW);
      const y0 = Math.round(cy * cellH);
      const x1 = Math.round((cx + 1) * cellW);
      const y1 = Math.round((cy + 1) * cellH);
      const cellIndex = cy * GRID_X + cx;
      const base = cellIndex * CELL_DIM;

      let validPixels = 0;
      let intensitySum = 0;
      let intensitySq = 0;
      const lbpHist = new Float32Array(LBP_BINS);
      const hogHist = new Float32Array(HOG_BINS);

      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const i = y * CANON_W + x;
          if (!mask[i]) continue;
          validPixels++;
          const v = gray.data[i];
          intensitySum += v;
          intensitySq += v * v;
          if (x > 0 && y > 0 && x < CANON_W - 1 && y < CANON_H - 1) {
            lbpHist[lbpCode(gray, x, y)]++;
            const ax = gx[i];
            const ay = gy[i];
            const mag = Math.hypot(ax, ay);
            // 无向梯度 [0, pi)，对鼻镜左右翻转之外的旋转由 LBP 兜底。
            let ang = Math.atan2(ay, ax);
            if (ang < 0) ang += Math.PI;
            const bin = Math.min(HOG_BINS - 1, Math.floor(ang / Math.PI * HOG_BINS));
            hogHist[bin] += mag;
          }
        }
      }

      if (validPixels < cellW * cellH * 0.45) {
        cellValid[cellIndex] = 0;
        continue;
      }
      cellValid[cellIndex] = 1;

      const mean = intensitySum / validPixels;
      const std = Math.sqrt(Math.max(0, intensitySq / validPixels - mean * mean));
      for (let b = 0; b < LBP_BINS; b++) descriptor[base + b] = lbpHist[b] / validPixels;
      for (let b = 0; b < HOG_BINS; b++) descriptor[base + LBP_BINS + b] = hogHist[b];
      descriptor[base + LBP_BINS + HOG_BINS] = mean / 255;
      descriptor[base + LBP_BINS + HOG_BINS + 1] = std / 128;

      // 格内 L2 归一化（消除该格整体增益）。
      let norm = 0;
      for (let k = 0; k < CELL_DIM; k++) norm += descriptor[base + k] ** 2;
      norm = Math.sqrt(norm) || 1;
      for (let k = 0; k < CELL_DIM; k++) descriptor[base + k] /= norm;
    }
  }

  // 全局温和标准化：只统一整体尺度，不做会抹平空间结构的块级二次归一化。
  let globalNorm = 0;
  for (let i = 0; i < descriptor.length; i++) globalNorm += descriptor[i] ** 2;
  globalNorm = Math.sqrt(globalNorm) || 1;
  for (let i = 0; i < descriptor.length; i++) descriptor[i] /= globalNorm;

  const coarse = computeCoarse(gray, mask);
  const keypoints = detectBeads(gray, mask);
  const beadTpl = describeBeads(gray, mask, keypoints, cowId);

  return { cowId, descriptor, coarse, cellValid, beads: beadTpl, thumb, createdAt };
}

/** 粗筛指纹：8x2 下采样亮度（16）+ 4x4 象限梯度能量（16），各自标准化。 */
function computeCoarse(gray: GrayImage, mask: Uint8Array): Float32Array {
  const out = new Float32Array(COARSE_DIM);
  const cols = 8;
  const rows = 2;
  let idx = 0;
  for (let ry = 0; ry < rows; ry++) {
    for (let rx = 0; rx < cols; rx++) {
      const x0 = Math.floor((rx / cols) * CANON_W);
      const x1 = Math.floor(((rx + 1) / cols) * CANON_W);
      const y0 = Math.floor((ry / rows) * CANON_H);
      const y1 = Math.floor(((ry + 1) / rows) * CANON_H);
      let sum = 0;
      let n = 0;
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const i = y * CANON_W + x;
          if (mask[i]) {
            sum += gray.data[i];
            n++;
          }
        }
      }
      out[idx++] = n ? sum / n / 255 : 0;
    }
  }
  const qx = 4;
  const qy = 4;
  for (let ry = 0; ry < qy; ry++) {
    for (let rx = 0; rx < qx; rx++) {
      const x0 = Math.floor((rx / qx) * CANON_W);
      const x1 = Math.floor(((rx + 1) / qx) * CANON_W);
      const y0 = Math.floor((ry / qy) * CANON_H);
      const y1 = Math.floor(((ry + 1) / qy) * CANON_H);
      let energy = 0;
      let n = 0;
      for (let y = Math.max(1, y0); y < Math.min(CANON_H - 1, y1); y++) {
        for (let x = Math.max(1, x0); x < Math.min(CANON_W - 1, x1); x++) {
          const i = y * CANON_W + x;
          if (!mask[i]) continue;
          const gxv =
            -gray.data[i - CANON_W - 1] + gray.data[i - CANON_W + 1] -
            2 * gray.data[i - 1] + 2 * gray.data[i + 1] -
            gray.data[i + CANON_W - 1] + gray.data[i + CANON_W + 1];
          const gyv =
            -gray.data[i - CANON_W - 1] - 2 * gray.data[i - CANON_W] - gray.data[i - CANON_W + 1] +
            gray.data[i + CANON_W - 1] + 2 * gray.data[i + CANON_W] + gray.data[i + CANON_W + 1];
          energy += gxv * gxv + gyv * gyv;
          n++;
        }
      }
      out[idx++] = n ? Math.sqrt(energy / n) / 128 : 0;
    }
  }
  // 全局 L2 标准化，使其与绝对增益无关。
  let norm = 0;
  for (let i = 0; i < out.length; i++) norm += out[i] * out[i];
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < out.length; i++) out[i] /= norm;
  return out;
}

/** 粗距离：L1（32 维，全量扫描极廉价）。 */
export function coarseDistance(a: Float32Array, b: Float32Array): number {
  let d = 0;
  for (let i = 0; i < COARSE_DIM; i++) d += Math.abs(a[i] - b[i]);
  return d;
}

/**
 * 细距离：仅累计双方都有效的格，允许 (shiftX, shiftY) 格移位。
 * 返回归一化到 [0,2] 量级的平均 L1 距离。
 */
export function fineDistance(
  query: FeatureTemplate,
  target: FeatureTemplate,
  shiftX: number,
  shiftY: number,
): number {
  let sum = 0;
  let cells = 0;
  for (let cy = 0; cy < GRID_Y; cy++) {
    const ty = cy + shiftY;
    if (ty < 0 || ty >= GRID_Y) continue;
    for (let cx = 0; cx < GRID_X; cx++) {
      const tx = cx + shiftX;
      if (tx < 0 || tx >= GRID_X) continue;
      const qCell = cy * GRID_X + cx;
      const tCell = ty * GRID_X + tx;
      if (!query.cellValid[qCell] || !target.cellValid[tCell]) continue;
      const qb = qCell * CELL_DIM;
      const tb = tCell * CELL_DIM;
      let d = 0;
      for (let k = 0; k < CELL_DIM; k++) d += Math.abs(query.descriptor[qb + k] - target.descriptor[tb + k]);
      sum += d;
      cells++;
    }
  }
  return cells ? sum / cells : 2;
}

const SHIFTS: Array<[number, number]> = (() => {
  const list: Array<[number, number]> = [];
  for (let dy = -2; dy <= 2; dy++) {
    for (let dx = -2; dx <= 2; dx++) list.push([dx, dy]);
  }
  return list;
})();
export { SHIFTS };

export interface MatchCandidate {
  cowId: string;
  distance: number;
  gridDistance: number;
  beadDistance: number;
  matchedBeads: number;
  beadRmse: number;
  coarse: number;
  similarity: number;
}

export interface MatchResult {
  topCandidates: MatchCandidate[];
  coarseScanned: number;
  fineCompared: number;
  timingMs: number;
}

/** 距离 -> 相似度（0..1），融合距离标定在基准中校准。 */
export function distanceToSimilarity(d: number): number {
  const x = (1.05 - d) / 0.42;
  return 1 / (1 + Math.exp(-x));
}

/**
 * 全库比对：
 * 阶段一 粗筛全量 L1，取最近 K 个；
 * 阶段二 对 K 个做 25 组移位的网格细距离 + 珠点 RANSAC 几何匹配，加权融合。
 */
export function search(
  query: FeatureTemplate,
  gallery: FeatureTemplate[],
  topK = 5,
  coarseKeep = 60,
): MatchResult {
  const t0 = now();
  const rough = new Array<{ cowId: string; d: number }>(gallery.length);
  for (let i = 0; i < gallery.length; i++) {
    rough[i] = { cowId: gallery[i].cowId, d: coarseDistance(query.coarse, gallery[i].coarse) };
  }
  rough.sort((a, b) => a.d - b.d);
  const keep = rough.slice(0, Math.min(coarseKeep, rough.length));
  const byId = new Map(gallery.map((t) => [t.cowId, t]));

  const scored: MatchCandidate[] = [];
  for (const r of keep) {
    const target = byId.get(r.cowId)!;
    let bestGrid = Infinity;
    for (const [sx, sy] of SHIFTS) {
      const d = fineDistance(query, target, sx, sy);
      if (d < bestGrid) bestGrid = d;
    }
    const beadStats = matchBeads(query.beads, target.beads);
    const bDist = beadDistance(beadStats);
    // 网格纹理与珠点几何等权融合；珠点是强身份信号，几何一致时显著拉低距离。
    const fused = 0.45 * bestGrid + 0.55 * bDist;
    scored.push({
      cowId: r.cowId,
      distance: fused,
      gridDistance: bestGrid,
      beadDistance: bDist,
      matchedBeads: beadStats.matchedPoints,
      beadRmse: beadStats.rmse,
      coarse: r.d,
      similarity: distanceToSimilarity(fused),
    });
  }
  scored.sort((a, b) => a.distance - b.distance);
  return {
    topCandidates: scored.slice(0, topK),
    coarseScanned: gallery.length,
    fineCompared: keep.length,
    timingMs: now() - t0,
  };
}

/** 抽象时间源，Node 与浏览器都可用；基准中可用注入计时。 */
let nowFn: () => number = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
export function setClock(fn: () => number): void {
  nowFn = fn;
}
function now(): number {
  return nowFn();
}

/** 模板序列化（IndexedDB 不直接存 TypedArray 视图时使用）。 */
export function serializeTemplate(t: FeatureTemplate): unknown {
  return {
    cowId: t.cowId,
    descriptor: Array.from(t.descriptor),
    coarse: Array.from(t.coarse),
    cellValid: Array.from(t.cellValid),
    beads: {
      cowId: t.beads.cowId,
      keypoints: t.beads.keypoints,
      descriptors: Array.from(t.beads.descriptors),
    },
    thumb: Array.from(t.thumb),
    createdAt: t.createdAt,
  };
}

export function deserializeTemplate(obj: {
  cowId: string;
  descriptor: number[];
  coarse: number[];
  cellValid: number[];
  beads: { cowId: string; keypoints: BeadTemplate['keypoints']; descriptors: number[] };
  thumb: number[];
  createdAt: number;
}): FeatureTemplate {
  return {
    cowId: obj.cowId,
    descriptor: Float32Array.from(obj.descriptor),
    coarse: Float32Array.from(obj.coarse),
    cellValid: Uint8Array.from(obj.cellValid),
    beads: {
      cowId: obj.beads.cowId,
      keypoints: obj.beads.keypoints,
      descriptors: Float32Array.from(obj.beads.descriptors),
    },
    thumb: Uint8Array.from(obj.thumb),
    createdAt: obj.createdAt,
  };
}
