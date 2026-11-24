/**
 * 鼻纹锚点身份指纹。
 *
 * 身份的本质是标准画布中珠点（鼻纹 minutia）的空间布局。注册阶段在矩形化、
 * 光照归一化后的标准鼻镜上确认一组锚点；理赔阶段对新采集做同样处理后比对。
 *
 * 两级检索（满足 5000 头 < 2s）：
 *  - 粗筛：标准画布划成 32x24 占用网格，多尺度高斯弥散成二值指纹，
 *    用 Jaccard/相关做全量廉价扫描，保留最近的若干头；
 *  - 精排：对候选做质心对齐后的对称 Chamfer 距离，精确度量布局差异。
 *
 * 锚点坐标在 256x192 标准画布，天然与光照、拍摄角度（矩形化已吸收）无关。
 */

import { CANON_W, CANON_H } from './synthesis';

export interface AnchorPoint {
  x: number;
  y: number;
}

export const GRID_COLS = 32;
export const GRID_ROWS = 24;
const CELL_W = CANON_W / GRID_COLS; // 8
const CELL_H = CANON_H / GRID_ROWS; // 8

export interface AnchorFingerprint {
  cowId: string;
  /** 归一化锚点坐标（标准画布像素）。 */
  points: AnchorPoint[];
  /** 32x24 二值占用网格（高斯弥散后阈值化），768 bit，打包成字节。 */
  bits: Uint8Array;
  /** bit 中置位的数量，Jaccard 用。 */
  popcount: number;
}

/** 由锚点构造二值占用指纹：每个锚点在网格上做半径 ~1 格的高斯弥散。 */
export function buildAnchorFingerprint(cowId: string, points: AnchorPoint[]): AnchorFingerprint {
  const grid = new Float32Array(GRID_COLS * GRID_ROWS);
  const sigma = 0.9;
  const reach = 2;
  for (const p of points) {
    const gx = p.x / CELL_W;
    const gy = p.y / CELL_H;
    const cx = Math.round(gx);
    const cy = Math.round(gy);
    for (let dy = -reach; dy <= reach; dy++) {
      for (let dx = -reach; dx <= reach; dx++) {
        const xx = cx + dx;
        const yy = cy + dy;
        if (xx < 0 || yy < 0 || xx >= GRID_COLS || yy >= GRID_ROWS) continue;
        const v = Math.exp(-((gx - xx) ** 2 + (gy - yy) ** 2) / (2 * sigma ** 2));
        const idx = yy * GRID_COLS + xx;
        if (v > grid[idx]) grid[idx] = v;
      }
    }
  }
  // 阈值化为二值（>0.35 视为占用），并打包。
  const bits = new Uint8Array(GRID_COLS * GRID_ROWS);
  let popcount = 0;
  for (let i = 0; i < grid.length; i++) {
    if (grid[i] > 0.35) {
      bits[i] = 1;
      popcount++;
    }
  }
  return { cowId, points: points.map((p) => ({ ...p })), bits, popcount };
}

/** 粗距离：二值占用网格的 1 - Jaccard（交并比），全量扫描极快。 */
export function fingerprintJaccardDistance(a: AnchorFingerprint, b: AnchorFingerprint): number {
  let inter = 0;
  const n = a.bits.length;
  for (let i = 0; i < n; i++) if (a.bits[i] && b.bits[i]) inter++;
  const union = a.popcount + b.popcount - inter;
  return union > 0 ? 1 - inter / union : 1;
}

/** 质心对齐后的对称 Chamfer 距离（精排），用空间网格把最近邻降到近常数时间。 */
export function chamferDistance(a: AnchorPoint[], b: AnchorPoint[]): number {
  if (!a.length || !b.length) return 999;
  const amx = a.reduce((s, p) => s + p.x, 0) / a.length;
  const amy = a.reduce((s, p) => s + p.y, 0) / a.length;
  const bmx = b.reduce((s, p) => s + p.x, 0) / b.length;
  const bmy = b.reduce((s, p) => s + p.y, 0) / b.length;
  const dx = bmx - amx;
  const dy = bmy - amy;

  // 在 b 上建空间哈希网格（格边长 8px），最近邻只查邻近 3x3 格。
  const CELL = 8;
  const grid = new Map<number, number[]>();
  const key = (gx: number, gy: number) => gy * 100000 + gx;
  for (let j = 0; j < b.length; j++) {
    const gx = Math.floor(b[j].x / CELL);
    const gy = Math.floor(b[j].y / CELL);
    const k = key(gx, gy);
    const arr = grid.get(k);
    if (arr) arr.push(j);
    else grid.set(k, [j]);
  }
  const nnToB = (px: number, py: number): number => {
    const gx = Math.floor(px / CELL);
    const gy = Math.floor(py / CELL);
    let md = Infinity;
    for (let ey = gy - 1; ey <= gy + 1; ey++) {
      for (let ex = gx - 1; ex <= gx + 1; ex++) {
        const arr = grid.get(key(ex, ey));
        if (!arr) continue;
        for (const j of arr) {
          const d = (px - b[j].x) ** 2 + (py - b[j].y) ** 2;
          if (d < md) md = d;
        }
      }
    }
    return md === Infinity ? nearestFallback(b, px, py) : Math.sqrt(md);
  };

  let d1 = 0;
  for (const p of a) d1 += nnToB(p.x + dx, p.y + dy);
  let d2 = 0;
  for (const q of b) {
    let md = Infinity;
    for (const p of a) {
      const d = (p.x + dx - q.x) ** 2 + (p.y + dy - q.y) ** 2;
      if (d < md) md = d;
    }
    d2 += Math.sqrt(md);
  }
  return (d1 / a.length + d2 / b.length) / 2;
}

function nearestFallback(b: AnchorPoint[], px: number, py: number): number {
  let md = Infinity;
  for (const q of b) md = Math.min(md, Math.hypot(px - q.x, py - q.y));
  return md;
}

export interface AnchorCandidate {
  cowId: string;
  coarse: number;
  chamfer: number;
  similarity: number;
}

/** Chamfer 距离 -> 相似度（0..1），标定参数见基准。 */
export function chamferToSimilarity(d: number): number {
  // 同牛 Chamfer 接近 0，异牛 > 2.5；用分段映射让分界清晰。
  const x = (1.1 - d) / 0.55;
  return 1 / (1 + Math.exp(-x));
}

export interface AnchorSearchResult {
  candidates: AnchorCandidate[];
  coarseScanned: number;
  fineCompared: number;
  timingMs: number;
}

/**
 * 全库检索：粗筛全量 Jaccard 取最近 coarseKeep 头，精排 Chamfer 后返回 topK。
 */
export function searchAnchors(
  query: AnchorFingerprint,
  gallery: AnchorFingerprint[],
  topK = 5,
  coarseKeep = 50,
  now: () => number = () => (typeof performance !== 'undefined' ? performance.now() : Date.now()),
): AnchorSearchResult {
  const t0 = now();
  const rough = new Array<{ cowId: string; d: number }>(gallery.length);
  for (let i = 0; i < gallery.length; i++) {
    rough[i] = { cowId: gallery[i].cowId, d: fingerprintJaccardDistance(query, gallery[i]) };
  }
  rough.sort((a, b) => a.d - b.d);
  const keep = rough.slice(0, Math.min(coarseKeep, rough.length));
  const byId = new Map(gallery.map((g) => [g.cowId, g]));
  const scored: AnchorCandidate[] = keep.map((r) => {
    const target = byId.get(r.cowId)!;
    const ch = chamferDistance(query.points, target.points);
    return {
      cowId: r.cowId,
      coarse: r.d,
      chamfer: ch,
      similarity: chamferToSimilarity(ch),
    };
  });
  scored.sort((a, b) => a.chamfer - b.chamfer);
  return {
    candidates: scored.slice(0, topK),
    coarseScanned: gallery.length,
    fineCompared: keep.length,
    timingMs: now() - t0,
  };
}
