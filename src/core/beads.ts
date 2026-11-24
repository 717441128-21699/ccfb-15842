/**
 * 珠状纹（bead）关键点检测与局部描述子。
 *
 * 牛鼻纹的身份信息主要来自一粒粒珠状突起的位置与局部纹理。
 * 检测：归一化图上的多尺度高斯拉普拉斯（LoG，用 DoG 近似）斑点响应，
 *      非极大值抑制 + 响应强度门限，输出亚像素级关键点。
 * 描述：以关键点为中心的 12x12 归一化贴片 LBP 直方图（旋转不变、灰度不变）。
 * 匹配：基于描述子最近邻 + 距离比门限 + 空间 RANSAC 几何一致性。
 */

import { GrayImage } from './geometry';
import { CANON_W, CANON_H } from './synthesis';

export interface Keypoint {
  x: number;
  y: number;
  scale: number;
  response: number;
}

export interface BeadTemplate {
  cowId: string;
  keypoints: Keypoint[];
  /** 与 keypoints 等长，每条 LBP_BINS=10 维。 */
  descriptors: Float32Array;
}

function gaussianKernel(sigma: number): Float32Array {
  const radius = Math.max(1, Math.ceil(sigma * 2.5));
  const k = new Float32Array(radius * 2 + 1);
  let norm = 0;
  for (let i = -radius; i <= radius; i++) {
    const v = Math.exp(-(i * i) / (2 * sigma * sigma));
    k[i + radius] = v;
    norm += v;
  }
  for (let i = 0; i < k.length; i++) k[i] /= norm;
  return k;
}

function blurGray(img: GrayImage, sigma: number): Float32Array {
  const W = img.width;
  const H = img.height;
  const k = gaussianKernel(sigma);
  const r = (k.length - 1) / 2;
  const tmp = new Float32Array(W * H);
  const out = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let acc = 0;
      for (let t = -r; t <= r; t++) {
        const xx = Math.min(W - 1, Math.max(0, x + t));
        acc += img.data[y * W + xx] * k[t + r];
      }
      tmp[y * W + x] = acc;
    }
  }
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let acc = 0;
      for (let t = -r; t <= r; t++) {
        const yy = Math.min(H - 1, Math.max(0, y + t));
        acc += tmp[yy * W + x] * k[t + r];
      }
      out[y * W + x] = acc;
    }
  }
  return out;
}

/** 积分图（用于任意圆盘/环带的快速区域均值）。 */
function buildIntegral(img: GrayImage): Float64Array {
  const W = img.width;
  const H = img.height;
  const integral = new Float64Array((W + 1) * (H + 1));
  for (let y = 0; y < H; y++) {
    let row = 0;
    for (let x = 0; x < W; x++) {
      row += img.data[y * W + x];
      integral[(y + 1) * (W + 1) + (x + 1)] = integral[y * (W + 1) + (x + 1)] + row;
    }
  }
  return integral;
}

function diskMean(integral: Float64Array, W: number, H: number, cx: number, cy: number, r: number): number {
  const x0 = Math.max(0, cx - r);
  const x1 = Math.min(W - 1, cx + r);
  const y0 = Math.max(0, cy - r);
  const y1 = Math.min(H - 1, cy + r);
  const area = (x1 - x0 + 1) * (y1 - y0 + 1);
  const sum =
    integral[(y1 + 1) * (W + 1) + (x1 + 1)] -
    integral[y0 * (W + 1) + (x1 + 1)] -
    integral[(y1 + 1) * (W + 1) + x0] +
    integral[y0 * (W + 1) + x0];
  return sum / area;
}

/** 三点抛物线极值的亚像素偏移（峰值在 center，返回相对 center 的 -0.5..0.5）。 */
function parabolaOffset(left: number, center: number, right: number): number {
  const denom = left - 2 * center + right;
  if (Math.abs(denom) < 1e-6) return 0;
  const d = 0.5 * (left - right) / denom;
  return Math.max(-0.5, Math.min(0.5, d));
}

/**
 * 珠点检测（环形匹配滤波器）。
 *
 * 珠状突起的稳定物理结构是「亮中心圆盘 + 外围暗环」。在每个像素计算
 * 响应 = 中心圆盘均值 − 外环带均值，该响应在珠点中心取正峰值，只对同心圆
 * 结构敏感，对整体亮度/渐变（已归一化）和单像素噪声不敏感，因此跨采集的
 * 可重复性远高于通用局部极大。随后局部极大提取 + 自适应门限 + 空间 NMS。
 */
export function detectBeads(
  img: GrayImage,
  mask: Uint8Array,
  _sigmas = [1.6, 2.4, 3.3],
  maxKeypoints = 100,
  nmsDist = 5.0,
): Keypoint[] {
  const W = img.width;
  const H = img.height;
  const smooth = blurGray(img, 1.0);
  const smoothImg = new GrayImage(W, H);
  for (let i = 0; i < smooth.length; i++) smoothImg.data[i] = Math.max(0, Math.min(255, smooth[i]));
  const integral = buildIntegral(smoothImg);

  // 珠点半径在标准画布上约 2.4..4.7：中心圆盘 r=2，外环 r=4。
  const rIn = 2;
  const rOut = 4;
  const response = new Float32Array(W * H);
  const border = rOut + 1;
  for (let y = border; y < H - border; y++) {
    for (let x = border; x < W - border; x++) {
      const i = y * W + x;
      if (!mask[i]) continue;
      const center = diskMean(integral, W, H, x, y, rIn);
      const outer = diskMean(integral, W, H, x, y, rOut);
      response[i] = center - outer;
    }
  }

  // 自适应门限：有效响应均值 + k*标准差。
  let sum = 0, sumSq = 0, n = 0;
  for (let i = 0; i < response.length; i++) {
    if (mask[i]) { sum += response[i]; sumSq += response[i] * response[i]; n++; }
  }
  const mean = sum / (n || 1);
  const std = Math.sqrt(Math.max(1e-6, sumSq / (n || 1) - mean * mean));
  const thresh = mean + 0.55 * std;

  // 局部极大（3x3）+ 二维二次曲面亚像素细化，降低模糊/噪声导致的定位抖动。
  const kps: Keypoint[] = [];
  for (let y = border; y < H - border; y++) {
    for (let x = border; x < W - border; x++) {
      const i = y * W + x;
      if (!mask[i] || response[i] < thresh) continue;
      const v = response[i];
      let isMax = true;
      for (let dy = -1; dy <= 1 && isMax; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dx === 0 && dy === 0) continue;
          if (response[(y + dy) * W + (x + dx)] > v) { isMax = false; break; }
        }
      }
      if (!isMax) continue;
      // 可分离二次曲面：x、y 方向各自抛物线求极值偏移。
      const ox = parabolaOffset(response[i - 1], v, response[i + 1]);
      const oy = parabolaOffset(response[i - W], v, response[i + W]);
      kps.push({
        x: x + ox,
        y: y + oy,
        scale: rOut,
        response: v,
      });
    }
  }

  // 响应排序 + 空间 NMS。
  kps.sort((a, b) => b.response - a.response);
  const kept: Keypoint[] = [];
  for (const kp of kps) {
    if (kept.length >= maxKeypoints) break;
    let dup = false;
    for (const k of kept) {
      if (Math.hypot(kp.x - k.x, kp.y - k.y) < nmsDist) { dup = true; break; }
    }
    if (!dup) kept.push(kp);
  }
  return kept;
}

/** Sobel 梯度方向（有向，0..2pi）与幅值。 */
function gradientAt(img: GrayImage, x: number, y: number): { mag: number; ang: number } {
  const W = img.width;
  const i = y * W + x;
  const gx =
    -img.data[i - W - 1] + img.data[i - W + 1] - 2 * img.data[i - 1] +
    2 * img.data[i + 1] - img.data[i + W - 1] + img.data[i + W + 1];
  const gy =
    -img.data[i - W - 1] - 2 * img.data[i - W] - img.data[i - W + 1] +
    img.data[i + W - 1] + 2 * img.data[i + W] + img.data[i + W + 1];
  let ang = Math.atan2(gy, gx);
  if (ang < 0) ang += Math.PI * 2;
  return { mag: Math.hypot(gx, gy), ang };
}

/**
 * 关键点主方向：邻域梯度方向直方图（8 桶）的峰值，做旋转不变性。
 */
function dominantOrientation(img: GrayImage, kp: Keypoint): number {
  const W = img.width;
  const H = img.height;
  const bins = new Float32Array(8);
  const r = 5;
  for (let dy = -r; dy <= r; dy++) {
    for (let dx = -r; dx <= r; dx++) {
      const x = Math.round(kp.x) + dx;
      const y = Math.round(kp.y) + dy;
      if (x < 1 || y < 1 || x >= W - 1 || y >= H - 1) continue;
      const g = gradientAt(img, x, y);
      const bin = Math.min(7, Math.floor(g.ang / (Math.PI * 2) * 8));
      const w = Math.exp(-(dx * dx + dy * dy) / (2 * r * r));
      bins[bin] += g.mag * w;
    }
  }
  let best = 0;
  for (let b = 1; b < 8; b++) if (bins[b] > bins[best]) best = b;
  return (best + 0.5) * (Math.PI * 2 / 8);
}

/**
 * SIFT 式局部描述子：以关键点为中心、按主方向旋转归一化的 4x4 子区，
 * 每区 8 方向梯度直方图，共 128 维。对旋转/光照（归一化后）稳健，
 * 且比纯 LBP 直方图判别力强得多。
 */
export const BEAD_DESC_DIM = 128;

export function describeBeads(
  img: GrayImage,
  mask: Uint8Array,
  kps: Keypoint[],
  cowId: string,
): BeadTemplate {
  const W = img.width;
  const H = img.height;
  const descriptors = new Float32Array(kps.length * BEAD_DESC_DIM);
  const valid: Keypoint[] = [];
  const patchR = 8; // 采样半径

  for (const kp of kps) {
    const cx = Math.round(kp.x);
    const cy = Math.round(kp.y);
    if (cx - patchR - 1 < 1 || cy - patchR - 1 < 1 || cx + patchR + 1 >= W - 1 || cy + patchR + 1 >= H - 1) {
      continue;
    }
    const orient = dominantOrientation(img, kp);
    const cos = Math.cos(-orient);
    const sin = Math.sin(-orient);
    const hist = new Float32Array(BEAD_DESC_DIM);
    let covered = 0;

    for (let dy = -patchR; dy <= patchR; dy++) {
      for (let dx = -patchR; dx <= patchR; dx++) {
        // 旋转到主方向坐标系。
        const rx = cos * dx - sin * dy;
        const ry = sin * dx + cos * dy;
        const sx = cx + Math.round(rx);
        const sy = cy + Math.round(ry);
        if (sx < 1 || sy < 1 || sx >= W - 1 || sy >= H - 1) continue;
        if (!mask[sy * W + sx]) continue;
        covered++;
        const g = gradientAt(img, sx, sy);
        // 相对主方向的方向角。
        let rel = g.ang - orient;
        rel = ((rel % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2);
        const gxCell = Math.min(3, Math.max(0, Math.floor((rx / patchR + 1) * 2)));
        const gyCell = Math.min(3, Math.max(0, Math.floor((ry / patchR + 1) * 2)));
        const angBin = Math.min(7, Math.floor(rel / (Math.PI * 2) * 8));
        const cell = gyCell * 4 + gxCell;
        hist[cell * 8 + angBin] += g.mag;
      }
    }
    if (covered < (patchR * 2 + 1) ** 2 * 0.6) continue;

    const base = valid.length * BEAD_DESC_DIM;
    let norm = 0;
    for (let b = 0; b < BEAD_DESC_DIM; b++) {
      // SIFT 截断，抑制少量极端梯度（反光残留）。
      const v = Math.min(0.2, hist[b]);
      descriptors[base + b] = v;
      norm += v * v;
    }
    norm = Math.sqrt(norm) || 1;
    for (let b = 0; b < BEAD_DESC_DIM; b++) descriptors[base + b] /= norm;
    valid.push(kp);
  }

  return {
    cowId,
    keypoints: valid,
    descriptors: descriptors.slice(0, valid.length * BEAD_DESC_DIM),
  };
}

export interface BeadMatchStats {
  matchedPoints: number;
  /** 匹配点对的平均空间残差（标准画布像素）。 */
  rmse: number;
  /** 一致性内点比例。 */
  inlierRatio: number;
}

/**
 * 两套珠点模板匹配（纯几何）。
 *
 * 两视图经矩形化后只剩小相似变换残差。用「两点相似变换投票」估计全局对齐
 * （吸收平移/微旋转/微缩放），再以严格小容差统计双向最近邻匹配的对称 F1。
 * 全局对齐让同牛真正对应的点重合；严格容差使异牛即便被对齐也无法大量匹配，
 * 从而拉开同/异牛差距。
 */
export function matchBeads(
  query: BeadTemplate,
  target: BeadTemplate,
): BeadMatchStats {
  const q = query.keypoints;
  const t = target.keypoints;
  if (q.length < 8 || t.length < 8) return { matchedPoints: 0, rmse: 999, inlierRatio: 0 };

  interface Sim { cos: number; sin: number; scale: number; tx: number; ty: number; }
  const apply = (a: Sim, x: number, y: number) => ({
    x: a.scale * (a.cos * x - a.sin * y) + a.tx,
    y: a.scale * (a.sin * x + a.cos * y) + a.ty,
  });
  const qmx = q.reduce((s, p) => s + p.x, 0) / q.length;
  const qmy = q.reduce((s, p) => s + p.y, 0) / q.length;
  const tmx = t.reduce((s, p) => s + p.x, 0) / t.length;
  const tmy = t.reduce((s, p) => s + p.y, 0) / t.length;
  const nnT = (x: number, y: number) => {
    let md = Infinity;
    for (const r of t) md = Math.min(md, Math.hypot(x - r.x, y - r.y));
    return md;
  };

  // 候选变换：质心平移 + 少量两点相似变换假设。
  const candidates: Sim[] = [
    { cos: 1, sin: 0, scale: 1, tx: tmx - qmx, ty: tmy - qmy },
  ];
  let seed = 987654321;
  const rand = () => { seed = (Math.imul(seed, 1103515245) + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
  for (let it = 0; it < 40; it++) {
    const i = Math.floor(rand() * q.length);
    const j = Math.floor(rand() * q.length);
    if (i === j) continue;
    // 用质心平移后的最近邻找对应。
    const dx0 = tmx - qmx, dy0 = tmy - qmy;
    let ti1 = -1, ti2 = -1, b1 = Infinity, b2 = Infinity;
    for (let k = 0; k < t.length; k++) {
      const d1 = Math.hypot((q[i].x + dx0) - t[k].x, (q[i].y + dy0) - t[k].y);
      if (d1 < b1) { b1 = d1; ti1 = k; }
      const d2 = Math.hypot((q[j].x + dx0) - t[k].x, (q[j].y + dy0) - t[k].y);
      if (d2 < b2) { b2 = d2; ti2 = k; }
    }
    if (ti1 < 0 || ti2 < 0 || ti1 === ti2 || b1 > 12 || b2 > 12) continue;
    const qdx = q[j].x - q[i].x, qdy = q[j].y - q[i].y;
    const tdx = t[ti2].x - t[ti1].x, tdy = t[ti2].y - t[ti1].y;
    const ql = Math.hypot(qdx, qdy), tl = Math.hypot(tdx, tdy);
    if (ql < 1e-3 || tl < 1e-3) continue;
    const scale = tl / ql;
    if (scale < 0.8 || scale > 1.25) continue;
    const theta = Math.atan2(tdy, tdx) - Math.atan2(qdy, qdx);
    if (Math.abs(theta) > 0.25) continue;
    const cos = Math.cos(theta), sin = Math.sin(theta);
    const mp = apply({ cos, sin, scale, tx: 0, ty: 0 }, q[i].x, q[i].y);
    candidates.push({ cos, sin, scale, tx: t[ti1].x - mp.x, ty: t[ti1].y - mp.y });
  }

  // 选粗容差内点最多的变换，再 ICP 精修一次。
  const coarseTol = 5;
  let best = candidates[0];
  let bestCount = -1;
  for (const c of candidates) {
    let cnt = 0;
    for (const p of q) if (nnT(apply(c, p.x, p.y).x, apply(c, p.x, p.y).y) < coarseTol) cnt++;
    if (cnt > bestCount) { bestCount = cnt; best = c; }
  }
  // ICP 精修（最近邻对应最小二乘相似变换）。
  let sim = best;
  for (let it = 0; it < 4; it++) {
    const cq: Keypoint[] = [], ct: Keypoint[] = [];
    for (const p of q) {
      const m = apply(sim, p.x, p.y);
      let mi = 0, md = Infinity;
      for (let k = 0; k < t.length; k++) { const d = Math.hypot(m.x - t[k].x, m.y - t[k].y); if (d < md) { md = d; mi = k; } }
      if (md < coarseTol) { cq.push(p); ct.push(t[mi]); }
    }
    if (cq.length < 4) break;
    const ax = cq.reduce((s, p) => s + p.x, 0) / cq.length, ay = cq.reduce((s, p) => s + p.y, 0) / cq.length;
    const bx = ct.reduce((s, p) => s + p.x, 0) / cq.length, by = ct.reduce((s, p) => s + p.y, 0) / cq.length;
    let num = 0, den = 0;
    for (let k = 0; k < cq.length; k++) {
      num += (cq[k].x - ax) * (ct[k].y - by) - (cq[k].y - ay) * (ct[k].x - bx);
      den += (cq[k].x - ax) * (ct[k].x - bx) + (cq[k].y - ay) * (ct[k].y - by);
    }
    const theta = Math.atan2(num, den), cos = Math.cos(theta), sin = Math.sin(theta);
    let sn = 0, sd = 0;
    for (let k = 0; k < cq.length; k++) {
      sn += Math.hypot(ct[k].x - bx, ct[k].y - by);
      sd += Math.hypot(cq[k].x - ax, cq[k].y - ay);
    }
    const scale = sd > 1e-6 ? sn / sd : 1;
    const mc = apply({ cos, sin, scale, tx: 0, ty: 0 }, ax, ay);
    sim = { cos, sin, scale, tx: bx - mc.x, ty: by - mc.y };
  }

  // 严格容差双向 F1。
  const tol = 2.2;
  let fwd = 0, se = 0, cnt = 0;
  for (const p of q) {
    const m = apply(sim, p.x, p.y);
    const d = nnT(m.x, m.y);
    if (d < tol) { fwd++; se += d * d; cnt++; }
  }
  let bwd = 0;
  const inv = 1 / sim.scale;
  for (const r of t) {
    const dx = r.x - sim.tx, dy = r.y - sim.ty;
    const qx = inv * (sim.cos * dx + sim.sin * dy);
    const qy = inv * (-sim.sin * dx + sim.cos * dy);
    let md = Infinity;
    for (const p of q) md = Math.min(md, Math.hypot(qx - p.x, qy - p.y));
    if (md < tol) bwd++;
  }
  const precision = bwd / t.length, recall = fwd / q.length;
  const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
  return {
    matchedPoints: Math.round(f1 * Math.min(q.length, t.length)),
    rmse: cnt ? Math.sqrt(se / cnt) : 999,
    inlierRatio: f1,
  };
}

/** 珠点匹配 -> 距离（越小越相似）。 */
export function beadDistance(stats: BeadMatchStats): number {
  return 1 - stats.inlierRatio + 0.2 * Math.min(1, stats.rmse / 6);
}

export { CANON_W, CANON_H };
