/**
 * 图像增强流水线（CPU 参考实现）。
 *
 * 步骤与 WebGL2 shader 完全对应：
 *  1) 鼻镜矩形化：按检测四边形透视到标准 256x192 画布，并输出鼻镜掩膜；
 *  2) 湿反光检测：局部邻域亮异常 + 饱和度缺失 -> 镜面掩膜；
 *  3) 扩散修复：掩膜像素由邻域非掩膜像素迭代扩散填补，消灭亮斑；
 *  4) 光照归一化：大尺度高斯背景作除法式归一化，去掉方向光照/渐变；
 *  5) 对比度标准化到 8 位。
 */

import {
  GrayImage,
  Point,
  NostrilPair,
  RgbaImage,
  warpPerspective,
  sampleGrayBilinear,
  resizeGray,
  findHomography,
} from './geometry';
import { CANON_W, CANON_H } from './synthesis';

// 标准画布中两个鼻孔的目标位置（与合成器鼻孔标定一致，间距 80）。
const CANON_LEFT = { x: 88, y: 126 };
const CANON_RIGHT = { x: 168, y: 126 };

export interface EnhancedResult {
  /** 256x192 增强后灰度。 */
  gray: GrayImage;
  /** 鼻镜有效区域掩膜（鼻孔内也置 0，特征只在有效皮肤上提取）。 */
  mask: Uint8Array;
  /** 被修复的镜面像素比例（质量门限用）。 */
  specularRatio: number;
  /** 归一化前的动态范围。 */
  dynamicRange: number;
  /** 高频能量（清晰度，质量门限用）。 */
  sharpness: number;
}

/**
 * 由两个鼻孔中心构造相似变换（仅旋转+缩放+平移，无透视外推），
 * 把采集图映射到标准画布。鼻孔中心检测精度高（亚像素），相似变换比
 * 四边形外推单应稳定得多，跨采集的对齐残差由鼻孔测量误差决定。
 */
export function similarityQuadFromNostrils(pair: NostrilPair): Point[] {
  const dl = pair.left;
  const dr = pair.right;
  const dx = dr.x - dl.x;
  const dy = dr.y - dl.y;
  const srcS = Math.hypot(dx, dy);
  const canonS = CANON_RIGHT.x - CANON_LEFT.x; // 80
  const scale = canonS / srcS;
  // 采集 -> 标准：先平移左鼻孔到原点，旋转使两鼻孔水平，缩放，再平移到标准位置。
  const ang = Math.atan2(dy, dx);
  const cos = Math.cos(-ang) * scale;
  const sin = Math.sin(-ang) * scale;
  const map = (p: Point): Point => {
    const rx = p.x - dl.x;
    const ry = p.y - dl.y;
    return {
      x: cos * rx - sin * ry + CANON_LEFT.x,
      y: sin * rx + cos * ry + CANON_LEFT.y,
    };
  };
  // 用标准四角反推其在采集图中的位置，复用 warpPerspective（单应可精确表示相似变换）。
  const canonCorners: Point[] = [
    { x: 0, y: 0 },
    { x: CANON_W - 1, y: 0 },
    { x: CANON_W - 1, y: CANON_H - 1 },
    { x: 0, y: CANON_H - 1 },
  ];
  // map 的逆：标准 -> 采集。
  const invScale = 1 / scale;
  const invAng = ang;
  const ic = Math.cos(invAng) * invScale;
  const isn = Math.sin(invAng) * invScale;
  const unmap = (c: Point): Point => {
    const rx = c.x - CANON_LEFT.x;
    const ry = c.y - CANON_LEFT.y;
    const sx = ic * rx - isn * ry;
    const sy = isn * rx + ic * ry;
    return { x: sx + dl.x, y: sy + dl.y };
  };
  void map;
  const srcQuad = canonCorners.map(unmap);
  return srcQuad;
}

/** 步骤 1：矩形化。muzzleQuad 为检测到的四边形（左上起顺时针）。 */
export function rectify(img: GrayImage, muzzleQuad: Point[]): GrayImage {
  return warpPerspective(img, muzzleQuad, CANON_W, CANON_H);
}

/** 基于鼻孔相似变换的矩形化（推荐，几何更稳）。 */
export function rectifyByNostrils(img: GrayImage, pair: NostrilPair): GrayImage {
  const srcQuad = similarityQuadFromNostrils(pair);
  return warpPerspective(img, srcQuad, CANON_W, CANON_H);
}

void findHomography;

/** 鼻镜掩膜：标准画布内的圆角区域，去掉鼻孔（用标准几何的近似椭圆）。 */
export function buildMuzzleMask(): Uint8Array {
  const mask = new Uint8Array(CANON_W * CANON_H);
  const cx = CANON_W / 2;
  const cy = 104;
  const rx = 100;
  const ry = 74;
  const nostrils = [
    { cx: 88, cy: 126, rx: 30, ry: 22 },
    { cx: 168, cy: 126, rx: 30, ry: 22 },
  ];
  for (let y = 0; y < CANON_H; y++) {
    for (let x = 0; x < CANON_W; x++) {
      const q = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2;
      if (q > 1) continue;
      let inNostril = false;
      for (const n of nostrils) {
        if (((x - n.cx) / n.rx) ** 2 + ((y - n.cy) / n.ry) ** 2 < 1) {
          inNostril = true;
          break;
        }
      }
      if (!inNostril) mask[y * CANON_W + x] = 1;
    }
  }
  return mask;
}

/** 步骤 2：镜面反光掩膜。 */
export function detectSpecular(
  gray: GrayImage,
  mask: Uint8Array,
  radius = 7,
  kLocal = 26,
  kAbs = 218,
): Uint8Array {
  const W = gray.width;
  const H = gray.height;
  const localMean = boxMean(gray, radius);
  const spec = new Uint8Array(W * H);
  const d = gray.data;
  for (let i = 0; i < d.length; i++) {
    if (!mask[i]) continue;
    if (d[i] >= kAbs && d[i] - localMean[i] >= kLocal) spec[i] = 1;
  }
  // 形态学膨胀 1 像素，盖住高光边缘的过渡带。
  return dilate(spec, W, H);
}

function boxMean(img: GrayImage, radius: number): Float32Array {
  const W = img.width;
  const H = img.height;
  const integral = new Float64Array((W + 1) * (H + 1));
  for (let y = 0; y < H; y++) {
    let rowSum = 0;
    for (let x = 0; x < W; x++) {
      rowSum += img.data[y * W + x];
      integral[(y + 1) * (W + 1) + (x + 1)] = integral[y * (W + 1) + (x + 1)] + rowSum;
    }
  }
  const out = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const x0 = Math.max(0, x - radius);
      const x1 = Math.min(W - 1, x + radius);
      const y0 = Math.max(0, y - radius);
      const y1 = Math.min(H - 1, y + radius);
      const area = (x1 - x0 + 1) * (y1 - y0 + 1);
      const sum =
        integral[(y1 + 1) * (W + 1) + (x1 + 1)] -
        integral[y0 * (W + 1) + (x1 + 1)] -
        integral[(y1 + 1) * (W + 1) + x0] +
        integral[y0 * (W + 1) + x0];
      out[y * W + x] = sum / area;
    }
  }
  return out;
}

function dilate(mask: Uint8Array, W: number, H: number): Uint8Array {
  const out = new Uint8Array(mask.length);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      if (!mask[y * W + x]) continue;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          const yy = y + dy;
          if (xx >= 0 && yy >= 0 && xx < W && yy < H) out[yy * W + xx] = 1;
        }
      }
    }
  }
  return out;
}

/** 步骤 3：扩散修复（Jacobi 迭代，掩膜像素取四邻域均值）。 */
export function diffuseInpaint(
  gray: GrayImage,
  _mask: Uint8Array,
  spec: Uint8Array,
  iterations = 140,
): GrayImage {
  const W = gray.width;
  const H = gray.height;
  let cur: Float32Array = Float32Array.from(gray.data);
  let nxt: Float32Array = new Float32Array(cur.length);
  for (let it = 0; it < iterations; it++) {
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const i = y * W + x;
        if (!spec[i]) {
          nxt[i] = cur[i];
          continue;
        }
        let sum = 0;
        let n = 0;
        if (y > 0) { sum += cur[i - W]; n++; }
        if (y < H - 1) { sum += cur[i + W]; n++; }
        if (x > 0) { sum += cur[i - 1]; n++; }
        if (x < W - 1) { sum += cur[i + 1]; n++; }
        nxt[i] = n ? sum / n : cur[i];
      }
    }
    [cur, nxt] = [nxt, cur];
  }
  const out = new GrayImage(W, H);
  for (let i = 0; i < cur.length; i++) out.data[i] = Math.round(Math.max(0, Math.min(255, cur[i])));
  return out;
}

/** 可分离高斯模糊（浮点）。 */
export function gaussianBlurGray(img: GrayImage, sigma: number): Float32Array {
  const W = img.width;
  const H = img.height;
  const radius = Math.max(1, Math.ceil(sigma * 2.5));
  const k = new Float32Array(radius * 2 + 1);
  let norm = 0;
  for (let i = -radius; i <= radius; i++) {
    const v = Math.exp(-(i * i) / (2 * sigma * sigma));
    k[i + radius] = v;
    norm += v;
  }
  for (let i = 0; i < k.length; i++) k[i] /= norm;

  const tmp = new Float32Array(W * H);
  const out = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let acc = 0;
      for (let t = -radius; t <= radius; t++) {
        const xx = Math.min(W - 1, Math.max(0, x + t));
        acc += img.data[y * W + xx] * k[t + radius];
      }
      tmp[y * W + x] = acc;
    }
  }
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let acc = 0;
      for (let t = -radius; t <= radius; t++) {
        const yy = Math.min(H - 1, Math.max(0, y + t));
        acc += tmp[yy * W + x] * k[t + radius];
      }
      out[y * W + x] = acc;
    }
  }
  return out;
}

/** 局部标准差（在给定均值图上，方形窗口）。 */
function localStd(img: GrayImage, mean: Float32Array, radius: number): Float32Array {
  const W = img.width;
  const H = img.height;
  const integral = new Float64Array((W + 1) * (H + 1));
  for (let y = 0; y < H; y++) {
    let row = 0;
    for (let x = 0; x < W; x++) {
      const v = img.data[y * W + x];
      row += v * v;
      integral[(y + 1) * (W + 1) + (x + 1)] = integral[y * (W + 1) + (x + 1)] + row;
    }
  }
  const out = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const x0 = Math.max(0, x - radius);
      const x1 = Math.min(W - 1, x + radius);
      const y0 = Math.max(0, y - radius);
      const y1 = Math.min(H - 1, y + radius);
      const area = (x1 - x0 + 1) * (y1 - y0 + 1);
      const sumSq =
        integral[(y1 + 1) * (W + 1) + (x1 + 1)] -
        integral[y0 * (W + 1) + (x1 + 1)] -
        integral[(y1 + 1) * (W + 1) + x0] +
        integral[y0 * (W + 1) + x0];
      const variance = Math.max(0, sumSq / area - mean[y * W + x] ** 2);
      out[y * W + x] = Math.sqrt(variance);
    }
  }
  return out;
}

/**
 * 两级带通式光照归一化 + 对比度标准化。
 *  1) 减去大尺度高斯背景，移除方向光照与平滑渐变（保留珠点级高频）；
 *  2) 局部标准化（减局部均值、除局部标准差），让全画面珠点对比度一致；
 *  3) 线性映射到 8 位。
 */
export function normalizeIllumination(
  gray: GrayImage,
  mask: Uint8Array,
  sigma = 64,
): { gray: GrayImage; dynamicRange: number } {
  const W = gray.width;
  const H = gray.height;
  const bg = gaussianBlurGray(gray, sigma);
  const highpass = new Float32Array(W * H);
  for (let i = 0; i < highpass.length; i++) {
    if (mask[i]) highpass[i] = gray.data[i] - bg[i];
  }

  // 局部标准化（窗口约一个珠点间距的 3 倍，覆盖若干珠点而非单个）。
  const hpImg = new GrayImage(W, H);
  for (let i = 0; i < highpass.length; i++) hpImg.data[i] = Math.max(0, Math.min(255, 128 + highpass[i] * 2));
  const lMean = boxMean(hpImg, 9);
  const lStd = localStd(hpImg, lMean, 9);

  const normalized = new Float32Array(W * H);
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < normalized.length; i++) {
    if (!mask[i]) continue;
    const v = (highpass[i]) / Math.max(6, lStd[i] * 1.4);
    normalized[i] = v;
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const range = max - min || 1;
  const out = new GrayImage(W, H);
  for (let i = 0; i < normalized.length; i++) {
    if (!mask[i]) {
      out.data[i] = 0;
      continue;
    }
    const t = (normalized[i] - min) / range;
    // 拉伸到 28..228，保留余量避免量化噪声放大。
    out.data[i] = Math.round(28 + t * 200);
  }
  return { gray: out, dynamicRange: range };
}

/** Tenengrad 清晰度：Sobel 能量均值，只统计有效皮肤。 */
export function tenengrad(gray: GrayImage, mask: Uint8Array): number {
  const W = gray.width;
  const H = gray.height;
  let sum = 0;
  let n = 0;
  for (let y = 1; y < H - 1; y++) {
    for (let x = 1; x < W - 1; x++) {
      const i = y * W + x;
      if (!mask[i]) continue;
      const gx =
        -gray.data[i - W - 1] + gray.data[i - W + 1] - 2 * gray.data[i - 1] +
        2 * gray.data[i + 1] - gray.data[i + W - 1] + gray.data[i + W + 1];
      const gy =
        -gray.data[i - W - 1] - 2 * gray.data[i - W] - gray.data[i - W + 1] +
        gray.data[i + W - 1] + 2 * gray.data[i + W] + gray.data[i + W + 1];
      sum += gx * gx + gy * gy;
      n++;
    }
  }
  return n ? sum / n : 0;
}

/** 完整增强流水线（CPU 参考版）。 */
export function enhanceCpu(img: GrayImage, muzzleQuad: Point[]): EnhancedResult {
  const rect = rectify(img, muzzleQuad);
  const mask = buildMuzzleMask();
  const spec = detectSpecular(rect, mask);
  const inpainted = diffuseInpaint(rect, mask, spec);
  const { gray: normalized, dynamicRange } = normalizeIllumination(inpainted, mask);
  const sharpness = tenengrad(normalized, mask);

  let specCount = 0;
  let maskCount = 0;
  for (let i = 0; i < spec.length; i++) {
    if (spec[i]) specCount++;
    if (mask[i]) maskCount++;
  }
  return {
    gray: normalized,
    mask,
    specularRatio: maskCount ? specCount / maskCount : 1,
    dynamicRange,
    sharpness,
  };
}

/** RGBA -> Gray 便捷封装（Worker 入口用）。 */
export function rgbaToGray(img: RgbaImage): GrayImage {
  return img.toGray();
}

/** 供 WebGL 回退/小图使用的缩放封装。 */
export function downscaleGray(img: GrayImage, maxDim: number): GrayImage {
  const scale = Math.min(1, maxDim / Math.max(img.width, img.height));
  if (scale === 1) return img;
  return resizeGray(img, Math.round(img.width * scale), Math.round(img.height * scale));
}

/** 用双线性采样生成预览小图（UI 用）。 */
export function grayToPreview(img: GrayImage, w: number, h: number): GrayImage {
  return resizeGray(img, w, h);
}

export { sampleGrayBilinear };
