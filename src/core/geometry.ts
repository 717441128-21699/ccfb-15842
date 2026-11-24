/**
 * 基础几何与图像原语：点、鼻孔地标、透视变换、灰度图容器。
 * 这些类型不依赖 DOM，可同时在浏览器与 Node 基准中使用。
 */

export interface Point {
  x: number;
  y: number;
}

export interface Size {
  width: number;
  height: number;
}

/** 两个鼻孔的中心地标，检测/合成阶段共用。 */
export interface NostrilPair {
  left: Point;
  right: Point;
}

/** 单通道 8 位灰度图，行优先。 */
export class GrayImage {
  readonly width: number;
  readonly height: number;
  readonly data: Uint8ClampedArray;

  constructor(width: number, height: number, data?: Uint8ClampedArray) {
    this.width = width;
    this.height = height;
    this.data = data ?? new Uint8ClampedArray(width * height);
  }

  idx(x: number, y: number): number {
    return y * this.width + x;
  }

  get(x: number, y: number): number {
    if (x < 0 || y < 0 || x >= this.width || y >= this.height) return 0;
    return this.data[y * this.width + x];
  }

  set(x: number, y: number, v: number): void {
    this.data[y * this.width + x] = v;
  }

  static copyFrom(src: GrayImage): GrayImage {
    return new GrayImage(src.width, src.height, new Uint8ClampedArray(src.data));
  }
}

/** RGBA 8 位图像。 */
export class RgbaImage {
  readonly width: number;
  readonly height: number;
  readonly data: Uint8ClampedArray;

  constructor(width: number, height: number, data?: Uint8ClampedArray) {
    this.width = width;
    this.height = height;
    this.data = data ?? new Uint8ClampedArray(width * height * 4);
  }

  toGray(): GrayImage {
    const out = new GrayImage(this.width, this.height);
    const d = this.data;
    const o = out.data;
    for (let i = 0, j = 0; i < d.length; i += 4, j++) {
      o[j] = Math.round(0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]);
    }
    return out;
  }

  static fromGray(gray: GrayImage): RgbaImage {
    const out = new RgbaImage(gray.width, gray.height);
    for (let i = 0, j = 0; i < out.data.length; i += 4, j++) {
      const v = gray.data[j];
      out.data[i] = v;
      out.data[i + 1] = v;
      out.data[i + 2] = v;
      out.data[i + 3] = 255;
    }
    return out;
  }
}

export function dist(a: Point, b: Point): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

export function midpoint(a: Point, b: Point): Point {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

/**
 * 3x3 透视变换（列向量），由 4 个源点 -> 4 个目标点求解。
 * 用于把检测到的鼻镜四边形矩形化为标准画布。
 */
export type Mat3 = number[]; // 长度 9，行优先

export function identityMat3(): Mat3 {
  return [1, 0, 0, 0, 1, 0, 0, 0, 1];
}

function solveLinear8(rows: number[][], rhs: number[]): number[] {
  // 高斯消元（8 元），方程规模固定且小，无需通用库。
  const n = 8;
  const a = rows.map((r, i) => [...r, rhs[i]]);
  for (let col = 0; col < n; col++) {
    let piv = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(a[r][col]) > Math.abs(a[piv][col])) piv = r;
    }
    if (piv !== col) {
      const tmp = a[piv];
      a[piv] = a[col];
      a[col] = tmp;
    }
    const d = a[col][col] || 1e-12;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = a[r][col] / d;
      if (f === 0) continue;
      for (let c = col; c <= n; c++) a[r][c] -= f * a[col][c];
    }
  }
  const x = new Array<number>(n);
  for (let i = 0; i < n; i++) x[i] = a[i][n] / a[i][i];
  return x;
}

/** 计算 src -> dst 的透视矩阵 H（dst = H * src，齐次坐标）。 */
export function findHomography(src: Point[], dst: Point[]): Mat3 {
  const rows: number[][] = [];
  const rhs: number[] = [];
  for (let i = 0; i < 4; i++) {
    const { x, y } = src[i];
    const { x: u, y: v } = dst[i];
    rows.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    rhs.push(u);
    rows.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
    rhs.push(v);
  }
  const h = solveLinear8(rows, rhs);
  return [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1];
}

/** 透视矩阵求逆（伴随矩阵法）。 */
export function invertMat3(m: Mat3): Mat3 {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const D = -(b * i - c * h);
  const E = a * i - c * g;
  const F = -(a * h - b * g);
  const G = b * f - c * e;
  const H = -(a * f - c * d);
  const I = a * e - b * d;
  const det = a * A + b * B + c * C;
  const s = 1 / (det || 1e-12);
  return [A * s, D * s, G * s, B * s, E * s, H * s, C * s, F * s, I * s];
}

export function applyMat3(m: Mat3, p: Point): Point {
  const w = m[6] * p.x + m[7] * p.y + m[8];
  return {
    x: (m[0] * p.x + m[1] * p.y + m[2]) / w,
    y: (m[3] * p.x + m[4] * p.y + m[5]) / w,
  };
}

/** 双线性采样灰度图，越界返回 0。 */
export function sampleGrayBilinear(img: GrayImage, x: number, y: number): number {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  if (x0 < -1 || y0 < -1 || x0 >= img.width || y0 >= img.height) return 0;
  const fx = x - x0;
  const fy = y - y0;
  const v00 = img.get(x0, y0);
  const v10 = img.get(x0 + 1, y0);
  const v01 = img.get(x0, y0 + 1);
  const v11 = img.get(x0 + 1, y0 + 1);
  const top = v00 * (1 - fx) + v10 * fx;
  const bot = v01 * (1 - fx) + v11 * fx;
  return top * (1 - fy) + bot * fy;
}

/** 用透视变换把 src 矩形化到 dst 尺寸（反向映射）。 */
export function warpPerspective(
  src: GrayImage,
  srcQuad: Point[],
  outW: number,
  outH: number,
): GrayImage {
  const dstQuad: Point[] = [
    { x: 0, y: 0 },
    { x: outW - 1, y: 0 },
    { x: outW - 1, y: outH - 1 },
    { x: 0, y: outH - 1 },
  ];
  const dstToSrc = invertMat3(findHomography(dstQuad, srcQuad));
  const out = new GrayImage(outW, outH);
  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      const p = applyMat3(dstToSrc, { x, y });
      out.data[y * outW + x] = Math.round(sampleGrayBilinear(src, p.x, p.y));
    }
  }
  return out;
}

/** 双线性缩放灰度图。 */
export function resizeGray(src: GrayImage, outW: number, outH: number): GrayImage {
  const out = new GrayImage(outW, outH);
  const sx = src.width / outW;
  const sy = src.height / outH;
  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      out.data[y * outW + x] = Math.round(
        sampleGrayBilinear(src, (x + 0.5) * sx - 0.5, (y + 0.5) * sy - 0.5),
      );
    }
  }
  return out;
}
