/**
 * 鼻镜检测：在采集图中找两个鼻孔，按鼻孔位置摆正鼻镜区域。
 *
 * 鼻孔是图中最暗、成对出现、近似水平排列的两个椭圆。算法：
 * 多档暗阈值 -> 连通域（BFS）-> 图像矩拟合椭圆 -> 候选对评分
 * （尺寸相似、近水平、间距合理、推算的鼻镜四边形落在画面内）。
 * 不依赖真值，真实拍摄场景同样适用。
 */

import { GrayImage, Point, NostrilPair } from './geometry';

export interface DetectedNostril {
  center: Point;
  axisA: number; // 长半轴
  axisB: number; // 短半轴
  angle: number;
  area: number;
  meanDark: number;
}

export interface DetectionResult {
  ok: boolean;
  confidence: number;
  nostrils: NostrilPair | null;
  components: DetectedNostril[];
  /** 鼻镜四边形（左上、右上、右下、左下），图像坐标。 */
  muzzleQuad: Point[] | null;
  reasons: string[];
}

interface Component {
  pixels: number;
  sumX: number;
  sumY: number;
  sumXX: number;
  sumYY: number;
  sumXY: number;
  sumGray: number;
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

function connectedComponents(img: GrayImage, threshold: number): Component[] {
  const W = img.width;
  const H = img.height;
  const labels = new Int32Array(W * H).fill(-1);
  const comps: Component[] = [];
  const stack: number[] = [];

  for (let start = 0; start < W * H; start++) {
    if (labels[start] !== -1 || img.data[start] >= threshold) continue;
    const id = comps.length;
    const c: Component = {
      pixels: 0,
      sumX: 0,
      sumY: 0,
      sumXX: 0,
      sumYY: 0,
      sumXY: 0,
      sumGray: 0,
      minX: W,
      maxX: -1,
      minY: H,
      maxY: -1,
    };
    labels[start] = id;
    stack.push(start);
    while (stack.length) {
      const idx = stack.pop()!;
      const x = idx % W;
      const y = (idx / W) | 0;
      c.pixels++;
      c.sumX += x;
      c.sumY += y;
      c.sumXX += x * x;
      c.sumYY += y * y;
      c.sumXY += x * y;
      c.sumGray += img.data[idx];
      if (x < c.minX) c.minX = x;
      if (x > c.maxX) c.maxX = x;
      if (y < c.minY) c.minY = y;
      if (y > c.maxY) c.maxY = y;

      const x0 = x > 0 ? x - 1 : 0;
      const x1 = x < W - 1 ? x + 1 : W - 1;
      const y0 = y > 0 ? y - 1 : 0;
      const y1 = y < H - 1 ? y + 1 : H - 1;
      for (let ny = y0; ny <= y1; ny++) {
        for (let nx = x0; nx <= x1; nx++) {
          const ni = ny * W + nx;
          if (labels[ni] === -1 && img.data[ni] < threshold) {
            labels[ni] = id;
            stack.push(ni);
          }
        }
      }
    }
    comps.push(c);
  }
  return comps;
}

function fitComponent(c: Component): DetectedNostril | null {
  const n = c.pixels;
  const cx = c.sumX / n;
  const cy = c.sumY / n;
  const mxx = c.sumXX / n - cx * cx;
  const myy = c.sumYY / n - cy * cy;
  const mxy = c.sumXY / n - cx * cy;
  const common = Math.sqrt(((mxx - myy) / 2) ** 2 + mxy * mxy);
  const lambda1 = (mxx + myy) / 2 + common;
  const lambda2 = (mxx + myy) / 2 - common;
  if (lambda1 <= 0 || lambda2 <= 0) return null;
  const axisA = Math.sqrt(lambda1) * 2;
  const axisB = Math.sqrt(lambda2) * 2;
  return {
    center: { x: cx, y: cy },
    axisA,
    axisB,
    angle: 0.5 * Math.atan2(2 * mxy, mxx - myy),
    area: n,
    meanDark: c.sumGray / n,
  };
}

/** 依据鼻孔对推算鼻镜四边形，几何比例由合成器真值轮廓标定（见 test/calibrate-quad）。 */
export function muzzleQuadFromPair(
  pair: NostrilPair,
  imgW: number,
  imgH: number,
  margin = 1.04,
): { quad: Point[]; inFrameRatio: number } {
  const left = pair.left;
  const right = pair.right;
  const dx = right.x - left.x;
  const dy = right.y - left.y;
  const s = Math.hypot(dx, dy);
  const ux = dx / s;
  const uy = dy / s;
  const vx = -uy; // u 顺时针 90°
  const vy = ux;
  const mx = (left.x + right.x) / 2;
  const my = (left.y + right.y) / 2;

  // 真值标定（p95，含小边距）：halfW≈1.46、top≈1.37、bottom≈0.80（单位=鼻孔间距 s）。
  const halfW = 1.40 * s * margin;
  const top = 1.31 * s * margin;
  const bottom = 0.76 * s * margin;

  const corner = (su: number, sv: number): Point => ({
    x: mx + ux * su + vx * sv,
    y: my + uy * su + vy * sv,
  });
  const quad = [
    corner(-halfW, -top),
    corner(halfW, -top),
    corner(halfW, bottom),
    corner(-halfW, bottom),
  ];

  let inside = 0;
  for (const q of quad) {
    if (q.x >= 0 && q.y >= 0 && q.x < imgW && q.y < imgH) inside++;
  }
  return { quad, inFrameRatio: inside / 4 };
}

interface PairScore {
  i: number;
  j: number;
  score: number;
  quad: Point[];
  inFrame: number;
}

function scorePair(a: DetectedNostril, b: DetectedNostril, imgW: number, imgH: number): PairScore | null {
  // 保证 a 在左、b 在右。
  let [l, r] = a.center.x <= b.center.x ? [a, b] : [b, a];
  const dx = r.center.x - l.center.x;
  const dy = r.center.y - l.center.y;
  const s = Math.hypot(dx, dy);
  if (s < imgW * 0.16 || s > imgW * 0.62) return null;
  const tilt = Math.abs(Math.atan2(dy, dx));
  if (tilt > 30 * Math.PI / 180) return null;

  const areaRatio = Math.min(l.area, r.area) / Math.max(l.area, r.area);
  if (areaRatio < 0.4) return null;
  const elongOk =
    l.axisB / l.axisA > 0.35 && r.axisB / r.axisA > 0.35 && l.axisB / l.axisA < 1.05 && r.axisB / r.axisA < 1.05;

  const pair: NostrilPair = { left: l.center, right: r.center };
  // margin=1：取景较紧时允许鼻镜四边形部分越界，只要鼻孔完整即可。
  const { quad, inFrameRatio } = muzzleQuadFromPair(pair, imgW, imgH, 1.0);
  // 至少两个角在画面内，且两个鼻孔中心本身都在画面内（手机取景的常见情形）。
  if (inFrameRatio < 0.25) return null;
  if (
    l.center.x < 0 || l.center.y < 0 || l.center.x >= imgW || l.center.y >= imgH ||
    r.center.x < 0 || r.center.y < 0 || r.center.x >= imgW || r.center.y >= imgH
  ) {
    return null;
  }

  let score = 0;
  score += areaRatio * 0.3;
  score += (1 - Math.min(1, tilt / (30 * Math.PI / 180))) * 0.25;
  score += inFrameRatio * 0.15;
  score += (elongOk ? 0.15 : 0);
  score += Math.max(0, 1 - Math.abs(l.meanDark - r.meanDark) / 40) * 0.15;
  return { i: -1, j: -1, score, quad, inFrame: inFrameRatio };
}

export function detectMuzzle(img: GrayImage): DetectionResult {
  const W = img.width;
  const H = img.height;
  const reasons: string[] = [];
  const minArea = W * H * 0.006;
  const maxArea = W * H * 0.1;

  interface BestCandidate {
    score: number;
    pair: NostrilPair;
    quad: Point[];
    comps: DetectedNostril[];
  }
  // 用持有对象规避 TS 对嵌套循环内 let 赋值的窄化限制。
  const bestRef: { value: BestCandidate | null } = { value: null };

  for (const threshold of [55, 70, 85, 100, 115]) {
    const raw = connectedComponents(img, threshold);
    const fitted: DetectedNostril[] = [];
    for (const c of raw) {
      if (c.pixels < minArea || c.pixels > maxArea) continue;
      const f = fitComponent(c);
      if (f) fitted.push(f);
    }
    if (fitted.length < 2) continue;

    for (let i = 0; i < fitted.length; i++) {
      for (let j = i + 1; j < fitted.length; j++) {
        const ps = scorePair(fitted[i], fitted[j], W, H);
        if (!ps) continue;
        const l = fitted[i].center.x <= fitted[j].center.x ? fitted[i] : fitted[j];
        const r = l === fitted[i] ? fitted[j] : fitted[i];
        const candidate: BestCandidate = {
          score: ps.score,
          pair: { left: l.center, right: r.center },
          quad: ps.quad,
          comps: [l, r],
        };
        const prev = bestRef.value;
        if (!prev || candidate.score > prev.score) bestRef.value = candidate;
      }
    }
    if (bestRef.value && bestRef.value.score > 0.72) break; // 足够可信就不再放宽阈值
  }

  const best = bestRef.value;
  if (!best) {
    reasons.push('未能找到成对的鼻孔，请让两个鼻孔都入镜并避免阴影遮挡');
    return { ok: false, confidence: 0, nostrils: null, components: [], muzzleQuad: null, reasons };
  }

  const confidence = Math.max(0, Math.min(1, best.score));
  if (confidence < 0.55) {
    reasons.push('鼻孔定位置信度偏低，建议正对牛鼻重拍');
  }
  return {
    ok: confidence >= 0.55,
    confidence,
    nostrils: best.pair,
    components: best.comps,
    muzzleQuad: best.quad,
    reasons,
  };
}
