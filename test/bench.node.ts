/**
 * 前端流水线基准（CPU 参考实现）。
 *
 * 评测识别之前的两道关键工序，均基于带真值的合成数据：
 *  A. 鼻孔检测：成功率、鼻孔中心归一化误差、鼻镜四边形 IoU；
 *  B. 质量门限：正常采集通过率 vs 退化采集拦截率（当场重拍提示的依据）。
 *
 * 身份识别指标（rank-1 / FAR / 检索耗时）见 bench-anchor.ts。
 *
 * 用法：npm run bench:quick（快速）或 npm run bench（完整）。
 */

import {
  generateIdentity,
  renderCapture,
  makeCaptureParams,
  CANON_W,
  CANON_H,
} from '../src/core/synthesis';
import { mulberry32 } from '../src/core/rng';
import { detectMuzzle, muzzleQuadFromPair } from '../src/core/detect';
import { enhanceCpu } from '../src/core/enhancement';
import { assessQuality } from '../src/core/quality';
import { Point, dist } from '../src/core/geometry';

const quick = process.argv.includes('--quick');
const DET_N = quick ? 120 : 400;
const QUALITY_N = quick ? 80 : 300;
const DEGRADED_N = quick ? 60 : 250;

function percentile(arr: number[], p: number): number {
  const s = [...arr].sort((a, b) => a - b);
  return s[Math.min(s.length - 1, Math.floor((p / 100) * s.length))];
}
const mean = (arr: number[]) => arr.reduce((a, b) => a + b, 0) / (arr.length || 1);

function rasterize(quad: Point[], W: number, H: number): Uint8Array {
  const mask = new Uint8Array(W * H);
  for (let py = 0; py < H; py++) {
    for (let px = 0; px < W; px++) {
      const x = (px / (W - 1)) * (CANON_W - 1);
      const y = (py / (H - 1)) * (CANON_H - 1);
      let inside = false;
      for (let i = 0, j = quad.length - 1; i < quad.length; j = i++) {
        const xi = quad[i].x, yi = quad[i].y, xj = quad[j].x, yj = quad[j].y;
        const hit = yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi;
        if (hit) inside = !inside;
      }
      mask[py * W + px] = inside ? 1 : 0;
    }
  }
  return mask;
}

function quadIoU(a: Point[], b: Point[]): number {
  const W = 100, H = 75;
  const m1 = rasterize(a, W, H);
  const m2 = rasterize(b, W, H);
  let inter = 0, uni = 0;
  for (let i = 0; i < m1.length; i++) {
    if (m1[i] && m2[i]) inter++;
    if (m1[i] || m2[i]) uni++;
  }
  return uni ? inter / uni : 0;
}

async function main(): Promise<void> {
  const rng = mulberry32(20261124);
  console.log(`=== 流水线基准（检测+质量门限，quick=${quick}）===\n`);

  // A. 鼻孔检测专项。
  console.log('[1/2] 鼻孔检测评测...');
  const errs: number[] = [];
  const ious: number[] = [];
  let detectFail = 0;
  for (let i = 0; i < DET_N; i++) {
    const cowId = `COW-${String(i + 1).padStart(5, '0')}`;
    const identity = generateIdentity(cowId);
    const cap = renderCapture(identity, makeCaptureParams(rng, 'field'));
    const det = detectMuzzle(cap.image.toGray());
    if (!det.ok || !det.nostrils || !det.muzzleQuad) {
      detectFail++;
      continue;
    }
    const s = dist(cap.nostrils.left, cap.nostrils.right);
    errs.push(
      (dist(det.nostrils.left, cap.nostrils.left) + dist(det.nostrils.right, cap.nostrils.right)) / (2 * s),
    );
    const { quad: gt } = muzzleQuadFromPair(cap.nostrils, cap.image.width, cap.image.height);
    const sxq = CANON_W / cap.image.width;
    const syq = CANON_H / cap.image.height;
    const scale = (q: Point[]) => q.map((p) => ({ x: p.x * sxq, y: p.y * syq }));
    ious.push(quadIoU(scale(det.muzzleQuad), scale(gt)));
  }
  console.log(
    `检测成功率 ${(((DET_N - detectFail) / DET_N) * 100).toFixed(1)}%，` +
      `鼻孔归一化误差 均值=${mean(errs).toFixed(4)} p95=${percentile(errs, 95).toFixed(4)}，` +
      `鼻镜四边形 IoU 均值=${mean(ious).toFixed(4)}\n`,
  );

  // B. 质量门限。
  console.log('[2/2] 质量门限评测...');
  let fieldPass = 0;
  for (let i = 0; i < QUALITY_N; i++) {
    const cowId = `COW-${String((i % 200) + 1).padStart(5, '0')}`;
    const cap = renderCapture(generateIdentity(cowId), makeCaptureParams(rng, 'field'));
    const det = detectMuzzle(cap.image.toGray());
    const enh = det.ok && det.muzzleQuad ? enhanceCpu(cap.image.toGray(), det.muzzleQuad) : null;
    if (assessQuality(det, enh).pass) fieldPass++;
  }
  let degradedBlock = 0;
  for (let i = 0; i < DEGRADED_N; i++) {
    const cowId = `COW-${String((i % 200) + 1).padStart(5, '0')}`;
    const cap = renderCapture(generateIdentity(cowId), makeCaptureParams(rng, 'degraded'));
    const det = detectMuzzle(cap.image.toGray());
    const enh = det.ok && det.muzzleQuad ? enhanceCpu(cap.image.toGray(), det.muzzleQuad) : null;
    if (!assessQuality(det, enh).pass) degradedBlock++;
  }
  console.log(
    `正常采集通过率 ${((fieldPass / QUALITY_N) * 100).toFixed(1)}%，` +
      `退化采集拦截率 ${((degradedBlock / DEGRADED_N) * 100).toFixed(1)}%\n`,
  );

  console.log(
    'SUMMARY_JSON=' +
      JSON.stringify({
        detectSuccessRate: (DET_N - detectFail) / DET_N,
        nostrilErrMean: mean(errs),
        nostrilErrP95: percentile(errs, 95),
        quadIoU: mean(ious),
        fieldPassRate: fieldPass / QUALITY_N,
        degradedBlockRate: degradedBlock / DEGRADED_N,
      }),
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
