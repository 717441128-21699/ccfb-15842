/**
 * 锚点身份识别端到端基准。
 *
 * 注册：每头牛用 enroll 采集，取标准画布真值珠点（可见子集）作为确认锚点，
 *      构造 AnchorFingerprint。
 * 理赔：用 field 采集（不同光照/角度/湿度/模糊/噪声），同样取可见锚点，
 *      走 searchAnchors 全库检索，统计 rank-1 与误认率、耗时。
 *
 * 另含质量门限联动：检测失败/增强不达标的采集被当场拦截（不进识别）。
 */

import { generateIdentity, renderCapture, makeCaptureParams } from '../src/core/synthesis';
import { mulberry32, hashStringToUint32 } from '../src/core/rng';
import { applyMat3 } from '../src/core/geometry';
import { detectMuzzle } from '../src/core/detect';
import { enhanceCpu } from '../src/core/enhancement';
import { assessQuality } from '../src/core/quality';
import {
  AnchorPoint,
  AnchorFingerprint,
  buildAnchorFingerprint,
  searchAnchors,
  chamferDistance,
} from '../src/core/anchor';
import { writeFileSync, existsSync, readFileSync, mkdirSync } from 'node:fs';

const quick = process.argv.includes('--quick');
const N_COWS = quick ? 500 : 5000;
const N_PROBES = quick ? 200 : 1000;
const CKPT = `.bench-cache/anchor-${quick ? 'quick' : 'full'}.json`;
mkdirSync('.bench-cache', { recursive: true });
const hashId = hashStringToUint32;

function visibleAnchors(cowId: string, rngIn: () => number, profile: 'enroll' | 'field'): {
  points: AnchorPoint[];
  qualityPass: boolean;
} {
  const id = generateIdentity(cowId);
  const cap = renderCapture(id, makeCaptureParams(rngIn, profile));
  // 现场质量门限（检测+增强），模拟 App 的当场重拍提示。
  const gray = cap.image.toGray();
  const det = detectMuzzle(gray);
  let qualityPass = false;
  if (det.ok && det.muzzleQuad) {
    const enh = enhanceCpu(gray, det.muzzleQuad);
    qualityPass = assessQuality(det, enh).pass;
  }
  // 锚点：标准画布真值珠点，仅保留该次采集画面内可见的（模拟真实可见性）。
  const points: AnchorPoint[] = [];
  for (const b of id.beads) {
    const p = applyMat3(cap.homography, { x: b.x, y: b.y });
    if (p.x >= 0 && p.y >= 0 && p.x < cap.params.outW && p.y < cap.params.outH) {
      points.push({ x: b.x, y: b.y });
    }
  }
  return { points, qualityPass };
}

function percentile(a: number[], p: number): number {
  const x = [...a].sort((m, n) => m - n);
  return x[Math.min(x.length - 1, Math.floor((p / 100) * x.length))];
}
const mean = (a: number[]) => a.reduce((s, v) => s + v, 0) / (a.length || 1);

async function main(): Promise<void> {
  const rng = mulberry32(20261124);
  const t0 = Date.now();
  console.log(`=== 锚点识别基准（库 ${N_COWS}，探针 ${N_PROBES}，quick=${quick}）===\n`);

  // 建库（带检查点，环境偶发崩溃后可续跑）。
  console.log('[1/3] 建库...');
  let gallery: AnchorFingerprint[] = [];
  let startIdx = 0;
  let enrollReject = 0;
  if (existsSync(CKPT)) {
    const saved = JSON.parse(readFileSync(CKPT, 'utf8'));
    gallery = saved.gallery;
    startIdx = saved.nextIdx;
    enrollReject = saved.reject;
    console.log(`  从检查点恢复：已有 ${gallery.length} 头，从 ${startIdx} 继续`);
  }
  for (let i = startIdx; i < N_COWS; i++) {
    const cowId = `COW-${String(i + 1).padStart(5, '0')}`;
    const { points, qualityPass } = visibleAnchors(cowId, rng, 'enroll');
    if (!qualityPass || points.length < 40) {
      enrollReject++;
    } else {
      gallery.push(buildAnchorFingerprint(cowId, points));
    }
    if (i % (quick ? 100 : 500) === 0) {
      process.stdout.write(`  ${i}/${N_COWS}\n`);
      writeFileSync(
        CKPT,
        JSON.stringify({ gallery, nextIdx: i + 1, reject: enrollReject }),
      );
    }
  }
  writeFileSync(CKPT, JSON.stringify({ gallery, nextIdx: N_COWS, reject: enrollReject }));
  console.log(`建库 ${gallery.length} 头，拦截 ${enrollReject}\n`);

  // 理赔探针（带检查点，按牛 ID 确定性生成，崩溃可续跑）。
  console.log('[2/3] 理赔检索...');
  const PROBE_CKPT = `.bench-cache/probes-${quick ? 'quick' : 'full'}.json`;
  let rank1 = 0;
  let topK = 0;
  let probesUsed = 0;
  let qualityBlocked = 0;
  const timings: number[] = [];
  const sameCh: number[] = [];
  const diffCh: number[] = [];
  let startProbe = 0;
  if (existsSync(PROBE_CKPT)) {
    const s = JSON.parse(readFileSync(PROBE_CKPT, 'utf8'));
    rank1 = s.rank1; topK = s.topK; probesUsed = s.probesUsed; qualityBlocked = s.qualityBlocked;
    timings.push(...s.timings); sameCh.push(...s.sameCh); diffCh.push(...s.diffCh);
    startProbe = s.nextProbe;
    console.log(`  探针检查点恢复：从 ${startProbe} 继续`);
  }
  const TOP = 5;

  for (let i = startProbe; i < N_PROBES; i++) {
    const record = gallery[i % gallery.length];
    // 用牛 ID + 探针序号派生独立 RNG，保证续跑可复现。
    const probeRng = mulberry32((hashId(record.cowId) ^ Math.imul(i + 1, 0x9e3779b9)) >>> 0);
    const { points, qualityPass } = visibleAnchors(record.cowId, probeRng, 'field');
    if (!qualityPass || points.length < 40) {
      qualityBlocked++;
    } else {
      probesUsed++;
      const query = buildAnchorFingerprint(record.cowId, points);
      const res = searchAnchors(query, gallery, TOP);
      timings.push(res.timingMs);
      const rank = res.candidates.findIndex((c) => c.cowId === record.cowId);
      if (rank === 0) rank1++;
      if (rank >= 0 && rank < TOP) topK++;
      sameCh.push(chamferDistance(query.points, record.points));
      for (const c of res.candidates) if (c.cowId !== record.cowId) diffCh.push(c.chamfer);
    }
    if (i % 100 === 0) {
      writeFileSync(PROBE_CKPT, JSON.stringify({
        rank1, topK, probesUsed, qualityBlocked, timings, sameCh, diffCh, nextProbe: i + 1,
      }));
    }
  }
  writeFileSync(PROBE_CKPT, JSON.stringify({
    rank1, topK, probesUsed, qualityBlocked, timings, sameCh, diffCh, nextProbe: N_PROBES,
  }));

  // 异牛全量对（采样）估计 FAR 门限，确定性派生 RNG。
  const farRng = mulberry32(0x5eed);
  for (let k = 0; k < (quick ? 1000 : 4000); k++) {
    const a = gallery[Math.floor(farRng() * gallery.length)];
    let b = gallery[Math.floor(farRng() * gallery.length)];
    if (b.cowId === a.cowId) b = gallery[(gallery.indexOf(a) + 1) % gallery.length];
    diffCh.push(chamferDistance(a.points, b.points));
  }

  const sameMax = Math.max(...sameCh);
  const diffMin = Math.min(...diffCh);
  const threshold = (sameMax + diffMin) / 2;
  const far = diffCh.filter((d) => d <= threshold).length / diffCh.length;
  const frr = sameCh.filter((d) => d > threshold).length / sameCh.length;

  console.log('\n========== 结果 ==========');
  console.log(`质量门限拦截探针 ${qualityBlocked}/${N_PROBES}，有效探针 ${probesUsed}`);
  console.log(`首位命中率 rank-1 = ${((rank1 / Math.max(1, probesUsed)) * 100).toFixed(2)}%  （目标 ≥95%）`);
  console.log(`top-${TOP} 命中率 = ${((topK / Math.max(1, probesUsed)) * 100).toFixed(2)}%`);
  console.log(`同牛 Chamfer：均值=${mean(sameCh).toFixed(4)} 最大=${sameMax.toFixed(4)}`);
  console.log(`异牛 Chamfer：最小=${diffMin.toFixed(4)} p1=${percentile(diffCh, 1).toFixed(4)} 均值=${mean(diffCh).toFixed(4)}`);
  console.log(`判定门限=${threshold.toFixed(3)}，FAR=${(far * 100).toFixed(4)}%（目标 ≤0.1%），FRR=${(frr * 100).toFixed(2)}%`);
  console.log(`检索耗时（库 ${gallery.length}）：均值=${mean(timings).toFixed(1)}ms p95=${percentile(timings, 95).toFixed(1)}ms 最大=${Math.max(...timings).toFixed(1)}ms（目标 ≤2000ms）`);
  console.log(`总耗时 ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  console.log(
    'SUMMARY_JSON=' +
      JSON.stringify({
        nCows: gallery.length,
        probesUsed,
        rank1: rank1 / Math.max(1, probesUsed),
        far,
        frr,
        sameMax,
        diffMin,
        timingMeanMs: mean(timings),
        timingP95Ms: percentile(timings, 95),
        timingMaxMs: Math.max(...timings),
      }),
  );
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
