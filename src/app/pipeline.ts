/**
 * 浏览器端图像处理服务：ImageData -> 灰度 -> 鼻孔检测 -> 矩形化增强 ->
 * 质量门限 -> 锚点指纹。CPU 参考实现，与 WebGL2 增强流水线算法一致；
 * 特征提取与比对在 Web Worker 中调用本模块的纯函数。
 */

import { GrayImage, RgbaImage, Point } from '../core/geometry';
import { detectMuzzle, DetectionResult } from '../core/detect';
import {
  enhanceCpu,
  EnhancedResult,
  rectifyByNostrils,
  buildMuzzleMask,
} from '../core/enhancement';
import { assessQuality, QualityReport } from '../core/quality';
import {
  AnchorFingerprint,
  AnchorPoint,
  buildAnchorFingerprint,
} from '../core/anchor';
import { detectBeads } from '../core/beads';
import { CANON_W, CANON_H } from '../core/synthesis';
import { GlEnhancer } from './gl-enhancer';
import { tenengrad } from '../core/enhancement';

export interface ProcessedCapture {
  detection: DetectionResult;
  enhanced: EnhancedResult | null;
  quality: QualityReport;
  fingerprint: AnchorFingerprint | null;
  /** 供 UI 叠加显示的检测四边形/鼻孔。 */
  overlay: { quad: Point[] | null; nostrils: DetectionResult['nostrils'] };
}

export function imageDataToGray(img: ImageData): GrayImage {
  const rgba = new RgbaImage(img.width, img.height, new Uint8ClampedArray(img.data));
  return rgba.toGray();
}

/** 生成 96x72 缩略图（灰度数组）。 */
export function makeThumb(gray: GrayImage): number[] {
  const tw = 96;
  const th = 72;
  const out: number[] = new Array(tw * th);
  for (let y = 0; y < th; y++) {
    for (let x = 0; x < tw; x++) {
      const sx = Math.min(gray.width - 1, Math.floor((x / tw) * gray.width));
      const sy = Math.min(gray.height - 1, Math.floor((y / th) * gray.height));
      out[y * tw + x] = gray.data[sy * gray.width + sx];
    }
  }
  return out;
}

/**
 * 处理一次采集：检测鼻孔 -> 增强（WebGL2 优先，CPU 回退）-> 质量门限 -> 锚点指纹。
 * 正常流程只在质量通过时注册或比对；confirmedAnchors 用于传入注册确认的 minutiae。
 */
export function processCapture(
  img: ImageData,
  cowId: string,
  opts: { useNostrilRectify?: boolean; confirmedAnchors?: AnchorPoint[] } = {},
): ProcessedCapture {
  const gray = imageDataToGray(img);
  const detection = detectMuzzle(gray);
  let enhanced: EnhancedResult | null = null;
  let fingerprint: AnchorFingerprint | null = null;

  if (detection.ok && detection.muzzleQuad && detection.nostrils) {
    enhanced = enhanceWithPreferredBackend(img, gray, detection, opts.useNostrilRectify ?? true);
    if (enhanced) {
      // 优先使用确认锚点（注册确认的 minutiae）；否则从增强图检测。
      const anchors = opts.confirmedAnchors && opts.confirmedAnchors.length >= 40
        ? opts.confirmedAnchors
        : anchorsFromEnhanced(enhanced);
      fingerprint = buildAnchorFingerprint(cowId, anchors);
    }
  }

  const quality = assessQuality(detection, enhanced);
  return {
    detection,
    enhanced,
    quality,
    fingerprint,
    overlay: { quad: detection.muzzleQuad, nostrils: detection.nostrils },
  };
}

/** 对已矩形化的标准画布图执行反光修复+光照归一化（增强的 2..5 步）。 */
function enhanceOnRect(rect: GrayImage): EnhancedResult {
  // 复用完整流水线：标准画布的四角即四边形，结果与输入等大。
  const quad: Point[] = [
    { x: 0, y: 0 },
    { x: CANON_W - 1, y: 0 },
    { x: CANON_W - 1, y: CANON_H - 1 },
    { x: 0, y: CANON_H - 1 },
  ];
  return enhanceCpu(rect, quad);
}

let glEnhancer: GlEnhancer | null | undefined;

/**
 * 优先 WebGL2 增强，失败/不支持时回退 CPU 参考实现。
 * Worker 中没有 DOM 时 GlEnhancer 构造会抛错，自动走 CPU。
 */
function enhanceWithPreferredBackend(
  img: ImageData,
  gray: GrayImage,
  detection: DetectionResult,
  useNostrilRectify: boolean,
): EnhancedResult {
  if (glEnhancer === undefined && typeof document !== 'undefined' && GlEnhancer.isSupported()) {
    try {
      const canvas = document.createElement('canvas');
      canvas.width = CANON_W;
      canvas.height = CANON_H;
      glEnhancer = new GlEnhancer(canvas);
    } catch {
      glEnhancer = null;
    }
  }
  if (glEnhancer && detection.nostrils) {
    try {
      const gpu = glEnhancer.enhance(img, {
        quad: detection.muzzleQuad ?? undefined,
        nostrils: useNostrilRectify ? detection.nostrils : undefined,
      });
      const mask = buildMuzzleMask();
      return {
        gray: gpu.gray,
        mask,
        specularRatio: gpu.specularRatio,
        // GPU 路径的动态范围用归一化图标准差近似，清晰度仍用 Tenengrad。
        dynamicRange: computeDynamicRange(gpu.gray, mask),
        sharpness: tenengrad(gpu.gray, mask),
      };
    } catch {
      // GPU 异常则回退 CPU。
      glEnhancer = null;
    }
  }
  if (useNostrilRectify && detection.nostrils) {
    return enhanceOnRect(rectifyByNostrils(gray, detection.nostrils));
  }
  return enhanceCpu(gray, detection.muzzleQuad!);
}

function computeDynamicRange(gray: GrayImage, mask: Uint8Array): number {
  let sum = 0, sumSq = 0, n = 0;
  for (let i = 0; i < gray.data.length; i++) {
    if (!mask[i]) continue;
    const v = gray.data[i];
    sum += v; sumSq += v * v; n++;
  }
  const mean = sum / (n || 1);
  return Math.sqrt(Math.max(0, sumSq / (n || 1) - mean * mean));
}

/**
 * 从增强后的标准画布提取锚点。
 * 生产环境这里接珠点检测器并经注册确认；在本合成数据系统中，锚点由
 * 增强图上稳定的高响应环形滤波器珠点构成，跨采集可重复。
 */
export function anchorsFromEnhanced(enhanced: EnhancedResult): AnchorPoint[] {
  const kps = detectBeads(enhanced.gray, enhanced.mask);
  return kps.map((k) => ({ x: k.x, y: k.y }));
}
