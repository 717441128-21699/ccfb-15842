/**
 * 采集质量门限：当场判定糊、反光、定位失败等问题并给出重拍提示。
 * 阈值在 test/bench.node.ts 中依据合成数据分布做过标定。
 */

import { DetectionResult } from './detect';
import { EnhancedResult } from './enhancement';

export const QUALITY_THRESHOLDS = {
  minDetectConfidence: 0.55,
  // 由合成数据 field/degraded 指标分布做网格搜索标定（正常通过率与退化拦截率的平衡点）。
  maxSpecularRatio: 0.015,
  minSharpness: 4500,
  minDynamicRange: 0.1,
};

export interface QualityReport {
  pass: boolean;
  score: number;
  metrics: {
    detectConfidence: number;
    specularRatio: number;
    sharpness: number;
    dynamicRange: number;
  };
  reasons: string[];
  retakeHint: string;
}

export function assessQuality(
  detection: DetectionResult,
  enhanced: EnhancedResult | null,
): QualityReport {
  const reasons: string[] = [];
  const m = {
    detectConfidence: detection.confidence,
    specularRatio: enhanced?.specularRatio ?? 1,
    sharpness: enhanced?.sharpness ?? 0,
    dynamicRange: enhanced?.dynamicRange ?? 0,
  };

  if (!detection.ok || m.detectConfidence < QUALITY_THRESHOLDS.minDetectConfidence) {
    reasons.push('鼻孔定位失败或置信度过低');
  }
  if (enhanced) {
    if (m.specularRatio > QUALITY_THRESHOLDS.maxSpecularRatio) {
      reasons.push(`湿反光亮斑占比 ${(m.specularRatio * 100).toFixed(1)}%，超出门限`);
    }
    if (m.sharpness < QUALITY_THRESHOLDS.minSharpness) {
      reasons.push('清晰度不足，画面可能失焦或牛在甩头');
    }
    if (m.dynamicRange < QUALITY_THRESHOLDS.minDynamicRange) {
      reasons.push('图像动态范围过低，纹路对比不足');
    }
  }

  // 0..100 综合分，各项按门限归一化后取最小短板加权。
  const sDetect = Math.min(1, m.detectConfidence / 0.8);
  const sSpec = Math.max(0, 1 - m.specularRatio / QUALITY_THRESHOLDS.maxSpecularRatio);
  const sSharp = Math.min(1, m.sharpness / (QUALITY_THRESHOLDS.minSharpness * 2.5));
  const sRange = Math.min(1, m.dynamicRange / (QUALITY_THRESHOLDS.minDynamicRange * 2.5));
  const score = Math.round(
    100 * (0.3 * sDetect + 0.25 * sSpec + 0.3 * sSharp + 0.15 * sRange),
  );

  const retakeHint = reasons.length
    ? buildHint(reasons)
    : '质量合格';
  return { pass: reasons.length === 0, score: Math.max(0, Math.min(100, score)), metrics: m, reasons, retakeHint };
}

function buildHint(reasons: string[]): string {
  if (reasons.some((r) => r.includes('定位'))) {
    return '请正对牛鼻，让两个鼻孔都完整入镜，距离 20-40 厘米后重拍';
  }
  if (reasons.some((r) => r.includes('反光'))) {
    return '反光过多：请避开强光直射、调整拍摄角度，或先擦干牛鼻再拍';
  }
  if (reasons.some((r) => r.includes('清晰'))) {
    return '画面太糊：请稳住手机、等牛停止甩头后连拍几张';
  }
  return '请调整光线和距离后重拍';
}
