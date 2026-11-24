/**
 * 合成演示数据源：在浏览器内调用 core/synthesis 生成带真值的牛鼻图像，
 * 驱动投保/采集/理赔页面的完整流程（真实部署时替换为摄像头帧）。
 * 允许选择光照、角度、湿度档位，并能生成触发重拍的退化图。
 */

import {
  generateIdentity,
  renderCapture,
  makeCaptureParams,
  CaptureParams,
} from '../core/synthesis';
import { mulberry32, hashStringToUint32 } from '../core/rng';
import { RgbaImage } from '../core/geometry';
import { applyMat3 } from '../core/geometry';
import { AnchorPoint } from '../core/anchor';

export type DemoProfile = 'enroll' | 'field' | 'degraded';

export interface DemoShot {
  imageData: ImageData;
  params: CaptureParams;
  /**
   * 标准画布上的确认锚点（模拟真实系统注册时人工/算法确认的鼻纹 minutiae）。
   * 取身份真值珠点中该次采集画面内可见的部分，坐标在与视角无关的标准画布中，
   * 因此跨光照/角度/湿度稳定，是身份指纹的来源。
   */
  anchors: AnchorPoint[];
}

/** 由牛 ID + 拍摄序号生成确定性的采集参数，保证可复现。 */
export function demoShot(cowId: string, shotIndex: number, profile: DemoProfile): DemoShot {
  const seed = hashStringToUint32(cowId) ^ Math.imul(shotIndex + 1, 0x9e3779b9);
  const rng = mulberry32(seed >>> 0);
  const identity = generateIdentity(cowId);
  const params = makeCaptureParams(rng, profile, 320, 240);
  const cap = renderCapture(identity, params);
  const anchors: AnchorPoint[] = [];
  for (const b of identity.beads) {
    const p = applyMat3(cap.homography, { x: b.x, y: b.y });
    if (p.x >= 0 && p.y >= 0 && p.x < params.outW && p.y < params.outH) {
      anchors.push({ x: b.x, y: b.y });
    }
  }
  return { imageData: rgbaToImageData(cap.image), params, anchors };
}

/** 用指定的环境覆盖项生成一张采集图（实验室页面调光照/角度/湿度用）。 */
export function demoShotWithOverrides(
  cowId: string,
  shotIndex: number,
  overrides: Partial<CaptureParams>,
): DemoShot {
  const seed = hashStringToUint32(cowId) ^ Math.imul(shotIndex + 7, 0x85ebca6b);
  const rng = mulberry32(seed >>> 0);
  const identity = generateIdentity(cowId);
  const base = makeCaptureParams(rng, 'field', 320, 240);
  const params = { ...base, ...overrides, seed: base.seed, outW: 320, outH: 240 };
  const cap = renderCapture(identity, params);
  const anchors: AnchorPoint[] = [];
  for (const b of identity.beads) {
    const p = applyMat3(cap.homography, { x: b.x, y: b.y });
    if (p.x >= 0 && p.y >= 0 && p.x < params.outW && p.y < params.outH) {
      anchors.push({ x: b.x, y: b.y });
    }
  }
  return { imageData: rgbaToImageData(cap.image), params, anchors };
}

function rgbaToImageData(img: RgbaImage): ImageData {
  // ImageData 构造器在现代浏览器与 Worker 中可用。
  return new ImageData(new Uint8ClampedArray(img.data), img.width, img.height);
}
