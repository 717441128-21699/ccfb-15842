/** 应用层共享类型。 */

export interface CowRecord {
  cowId: string;
  /** 耳标号（可能脱落/更换，仅作辅助，不作为身份依据）。 */
  earTag?: string;
  breed?: string;
  birthDate?: string;
  createdAt: number;
  updatedAt: number;
  /** 注册时的增强缩略图（96x72 灰度）。 */
  thumb: number[];
  /** 锚点数。 */
  anchorCount: number;
  /** 最近一次理赔比对结果摘要。 */
  lastClaim?: {
    at: number;
    matchedCowId: string | null;
    topSimilarity: number;
    decision: 'confirmed' | 'rejected' | 'inconclusive';
  };
}

export interface EnrollSubmission {
  cowId: string;
  earTag?: string;
  breed?: string;
  birthDate?: string;
  /** 采集到的 RGBA 图像位图（来自摄像头或合成演示）。 */
  image: ImageData;
}

export type PageName = 'enroll' | 'capture' | 'claim' | 'herd' | 'lab';

export interface QualitySummary {
  pass: boolean;
  score: number;
  detectConfidence: number;
  specularRatio: number;
  sharpness: number;
  dynamicRange: number;
  reasons: string[];
  retakeHint: string;
}
