/**
 * 特征提取与比对 Web Worker。
 *
 * 主线程把采集到的 ImageData（或合成位图）与任务参数 postMessage 进来，
 * Worker 内跑检测/增强/质量门限/锚点提取，以及全库检索，避免阻塞 UI。
 * 纯 TS 算法与 Node 基准共用同一套 core 模块。
 */

import { processCapture, ProcessedCapture } from './pipeline';
import {
  AnchorFingerprint,
  AnchorPoint,
  searchAnchors,
  AnchorSearchResult,
} from '../core/anchor';

export type WorkerRequest =
  | {
      seq: number;
      type: 'process';
      cowId: string;
      image: ImageData;
      useNostrilRectify?: boolean;
      confirmedAnchors?: AnchorPoint[];
    }
  | {
      seq: number;
      type: 'search';
      query: AnchorFingerprint;
      gallery: AnchorFingerprint[];
      topK?: number;
    };

export type WorkerResponse =
  | { seq: number; type: 'process-result'; result: ProcessedCapture }
  | { seq: number; type: 'search-result'; result: AnchorSearchResult }
  | { seq: number; type: 'error'; message: string };

const ctx = self as unknown as DedicatedWorkerGlobalScope;

ctx.onmessage = (ev: MessageEvent<WorkerRequest>) => {
  const req = ev.data;
  const seq = req.seq;
  try {
    if (req.type === 'process') {
      const result = processCapture(
        req.image,
        req.cowId,
        {
          useNostrilRectify: req.useNostrilRectify ?? true,
          confirmedAnchors: req.confirmedAnchors,
        },
      );
      const res: WorkerResponse = { seq, type: 'process-result', result };
      ctx.postMessage(res);
    } else if (req.type === 'search') {
      const result = searchAnchors(req.query, req.gallery, req.topK ?? 5);
      const res: WorkerResponse = { seq, type: 'search-result', result };
      ctx.postMessage(res);
    }
  } catch (e) {
    const res: WorkerResponse = {
      seq,
      type: 'error',
      message: e instanceof Error ? e.message : String(e),
    };
    ctx.postMessage(res);
  }
};
