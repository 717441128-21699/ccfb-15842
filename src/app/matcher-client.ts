/**
 * Web Worker 客户端封装：主线程通过 Promise 调用处理/检索，
 * Worker 构造失败（老旧浏览器）时回退到主线程同步实现。
 */

import type { WorkerRequest, WorkerResponse } from './matcher.worker';
import { processCapture, ProcessedCapture } from './pipeline';
import { AnchorFingerprint, AnchorPoint, searchAnchors, AnchorSearchResult } from '../core/anchor';

type RequestWithoutSeq =
  | { type: 'process'; cowId: string; image: ImageData; useNostrilRectify?: boolean; confirmedAnchors?: AnchorPoint[] }
  | { type: 'search'; query: AnchorFingerprint; gallery: AnchorFingerprint[]; topK?: number };

export class MatcherClient {
  private worker: Worker | null = null;
  private seq = 0;
  private pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();

  constructor() {
    try {
      this.worker = new Worker(new URL('./matcher.worker.ts', import.meta.url), { type: 'module' });
      this.worker.onmessage = (ev: MessageEvent<WorkerResponse>) => {
        const msg = ev.data;
        const waiter = this.pending.get(msg.seq);
        if (!waiter) return;
        this.pending.delete(msg.seq);
        if (msg.type === 'error') waiter.reject(new Error(msg.message));
        else waiter.resolve(msg);
      };
      this.worker.onerror = (e) => {
        for (const { reject } of this.pending.values()) reject(new Error(e.message));
        this.pending.clear();
      };
    } catch {
      this.worker = null;
    }
  }

  private call(req: RequestWithoutSeq): Promise<WorkerResponse> {
    const id = ++this.seq;
    const full = { seq: id, ...req } as WorkerRequest;
    if (!this.worker) return Promise.resolve(this.runOnMain(full));
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
      this.worker!.postMessage(full);
    });
  }

  private runOnMain(req: WorkerRequest): WorkerResponse {
    if (req.type === 'process') {
      return {
        seq: req.seq,
        type: 'process-result',
        result: processCapture(req.image, req.cowId, {
          useNostrilRectify: req.useNostrilRectify ?? true,
          confirmedAnchors: req.confirmedAnchors,
        }),
      };
    }
    return { seq: req.seq, type: 'search-result', result: searchAnchors(req.query, req.gallery, req.topK ?? 5) };
  }

  async process(
    image: ImageData,
    cowId: string,
    useNostrilRectify = true,
    confirmedAnchors?: AnchorPoint[],
  ): Promise<ProcessedCapture> {
    const res = await this.call({ type: 'process', cowId, image, useNostrilRectify, confirmedAnchors });
    if (res.type !== 'process-result') throw new Error('意外的 Worker 响应');
    return res.result;
  }

  async search(
    query: AnchorFingerprint,
    gallery: AnchorFingerprint[],
    topK = 5,
  ): Promise<AnchorSearchResult> {
    const res = await this.call({ type: 'search', query, gallery, topK });
    if (res.type !== 'search-result') throw new Error('意外的 Worker 响应');
    return res.result;
  }

  dispose(): void {
    this.worker?.terminate();
    this.worker = null;
  }
}
