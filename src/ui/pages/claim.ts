import { h, drawImageData, showToast } from '../dom';
import { demoShot } from '../../app/demo-data';
import { matcher } from '../../app/app';
import { getAllTemplates, getCow, listCows, saveCow } from '../../app/db';
import { AnchorSearchResult } from '../../core/anchor';

// 判定门限：相似度高于此值才确认同一头牛（基准标定，留保守余量）。
const CONFIRM_SIMILARITY = 0.82;

export function renderClaimPage(root: HTMLElement): void {
  root.append(
    h('h1', { class: 'page-title' }, ['理赔比对']),
    h('p', { class: 'page-sub' }, ['对理赔现场拍摄的鼻纹在整群牛档案中检索，返回最相似的几头与相似度，解决耳标脱落/更换后的身份争议。']),
  );

  const card = h('div', { class: 'card' });
  const claimedIdInput = h('input', { placeholder: '报案声称的牛只编号（可选）' }) as HTMLInputElement;
  const canvas = document.createElement('canvas');
  const wrap = h('div', { class: 'canvas-wrap' }, [canvas]);
  const resultBox = h('div', {});
  const timingBox = h('div', { class: 'small' }, ['']);
  let currentImage: ImageData | null = null;
  let currentAnchors: import('../../core/anchor').AnchorPoint[] = [];

  const shootBtn = h('button', { class: 'secondary' }, ['现场拍摄（合成）']);
  const searchBtn = h('button', { disabled: true }, ['全库比对']);

  shootBtn.onclick = () => {
    const id = claimedIdInput.value.trim() || `COW-${String(1 + Math.floor(Math.random() * 5000)).padStart(5, '0')}`;
    const shot = demoShot(id, 70000 + Math.floor(Math.random() * 9999), 'field');
    currentImage = shot.imageData;
    currentAnchors = shot.anchors;
    drawImageData(canvas, shot.imageData);
    searchBtn.disabled = false;
    resultBox.replaceChildren();
  };

  searchBtn.onclick = async () => {
    if (!currentImage) return;
    const gallery = await getAllTemplates();
    if (!gallery.length) return showToast('档案库为空，请先投保登记', 'error');
    const claimed = claimedIdInput.value.trim();
    const processed = await matcher.process(currentImage, claimed || 'CLAIM-PROBE', true, currentAnchors);
    if (!processed.fingerprint) return showToast('图像质量不合格，请重拍', 'error');
    const t0 = performance.now();
    const result: AnchorSearchResult = await matcher.search(processed.fingerprint, gallery, 5);
    const elapsed = performance.now() - t0;
    timingBox.textContent = `扫描 ${result.coarseScanned} 头，精排 ${result.fineCompared} 头，耗时 ${elapsed.toFixed(0)} ms（门限 2000 ms）`;
    renderResult(result, claimed);
  };

  async function renderResult(result: AnchorSearchResult, claimed: string): Promise<void> {
    resultBox.replaceChildren();
    const top = result.candidates[0];
    const confirmed = top && top.similarity >= CONFIRM_SIMILARITY;
    const decision = confirmed ? 'confirmed' : top ? 'inconclusive' : 'rejected';
    const banner = h('div', { class: `quality ${confirmed ? 'pass' : 'fail'}` }, [
      h('div', { style: 'font-size:16px;font-weight:700' }, [
        confirmed
          ? `✓ 身份确认：最匹配 ${top!.cowId}，相似度 ${(top!.similarity * 100).toFixed(1)}%`
          : '✗ 未在档案中找到足够相似的牛只，身份无法确认',
      ]),
      claimed ? h('div', { class: 'small' }, [`报案编号 ${claimed}，${top && top.cowId === claimed ? '与比对结果一致' : '与比对结果不一致，请人工复核'}`]) : h('div', {}),
    ]);
    if (confirmed && claimed) {
      const rec = await getCow(top!.cowId);
      if (rec) {
        rec.lastClaim = {
          at: Date.now(),
          matchedCowId: top!.cowId,
          topSimilarity: top!.similarity,
          decision,
        };
        rec.updatedAt = Date.now();
        await saveCow(rec);
      }
    }
    const list = h('div', {}, result.candidates.map((c, i) => {
      const pct = Math.round(c.similarity * 100);
      return h('div', { class: 'candidate' }, [
        h('div', { class: 'rank' }, [`#${i + 1}`]),
        h('div', { style: 'width:130px' }, [h('div', { class: 'mono' }, [c.cowId]), h('div', { class: 'small' }, [`Chamfer ${c.chamfer.toFixed(3)}`])]),
        h('div', { class: 'sim-bar' }, [h('i', { style: `width:${pct}%` }, [])]),
        h('div', { class: 'sim-val' }, [pct + '%']),
      ]);
    }));
    resultBox.append(banner, list);
    void listCows;
  }

  card.append(
    h('div', { class: 'grid-2' }, [
      h('div', {}, [
        h('h2', {}, ['理赔现场']),
        h('label', { class: 'field' }, [h('span', {}, ['报案牛只编号']), claimedIdInput]),
        h('div', { class: 'btn-row' }, [shootBtn, searchBtn]),
        timingBox,
      ]),
      h('div', {}, [h('h2', {}, ['采集画面']), wrap]),
    ]),
    h('h2', {}, ['比对结果']),
    resultBox,
  );
  root.append(card);
}
