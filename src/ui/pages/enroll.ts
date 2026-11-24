import { h } from '../dom';
import { demoShot } from '../../app/demo-data';
import { matcher } from '../../app/app';
import { saveCow, saveTemplate, getCow } from '../../app/db';
import { drawImageData, grayThumbToDataURL, showToast } from '../dom';
import { makeThumb } from '../../app/pipeline';
import { ProcessedCapture } from '../../app/pipeline';
import { CowRecord } from '../../app/types';

export async function renderEnrollPage(root: HTMLElement): Promise<void> {
  root.append(
    h('h1', { class: 'page-title' }, ['投保登记']),
    h('p', { class: 'page-sub' }, ['录入牛只信息并采集鼻纹，系统当场做质量门限，合格后生成唯一身份指纹入库。']),
  );

  const card = h('div', { class: 'card' });
  const cowIdInput = h('input', { value: `COW-${String(Date.now()).slice(-6)}`, placeholder: '牛只编号' }) as HTMLInputElement;
  const earTagInput = h('input', { placeholder: '耳标号（可选，脱落不影响身份）' }) as HTMLInputElement;
  const breedInput = h('input', { placeholder: '品种（可选）' }) as HTMLInputElement;
  const birthInput = h('input', { type: 'date' }) as HTMLInputElement;

  const form = h('div', {}, [
    h('label', { class: 'field' }, [h('span', {}, ['牛只编号 *']), cowIdInput]),
    h('label', { class: 'field' }, [h('span', {}, ['耳标号']), earTagInput]),
    h('div', { class: 'grid-2' }, [
      h('label', { class: 'field' }, [h('span', {}, ['品种']), breedInput]),
      h('label', { class: 'field' }, [h('span', {}, ['出生日期']), birthInput]),
    ]),
  ]);

  const canvas = document.createElement('canvas');
  const canvasWrap = h('div', { class: 'canvas-wrap' }, [canvas]);
  const shotMeta = h('div', { class: 'small mono' }, ['尚未采集']);
  const qualityBox = h('div', {});
  let currentShot: ReturnType<typeof demoShot> | null = null;
  let lastProcessed: ProcessedCapture | null = null;

  const enrollBtn = h('button', { disabled: true }, ['登记入库']);
  const captureBtn = h('button', { class: 'secondary' }, ['拍摄鼻纹（合成采集）']);
  const retakeBtn = h('button', { class: 'secondary' }, ['换一张']);

  async function capture(): Promise<void> {
    const cowId = cowIdInput.value.trim();
    if (!cowId) {
      showToast('请先填写牛只编号', 'error');
      return;
    }
    currentShot = demoShot(cowId, Math.floor(Math.random() * 1e6), 'enroll');
    drawImageData(canvas, currentShot.imageData);
    shotMeta.textContent = `采集参数 角度=${currentShot.params.yawDeg.toFixed(1)}° 湿度=${currentShot.params.humidity.toFixed(2)} 模糊σ=${currentShot.params.blurSigma.toFixed(2)}`;
    lastProcessed = await matcher.process(currentShot.imageData, cowId, true, currentShot.anchors);
    const q = lastProcessed.quality;
    qualityBox.replaceChildren(
      h('div', { class: `quality ${q.pass ? 'pass' : 'fail'}` }, [
        h('div', { style: 'display:flex;align-items:center;gap:14px' }, [
          h('div', { class: 'score-ring' }, [String(q.score)]),
          h('div', {}, [
            h('div', { class: q.pass ? 'hint ok' : 'hint' }, [q.pass ? '✓ 质量合格，可登记' : '✗ 质量不合格，建议重拍']),
            h('div', { class: 'small' }, [q.retakeHint]),
          ]),
        ]),
        h('div', { class: 'metric-row' }, [
          h('span', {}, ['定位置信 ', h('b', {}, [q.metrics.detectConfidence.toFixed(2)])]),
          h('span', {}, ['反光占比 ', h('b', {}, [(q.metrics.specularRatio * 100).toFixed(1) + '%'])]),
          h('span', {}, ['清晰度 ', h('b', {}, [q.metrics.sharpness.toFixed(0)])]),
          h('span', {}, ['动态范围 ', h('b', {}, [q.metrics.dynamicRange.toFixed(2)])]),
        ]),
      ]),
    );
    if (lastProcessed.overlay.quad) drawImageData(canvas, currentShot!.imageData, lastProcessed.overlay);
    enrollBtn.disabled = !(q.pass && lastProcessed.fingerprint);
  }

  captureBtn.onclick = () => void capture();
  retakeBtn.onclick = () => void capture();

  enrollBtn.onclick = async () => {
    if (!currentShot || !lastProcessed?.fingerprint || !lastProcessed.enhanced) return;
    const cowId = cowIdInput.value.trim();
    if (await getCow(cowId)) {
      showToast('该牛只编号已登记', 'error');
      return;
    }
    const record: CowRecord = {
      cowId,
      earTag: earTagInput.value.trim() || undefined,
      breed: breedInput.value.trim() || undefined,
      birthDate: birthInput.value || undefined,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      thumb: makeThumb(lastProcessed.enhanced.gray),
      anchorCount: lastProcessed.fingerprint.points.length,
    };
    await saveCow(record);
    await saveTemplate(lastProcessed.fingerprint);
    showToast(`登记成功：${cowId}，锚点 ${lastProcessed.fingerprint.points.length} 个`, 'success');
    enrollBtn.disabled = true;
  };

  card.append(
    h('div', { class: 'grid-2' }, [
      h('div', {}, [
        h('h2', {}, ['牛只信息']),
        form,
        h('div', { class: 'btn-row' }, [captureBtn, retakeBtn, enrollBtn]),
      ]),
      h('div', {}, [
        h('h2', {}, ['鼻纹采集与质量']),
        canvasWrap,
        shotMeta,
        qualityBox,
      ]),
    ]),
  );
  root.append(card);
  void grayThumbToDataURL;
}
