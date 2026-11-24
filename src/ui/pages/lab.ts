import { h, drawImageData } from '../dom';
import { demoShotWithOverrides } from '../../app/demo-data';
import { matcher } from '../../app/app';
import { CaptureParams } from '../../core/synthesis';

function slider(label: string, min: number, max: number, step: number, value: number): { wrap: HTMLElement; input: HTMLInputElement; val: HTMLElement } {
  const input = h('input', { type: 'range', min: String(min), max: String(max), step: String(step), value: String(value) }) as HTMLInputElement;
  const val = h('b', {}, [value.toFixed(step < 1 ? 2 : 0)]);
  input.oninput = () => { val.textContent = step < 1 ? Number(input.value).toFixed(2) : input.value; };
  const wrap = h('label', { class: 'field' }, [h('span', {}, [label, ' ', val]), input]);
  return { wrap, input, val };
}

export function renderLabPage(root: HTMLElement): void {
  root.append(
    h('h1', { class: 'page-title' }, ['合成实验室']),
    h('p', { class: 'page-sub' }, ['按不同光照、角度、湿度、模糊自行合成带真值的牛鼻图像，观察检测、增强与质量门限的反应。']),
  );
  const card = h('div', { class: 'card' });
  const cowIdInput = h('input', { value: 'COW-LAB-01' }) as HTMLInputElement;

  const yaw = slider('偏航角（°）', -30, 30, 1, 0);
  const pitch = slider('俯仰角（°）', -25, 25, 1, 0);
  const roll = slider('翻滚角（°）', -25, 25, 1, 0);
  const light = slider('光照强度', 0, 0.6, 0.02, 0.25);
  const humidity = slider('湿度', 0, 1, 0.02, 0.4);
  const specular = slider('反光斑数量', 0, 20, 1, 5);
  const blur = slider('模糊 σ', 0, 4, 0.1, 0.8);
  const noise = slider('噪声 σ', 0, 14, 0.5, 4);

  const canvas = document.createElement('canvas');
  const wrap = h('div', { class: 'canvas-wrap' }, [canvas]);
  const qBox = h('div', {});

  const controls = h('div', { class: 'control-grid' }, [
    yaw.wrap, pitch.wrap, roll.wrap, light.wrap, humidity.wrap,
    specular.wrap, blur.wrap, noise.wrap,
    h('label', { class: 'field' }, [h('span', {}, ['牛只编号']), cowIdInput]),
  ]);

  const gen = h('button', {}, ['生成并检测']);
  gen.onclick = async () => {
    const overrides: Partial<CaptureParams> = {
      yawDeg: Number(yaw.input.value),
      pitchDeg: Number(pitch.input.value),
      rollDeg: Number(roll.input.value),
      lightStrength: Number(light.input.value),
      humidity: Number(humidity.input.value),
      specularCount: Number(specular.input.value),
      blurSigma: Number(blur.input.value),
      noiseSigma: Number(noise.input.value),
    };
    const shot = demoShotWithOverrides(cowIdInput.value.trim(), Date.now() % 100000, overrides);
    drawImageData(canvas, shot.imageData);
    const processed = await matcher.process(shot.imageData, cowIdInput.value.trim(), true, shot.anchors);
    if (processed.overlay.quad) drawImageData(canvas, shot.imageData, processed.overlay);
    const q = processed.quality;
    qBox.replaceChildren();
    qBox.append(
      h('div', { class: `quality ${q.pass ? 'pass' : 'fail'}` }, [
        h('div', { style: 'display:flex;gap:14px;align-items:center' }, [
          h('div', { class: 'score-ring' }, [String(q.score)]),
          h('div', {}, [
            h('div', { class: q.pass ? 'hint ok' : 'hint' }, [q.pass ? '✓ 合格' : '✗ ' + q.retakeHint]),
            h('div', { class: 'metric-row' }, [
              h('span', {}, ['定位 ', h('b', {}, [q.metrics.detectConfidence.toFixed(2)])]),
              h('span', {}, ['反光 ', h('b', {}, [(q.metrics.specularRatio * 100).toFixed(1) + '%'])]),
              h('span', {}, ['清晰度 ', h('b', {}, [q.metrics.sharpness.toFixed(0)])]),
              h('span', {}, ['锚点 ', h('b', {}, [String(processed.fingerprint?.points.length ?? 0)])]),
            ]),
          ]),
        ]),
      ]),
    );
  };

  card.append(
    h('h2', {}, ['采集环境控制（带纹路真值的合成）']),
    controls,
    h('div', { class: 'btn-row' }, [gen]),
    h('div', { class: 'grid-2', style: 'margin-top:16px' }, [wrap, qBox]),
  );
  root.append(card);
}
