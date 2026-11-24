import { h, drawImageData, showToast } from '../dom';
import { demoShot, DemoProfile } from '../../app/demo-data';
import { matcher } from '../../app/app';

export function renderCapturePage(root: HTMLElement): void {
  root.append(
    h('h1', { class: 'page-title' }, ['鼻纹采集']),
    h('p', { class: 'page-sub' }, ['模拟手机对准牛鼻连拍：当场检测鼻孔、去湿反光并评估清晰度，太糊或反光过多立即提示重拍。']),
  );

  const card = h('div', { class: 'card' });
  const cowIdInput = h('input', { value: 'COW-DEMO-01', placeholder: '牛只编号' }) as HTMLInputElement;
  const profileSelect = h('select', {}, [
    h('option', { value: 'enroll' }, ['正常采集']),
    h('option', { value: 'field' }, ['野外复杂光照']),
    h('option', { value: 'degraded' }, ['退化（糊/强反光，应被拦截）']),
  ]) as HTMLSelectElement;

  const canvas = document.createElement('canvas');
  const wrap = h('div', { class: 'canvas-wrap' }, [canvas]);
  const meta = h('div', { class: 'small mono' }, ['点击下方按钮采集']);
  const qBox = h('div', {});
  const history = h('div', {});
  const shots: { profile: DemoProfile; pass: boolean; score: number }[] = [];

  const btn = h('button', {}, ['采集一张']);
  btn.onclick = async () => {
    const profile = profileSelect.value as DemoProfile;
    const shot = demoShot(cowIdInput.value.trim() || 'COW-DEMO-01', Date.now() % 100000, profile);
    drawImageData(canvas, shot.imageData);
    meta.textContent = `角度=${shot.params.yawDeg.toFixed(1)}° 俯仰=${shot.params.pitchDeg.toFixed(1)}° 湿度=${shot.params.humidity.toFixed(2)} 反光斑=${shot.params.specularCount} 模糊σ=${shot.params.blurSigma.toFixed(2)} 噪声σ=${shot.params.noiseSigma.toFixed(1)}`;
    const processed = await matcher.process(shot.imageData, cowIdInput.value.trim() || 'COW-DEMO-01', true, shot.anchors);
    if (processed.overlay.quad) drawImageData(canvas, shot.imageData, processed.overlay);
    const q = processed.quality;
    qBox.replaceChildren();
    qBox.append(
      h('div', { class: `quality ${q.pass ? 'pass' : 'fail'}` }, [
        h('div', { style: 'display:flex;gap:14px;align-items:center' }, [
          h('div', { class: 'score-ring' }, [String(q.score)]),
          h('div', {}, [
            h('div', { class: q.pass ? 'hint ok' : 'hint' }, [q.pass ? '✓ 采集合格' : '✗ ' + q.retakeHint]),
            h('div', { class: 'metric-row' }, [
              h('span', {}, ['定位 ', h('b', {}, [q.metrics.detectConfidence.toFixed(2)])]),
              h('span', {}, ['反光 ', h('b', {}, [(q.metrics.specularRatio * 100).toFixed(1) + '%'])]),
              h('span', {}, ['清晰度 ', h('b', {}, [q.metrics.sharpness.toFixed(0)])]),
            ]),
          ]),
        ]),
      ]),
    );
    shots.unshift({ profile, pass: q.pass, score: q.score });
    renderHistory();
    if (!q.pass) showToast('质量不达标，已提示重拍', 'error');
  };

  function renderHistory(): void {
    history.replaceChildren();
    if (!shots.length) return;
    history.append(
      h('h2', {}, ['连拍记录']),
      h('table', {}, [
        h('thead', {}, h('tr', {}, [h('th', {}, ['#']), h('th', {}, ['档位']), h('th', {}, ['评分']), h('th', {}, ['结果'])])),
        h('tbody', {}, shots.slice(0, 8).map((s, i) =>
          h('tr', {}, [
            h('td', {}, [String(i + 1)]),
            h('td', {}, [s.profile]),
            h('td', {}, [String(s.score)]),
            h('td', {}, h('span', { class: `badge ${s.pass ? 'ok' : 'no'}` }, [s.pass ? '合格' : '重拍'])),
          ]),
        )),
      ]),
    );
  }

  card.append(
    h('div', { class: 'grid-2' }, [
      h('div', {}, [
        h('h2', {}, ['采集控制']),
        h('label', { class: 'field' }, [h('span', {}, ['牛只编号']), cowIdInput]),
        h('label', { class: 'field' }, [h('span', {}, ['场景档位']), profileSelect]),
        h('div', { class: 'btn-row' }, [btn]),
        meta,
        qBox,
      ]),
      h('div', {}, [h('h2', {}, ['画面']), wrap, history]),
    ]),
  );
  root.append(card);
}
