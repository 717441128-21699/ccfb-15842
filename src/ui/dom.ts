/** 极简 DOM 构建辅助。 */
type ElProps<K extends keyof HTMLElementTagNameMap> = Partial<Omit<HTMLElementTagNameMap[K], 'style' | 'children'>> & {
  class?: string;
  style?: string | Partial<CSSStyleDeclaration>;
};

export function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  props: ElProps<K> = {},
  children: (Node | string)[] | Node | string = [],
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === 'class') el.className = v as string;
    else if (k === 'style' && typeof v === 'string') el.setAttribute('style', v);
    else (el as unknown as Record<string, unknown>)[k] = v;
  }
  const list = Array.isArray(children) ? children : [children];
  for (const c of list) {
    if (Array.isArray(c)) el.append(...(c as Node[]));
    else el.append(typeof c === 'string' ? document.createTextNode(c) : c);
  }
  return el;
}

export function clear(el: HTMLElement): void {
  el.replaceChildren();
}

export function showToast(message: string, kind: 'info' | 'error' | 'success' = 'info', ms = 2600): void {
  const t = h('div', { class: `toast ${kind === 'info' ? '' : kind}` }, [message]);
  document.body.append(t);
  setTimeout(() => t.remove(), ms);
}

/** 把 ImageData 画到 canvas（可叠加检测四边形与鼻孔）。 */
export function drawImageData(
  canvas: HTMLCanvasElement,
  img: ImageData,
  overlay?: { quad?: { x: number; y: number }[] | null; nostrils?: { left: { x: number; y: number }; right: { x: number; y: number } } | null },
): void {
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext('2d')!;
  ctx.putImageData(img, 0, 0);
  if (!overlay) return;
  if (overlay.quad && overlay.quad.length === 4) {
    ctx.strokeStyle = '#4c9aff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    overlay.quad.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
    ctx.closePath();
    ctx.stroke();
  }
  if (overlay.nostrils) {
    ctx.fillStyle = '#2bd17e';
    for (const p of [overlay.nostrils.left, overlay.nostrils.right]) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

/** 灰度数组（96x72）转 dataURL 用于档案缩略图。 */
export function grayThumbToDataURL(gray: number[], w = 96, h = 72): string {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d')!;
  const img = ctx.createImageData(w, h);
  for (let i = 0; i < gray.length; i++) {
    img.data[i * 4] = gray[i];
    img.data[i * 4 + 1] = gray[i];
    img.data[i * 4 + 2] = gray[i];
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return c.toDataURL('image/png');
}

export function fmtTime(ms: number): string {
  const d = new Date(ms);
  const p = (n: number) => String(n).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}
