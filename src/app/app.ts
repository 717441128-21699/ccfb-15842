import { MatcherClient } from './matcher-client';
import { renderEnrollPage } from '../ui/pages/enroll';
import { renderCapturePage } from '../ui/pages/capture';
import { renderClaimPage } from '../ui/pages/claim';
import { renderHerdPage } from '../ui/pages/herd';
import { renderLabPage } from '../ui/pages/lab';
import { PageName } from './types';
import { h, clear } from '../ui/dom';
import { countCows } from './db';

export const matcher = new MatcherClient();

const PAGES: Record<PageName, (root: HTMLElement) => Promise<void> | void> = {
  enroll: renderEnrollPage,
  capture: renderCapturePage,
  claim: renderClaimPage,
  herd: renderHerdPage,
  lab: renderLabPage,
};

const NAV: { key: PageName; label: string }[] = [
  { key: 'enroll', label: '投保登记' },
  { key: 'capture', label: '鼻纹采集' },
  { key: 'claim', label: '理赔比对' },
  { key: 'herd', label: '牛群档案' },
  { key: 'lab', label: '合成实验室' },
];

function currentPage(): PageName {
  const raw = (location.hash.replace('#/', '') || 'enroll') as PageName;
  return NAV.some((n) => n.key === raw) ? raw : 'enroll';
}

export async function refreshCount(): Promise<void> {
  const el = document.getElementById('cow-count');
  if (el) el.innerHTML = `在档 <b>${await countCows()}</b> 头`;
}

export async function navigate(page: PageName): Promise<void> {
  if (location.hash !== `#/${page}`) location.hash = `#/${page}`;
  else await render();
}

export async function render(): Promise<void> {
  const page = currentPage();
  document.querySelectorAll<HTMLAnchorElement>('.nav a').forEach((a) => {
    a.classList.toggle('active', a.dataset.page === page);
  });
  const root = document.getElementById('page-root')!;
  clear(root);
  await PAGES[page](root);
  await refreshCount();
}

export function mountApp(): void {
  const header = h('header', { class: 'topbar' }, [
    h('div', { class: 'brand' }, ['牛鼻纹识别系统', h('small', {}, ['MuzzlePrint ID'])]),
    h('nav', { class: 'nav' }, NAV.map((n) => {
      const a = h('a', { href: `#/${n.key}` }, [n.label]) as HTMLAnchorElement;
      a.dataset.page = n.key;
      return a;
    })),
    h('div', { class: 'spacer' }),
    h('div', { class: 'count-pill', id: 'cow-count' }, []),
  ]);
  const main = h('main', {}, [h('div', { id: 'page-root' }, [])]);
  const app = document.getElementById('app')!;
  app.append(h('div', { class: 'app' }, [header, main]));
  window.addEventListener('hashchange', () => void render());
  void render();
}
