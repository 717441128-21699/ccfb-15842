import { h, grayThumbToDataURL, showToast, fmtTime } from '../dom';
import { listCows, deleteCow } from '../../app/db';
import { navigate } from '../../app/app';
import { CowRecord } from '../../app/types';

export async function renderHerdPage(root: HTMLElement): Promise<void> {
  root.append(
    h('h1', { class: 'page-title' }, ['牛群档案']),
    h('p', { class: 'page-sub' }, ['所有已投保牛只的鼻纹档案与理赔记录。耳标仅作辅助，身份以鼻纹指纹为准。']),
  );
  const card = h('div', { class: 'card' });
  const cows = await listCows();
  if (!cows.length) {
    card.append(h('div', { class: 'empty' }, ['暂无档案，去「投保登记」录入第一头牛']));
    root.append(card);
    return;
  }

  const searchInput = h('input', { placeholder: '按编号/耳标/品种搜索...' }) as HTMLInputElement;
  const tableHost = h('div', {});
  card.append(h('label', { class: 'field' }, [searchInput]), tableHost);
  root.append(card);

  const renderTable = (rows: CowRecord[]): void => {
    tableHost.replaceChildren();
    tableHost.append(
      h('table', {}, [
        h('thead', {}, h('tr', {}, [
          h('th', {}, ['缩略图']), h('th', {}, ['牛只编号']), h('th', {}, ['耳标']),
          h('th', {}, ['品种']), h('th', {}, ['锚点数']), h('th', {}, ['登记时间']),
          h('th', {}, ['最近理赔']), h('th', {}, ['操作']),
        ])),
        h('tbody', {}, rows.map((c) => {
          const tr = h('tr', {}, [
            h('td', { class: 'thumb-cell' }, [h('img', { src: grayThumbToDataURL(c.thumb), alt: c.cowId })]),
            h('td', { class: 'mono' }, [c.cowId]),
            h('td', {}, [c.earTag || '—']),
            h('td', {}, [c.breed || '—']),
            h('td', {}, [String(c.anchorCount)]),
            h('td', {}, [fmtTime(c.createdAt)]),
            h('td', {}, c.lastClaim
              ? h('span', { class: `badge ${c.lastClaim.decision === 'confirmed' ? 'ok' : 'warn'}` }, [
                  c.lastClaim.decision === 'confirmed' ? '已确认' : '待复核',
                ])
              : h('span', { class: 'small' }, ['无'])),
            h('td', {}, (() => {
              const del = h('button', { class: 'danger' }, ['删除']);
              del.onclick = async () => {
                if (!confirm(`确认删除 ${c.cowId} 的档案与指纹？`)) return;
                await deleteCow(c.cowId);
                showToast('已删除', 'success');
                await navigate('herd');
              };
              return h('div', { class: 'btn-row' }, [del]);
            })()),
          ]);
          return tr;
        })),
      ]),
    );
  };
  renderTable(cows);
  searchInput.oninput = () => {
    const q = searchInput.value.trim().toLowerCase();
    renderTable(!q ? cows : cows.filter((c) =>
      [c.cowId, c.earTag, c.breed].some((v) => v?.toLowerCase().includes(q)),
    ));
  };
}
