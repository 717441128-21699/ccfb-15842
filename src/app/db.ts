/**
 * IndexedDB 持久层：牛群档案 + 锚点指纹模板。
 * 粗筛指纹（768 bit）随档案全量载入内存做检索；锚点坐标按 cowId 索引，
 * 精排时按需读取，控制 5000 头规模下的内存占用。
 */

import { AnchorFingerprint } from '../core/anchor';
import { CowRecord } from './types';

const DB_NAME = 'muzzle-id-db';
const DB_VERSION = 1;
const STORE_COWS = 'cows';
const STORE_TEMPLATES = 'templates';

let dbPromise: Promise<IDBDatabase> | null = null;

function openDb(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;
  dbPromise = new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE_COWS)) {
        db.createObjectStore(STORE_COWS, { keyPath: 'cowId' });
      }
      if (!db.objectStoreNames.contains(STORE_TEMPLATES)) {
        db.createObjectStore(STORE_TEMPLATES, { keyPath: 'cowId' });
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
  return dbPromise;
}

function tx<T>(store: string, mode: IDBTransactionMode, fn: (s: IDBObjectStore) => IDBRequest<T>): Promise<T> {
  return openDb().then(
    (db) =>
      new Promise<T>((resolve, reject) => {
        const t = db.transaction(store, mode);
        const req = fn(t.objectStore(store));
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
      }),
  );
}

export async function saveCow(record: CowRecord): Promise<void> {
  await tx(STORE_COWS, 'readwrite', (s) => s.put(record) as IDBRequest<unknown>);
}

export async function getCow(cowId: string): Promise<CowRecord | undefined> {
  return tx<CowRecord | undefined>(STORE_COWS, 'readonly', (s) => s.get(cowId) as IDBRequest<CowRecord | undefined>);
}

export async function deleteCow(cowId: string): Promise<void> {
  const db = await openDb();
  await new Promise<void>((resolve, reject) => {
    const t = db.transaction([STORE_COWS, STORE_TEMPLATES], 'readwrite');
    t.objectStore(STORE_COWS).delete(cowId);
    t.objectStore(STORE_TEMPLATES).delete(cowId);
    t.oncomplete = () => resolve();
    t.onerror = () => reject(t.error);
  });
}

export async function listCows(): Promise<CowRecord[]> {
  return tx<CowRecord[]>(STORE_COWS, 'readonly', (s) => s.getAll() as IDBRequest<CowRecord[]>);
}

export async function countCows(): Promise<number> {
  return tx<number>(STORE_COWS, 'readonly', (s) => s.count() as IDBRequest<number>);
}

export async function saveTemplate(tpl: AnchorFingerprint): Promise<void> {
  // IndexedDB 结构化克隆支持 TypedArray，直接存。
  await tx(STORE_TEMPLATES, 'readwrite', (s) => s.put(tpl) as IDBRequest<unknown>);
}

export async function getTemplate(cowId: string): Promise<AnchorFingerprint | undefined> {
  return tx<AnchorFingerprint | undefined>(
    STORE_TEMPLATES,
    'readonly',
    (s) => s.get(cowId) as IDBRequest<AnchorFingerprint | undefined>,
  );
}

export async function getAllTemplates(): Promise<AnchorFingerprint[]> {
  return tx<AnchorFingerprint[]>(
    STORE_TEMPLATES,
    'readonly',
    (s) => s.getAll() as IDBRequest<AnchorFingerprint[]>,
  );
}

export async function clearAll(): Promise<void> {
  const db = await openDb();
  await new Promise<void>((resolve, reject) => {
    const t = db.transaction([STORE_COWS, STORE_TEMPLATES], 'readwrite');
    t.objectStore(STORE_COWS).clear();
    t.objectStore(STORE_TEMPLATES).clear();
    t.oncomplete = () => resolve();
    t.onerror = () => reject(t.error);
  });
}
