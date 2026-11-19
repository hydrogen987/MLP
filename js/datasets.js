/* ============================================================
   datasets.js — 六个二维二分类数据集
   所有坐标最后都会归一化到 [-1, 1]，方便直接映射到画布。
   ============================================================ */
(function (global) {
  'use strict';

  function mulberry32(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function gaussian(rand) {
    let u = 0;
    let v = 0;
    while (u === 0) u = rand();
    while (v === 0) v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  const clamp = (v) => Math.max(-1, Math.min(1, v));

  const META = {
    xor: { label: '异或', en: 'XOR', note: '四个象限，必须弯折才能分开' },
    circle: { label: '圆环', en: 'Circle', note: '中心与外围，需要闭合边界' },
    moons: { label: '双月', en: 'Moons', note: '两个交错的半月，经典非线性题' },
    spiral: { label: '螺旋', en: 'Spiral', note: '两条缠绕的臂，考验容量' },
    blobs: { label: '高斯团', en: 'Blobs', note: '线性可分，用作基线' },
    checker: { label: '棋盘', en: 'Checker', note: '四乘四方格，需要足够宽度' },
  };

  function normalize(points, pad) {
    if (!points.length) return points;
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const p of points) {
      if (p.x[0] < minX) minX = p.x[0];
      if (p.x[0] > maxX) maxX = p.x[0];
      if (p.x[1] < minY) minY = p.x[1];
      if (p.x[1] > maxY) maxY = p.x[1];
    }
    const spanX = Math.max(1e-6, maxX - minX);
    const spanY = Math.max(1e-6, maxY - minY);
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    let scale = Math.min((2 * pad) / spanX, (2 * pad) / spanY);
    scale = Math.min(scale, 1.7);
    for (const p of points) {
      p.x[0] = clamp((p.x[0] - cx) * scale);
      p.x[1] = clamp((p.x[1] - cy) * scale);
    }
    return points;
  }

  function generate(name, n, noise, seed) {
    const rand = mulberry32(seed || 1);
    const pts = [];
    const jitter = (s) => gaussian(rand) * s;
    const count = Math.max(2, n | 0);

    if (name === 'circle') {
      for (let i = 0; i < count; i++) {
        const label = i % 2;
        const radius = label === 1 ? 0.16 + rand() * 0.26 : 0.62 + rand() * 0.34;
        const angle = rand() * Math.PI * 2;
        pts.push({
          x: [
            Math.cos(angle) * radius + jitter(noise * 0.32),
            Math.sin(angle) * radius + jitter(noise * 0.32),
          ],
          y: label,
        });
      }
    } else if (name === 'moons') {
      for (let i = 0; i < count; i++) {
        const label = i % 2;
        const t = Math.PI * (0.04 + 0.92 * rand());
        let x;
        let y;
        if (label === 0) {
          x = Math.cos(t) * 0.8 - 0.22;
          y = Math.sin(t) * 0.8 - 0.22;
        } else {
          x = 0.58 - Math.cos(t) * 0.8;
          y = 0.22 - Math.sin(t) * 0.8;
        }
        pts.push({
          x: [x + jitter(noise * 0.3), y + jitter(noise * 0.3)],
          y: label,
        });
      }
    } else if (name === 'spiral') {
      for (let i = 0; i < count; i++) {
        const label = i % 2;
        const t = Math.pow(rand(), 0.9);
        const radius = 0.08 + 0.92 * t;
        const angle = t * Math.PI * 2 * 1.5 + label * Math.PI;
        const local = noise * 0.22 * (0.35 + 0.65 * t);
        pts.push({
          x: [Math.cos(angle) * radius + jitter(local), Math.sin(angle) * radius + jitter(local)],
          y: label,
        });
      }
    } else if (name === 'blobs') {
      for (let i = 0; i < count; i++) {
        const label = i % 2;
        const cx = label === 1 ? 0.42 : -0.42;
        const cy = label === 1 ? 0.14 : -0.14;
        const spread = 0.22 + noise * 0.25;
        pts.push({
          x: [cx + jitter(spread), cy + jitter(spread)],
          y: label,
        });
      }
    } else if (name === 'checker') {
      for (let i = 0; i < count; i++) {
        let x = rand() * 2 - 1;
        let y = rand() * 2 - 1;
        x += jitter(noise * 0.18);
        y += jitter(noise * 0.18);
        const cellX = Math.min(3, Math.max(0, Math.floor(((x + 1) / 2) * 4)));
        const cellY = Math.min(3, Math.max(0, Math.floor(((y + 1) / 2) * 4)));
        pts.push({ x: [x, y], y: (cellX + cellY) % 2 });
      }
    } else {
      // XOR：四个象限
      for (let i = 0; i < count; i++) {
        const label = rand() < 0.5 ? 0 : 1;
        const sx = rand() < 0.5 ? -1 : 1;
        const sy = label === 0 ? sx : -sx;
        pts.push({
          x: [
            sx * (0.34 + rand() * 0.24) + jitter(noise * 0.42),
            sy * (0.34 + rand() * 0.24) + jitter(noise * 0.42),
          ],
          y: label,
        });
      }
    }

    return normalize(pts, 0.94);
  }

  const API = { generate, META };
  if (typeof module !== 'undefined' && module.exports) module.exports = API;
  else global.NeuralDatasets = API;
})(typeof globalThis !== 'undefined' ? globalThis : this);
