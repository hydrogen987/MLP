/* ============================================================
   viz.js — Canvas 可视化
   决策边界 / 损失曲线 / 网络与激活 / 对照实验图表
   ============================================================ */
(function (global) {
  'use strict';

  const PAD = 10;
  const GRID = 80;

  const C0 = [46, 230, 200]; // 类别 0 / 正权重
  const C1 = [255, 122, 89]; // 类别 1 / 负权重
  const FIELD0 = [20, 78, 74];
  const FIELD1 = [118, 54, 45];

  const MONO = '10px "JetBrains Mono", "SFMono-Regular", Consolas, monospace';
  const MONO_SM = '9px "JetBrains Mono", "SFMono-Regular", Consolas, monospace';

  function roundRectPath(ctx, x, y, w, h, r) {
    const rr = Math.min(r, Math.abs(w) / 2, Math.abs(h) / 2);
    ctx.beginPath();
    ctx.moveTo(x + rr, y);
    ctx.arcTo(x + w, y, x + w, y + h, rr);
    ctx.arcTo(x + w, y + h, x, y + h, rr);
    ctx.arcTo(x, y + h, x, y, rr);
    ctx.arcTo(x, y, x + w, y, rr);
    ctx.closePath();
  }

  function fitCanvas(canvas) {
    const dpr = Math.min(global.devicePixelRatio || 1, 2);
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));
    const pw = Math.round(w * dpr);
    const ph = Math.round(h * dpr);
    if (canvas.width !== pw || canvas.height !== ph) {
      canvas.width = pw;
      canvas.height = ph;
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return { ctx, w, h, dpr };
  }

  /* ---------------- 决策边界 ---------------- */
  let fieldCanvas = null;
  let fieldCtx = null;
  let fieldImage = null;
  let fieldNet = null;
  let fieldVersion = -1;

  function computeField(net) {
    if (fieldNet === net && fieldVersion === net.version && fieldCanvas) return fieldCanvas;
    if (!fieldCanvas) {
      fieldCanvas = document.createElement('canvas');
      fieldCanvas.width = GRID;
      fieldCanvas.height = GRID;
      fieldCtx = fieldCanvas.getContext('2d');
      fieldImage = fieldCtx.createImageData(GRID, GRID);
    }
    const data = fieldImage.data;
    const input = new Float64Array(2);
    const [r0, g0, b0] = FIELD0;
    const [r1, g1, b1] = FIELD1;
    let k = 0;
    for (let gy = 0; gy < GRID; gy++) {
      const y = 1 - (2 * (gy + 0.5)) / GRID;
      input[1] = y;
      for (let gx = 0; gx < GRID; gx++) {
        input[0] = (2 * (gx + 0.5)) / GRID - 1;
        let p = net.predict(input);
        if (!isFinite(p)) p = 0.5;
        data[k++] = (r0 + (r1 - r0) * p) | 0;
        data[k++] = (g0 + (g1 - g0) * p) | 0;
        data[k++] = (b0 + (b1 - b0) * p) | 0;
        data[k++] = 255;
      }
    }
    fieldCtx.putImageData(fieldImage, 0, 0);
    fieldNet = net;
    fieldVersion = net.version;
    return fieldCanvas;
  }

  function drawBoundary(canvas, net, opts) {
    const o = opts || {};
    const { ctx, w, h } = fitCanvas(canvas);
    if (w < 20 || h < 20) return;

    ctx.fillStyle = '#070c11';
    ctx.fillRect(0, 0, w, h);

    const pad = PAD;
    const iw = w - pad * 2;
    const ih = h - pad * 2;
    const field = computeField(net);

    ctx.save();
    roundRectPath(ctx, pad, pad, iw, ih, 12);
    ctx.clip();

    ctx.imageSmoothingEnabled = true;
    ctx.globalAlpha = 0.95;
    ctx.drawImage(field, pad, pad, iw, ih);
    ctx.globalAlpha = 1;

    // 网格
    ctx.strokeStyle = 'rgba(255,255,255,0.055)';
    ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
      const gx = pad + (iw * i) / 4;
      const gy = pad + (ih * i) / 4;
      ctx.beginPath();
      ctx.moveTo(gx, pad);
      ctx.lineTo(gx, pad + ih);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(pad, gy);
      ctx.lineTo(pad + iw, gy);
      ctx.stroke();
    }

    const toX = (v) => pad + ((v + 1) / 2) * iw;
    const toY = (v) => pad + ((1 - v) / 2) * ih;
    const r = Math.max(1.9, Math.min(3.2, iw / 170));

    const test = o.test || [];
    const train = o.train || [];

    // 测试点：白色外环
    ctx.lineWidth = 1.1;
    for (const p of test) {
      ctx.beginPath();
      ctx.arc(toX(p.x[0]), toY(p.x[1]), r + 1.7, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255,255,255,0.62)';
      ctx.stroke();
    }

    ctx.lineWidth = 1;
    for (const p of train) {
      ctx.beginPath();
      ctx.arc(toX(p.x[0]), toY(p.x[1]), r, 0, Math.PI * 2);
      ctx.fillStyle = p.y === 1 ? '#ff8f70' : '#4df3d6';
      ctx.fill();
      ctx.strokeStyle = 'rgba(4,9,13,0.8)';
      ctx.stroke();
    }

    for (const p of test) {
      ctx.beginPath();
      ctx.arc(toX(p.x[0]), toY(p.x[1]), r, 0, Math.PI * 2);
      ctx.fillStyle = p.y === 1 ? '#ff8f70' : '#4df3d6';
      ctx.fill();
    }

    // 探针
    if (o.probe) {
      const px = toX(o.probe[0]);
      const py = toY(o.probe[1]);
      ctx.strokeStyle = 'rgba(255,255,255,0.5)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(px, py, 6.5, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(px, py, 2.2, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,255,255,0.28)';
      ctx.beginPath();
      ctx.moveTo(px - 10, py);
      ctx.lineTo(px - 7, py);
      ctx.moveTo(px + 7, py);
      ctx.lineTo(px + 10, py);
      ctx.moveTo(px, py - 10);
      ctx.lineTo(px, py - 7);
      ctx.moveTo(px, py + 7);
      ctx.lineTo(px, py + 10);
      ctx.stroke();
    }

    ctx.restore();

    ctx.strokeStyle = 'rgba(140,175,200,0.2)';
    ctx.lineWidth = 1;
    roundRectPath(ctx, pad + 0.5, pad + 0.5, iw - 1, ih - 1, 12);
    ctx.stroke();
  }

  /* ---------------- 损失 / 准确率曲线 ---------------- */
  function drawLossChart(canvas, history) {
    const { ctx, w, h } = fitCanvas(canvas);
    if (w < 60 || h < 50) return;

    const padL = 40;
    const padR = 36;
    const padT = 12;
    const padB = 24;
    const iw = w - padL - padR;
    const ih = h - padT - padB;

    ctx.fillStyle = 'rgba(6,11,16,0.55)';
    roundRectPath(ctx, 0.5, 0.5, w - 1, h - 1, 11);
    ctx.fill();

    if (!history || history.length < 2) {
      ctx.fillStyle = 'rgba(140,165,180,0.5)';
      ctx.font = MONO;
      ctx.textAlign = 'center';
      ctx.fillText('点击「开始训练」记录曲线', w / 2, h / 2);
      return;
    }

    let maxLoss = 0.2;
    for (const p of history) {
      if (isFinite(p.loss) && p.loss > maxLoss) maxLoss = p.loss;
    }
    const yMax = Math.min(2.5, Math.max(0.35, maxLoss * 1.12));
    const maxEpoch = Math.max(10, history[history.length - 1].epoch);

    const x = (e) => padL + (e / maxEpoch) * iw;
    const yL = (v) => padT + (ih - 1) * (1 - Math.min(1, Math.max(0, v / yMax)));
    const yA = (v) => padT + (ih - 1) * (1 - Math.min(1, Math.max(0, v)));

    // 横向网格 + 左轴（损失）
    ctx.font = MONO_SM;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i <= 4; i++) {
      const gy = padT + (ih * i) / 4;
      ctx.strokeStyle = i === 4 ? 'rgba(140,175,200,0.16)' : 'rgba(140,175,200,0.07)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padL, gy + 0.5);
      ctx.lineTo(padL + iw, gy + 0.5);
      ctx.stroke();
      ctx.fillStyle = 'rgba(140,165,180,0.55)';
      ctx.fillText((yMax * (1 - i / 4)).toFixed(2), padL - 7, gy);
    }

    // 右轴（准确率）
    ctx.textAlign = 'left';
    for (let i = 0; i <= 2; i++) {
      const gy = padT + (ih * i) / 2;
      ctx.fillStyle = 'rgba(255,200,87,0.55)';
      ctx.fillText(`${100 - i * 50}%`, padL + iw + 7, gy);
    }

    // 底部 epoch 标签
    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillStyle = 'rgba(140,165,180,0.5)';
    ctx.fillText('0', padL, padT + ih + 7);
    ctx.textAlign = 'right';
    ctx.fillText(`${maxEpoch} epoch`, padL + iw, padT + ih + 7);

    const step = Math.max(1, Math.floor(history.length / 500));

    // 损失面积
    const grad = ctx.createLinearGradient(0, padT, 0, padT + ih);
    grad.addColorStop(0, 'rgba(46,230,200,0.24)');
    grad.addColorStop(1, 'rgba(46,230,200,0)');
    ctx.beginPath();
    ctx.moveTo(x(history[0].epoch), padT + ih);
    for (let i = 0; i < history.length; i += step) {
      const p = history[i];
      const v = isFinite(p.loss) ? p.loss : yMax;
      ctx.lineTo(x(p.epoch), yL(v));
    }
    const last = history[history.length - 1];
    ctx.lineTo(x(last.epoch), yL(isFinite(last.loss) ? last.loss : yMax));
    ctx.lineTo(x(last.epoch), padT + ih);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // 损失线
    ctx.beginPath();
    for (let i = 0; i < history.length; i += step) {
      const p = history[i];
      const v = isFinite(p.loss) ? p.loss : yMax;
      const px = x(p.epoch);
      const py = yL(v);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.lineTo(x(last.epoch), yL(isFinite(last.loss) ? last.loss : yMax));
    ctx.strokeStyle = '#2ee6c8';
    ctx.lineWidth = 1.7;
    ctx.lineJoin = 'round';
    ctx.stroke();

    // 准确率线
    ctx.beginPath();
    for (let i = 0; i < history.length; i += step) {
      const p = history[i];
      const px = x(p.epoch);
      const py = yA(p.acc);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.lineTo(x(last.epoch), yA(last.acc));
    ctx.strokeStyle = 'rgba(255,200,87,0.9)';
    ctx.lineWidth = 1.4;
    ctx.setLineDash([4, 3]);
    ctx.stroke();
    ctx.setLineDash([]);

    // 末端点
    const lx = x(last.epoch);
    const ly = yL(isFinite(last.loss) ? last.loss : yMax);
    ctx.beginPath();
    ctx.arc(lx, ly, 2.6, 0, Math.PI * 2);
    ctx.fillStyle = '#9dfff0';
    ctx.fill();
  }

  /* ---------------- 网络与激活 ---------------- */
  function drawNetwork(canvas, net, probe) {
    const { ctx, w, h } = fitCanvas(canvas);
    if (w < 40 || h < 40) return;

    ctx.fillStyle = 'rgba(6,11,16,0.55)';
    roundRectPath(ctx, 0.5, 0.5, w - 1, h - 1, 11);
    ctx.fill();

    const sizes = net.sizes;
    const cols = sizes.length;
    const padX = 30;
    const padTop = 16;
    const padBottom = 30;
    const colX = (i) => padX + (i * (w - padX * 2)) / (cols - 1 || 1);
    const rowY = (i, count) => {
      if (count === 1) return (padTop + h - padBottom) / 2;
      return padTop + (i * (h - padTop - padBottom)) / (count - 1);
    };

    let maxW = 0;
    for (const layer of net.W) {
      for (let i = 0; i < layer.length; i++) {
        const v = Math.abs(layer[i]);
        if (v > maxW) maxW = v;
      }
    }
    const wScale = Math.max(maxW, 1e-9);

    // 权重连线
    for (let l = 0; l < net.W.length; l++) {
      const W = net.W[l];
      const nOut = sizes[l + 1];
      const nIn = sizes[l];
      for (let i = 0; i < nOut; i++) {
        for (let j = 0; j < nIn; j++) {
          const wt = W[i * nIn + j];
          const t = Math.min(1, Math.abs(wt) / wScale);
          ctx.strokeStyle =
            wt >= 0
              ? `rgba(46,230,200,${0.05 + 0.34 * t})`
              : `rgba(255,122,89,${0.05 + 0.34 * t})`;
          ctx.lineWidth = 0.4 + 2 * t;
          ctx.beginPath();
          ctx.moveTo(colX(l), rowY(j, nIn));
          ctx.lineTo(colX(l + 1), rowY(i, nOut));
          ctx.stroke();
        }
      }
    }

    const acts = probe ? net.activations(probe) : null;
    const maxCount = Math.max.apply(null, sizes);
    const radius = Math.max(3.1, Math.min(8, ((h - padTop - padBottom) / Math.max(4, maxCount)) * 0.34));

    // 节点
    for (let l = 0; l < cols; l++) {
      const count = sizes[l];
      let maxA = 0.001;
      if (acts) {
        for (let i = 0; i < acts[l].length; i++) maxA = Math.max(maxA, Math.abs(acts[l][i]));
      }
      for (let i = 0; i < count; i++) {
        const cx = colX(l);
        const cy = rowY(i, count);
        let t = 0;
        if (acts) {
          if (l === cols - 1) t = acts[l][i] * 2 - 1;
          else t = Math.max(-1, Math.min(1, acts[l][i] / maxA));
        }
        const col = t >= 0 ? C0 : C1;
        const mixv = acts ? Math.abs(t) : 0;
        const base = [13, 23, 32];
        const fill =
          `rgb(${(base[0] + (col[0] - base[0]) * mixv * 0.8) | 0},` +
          `${(base[1] + (col[1] - base[1]) * mixv * 0.8) | 0},` +
          `${(base[2] + (col[2] - base[2]) * mixv * 0.8) | 0})`;
        ctx.beginPath();
        ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.fillStyle = fill;
        ctx.fill();
        ctx.strokeStyle = acts
          ? `rgba(220,245,250,${0.18 + 0.4 * mixv})`
          : 'rgba(140,175,200,0.35)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }

    // 列标签
    ctx.font = MONO_SM;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillStyle = 'rgba(140,165,180,0.5)';
    for (let l = 0; l < cols; l++) {
      const label = l === 0 ? '输入' : l === cols - 1 ? '输出' : `h${l}`;
      ctx.fillText(label, colX(l), h - 16);
    }

    if (maxW < 1e-7) {
      ctx.font = MONO_SM;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillStyle = 'rgba(255,200,87,0.8)';
      ctx.fillText('所有权重为 0 · 对称性未被打破', w / 2, 8);
    }

    // 输出概率
    if (acts) {
      const p = acts[cols - 1][0];
      ctx.font = MONO;
      ctx.textAlign = 'center';
      ctx.fillStyle = 'rgba(235,255,250,0.9)';
      ctx.fillText(`p = ${p.toFixed(2)}`, colX(cols - 1), Math.max(4, rowY(0, 1) - radius - 16));
    }

  }

  /* ---------------- 对照实验 ---------------- */
  function drawSweepChart(canvas, runs, maxEpoch) {
    const { ctx, w, h } = fitCanvas(canvas);
    if (w < 60 || h < 60) return;

    const padL = 42;
    const padR = 14;
    const padT = 14;
    const padB = 28;
    const iw = w - padL - padR;
    const ih = h - padT - padB;

    ctx.fillStyle = 'rgba(6,11,16,0.55)';
    roundRectPath(ctx, 0.5, 0.5, w - 1, h - 1, 11);
    ctx.fill();

    if (!runs || !runs.length) {
      ctx.fillStyle = 'rgba(140,165,180,0.45)';
      ctx.font = MONO;
      ctx.textAlign = 'center';
      ctx.fillText('选择参数后点击「运行对照实验」', w / 2, h / 2);
      return;
    }

    let maxLoss = 0.2;
    for (const r of runs) {
      for (const v of r.history) if (isFinite(v) && v > maxLoss) maxLoss = v;
    }
    const yMax = Math.min(2.5, Math.max(0.35, maxLoss * 1.1));

    ctx.font = MONO_SM;
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 4; i++) {
      const gy = padT + (ih * i) / 4;
      ctx.strokeStyle = i === 4 ? 'rgba(140,175,200,0.16)' : 'rgba(140,175,200,0.07)';
      ctx.beginPath();
      ctx.moveTo(padL, gy + 0.5);
      ctx.lineTo(padL + iw, gy + 0.5);
      ctx.stroke();
      ctx.fillStyle = 'rgba(140,165,180,0.5)';
      ctx.fillText((yMax * (1 - i / 4)).toFixed(2), padL - 7, gy);
    }

    ctx.textAlign = 'left';
    ctx.textBaseline = 'top';
    ctx.fillStyle = 'rgba(140,165,180,0.5)';
    ctx.fillText('0', padL, padT + ih + 8);
    ctx.textAlign = 'right';
    ctx.fillText(`${maxEpoch} 轮`, padL + iw, padT + ih + 8);

    const x = (e) => padL + (e / maxEpoch) * iw;
    const y = (v) => padT + (ih - 1) * (1 - Math.min(1, Math.max(0, v / yMax)));

    for (const r of runs) {
      if (!r.history.length) continue;
      ctx.beginPath();
      const stride = Math.max(1, Math.floor(r.history.length / 400));
      for (let i = 0; i < r.history.length; i += stride) {
        const v = isFinite(r.history[i]) ? r.history[i] : yMax;
        const px = x(i);
        const py = y(v);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      const lastV = r.history[r.history.length - 1];
      ctx.lineTo(x(r.history.length - 1), y(isFinite(lastV) ? lastV : yMax));
      ctx.strokeStyle = r.color;
      ctx.lineWidth = 1.8;
      ctx.lineJoin = 'round';
      ctx.stroke();

      ctx.beginPath();
      ctx.arc(x(r.history.length - 1), y(isFinite(lastV) ? lastV : yMax), 2.4, 0, Math.PI * 2);
      ctx.fillStyle = r.color;
      ctx.fill();
    }
  }

  function drawThumb(canvas, net, train, test) {
    const { ctx, w, h } = fitCanvas(canvas);
    if (w < 10 || h < 10) return;

    ctx.fillStyle = '#070c11';
    ctx.fillRect(0, 0, w, h);
    ctx.save();
    roundRectPath(ctx, 0, 0, w, h, 7);
    ctx.clip();

    const grid = 54;
    const img = ctx.createImageData(grid, grid);
    const data = img.data;
    const input = new Float64Array(2);
    let k = 0;
    for (let gy = 0; gy < grid; gy++) {
      const y = 1 - (2 * (gy + 0.5)) / grid;
      input[1] = y;
      for (let gx = 0; gx < grid; gx++) {
        input[0] = (2 * (gx + 0.5)) / grid - 1;
        let p = net.predict(input);
        if (!isFinite(p)) p = 0.5;
        data[k++] = (FIELD0[0] + (FIELD1[0] - FIELD0[0]) * p) | 0;
        data[k++] = (FIELD0[1] + (FIELD1[1] - FIELD0[1]) * p) | 0;
        data[k++] = (FIELD0[2] + (FIELD1[2] - FIELD0[2]) * p) | 0;
        data[k++] = 255;
      }
    }
    const off = document.createElement('canvas');
    off.width = grid;
    off.height = grid;
    off.getContext('2d').putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, 0, 0, w, h);

    const r = Math.max(0.7, w / 95);
    for (const p of train) {
      ctx.beginPath();
      ctx.arc(((p.x[0] + 1) / 2) * w, ((1 - p.x[1]) / 2) * h, r, 0, Math.PI * 2);
      ctx.fillStyle = p.y === 1 ? 'rgba(255,143,112,0.9)' : 'rgba(77,243,214,0.9)';
      ctx.fill();
    }
    ctx.restore();
  }

  global.Viz = {
    PAD,
    drawBoundary,
    drawLossChart,
    drawNetwork,
    drawSweepChart,
    drawThumb,
  };
})(typeof window !== 'undefined' ? window : globalThis);
