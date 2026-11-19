/* ============================================================
   app.js — 页面交互、训练循环与状态管理
   ============================================================ */
(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const LR_MIN = Math.log10(0.0002);
  const LR_MAX = Math.log10(8);
  const MAX_EPOCHS = 2400;

  const OPT_LABEL = { sgd: 'SGD', momentum: 'Momentum', rmsprop: 'RMSProp', adam: 'Adam' };

  const DEFAULTS = {
    dataset: 'xor',
    n: 400,
    noise: 0.18,
    layers: 2,
    width: 8,
    activation: 'relu',
    init: 'he',
    optimizer: 'adam',
    lr: 0.03,
    batchSize: 16,
    l2: 0,
    speed: 4,
    dataSeed: 20260419,
    netSeed: 7331,
  };

  const state = Object.assign({}, DEFAULTS, {
    running: false,
    epoch: 0,
    history: [],
    train: [],
    test: [],
    net: null,
    opt: null,
    probe: null,
    testEval: { loss: 0, acc: 0 },
    epochMs: 0,
    lastLoss: 0,
    lastAcc: 0,
  });

  let rafId = null;
  let toastTimer = null;

  /* ---------------- 工具 ---------------- */
  function lrToSlider(lr) {
    return ((Math.log10(lr) - LR_MIN) / (LR_MAX - LR_MIN)) * 100;
  }

  function sliderToLR(v) {
    return Math.pow(10, LR_MIN + ((LR_MAX - LR_MIN) * v) / 100);
  }

  function fmtLR(lr) {
    if (!isFinite(lr)) return '—';
    if (lr < 0.001) return lr.toFixed(5);
    if (lr < 0.01) return lr.toFixed(4);
    if (lr < 1) return lr.toFixed(3);
    return lr.toFixed(1);
  }

  function fmtLoss(v) {
    if (!isFinite(v)) return '发散';
    return v < 1 ? v.toFixed(3) : v.toFixed(2);
  }

  function pct(v) {
    if (!isFinite(v)) return '—';
    return (v * 100).toFixed(1) + '%';
  }

  function toast(msg) {
    const el = $('toast');
    if (!el) return;
    el.textContent = msg;
    el.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.remove('show'), 2400);
  }

  function scrollToSection(selector) {
    const el = document.querySelector(selector);
    if (!el) return;
    const top = el.getBoundingClientRect().top + window.scrollY - 78;
    window.scrollTo({ top, behavior: 'smooth' });
  }

  function netSizes() {
    const hidden = [];
    for (let i = 0; i < state.layers; i++) hidden.push(state.width);
    return [2].concat(hidden, [1]);
  }

  /* ---------------- 数据 ---------------- */
  function rebuildData() {
    const points = NeuralDatasets.generate(state.dataset, state.n, state.noise, state.dataSeed);
    const rnd = Neural.mulberry32(state.dataSeed + 991);
    const byClass = [[], []];
    for (const p of points) byClass[p.y].push(p);
    for (const arr of byClass) {
      for (let i = arr.length - 1; i > 0; i--) {
        const j = (rnd() * (i + 1)) | 0;
        const tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
      }
    }
    const test = [];
    for (const arr of byClass) {
      const k = Math.max(1, Math.round(arr.length * 0.22));
      for (let i = 0; i < k; i++) test.push(arr[i]);
    }
    const testIds = new Set(test);
    state.test = test;
    state.train = points.filter((p) => !testIds.has(p));
  }

  /* ---------------- 网络生命周期 ---------------- */
  function resetNetwork() {
    stopLoop();
    state.net = new Neural.MLP({
      sizes: netSizes(),
      activation: state.activation,
      init: state.init,
      seed: state.netSeed,
    });
    state.opt = new Neural.Optimizer(state.optimizer);
    state.opt.reset(state.net.params);
    state.epoch = 0;
    state.history = [];
    state.probe = null;
    state.epochMs = 0;

    const ev = state.net.evaluate(state.train);
    state.lastLoss = ev.loss;
    state.lastAcc = ev.acc;
    state.history.push({ epoch: 0, loss: ev.loss, acc: ev.acc });
    state.testEval = state.net.evaluate(state.test);

    renderAll();
  }

  function rebuildAll() {
    rebuildData();
    resetNetwork();
  }

  /* ---------------- 训练 ---------------- */
  function trainEpochs(count) {
    for (let i = 0; i < count; i++) {
      if (state.epoch >= MAX_EPOCHS) break;
      const res = state.net.trainEpoch(state.train, {
        lr: state.lr,
        batchSize: state.batchSize,
        l2: state.l2,
        optimizer: state.opt,
      });
      state.epoch += 1;
      state.lastLoss = res.loss;
      state.lastAcc = res.acc;
      state.history.push({ epoch: state.epoch, loss: res.loss, acc: res.acc });
    }
    state.testEval = state.net.evaluate(state.test);
  }

  function startLoop() {
    if (state.running) return;
    if (!state.net) return;
    if (state.epoch >= MAX_EPOCHS) {
      toast('已经训练到上限，先重置再继续');
      return;
    }
    state.running = true;
    updatePlayUI();
    updateStatus();

    const frame = () => {
      if (!state.running) return;
      const t0 = performance.now();
      trainEpochs(state.speed);
      state.epochMs = (performance.now() - t0) / Math.max(1, state.speed);
      renderAll();
      if (state.epoch >= MAX_EPOCHS) {
        stopLoop();
        toast('已训练 ' + MAX_EPOCHS + ' 轮，可重置后重新开始');
        return;
      }
      rafId = requestAnimationFrame(frame);
    };
    rafId = requestAnimationFrame(frame);
  }

  function stopLoop() {
    state.running = false;
    if (rafId) {
      cancelAnimationFrame(rafId);
      rafId = null;
    }
    updatePlayUI();
    updateStatus();
  }

  function updatePlayUI() {
    const label = $('playLabel');
    const icon = $('playIcon');
    if (label) label.textContent = state.running ? '暂停训练' : '开始训练';
    if (icon) icon.setAttribute('href', state.running ? '#i-pause' : '#i-play');
    const btn = $('btnPlay');
    if (btn) btn.classList.toggle('is-running', state.running);
    const dot = $('statusDot');
    if (dot) dot.classList.toggle('live', state.running);
  }

  function updateStatus() {
    const el = $('statusLine');
    if (!el) return;
    if (state.running) el.textContent = `训练中 · 第 ${state.epoch} 轮`;
    else if (state.epoch > 0) el.textContent = `已暂停 · 第 ${state.epoch} 轮`;
    else el.textContent = '准备就绪';
  }

  /* ---------------- 渲染 ---------------- */
  function renderAll() {
    if (!state.net) return;
    Viz.drawBoundary($('boundaryCanvas'), state.net, {
      train: state.train,
      test: state.test,
      probe: state.probe,
    });
    Viz.drawLossChart($('lossCanvas'), state.history);
    Viz.drawNetwork($('netCanvas'), state.net, state.probe);
    $('netMeta').textContent = `${state.net.sizes.join(' → ')} · ${
      Neural.ACTIVATIONS[state.activation].label
    } · ${OPT_LABEL[state.optimizer]}`;
    updateStats();
    updateProbeLabel();
    updateStatus();
  }

  function updateStats() {
    if (!state.net) return;
    $('statEpoch').textContent = String(state.epoch);
    $('statLoss').textContent = fmtLoss(state.lastLoss);
    $('statTrainAcc').textContent = pct(state.lastAcc);
    $('statTestAcc').textContent = pct(state.testEval.acc);
    $('statParams').textContent = String(state.net.paramCount);
    $('statSpeed').textContent = state.epoch ? state.epochMs.toFixed(1) + ' ms' : '—';

    const testEl = $('statTestAcc');
    testEl.classList.remove('up', 'down');
    if (state.epoch > 0 && isFinite(state.testEval.acc)) {
      if (state.testEval.acc >= 0.85) testEl.classList.add('up');
      else if (state.testEval.acc <= 0.65) testEl.classList.add('down');
    }
  }

  function updateProbeLabel() {
    const el = $('probeLabel');
    if (!el || !state.net) return;
    if (!state.probe) {
      el.textContent = '探针 —';
      return;
    }
    const p = state.net.predict(state.probe);
    el.textContent = `探针 (${state.probe[0].toFixed(2)}, ${state.probe[1].toFixed(
      2
    )}) → p = ${p.toFixed(2)}`;
  }

  /* ---------------- 控件同步 ---------------- */
  function setSeg(id, value) {
    const root = $(id);
    if (!root) return;
    root.querySelectorAll('button[data-value]').forEach((b) => {
      const on = b.dataset.value === String(value);
      b.classList.toggle('is-active', on);
      b.setAttribute('aria-pressed', String(on));
    });
  }

  function syncControls() {
    setSeg('datasetSeg', state.dataset);
    setSeg('layersSeg', state.layers);
    setSeg('activationSeg', state.activation);
    setSeg('optimizerSeg', state.optimizer);

    $('nRange').value = String(state.n);
    $('nVal').textContent = String(state.n);
    $('noiseRange').value = String(Math.round(state.noise * 100));
    $('noiseVal').textContent = state.noise.toFixed(2);
    $('widthRange').value = String(state.width);
    $('widthVal').textContent = String(state.width);
    $('layersVal').textContent = `${state.layers} 层`;
    $('lrRange').value = String(lrToSlider(state.lr));
    $('lrVal').textContent = fmtLR(state.lr);
    $('batchSelect').value = String(state.batchSize);
    $('l2Range').value = String(Math.round(state.l2 * 1000));
    $('l2Val').textContent = state.l2.toFixed(3);
    $('speedRange').value = String(state.speed);
    $('speedVal').textContent = `${state.speed} 轮 / 帧`;
    $('initSelect').value = state.init;
  }

  /* ---------------- 控件绑定 ---------------- */
  function bindSeg(id, key, after) {
    const root = $(id);
    root.addEventListener('click', (e) => {
      const btn = e.target.closest('button[data-value]');
      if (!btn || !root.contains(btn)) return;
      const raw = btn.dataset.value;
      state[key] = typeof state[key] === 'number' ? Number(raw) : raw;
      setSeg(id, raw);
      if (after) after();
    });
  }

  function bindControls() {
    bindSeg('datasetSeg', 'dataset', () => rebuildAll());
    bindSeg('layersSeg', 'layers', () => {
      $('layersVal').textContent = `${state.layers} 层`;
      resetNetwork();
    });
    bindSeg('activationSeg', 'activation', () => resetNetwork());
    bindSeg('optimizerSeg', 'optimizer', () => resetNetwork());

    const nRange = $('nRange');
    nRange.addEventListener('input', () => {
      state.n = Number(nRange.value);
      $('nVal').textContent = String(state.n);
    });
    nRange.addEventListener('change', () => rebuildAll());

    const noiseRange = $('noiseRange');
    noiseRange.addEventListener('input', () => {
      state.noise = Number(noiseRange.value) / 100;
      $('noiseVal').textContent = state.noise.toFixed(2);
    });
    noiseRange.addEventListener('change', () => rebuildAll());

    const widthRange = $('widthRange');
    widthRange.addEventListener('input', () => {
      state.width = Number(widthRange.value);
      $('widthVal').textContent = String(state.width);
    });
    widthRange.addEventListener('change', () => resetNetwork());

    const lrRange = $('lrRange');
    lrRange.addEventListener('input', () => {
      state.lr = sliderToLR(Number(lrRange.value));
      $('lrVal').textContent = fmtLR(state.lr);
    });

    $('batchSelect').addEventListener('change', (e) => {
      state.batchSize = Number(e.target.value);
    });

    const l2Range = $('l2Range');
    l2Range.addEventListener('input', () => {
      state.l2 = Number(l2Range.value) / 1000;
      $('l2Val').textContent = state.l2.toFixed(3);
    });

    const speedRange = $('speedRange');
    speedRange.addEventListener('input', () => {
      state.speed = Number(speedRange.value);
      $('speedVal').textContent = `${state.speed} 轮 / 帧`;
    });

    $('initSelect').addEventListener('change', (e) => {
      state.init = e.target.value;
      resetNetwork();
    });
  }

  /* ---------------- 探针 ---------------- */
  function bindProbe() {
    const canvas = $('boundaryCanvas');
    const move = (e) => {
      if (!state.net) return;
      const rect = canvas.getBoundingClientRect();
      const pad = Viz.PAD;
      const nx = (e.clientX - rect.left - pad) / Math.max(1, rect.width - pad * 2);
      const ny = (e.clientY - rect.top - pad) / Math.max(1, rect.height - pad * 2);
      if (nx < -0.04 || nx > 1.04 || ny < -0.04 || ny > 1.04) return;
      state.probe = [clamp(nx * 2 - 1, -1, 1), clamp(1 - ny * 2, -1, 1)];
      Viz.drawBoundary(canvas, state.net, {
        train: state.train,
        test: state.test,
        probe: state.probe,
      });
      Viz.drawNetwork($('netCanvas'), state.net, state.probe);
      updateProbeLabel();
    };

    canvas.addEventListener('pointermove', move);
    canvas.addEventListener('pointerdown', move);
    canvas.addEventListener('pointerleave', () => {
      state.probe = null;
      renderAll();
    });
  }

  /* ---------------- 预设实验 ---------------- */
  const PRESETS = [
    {
      id: 'lr-high',
      tag: '优化',
      title: '学习率过大',
      en: 'Too large η',
      desc: 'SGD 配 η = 8，一步跨过谷底。损失在 0.5 附近反复跳动，决策边界始终安定不下来。',
      config: { dataset: 'xor', n: 400, noise: 0.18, layers: 1, width: 8, activation: 'relu', init: 'he', optimizer: 'sgd', lr: 8, batchSize: 8, l2: 0 },
    },
    {
      id: 'lr-low',
      tag: '优化',
      title: '学习率过小',
      en: 'Too small η',
      desc: 'η = 0.0005，参数每一步只挪一点点。训练几百轮后仍然明显欠拟合，损失下降得非常慢。',
      config: { dataset: 'circle', n: 400, noise: 0.18, layers: 2, width: 8, activation: 'relu', init: 'he', optimizer: 'sgd', lr: 0.0005, batchSize: 16, l2: 0 },
    },
    {
      id: 'underfit',
      tag: '容量',
      title: '容量不足',
      en: 'Underfitting',
      desc: '隐藏层设为 0，网络退化成线性分类器，圆环被一条直线硬生生切开，训练损失卡在高位。',
      config: { dataset: 'circle', n: 400, noise: 0.18, layers: 0, width: 8, activation: 'relu', init: 'he', optimizer: 'adam', lr: 0.05, batchSize: 16, l2: 0 },
    },
    {
      id: 'overfit',
      tag: '泛化',
      title: '过拟合',
      en: 'Overfitting',
      desc: '只有 40 个棋盘样本，却用了 2×16 的网络。训练集被完全记住，测试点开始落在错误的一侧。',
      config: { dataset: 'checker', n: 40, noise: 0.22, layers: 2, width: 16, activation: 'relu', init: 'he', optimizer: 'adam', lr: 0.03, batchSize: 8, l2: 0 },
    },
    {
      id: 'zeros',
      tag: '初始化',
      title: '零初始化',
      en: 'Symmetry trap',
      desc: '所有权重初始化为 0，同层神经元的梯度永远相同。损失停在 ln 2 ≈ 0.693，网络什么也学不到。',
      config: { dataset: 'xor', n: 400, noise: 0.18, layers: 1, width: 8, activation: 'relu', init: 'zeros', optimizer: 'adam', lr: 0.03, batchSize: 16, l2: 0 },
    },
    {
      id: 'optimizer',
      tag: '对照',
      title: '优化器对照',
      en: 'Adam vs SGD',
      desc: '同一份棋盘数据交给四个优化器并行训练：SGD 几乎停在原地，Adam 最快把边界画出来。',
      action: 'sweep',
      sweepParam: 'optimizer',
      config: { dataset: 'checker', n: 400, noise: 0.16, layers: 2, width: 12, activation: 'relu', init: 'he', optimizer: 'adam', lr: 0.02, batchSize: 16, l2: 0 },
    },
  ];

  function configSummary(p) {
    if (p.action === 'sweep') {
      if (!p.config) return '四组并行 · 160 轮';
      return `${NeuralDatasets.META[p.config.dataset].label} · 4 组 · 160 轮`;
    }
    const c = p.config;
    return `${c.n} 样本 · ${c.layers}×${c.width} · ${OPT_LABEL[c.optimizer]} η=${fmtLR(c.lr)}`;
  }

  function renderPresets() {
    const grid = $('presetGrid');
    grid.innerHTML = PRESETS.map((p, i) => {
      const action = p.action === 'sweep' ? '去对照实验' : '载入并运行';
      return `<article class="preset-card">
        <div class="preset-head">
          <span class="preset-num">${String(i + 1).padStart(2, '0')}</span>
          <span class="preset-tag">${p.tag}</span>
        </div>
        <h3>${p.title}</h3>
        <p class="preset-en">${p.en}</p>
        <p class="preset-desc">${p.desc}</p>
        <div class="preset-foot">
          <span class="preset-config">${configSummary(p)}</span>
          <button class="preset-run" data-preset="${p.id}" type="button">${action}
            <svg class="icon" style="width:12px;height:12px"><use href="#i-arrow"/></svg>
          </button>
        </div>
      </article>`;
    }).join('');
  }

  function applyPreset(id) {
    const preset = PRESETS.find((p) => p.id === id);
    if (!preset) return;

    if (preset.action === 'sweep') {
      Object.assign(state, DEFAULTS, preset.config || {});
      state.netSeed = 7331;
      syncControls();
      rebuildAll();
      scrollToSection('#sweep');
      if (window.Sweep) {
        Sweep.setParam(preset.sweepParam);
        setTimeout(() => Sweep.start(), 700);
      }
      return;
    }

    Object.assign(state, DEFAULTS, preset.config);
    state.netSeed = 7331;
    syncControls();
    rebuildAll();
    scrollToSection('#lab');
    toast(`已载入「${preset.title}」，正在训练`);
    setTimeout(() => {
      if (!state.running) startLoop();
    }, 720);
  }

  /* ---------------- 其它 ---------------- */
  function bindButtons() {
    $('btnPlay').addEventListener('click', () => {
      if (state.running) stopLoop();
      else startLoop();
    });

    const heroStart = $('heroStart');
    if (heroStart) {
      heroStart.addEventListener('click', () => {
        setTimeout(() => {
          if (!state.running) startLoop();
        }, 620);
      });
    }

    $('btnStep').addEventListener('click', () => {
      if (!state.net) return;
      if (state.running) stopLoop();
      if (state.epoch >= MAX_EPOCHS) {
        toast('已经训练到上限，先重置再继续');
        return;
      }
      const t0 = performance.now();
      trainEpochs(1);
      state.epochMs = performance.now() - t0;
      renderAll();
    });

    $('btnReset').addEventListener('click', () => {
      resetNetwork();
      toast('已重置：重新初始化权重，数据保持不变');
    });

    $('btnShuffle').addEventListener('click', () => {
      state.dataSeed = (Math.random() * 1e9) | 0;
      rebuildAll();
      toast('已重新采样并打乱数据');
    });

    document.addEventListener('click', (e) => {
      const btn = e.target.closest('[data-preset]');
      if (!btn) return;
      applyPreset(btn.dataset.preset);
    });

    document.addEventListener('keydown', (e) => {
      if (e.code !== 'Space' || e.repeat) return;
      const tag = (document.activeElement && document.activeElement.tagName) || '';
      if (/^(INPUT|SELECT|TEXTAREA|BUTTON|A|OPTION)$/.test(tag)) return;
      e.preventDefault();
      if (state.running) stopLoop();
      else startLoop();
    });

    let resizeTimer = null;
    window.addEventListener('resize', () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        if (state.net) renderAll();
        if (window.Sweep) Sweep.redraw();
      }, 140);
    });
  }

  /* ---------------- 启动 ---------------- */
  function init() {
    renderPresets();
    syncControls();
    bindControls();
    bindButtons();
    bindProbe();
    rebuildAll();

    if (window.Sweep) {
      Sweep.init({
        getState: () => state,
        onStart: () => stopLoop(),
      });
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
