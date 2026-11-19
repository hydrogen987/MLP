/* ============================================================
   sweep.js — 参数对照实验
   固定数据与网络，只改变一个参数，四组模型并行训练。
   ============================================================ */
(function (global) {
  'use strict';

  const COLORS = ['#2ee6c8', '#ffc857', '#9d8cff', '#ff7a59'];
  const OPT_LABEL = { sgd: 'SGD', momentum: 'Momentum', rmsprop: 'RMSProp', adam: 'Adam' };
  const INIT_LABEL = { zeros: '全零', small: '小随机', xavier: 'Xavier', he: 'He' };

  const actLabel = (v) => (Neural.ACTIVATIONS[v] ? Neural.ACTIVATIONS[v].label : v);

  const PARAMS = {
    lr: { name: '学习率', values: [0.001, 0.01, 0.1, 0.6], label: (v) => 'η = ' + v },
    activation: { name: '激活函数', values: ['relu', 'tanh', 'sigmoid', 'leaky'], label: actLabel },
    width: { name: '隐藏层宽度', values: [4, 8, 12, 16], label: (v) => v + ' 个/层' },
    optimizer: { name: '优化器', values: ['sgd', 'momentum', 'rmsprop', 'adam'], label: (v) => OPT_LABEL[v] },
    init: { name: '初始化', values: ['zeros', 'small', 'xavier', 'he'], label: (v) => INIT_LABEL[v] },
  };

  const Sweep = {
    param: 'lr',
    runs: null,
    data: [],
    test: [],
    epoch: 0,
    maxEpoch: 160,
    running: false,
    raf: null,
    getState: null,
    onStart: null,
    els: null,
    _btnHTML: '',

    init(opts) {
      this.getState = opts.getState;
      this.onStart = opts.onStart;
      this.els = {
        seg: document.getElementById('sweepSeg'),
        btn: document.getElementById('btnSweep'),
        status: document.getElementById('sweepStatus'),
        chart: document.getElementById('sweepChart'),
        cards: document.getElementById('sweepCards'),
        legend: document.getElementById('sweepLegend'),
      };
      this._btnHTML = this.els.btn.innerHTML;
      this.els.seg.addEventListener('click', (e) => {
        const btn = e.target.closest('button[data-value]');
        if (!btn) return;
        this.setParam(btn.dataset.value);
      });
      this.els.btn.addEventListener('click', () => this.start());
      this.setParam('lr');
    },

    setParam(value) {
      if (!PARAMS[value]) return;
      this.param = value;
      if (this.els && this.els.seg) {
        this.els.seg.querySelectorAll('button[data-value]').forEach((b) => {
          const on = b.dataset.value === value;
          b.classList.toggle('is-active', on);
          b.setAttribute('aria-pressed', String(on));
        });
      }
      this.resetResults();
    },

    resetResults() {
      this.stop();
      this.runs = null;
      this.epoch = 0;
      this.done = false;
      if (!this.els) return;
      this.els.cards.innerHTML =
        '<p class="sweep-empty">运行后，这里会出现四张决策边界缩略图，以及各自的训练 / 测试准确率。</p>';
      this.els.legend.innerHTML = '';
      this.els.status.textContent = `当前数据集 · ${PARAMS[this.param].name} · ${this.maxEpoch} 轮`;
      Viz.drawSweepChart(this.els.chart, null, this.maxEpoch);
    },

    start() {
      if (this.running) return;
      const base = this.getState();
      if (!base || !base.net) return;
      this.stop();
      if (this.onStart) this.onStart();

      this.data = base.train.slice(0, 240);
      if (this.data.length < 8) this.data = base.train.slice();
      this.test = base.test;
      const spec = PARAMS[this.param];
      const baseBatch = base.batchSize || this.data.length;
      this.maxEpoch = 160;
      this.epoch = 0;
      this.midDrawn = false;
      this.done = false;

      this.runs = spec.values.map((value, i) => {
        const conf = {
          sizes: [2].concat(Array(base.layers).fill(base.width), [1]),
          activation: base.activation,
          init: base.init,
          optimizer: base.optimizer,
          lr: base.lr,
          batchSize: Math.max(1, Math.min(baseBatch, 32)),
          l2: base.l2,
        };

        if (this.param === 'lr') conf.lr = value;
        else if (this.param === 'activation') conf.activation = value;
        else if (this.param === 'width') conf.sizes = [2].concat(Array(base.layers).fill(value), [1]);
        else if (this.param === 'optimizer') conf.optimizer = value;
        else if (this.param === 'init') conf.init = value;

        const net = new Neural.MLP({
          sizes: conf.sizes,
          activation: conf.activation,
          init: conf.init,
          seed: 4321,
        });
        const opt = new Neural.Optimizer(conf.optimizer);
        opt.reset(net.params);

        return {
          conf,
          value,
          label: spec.label(value),
          color: COLORS[i],
          net,
          opt,
          history: [],
          loss: NaN,
          trainAcc: 0,
          testAcc: 0,
          card: null,
        };
      });

      this.buildCards();
      Viz.drawSweepChart(this.els.chart, this.runs, this.maxEpoch);
      this.els.status.textContent = `训练中 0 / ${this.maxEpoch} 轮`;
      this.els.btn.disabled = true;
      this.els.btn.textContent = '训练中…';
      this.running = true;
      this.raf = requestAnimationFrame(() => this.frame());
    },

    buildCards() {
      this.els.cards.innerHTML = '';
      this.els.legend.innerHTML = '';
      for (const run of this.runs) {
        const card = document.createElement('div');
        card.className = 'sweep-card';
        card.style.setProperty('--c', run.color);
        card.innerHTML = `
          <div class="sweep-thumb"><canvas></canvas></div>
          <div class="sweep-meta"><i style="background:${run.color}"></i><b title="${run.label}">${run.label}</b></div>
          <div class="sweep-nums"><span class="acc">—</span><span class="loss">—</span></div>`;
        this.els.cards.appendChild(card);
        run.card = card;

        const legend = document.createElement('span');
        legend.innerHTML = `<i style="background:${run.color}"></i>${run.label}`;
        this.els.legend.appendChild(legend);
      }
    },

    frame() {
      if (!this.running) return;
      for (const run of this.runs) {
        const bs = Math.min(run.conf.batchSize, this.data.length);
        for (let k = 0; k < 2; k++) {
          const res = run.net.trainEpoch(this.data, {
            lr: run.conf.lr,
            batchSize: bs,
            l2: run.conf.l2,
            optimizer: run.opt,
          });
          run.history.push(res.loss);
          run.loss = res.loss;
        }
      }
      this.epoch = Math.min(this.maxEpoch, this.epoch + 2);
      Viz.drawSweepChart(this.els.chart, this.runs, this.maxEpoch);
      this.els.status.textContent = `训练中 ${this.epoch} / ${this.maxEpoch} 轮`;

      if (!this.midDrawn && this.epoch >= this.maxEpoch / 2) {
        this.midDrawn = true;
        for (const run of this.runs) {
          if (run.card) Viz.drawThumb(run.card.querySelector('canvas'), run.net, this.data, this.test);
        }
      }

      if (this.epoch >= this.maxEpoch) this.finish();
      else this.raf = requestAnimationFrame(() => this.frame());
    },

    finish() {
      this.running = false;
      if (this.raf) cancelAnimationFrame(this.raf);
      this.raf = null;

      for (const run of this.runs) {
        run.trainAcc = run.net.evaluate(this.data).acc;
        run.testAcc = run.net.evaluate(this.test).acc;
        if (!run.card) continue;
        const accEl = run.card.querySelector('.acc');
        const lossEl = run.card.querySelector('.loss');
        accEl.textContent = `训 ${(run.trainAcc * 100).toFixed(0)}% / 测 ${(run.testAcc * 100).toFixed(0)}%`;
        accEl.title = '训练准确率 / 测试准确率';
        lossEl.textContent = isFinite(run.loss) ? 'L ' + run.loss.toFixed(3) : 'L 发散';
        Viz.drawThumb(run.card.querySelector('canvas'), run.net, this.data, this.test);
      }

      this.done = true;
      this.els.status.textContent = '完成 · 卡片内为 训练 / 测试 准确率';
      this.els.btn.disabled = false;
      this.els.btn.innerHTML = this._btnHTML;
    },

    stop() {
      this.running = false;
      if (this.raf) cancelAnimationFrame(this.raf);
      this.raf = null;
      if (this.els && this.els.btn && this.els.btn.disabled) {
        this.els.btn.disabled = false;
        this.els.btn.innerHTML = this._btnHTML;
      }
    },

    redraw() {
      if (!this.els) return;
      Viz.drawSweepChart(this.els.chart, this.runs, this.maxEpoch);
      if (!this.runs || this.running || !this.done) return;
      for (const run of this.runs) {
        if (!run.card) continue;
        Viz.drawThumb(run.card.querySelector('canvas'), run.net, this.data, this.test);
      }
    },
  };

  global.Sweep = Sweep;
})(typeof window !== 'undefined' ? window : globalThis);
