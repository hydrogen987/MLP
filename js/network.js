/* ============================================================
   network.js — 从零实现的多层感知机
   前向传播 / 反向传播 / SGD·Momentum·RMSProp·Adam / 四种初始化
   不依赖任何框架，浏览器与 Node 均可运行。
   ============================================================ */
(function (global) {
  'use strict';

  /* ---------- 可复现随机数 ---------- */
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

  function sigmoid(z) {
    if (z >= 0) return 1 / (1 + Math.exp(-z));
    const e = Math.exp(z);
    return e / (1 + e);
  }

  /* ---------- 激活函数 ---------- */
  const ACTIVATIONS = {
    relu: {
      label: 'ReLU',
      f: (z) => (z > 0 ? z : 0),
      df: (z) => (z > 0 ? 1 : 0),
    },
    leaky: {
      label: 'LeakyReLU',
      f: (z) => (z > 0 ? z : 0.01 * z),
      df: (z) => (z > 0 ? 1 : 0.01),
    },
    tanh: {
      label: 'Tanh',
      f: (z) => Math.tanh(z),
      df: (z, a) => 1 - a * a,
    },
    sigmoid: {
      label: 'Sigmoid',
      f: (z) => sigmoid(z),
      df: (z, a) => a * (1 - a),
    },
  };

  /* ---------- 优化器 ---------- */
  class Optimizer {
    constructor(type) {
      this.type = type || 'adam';
      this.t = 0;
      this.m = null;
      this.v = null;
    }

    reset(params) {
      this.t = 0;
      this.m = null;
      this.v = null;
      if (this.type === 'momentum') {
        this.m = params.map((p) => new Float64Array(p.length));
      } else if (this.type === 'adam') {
        this.m = params.map((p) => new Float64Array(p.length));
        this.v = params.map((p) => new Float64Array(p.length));
      } else if (this.type === 'rmsprop') {
        this.v = params.map((p) => new Float64Array(p.length));
      }
    }

    step(params, grads, lr) {
      const type = this.type;
      this.t += 1;

      if ((type === 'momentum' || type === 'adam') && !this.m) this.reset(params);
      if (type === 'rmsprop' && !this.v) this.reset(params);

      if (type === 'sgd') {
        for (let k = 0; k < params.length; k++) {
          const p = params[k];
          const g = grads[k];
          for (let i = 0; i < p.length; i++) p[i] -= lr * g[i];
        }
        return;
      }

      if (type === 'momentum') {
        for (let k = 0; k < params.length; k++) {
          const p = params[k];
          const g = grads[k];
          const v = this.m[k];
          for (let i = 0; i < p.length; i++) {
            v[i] = 0.9 * v[i] - lr * g[i];
            p[i] += v[i];
          }
        }
        return;
      }

      if (type === 'rmsprop') {
        for (let k = 0; k < params.length; k++) {
          const p = params[k];
          const g = grads[k];
          const s = this.v[k];
          for (let i = 0; i < p.length; i++) {
            s[i] = 0.9 * s[i] + 0.1 * g[i] * g[i];
            p[i] -= (lr * g[i]) / (Math.sqrt(s[i]) + 1e-8);
          }
        }
        return;
      }

      // Adam
      const b1 = 0.9;
      const b2 = 0.999;
      const c1 = 1 - Math.pow(b1, this.t);
      const c2 = 1 - Math.pow(b2, this.t);
      for (let k = 0; k < params.length; k++) {
        const p = params[k];
        const g = grads[k];
        const m = this.m[k];
        const v = this.v[k];
        for (let i = 0; i < p.length; i++) {
          m[i] = b1 * m[i] + (1 - b1) * g[i];
          v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
          p[i] -= (lr * (m[i] / c1)) / (Math.sqrt(v[i] / c2) + 1e-8);
        }
      }
    }
  }

  /* ---------- 多层感知机 ---------- */
  class MLP {
    constructor(opts) {
      const options = opts || {};
      this.sizes = (options.sizes || [2, 8, 8, 1]).slice();
      this.activation = options.activation || 'relu';
      this.init = options.init || 'he';
      this.act = ACTIVATIONS[this.activation] || ACTIVATIONS.relu;
      this.seed = options.seed == null ? 42 : options.seed;
      this.rand = mulberry32(this.seed);
      this.version = 0;

      this.W = [];
      this.b = [];
      for (let l = 1; l < this.sizes.length; l++) {
        this.W.push(this._initMatrix(this.sizes[l], this.sizes[l - 1]));
        this.b.push(new Float64Array(this.sizes[l]));
      }

      this.params = this.W.concat(this.b);
      this.grads = this.W
        .map((w) => new Float64Array(w.length))
        .concat(this.b.map((b) => new Float64Array(b.length)));

      const L = this.W.length;
      this._gW = this.grads.slice(0, L);
      this._gB = this.grads.slice(L);
      this._scratch = this.b.map((b) => [
        new Float64Array(b.length),
        new Float64Array(b.length),
      ]);
      this._order = null;
    }

    _initMatrix(nOut, nIn) {
      const W = new Float64Array(nOut * nIn);
      if (this.init === 'zeros') return W;
      let std;
      if (this.init === 'he') std = Math.sqrt(2 / nIn);
      else if (this.init === 'xavier') std = Math.sqrt(2 / (nIn + nOut));
      else std = 0.1; // 小随机
      for (let i = 0; i < W.length; i++) W[i] = gaussian(this.rand) * std;
      return W;
    }

    get paramCount() {
      let n = 0;
      for (const p of this.params) n += p.length;
      return n;
    }

    /* 只做预测：复用缓冲区，供决策边界逐像素采样使用 */
    predict(x) {
      let a = x;
      let nIn = x.length;
      for (let l = 0; l < this.W.length; l++) {
        const W = this.W[l];
        const b = this.b[l];
        const nOut = b.length;
        const z = this._scratch[l][0];
        const out = this._scratch[l][1];
        for (let i = 0; i < nOut; i++) {
          let s = b[i];
          const off = i * nIn;
          for (let j = 0; j < nIn; j++) s += W[off + j] * a[j];
          z[i] = s;
        }
        const last = l === this.W.length - 1;
        for (let i = 0; i < nOut; i++) out[i] = last ? sigmoid(z[i]) : this.act.f(z[i]);
        a = out;
        nIn = nOut;
      }
      return a[0];
    }

    /* 返回每一层的激活值，用于网络结构图 */
    activations(x) {
      const layers = [Array.from(x)];
      let a = x;
      let nIn = x.length;
      for (let l = 0; l < this.W.length; l++) {
        const W = this.W[l];
        const b = this.b[l];
        const nOut = b.length;
        const z = new Float64Array(nOut);
        const out = new Float64Array(nOut);
        for (let i = 0; i < nOut; i++) {
          let s = b[i];
          const off = i * nIn;
          for (let j = 0; j < nIn; j++) s += W[off + j] * a[j];
          z[i] = s;
        }
        const last = l === this.W.length - 1;
        for (let i = 0; i < nOut; i++) out[i] = last ? sigmoid(z[i]) : this.act.f(z[i]);
        layers.push(Array.from(out));
        a = out;
        nIn = nOut;
      }
      return layers;
    }

    /* 带缓存的前向传播，训练时使用 */
    forward(x) {
      const acts = [x];
      const zs = [];
      let a = x;
      let nIn = x.length;
      for (let l = 0; l < this.W.length; l++) {
        const W = this.W[l];
        const b = this.b[l];
        const nOut = b.length;
        const z = new Float64Array(nOut);
        for (let i = 0; i < nOut; i++) {
          let s = b[i];
          const off = i * nIn;
          for (let j = 0; j < nIn; j++) s += W[off + j] * a[j];
          z[i] = s;
        }
        zs.push(z);
        const out = new Float64Array(nOut);
        const last = l === this.W.length - 1;
        for (let i = 0; i < nOut; i++) out[i] = last ? sigmoid(z[i]) : this.act.f(z[i]);
        acts.push(out);
        a = out;
        nIn = nOut;
      }
      return { acts, zs, p: a[0] };
    }

    /* 反向传播：把梯度累加到 _gW / _gB */
    backward(cache, y) {
      const L = this.W.length;
      const acts = cache.acts;
      const zs = cache.zs;
      const deltas = new Array(L);

      // sigmoid + 二元交叉熵 ⇒ δ = p − y
      deltas[L - 1] = new Float64Array([cache.p - y]);

      for (let l = L - 1; l >= 0; l--) {
        const d = deltas[l];
        const aPrev = acts[l];
        const nOut = d.length;
        const nIn = aPrev.length;
        const gW = this._gW[l];
        const gB = this._gB[l];
        const W = this.W[l];

        for (let i = 0; i < nOut; i++) {
          const di = d[i];
          gB[i] += di;
          const off = i * nIn;
          for (let j = 0; j < nIn; j++) gW[off + j] += di * aPrev[j];
        }

        if (l > 0) {
          const nd = new Float64Array(nIn);
          for (let j = 0; j < nIn; j++) {
            let s = 0;
            for (let i = 0; i < nOut; i++) s += W[i * nIn + j] * d[i];
            nd[j] = s * this.act.df(zs[l - 1][j], acts[l][j]);
          }
          deltas[l - 1] = nd;
        }
      }
    }

    /* 训练一个 epoch：打乱数据 → 小批量前向/反向 → 更新参数 */
    trainEpoch(data, opts) {
      const options = opts || {};
      const n = data.length;
      const lr = options.lr == null ? 0.03 : options.lr;
      const l2 = options.l2 || 0;
      const optimizer = options.optimizer;
      const batchSize = Math.max(1, Math.min(options.batchSize || n, n));
      const L = this.W.length;

      if (!this._order || this._order.length !== n) {
        this._order = new Int32Array(n);
        for (let i = 0; i < n; i++) this._order[i] = i;
      }
      const order = this._order;
      for (let i = n - 1; i > 0; i--) {
        const j = (this.rand() * (i + 1)) | 0;
        const tmp = order[i];
        order[i] = order[j];
        order[j] = tmp;
      }

      let lossSum = 0;
      let correct = 0;

      for (let start = 0; start < n; start += batchSize) {
        const end = Math.min(start + batchSize, n);
        const count = end - start;
        const inv = 1 / count;

        for (let k = start; k < end; k++) {
          const sample = data[order[k]];
          const cache = this.forward(sample.x);
          let p = cache.p;
          if (!isFinite(p)) p = 0.5;
          p = Math.min(1 - 1e-7, Math.max(1e-7, p));
          lossSum += -(sample.y * Math.log(p) + (1 - sample.y) * Math.log(1 - p));
          if ((cache.p >= 0.5 ? 1 : 0) === sample.y) correct += 1;
          this.backward(cache, sample.y);
        }

        for (let l = 0; l < L; l++) {
          const gw = this._gW[l];
          const gb = this._gB[l];
          const W = this.W[l];
          if (l2 > 0) {
            for (let i = 0; i < gw.length; i++) gw[i] = gw[i] * inv + l2 * W[i];
          } else {
            for (let i = 0; i < gw.length; i++) gw[i] *= inv;
          }
          for (let i = 0; i < gb.length; i++) gb[i] *= inv;
        }

        optimizer.step(this.params, this.grads, lr);

        if (end < n) {
          for (let g = 0; g < this.grads.length; g++) this.grads[g].fill(0);
        }
      }

      this.version += 1;
      return { loss: lossSum / n, acc: correct / n };
    }

    evaluate(data) {
      if (!data.length) return { loss: 0, acc: 0 };
      let loss = 0;
      let correct = 0;
      for (const sample of data) {
        let p = this.predict(sample.x);
        if (!isFinite(p)) p = 0.5;
        const q = Math.min(1 - 1e-7, Math.max(1e-7, p));
        loss += -(sample.y * Math.log(q) + (1 - sample.y) * Math.log(1 - q));
        if ((p >= 0.5 ? 1 : 0) === sample.y) correct += 1;
      }
      return { loss: loss / data.length, acc: correct / data.length };
    }
  }

  const API = { MLP, Optimizer, ACTIVATIONS, mulberry32, gaussian };
  if (typeof module !== 'undefined' && module.exports) module.exports = API;
  else global.Neural = API;
})(typeof globalThis !== 'undefined' ? globalThis : this);
