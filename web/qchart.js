// Rolling Q-value time-series chart. Vanilla canvas, no chart library.
// One QChart per canvas; the caller decides which agent to display via
// draw(currentAgentName). Internally we keep a per-agent ring buffer so
// switching the "Show Q for" radio is instantaneous.
// Globals: QChart.

class QChart {
  constructor(canvas, historySize = 200) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.historySize = historySize;
    // name -> { qs: Array<Float32Array(6)>, argmax: Array<int> }
    this.buffers = new Map();

    // Color per discrete action 0..5 (NOOP, FWD, FWD+J, JUMP, BCK+J, BACK).
    this.actionColors = [
      '#9aa6c2', // 0 NOOP
      '#3a7bd5', // 1 FWD
      '#7dd87d', // 2 FWD+J
      '#ffcf6b', // 3 JUMP
      '#d05050', // 4 BCK+J
      '#c884ff', // 5 BACK
    ];
    this.actionLabels = ['NOOP', 'FWD', 'FWD+J', 'JUMP', 'BCK+J', 'BACK'];
    // Display mode: 'q' shows raw Q-values; 'advantage' shows Q - mean(Q) per
    // frame (= the Dueling A-stream signal, with V baseline removed). Useful
    // because Dueling DQN's V dominates Q magnitude → all 6 lines overlap on
    // the raw-Q view; advantage mode amplifies the action-differentiating signal.
    this.mode = 'q';
  }

  setMode(mode) {
    if (mode === 'q' || mode === 'advantage') this.mode = mode;
  }

  ensureBuffer(name) {
    if (!this.buffers.has(name)) {
      this.buffers.set(name, { qs: [], argmax: [] });
    }
    return this.buffers.get(name);
  }

  // Append one frame for a given agent. qs is a Float32Array length 6.
  push(name, qs, argmax) {
    const buf = this.ensureBuffer(name);
    // Defensive copy so later session.run() invocations cannot mutate history.
    const copy = new Float32Array(qs.length);
    for (let i = 0; i < qs.length; i++) copy[i] = qs[i];
    buf.qs.push(copy);
    buf.argmax.push(argmax);
    if (buf.qs.length > this.historySize) {
      buf.qs.shift();
      buf.argmax.shift();
    }
  }

  clear(name) {
    if (name === undefined) {
      this.buffers.clear();
    } else if (this.buffers.has(name)) {
      this.buffers.set(name, { qs: [], argmax: [] });
    }
  }

  draw(currentAgentName) {
    const ctx = this.ctx;
    const W = this.canvas.width;
    const H = this.canvas.height;
    const padL = 36, padR = 110, padT = 8, padB = 18;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    ctx.fillStyle = '#0e1116';
    ctx.fillRect(0, 0, W, H);

    // Frame.
    ctx.strokeStyle = '#2a2f3a';
    ctx.lineWidth = 1;
    ctx.strokeRect(padL + 0.5, padT + 0.5, plotW, plotH);

    const buf = currentAgentName ? this.buffers.get(currentAgentName) : null;
    if (!buf || buf.qs.length === 0) {
      ctx.fillStyle = '#8a93a6';
      ctx.font = '12px ui-sans-serif, system-ui, sans-serif';
      ctx.fillText(currentAgentName ? `No Q data yet for ${currentAgentName}` : 'No agent selected',
                   padL + 8, padT + 16);
      this._drawLegend(padL + plotW + 12, padT, -1);
      return;
    }

    // Build the per-frame transformed values according to display mode.
    // In advantage mode each frame's values are centered (sub mean) so the
    // V-stream baseline is removed, exposing per-action differences.
    const N0 = buf.qs.length;
    const view = new Array(N0);
    for (let t = 0; t < N0; t++) {
      const q = buf.qs[t];
      if (this.mode === 'advantage') {
        let m = 0; for (let a = 0; a < 6; a++) m += q[a]; m /= 6;
        const adv = new Float32Array(6);
        for (let a = 0; a < 6; a++) adv[a] = q[a] - m;
        view[t] = adv;
      } else {
        view[t] = q;
      }
    }

    // Auto-scale Y to recent min/max across all 6 channels.
    let qmin = Infinity, qmax = -Infinity;
    for (let t = 0; t < view.length; t++) {
      const q = view[t];
      for (let a = 0; a < 6; a++) {
        if (q[a] < qmin) qmin = q[a];
        if (q[a] > qmax) qmax = q[a];
      }
    }
    if (!isFinite(qmin) || !isFinite(qmax)) { qmin = 0; qmax = 1; }
    if (qmax - qmin < 1e-3) { qmax = qmin + 1e-3; }
    const qSpan = qmax - qmin;
    // Visual breathing room.
    const padFrac = 0.08;
    qmin -= qSpan * padFrac;
    qmax += qSpan * padFrac;

    const N = buf.qs.length;
    // X is frame index inside the rolling window: oldest sample at the left edge,
    // newest at the right. We always render across the full plot width; if N <
    // historySize the curves grow from the right toward the left as data fills.
    const xStep = N > 1 ? plotW / (this.historySize - 1) : plotW;
    const xOffset = N < this.historySize ? plotW - (N - 1) * xStep : 0;

    const yFor = (v) => {
      const norm = (v - qmin) / (qmax - qmin);
      return padT + (1 - norm) * plotH;
    };

    // Y-axis tick labels at min, mid, max.
    ctx.fillStyle = '#8a93a6';
    ctx.font = '10px ui-monospace, monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const yTicks = [qmin, (qmin + qmax) / 2, qmax];
    for (const v of yTicks) {
      const y = yFor(v);
      ctx.fillText(v.toFixed(2), padL - 4, y);
      ctx.strokeStyle = '#1c2230';
      ctx.beginPath();
      ctx.moveTo(padL, y + 0.5);
      ctx.lineTo(padL + plotW, y + 0.5);
      ctx.stroke();
    }
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';

    // X-axis label (frame index relative to "now").
    ctx.fillStyle = '#8a93a6';
    ctx.font = '10px ui-monospace, monospace';
    ctx.fillText(`-${this.historySize - 1}`, padL, H - 4);
    ctx.fillText('0', padL + plotW - 8, H - 4);

    // Currently-argmax action (in the latest sample) is drawn last + thicker.
    const currentArg = buf.argmax[buf.argmax.length - 1];

    const drawLineFor = (a, isCurrent) => {
      ctx.strokeStyle = this.actionColors[a];
      ctx.lineWidth = isCurrent ? 2.5 : 1.0;
      ctx.globalAlpha = isCurrent ? 1.0 : 0.85;
      ctx.beginPath();
      for (let t = 0; t < N; t++) {
        const x = padL + xOffset + t * xStep;
        const y = yFor(view[t][a]);
        if (t === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }
      ctx.stroke();
    };

    // Draw non-current first (background), then current on top.
    for (let a = 0; a < 6; a++) {
      if (a === currentArg) continue;
      drawLineFor(a, false);
    }
    drawLineFor(currentArg, true);
    ctx.globalAlpha = 1.0;

    this._drawLegend(padL + plotW + 12, padT, currentArg);

    // Title strip.
    ctx.fillStyle = '#e6e8ef';
    ctx.font = '11px ui-sans-serif, system-ui, sans-serif';
    const titleLabel = this.mode === 'advantage' ? 'Advantage (Q - mean Q)' : 'Q-values';
    ctx.fillText(`${titleLabel} - ${currentAgentName}`, padL + 4, padT + 12);
  }

  _drawLegend(x, y, currentArg) {
    const ctx = this.ctx;
    ctx.font = '11px ui-monospace, monospace';
    for (let a = 0; a < 6; a++) {
      const ly = y + 6 + a * 16;
      ctx.fillStyle = this.actionColors[a];
      ctx.fillRect(x, ly, 12, 3);
      ctx.fillStyle = (a === currentArg) ? '#ffffff' : '#8a93a6';
      ctx.fillText(this.actionLabels[a], x + 18, ly + 6);
    }
  }
}
