// Replay-mode renderer. Loads a JSON dump produced by
// scripts/record_replays.py and renders frames directly onto the existing
// <canvas id="game"> WITHOUT re-running physics — the recorded ball/agent
// state IS the source of truth for visualization. This isolates whether the
// browser's fast-sim discrepancy is in JS physics (suspected) or elsewhere.
//
// Frame contract (see scripts/record_replays.py for the producer side):
//   meta:   { left_variant, right_variant, seed, outcome:{winner,steps,...} }
//   frames: [
//     { t, ball:[x,y,vx,vy], left:[x,y,vx,vy], right:[x,y,vx,vy],
//       lives:[L,R], actions:[aL,aR],
//       obs_left:[12], obs_right:[12], q_left:6|null, q_right:6|null }
//   ]
//
// Coordinates are in slimevolley world units (REF_W=48, REF_H=48 reference).
// We reuse the same toX/toY scaling and palette as web/game.js. To avoid
// duplicating long render code paths, we instantiate a one-off World, copy
// the recorded ball + agent positions into it on every frame, and then call
// game.js's existing draw helpers... except game.js doesn't expose them as
// module exports. So we re-implement the (short) drawing inline below;
// they're parameter-stable so visual parity is trivial.
//
// Globals: ReplayPlayer.

// eslint-disable-next-line no-unused-vars
class ReplayPlayer {
  constructor(canvas, qchart) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.qchart = qchart;
    this.frames = [];
    this.meta = null;
    this.frameIdx = 0;
    this.playing = false;
    this.speed = 1.0;          // multiplier on native 30 FPS playback
    this._lastTickMs = 0;
    this._frameAccumMs = 0;
    this._rafId = null;
    this.onFrameAdvance = null; // optional callback (idx, total) for UI updates
    this.currentAgentName = null; // last name pushed to qchart (for draw())

    // Cached drawing constants (mirror game.js).
    this.CANVAS_W = canvas.width;
    this.CANVAS_H = canvas.height;
    this.REF_W = 48; // matches physics.js REF_W (24*2)
    this.REF_H = 48;
    this.REF_U = 1.5;
    this.REF_WALL_WIDTH  = 1.0;
    this.REF_WALL_HEIGHT = 3.5;
    this.FACTOR = this.CANVAS_W / this.REF_W;
    this.AGENT_R = 1.5;
    this.BALL_R  = 0.5;
    this.FENCE_STUB_R = this.REF_WALL_WIDTH / 2;

    // Palette (kept in sync with game.js).
    this.COLOR_BG        = '#0e1116';
    this.COLOR_GROUND    = '#3a8a4a';
    this.COLOR_FENCE     = '#dcc05c';
    this.COLOR_BALL      = '#3a7bd5';
    this.COLOR_LEFT      = '#3a7bd5';
    this.COLOR_RIGHT     = '#d05050';
    this.COLOR_EYE_WHITE = '#ffffff';
    this.COLOR_EYE_PUPIL = '#000000';
    this.COLOR_LIFE      = '#dcc05c';
  }

  // --- world->screen helpers (identical to game.js) ----------------------
  toX(x) { return (x + this.REF_W / 2) * this.FACTOR; }
  toY(y) { return this.CANVAS_H - y * this.FACTOR; }
  toP(r) { return r * this.FACTOR; }

  async loadFromUrl(url) {
    const resp = await fetch(url, { cache: 'no-store' });
    if (!resp.ok) throw new Error(`replay fetch failed: HTTP ${resp.status}`);
    const data = await resp.json();
    if (!data || !Array.isArray(data.frames) || !data.meta) {
      throw new Error('replay JSON missing meta or frames');
    }
    this.pause();
    this.meta = data.meta;
    this.frames = data.frames;
    this.frameIdx = 0;
    this._frameAccumMs = 0;
    if (this.qchart && typeof this.qchart.clear === 'function') {
      this.qchart.clear();
    }
    // Render the first frame so the canvas isn't blank before play.
    this.renderCurrent();
    if (this.onFrameAdvance) this.onFrameAdvance(this.frameIdx, this.frames.length);
    return this.meta;
  }

  setSpeed(mul) {
    this.speed = Math.max(0.05, Number(mul) || 1.0);
  }

  // Move to a specific frame. Clears the qchart history and re-pushes
  // [0..idx] so the rolling chart matches what was visible at that moment.
  // For long replays this is O(N); the chart's history window naturally
  // bounds the visible work to historySize=200 entries.
  seek(idx) {
    if (!this.frames.length) return;
    idx = Math.max(0, Math.min(this.frames.length - 1, idx | 0));
    this.frameIdx = idx;
    if (this.qchart && typeof this.qchart.clear === 'function') {
      this.qchart.clear();
    }
    // Re-push Q history up to the seek target so chart isn't empty after
    // a scrub. We push for whichever side is currently selected by
    // getQSide() — the host wires this to the "Show Q for" radio.
    const side = this._currentQSide();
    const key = this._qKeyForSide(side);
    if (key) {
      for (let i = 0; i <= idx; i++) {
        const q = this._qForSide(this.frames[i], side);
        if (q) {
          const arr = new Float32Array(q);
          let argmax = 0, best = arr[0];
          for (let a = 1; a < arr.length; a++) {
            if (arr[a] > best) { best = arr[a]; argmax = a; }
          }
          this.qchart.push(key, arr, argmax);
        }
      }
      this.currentAgentName = key;
    }
    this.renderCurrent();
    if (this.onFrameAdvance) this.onFrameAdvance(this.frameIdx, this.frames.length);
  }

  play() {
    if (this.playing || !this.frames.length) return;
    if (this.frameIdx >= this.frames.length - 1) this.frameIdx = 0;
    this.playing = true;
    this._lastTickMs = performance.now();
    this._frameAccumMs = 0;
    const tick = () => {
      if (!this.playing) { this._rafId = null; return; }
      const now = performance.now();
      const dt = now - this._lastTickMs;
      this._lastTickMs = now;
      // 30 Hz native playback (matches slimevolley TIMESTEP=1/30s).
      this._frameAccumMs += dt * this.speed;
      const frameMs = 1000 / 30;
      let advanced = false;
      while (this._frameAccumMs >= frameMs) {
        this._frameAccumMs -= frameMs;
        if (this.frameIdx < this.frames.length - 1) {
          this.frameIdx++;
          this._pushQForCurrent();
          advanced = true;
        } else {
          this.playing = false;
          break;
        }
      }
      if (advanced) {
        this.renderCurrent();
        if (this.onFrameAdvance) this.onFrameAdvance(this.frameIdx, this.frames.length);
      }
      if (this.playing) this._rafId = requestAnimationFrame(tick);
      else this._rafId = null;
    };
    this._rafId = requestAnimationFrame(tick);
  }

  pause() {
    this.playing = false;
    if (this._rafId !== null) {
      cancelAnimationFrame(this._rafId);
      this._rafId = null;
    }
  }

  restart() {
    this.pause();
    this.frameIdx = 0;
    this._frameAccumMs = 0;
    if (this.qchart && typeof this.qchart.clear === 'function') {
      this.qchart.clear();
    }
    this.renderCurrent();
    if (this.onFrameAdvance) this.onFrameAdvance(this.frameIdx, this.frames.length);
  }

  // --- Q-chart wiring ----------------------------------------------------
  // Host calls setQSideGetter(fn) with a function returning 'left'|'right'
  // so the player can read the current radio without DOM coupling.
  setQSideGetter(fn) { this._getQSide = fn; }
  _currentQSide() { return (this._getQSide && this._getQSide()) || 'right'; }

  // Stable chart key per replay+side so QChart's per-name buffer is reused
  // across the playback and reset between replays. We embed the variant key
  // so left=baseline / right=v1_selfplay both have distinct buffers.
  _qKeyForSide(side) {
    if (!this.meta) return null;
    const variant = side === 'left' ? this.meta.left_variant : this.meta.right_variant;
    if (!variant || variant === 'baseline') return null; // baseline has no Q
    return `${side}:${variant} (replay)`;
  }

  _qForSide(frame, side) {
    return side === 'left' ? frame.q_left : frame.q_right;
  }

  _pushQForCurrent() {
    if (!this.qchart || typeof this.qchart.push !== 'function') return;
    const side = this._currentQSide();
    const key = this._qKeyForSide(side);
    if (!key) { this.currentAgentName = null; return; }
    const frame = this.frames[this.frameIdx];
    const q = this._qForSide(frame, side);
    if (!q) return; // some frames may not have Q (t=0 init frame)
    const arr = new Float32Array(q);
    let argmax = 0, best = arr[0];
    for (let a = 1; a < arr.length; a++) {
      if (arr[a] > best) { best = arr[a]; argmax = a; }
    }
    this.qchart.push(key, arr, argmax);
    this.currentAgentName = key;
  }

  // --- rendering ---------------------------------------------------------
  renderCurrent() {
    if (!this.frames.length) {
      this._drawBackground();
      return;
    }
    const frame = this.frames[this.frameIdx];
    this._drawBackground();
    this._drawGround();
    this._drawFence();
    this._drawSlime(frame.left,  -1, this.COLOR_LEFT,  frame.ball);
    this._drawSlime(frame.right, +1, this.COLOR_RIGHT, frame.ball);
    this._drawBall(frame.ball);
    this._drawLives(frame.lives);
    this._drawHud(frame);
  }

  _drawBackground() {
    const ctx = this.ctx;
    ctx.fillStyle = this.COLOR_BG;
    ctx.fillRect(0, 0, this.CANVAS_W, this.CANVAS_H);
  }

  _drawGround() {
    // Ground is a thin horizontal band at y=0.75±REF_U/2 in world coords —
    // matches World.reset() in physics.js.
    const ctx = this.ctx;
    const gy = 0.75; // center y
    const gh = this.REF_U;
    const gw = this.REF_W;
    const x = this.toX(-gw / 2);
    const w = this.toP(gw);
    const yTop = this.toY(gy + gh / 2);
    const h = this.toP(gh);
    ctx.fillStyle = this.COLOR_GROUND;
    ctx.fillRect(x, yTop, w, h);
  }

  _drawFence() {
    // Fence centered on x=0, full height REF_WALL_HEIGHT-1.5 (per
    // physics.js World.reset). Fence stub on top.
    const ctx = this.ctx;
    const fy = 0.75 + this.REF_WALL_HEIGHT / 2;
    const fh = this.REF_WALL_HEIGHT - 1.5;
    const fw = this.REF_WALL_WIDTH;
    const x = this.toX(-fw / 2);
    const w = this.toP(fw);
    const yTop = this.toY(fy + fh / 2);
    const h = this.toP(fh);
    ctx.fillStyle = this.COLOR_FENCE;
    ctx.fillRect(x, yTop, w, h);
    // Stub.
    ctx.beginPath();
    ctx.arc(this.toX(0), this.toY(this.REF_WALL_HEIGHT),
            this.toP(this.FENCE_STUB_R), 0, Math.PI * 2);
    ctx.fill();
  }

  _drawBall(ball) {
    const ctx = this.ctx;
    ctx.fillStyle = this.COLOR_BALL;
    ctx.beginPath();
    ctx.arc(this.toX(ball[0]), this.toY(ball[1]),
            this.toP(this.BALL_R), 0, Math.PI * 2);
    ctx.fill();
  }

  _drawSlime(agent, dir, color, ball) {
    // agent = [x, y, vx, vy]; we only need x,y. dir tells eye orientation.
    const ctx = this.ctx;
    const ax = agent[0];
    const ay = agent[1];
    const cx = this.toX(ax);
    const cy = this.toY(ay);
    const r  = this.toP(this.AGENT_R);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(cx, cy, r, Math.PI, 0, false);
    ctx.closePath();
    ctx.fill();

    // Eye orientation toward fence (matches game.js drawSlime).
    const angle = dir === 1 ? Math.PI * 120 / 180 : Math.PI * 60 / 180;
    const eyeBaseX = ax + 0.6 * this.AGENT_R * Math.cos(angle);
    const eyeBaseY = ay + 0.6 * this.AGENT_R * Math.sin(angle);
    const eyePxX = this.toX(eyeBaseX);
    const eyePxY = this.toY(eyeBaseY);
    const eyeR  = r * 0.3;
    ctx.fillStyle = this.COLOR_EYE_WHITE;
    ctx.beginPath();
    ctx.arc(eyePxX, eyePxY, eyeR, 0, Math.PI * 2);
    ctx.fill();

    // Pupil tracks ball.
    let dx = ball[0] - eyeBaseX;
    let dy = ball[1] - eyeBaseY;
    const d = Math.hypot(dx, dy) || 1;
    dx /= d; dy /= d;
    const pupilOffset = 0.15 * this.AGENT_R;
    ctx.fillStyle = this.COLOR_EYE_PUPIL;
    ctx.beginPath();
    ctx.arc(this.toX(eyeBaseX + dx * pupilOffset),
            this.toY(eyeBaseY + dy * pupilOffset),
            r * 0.1, 0, Math.PI * 2);
    ctx.fill();
  }

  _drawLives(lives) {
    // Match game.js: render (life - 1) coins along the top of each side.
    const ctx = this.ctx;
    const [lLife, rLife] = lives;
    ctx.fillStyle = this.COLOR_LIFE;
    for (let i = 1; i < lLife; i++) {
      const wx = -this.REF_W / 2 + i * 2;
      ctx.beginPath();
      ctx.arc(this.toX(wx), this.toY(this.REF_H - 2),
              this.toP(0.5), 0, Math.PI * 2);
      ctx.fill();
    }
    for (let i = 1; i < rLife; i++) {
      const wx = this.REF_W / 2 - i * 2;
      ctx.beginPath();
      ctx.arc(this.toX(wx), this.toY(this.REF_H - 2),
              this.toP(0.5), 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Small overlay top-center: variants + frame counter so the user can see
  // exactly which replay is playing (the canvas otherwise has no labels).
  _drawHud(frame) {
    if (!this.meta) return;
    const ctx = this.ctx;
    const total = this.frames.length;
    const text = `${this.meta.left_variant} (L) vs ${this.meta.right_variant} (R)  -  frame ${this.frameIdx + 1}/${total}`;
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    ctx.font = '13px ui-monospace, monospace';
    const w = ctx.measureText(text).width;
    ctx.fillRect(this.CANVAS_W / 2 - w / 2 - 8, 4, w + 16, 22);
    ctx.fillStyle = '#e6e8ef';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText(text, this.CANVAS_W / 2, 8);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
  }
}
