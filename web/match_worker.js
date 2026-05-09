// Match worker: runs ONE headless slime volleyball match in an isolated WASM
// context, then terminates itself. Master spawns a fresh worker per match so
// onnxruntime-web's WASM heap (kernel caches, intermediate activations) can
// never accumulate across matches — it dies with the worker.
//
// Self-contained: re-implements the physics loop AND the BaselinePolicy inline
// rather than importScripts()'ing the page's web/*.js (those expect a window/
// document context and would re-export classes through globals the page already
// owns). Keep this file in sync with physics.js + baseline.js if either is
// edited.
//
// Wire protocol:
//   master -> worker (single message):
//     { leftVariantKey, rightVariantKey, leftModelUrl, rightModelUrl,
//       seed, maxSteps }
//   worker -> master (single message):
//     { leftLives, rightLives, steps }
//   The worker then self.close()'s. Master ALSO calls terminate() after
//   receiving the message so a misbehaving worker can't survive.

importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.25.1/dist/ort.min.js');

// --- ort wasm setup (must match the page's own setup so we hit the same CDN
// cache instead of the worker re-downloading the .wasm artifacts on every
// spawn).
ort.env.wasm.numThreads = 1;
ort.env.wasm.wasmPaths  = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.25.1/dist/';

// --- physics constants (verbatim from physics.js) ----------------------------
const REF_W = 24 * 2;
const REF_H = REF_W;
const REF_U = 1.5;
const REF_WALL_WIDTH = 1.0;
const REF_WALL_HEIGHT = 3.5;
const PLAYER_SPEED_X = 10 * 1.75;
const PLAYER_SPEED_Y = 10 * 1.35;
const MAX_BALL_SPEED = 15 * 1.5;
const TIMESTEP = 1 / 30;
const NUDGE = 0.1;
const FRICTION = 1.0;
const INIT_DELAY_FRAMES = 30;
const GRAVITY = -9.8 * 2 * 1.5;
const MAXLIVES = 5;
const MAX_EPISODE_STEPS = 3000;

const ACTION_TABLE = [
  [0, 0, 0], [1, 0, 0], [1, 0, 1],
  [0, 0, 1], [0, 1, 1], [0, 1, 0],
];

class Particle {
  constructor(x, y, vx, vy, r) {
    this.x = x; this.y = y;
    this.prev_x = x; this.prev_y = y;
    this.vx = vx; this.vy = vy;
    this.r = r;
  }
  move() {
    this.prev_x = this.x; this.prev_y = this.y;
    this.x += this.vx * TIMESTEP;
    this.y += this.vy * TIMESTEP;
  }
  applyAcceleration(ax, ay) {
    this.vx += ax * TIMESTEP;
    this.vy += ay * TIMESTEP;
  }
  checkEdges() {
    if (this.x <= this.r - REF_W / 2) {
      this.vx *= -FRICTION;
      this.x = this.r - REF_W / 2 + NUDGE * TIMESTEP;
    }
    if (this.x >= REF_W / 2 - this.r) {
      this.vx *= -FRICTION;
      this.x = REF_W / 2 - this.r - NUDGE * TIMESTEP;
    }
    if (this.y <= this.r + REF_U) {
      this.vy *= -FRICTION;
      this.y = this.r + REF_U + NUDGE * TIMESTEP;
      if (this.x <= 0) return -1;
      return 1;
    }
    if (this.y >= REF_H - this.r) {
      this.vy *= -FRICTION;
      this.y = REF_H - this.r - NUDGE * TIMESTEP;
    }
    if ((this.x <= REF_WALL_WIDTH / 2 + this.r) &&
        (this.prev_x > REF_WALL_WIDTH / 2 + this.r) &&
        (this.y <= REF_WALL_HEIGHT)) {
      this.vx *= -FRICTION;
      this.x = REF_WALL_WIDTH / 2 + this.r + NUDGE * TIMESTEP;
    }
    if ((this.x >= -REF_WALL_WIDTH / 2 - this.r) &&
        (this.prev_x < -REF_WALL_WIDTH / 2 - this.r) &&
        (this.y <= REF_WALL_HEIGHT)) {
      this.vx *= -FRICTION;
      this.x = -REF_WALL_WIDTH / 2 - this.r - NUDGE * TIMESTEP;
    }
    return 0;
  }
  getDist2(p) {
    const dx = p.x - this.x;
    const dy = p.y - this.y;
    return dx * dx + dy * dy;
  }
  isColliding(p) {
    const r = this.r + p.r;
    return r * r > this.getDist2(p);
  }
  bounce(p) {
    let abx = this.x - p.x;
    let aby = this.y - p.y;
    const abd = Math.sqrt(abx * abx + aby * aby);
    abx /= abd; aby /= abd;
    const nx = abx, ny = aby;
    abx *= NUDGE; aby *= NUDGE;
    while (this.isColliding(p)) {
      this.x += abx; this.y += aby;
    }
    let ux = this.vx - p.vx;
    let uy = this.vy - p.vy;
    const un = ux * nx + uy * ny;
    const unx = nx * (un * 2.0);
    const uny = ny * (un * 2.0);
    ux -= unx; uy -= uny;
    this.vx = ux + p.vx;
    this.vy = uy + p.vy;
  }
  limitSpeed(minSpeed, maxSpeed) {
    const mag2 = this.vx * this.vx + this.vy * this.vy;
    if (mag2 > maxSpeed * maxSpeed) {
      const mag = Math.sqrt(mag2);
      this.vx = (this.vx / mag) * maxSpeed;
      this.vy = (this.vy / mag) * maxSpeed;
    }
    if (mag2 < minSpeed * minSpeed && mag2 > 0) {
      const mag = Math.sqrt(mag2);
      this.vx = (this.vx / mag) * minSpeed;
      this.vy = (this.vy / mag) * minSpeed;
    }
  }
}

class Wall {
  constructor(x, y, w, h) { this.x = x; this.y = y; this.w = w; this.h = h; }
}

class Agent {
  constructor(dir, x, y) {
    this.dir = dir;
    this.x = x; this.y = y;
    this.r = 1.5;
    this.vx = 0; this.vy = 0;
    this.desired_vx = 0; this.desired_vy = 0;
    this.life = MAXLIVES;
    this.state = {
      x: 0, y: 0, vx: 0, vy: 0,
      bx: 0, by: 0, bvx: 0, bvy: 0,
      ox: 0, oy: 0, ovx: 0, ovy: 0,
    };
  }
  setAction(action) {
    const forward  = action[0] > 0;
    const backward = action[1] > 0;
    const jump     = action[2] > 0;
    this.desired_vx = 0; this.desired_vy = 0;
    if (forward && !backward)  this.desired_vx = -PLAYER_SPEED_X;
    if (backward && !forward)  this.desired_vx =  PLAYER_SPEED_X;
    if (jump) this.desired_vy = PLAYER_SPEED_Y;
  }
  move() {
    this.x += this.vx * TIMESTEP;
    this.y += this.vy * TIMESTEP;
  }
  update() {
    this.vy += GRAVITY * TIMESTEP;
    if (this.y <= REF_U + NUDGE * TIMESTEP) this.vy = this.desired_vy;
    this.vx = this.desired_vx * this.dir;
    this.move();
    if (this.y <= REF_U) { this.y = REF_U; this.vy = 0; }
    if (this.x * this.dir <= REF_WALL_WIDTH / 2 + this.r) {
      this.vx = 0;
      this.x = this.dir * (REF_WALL_WIDTH / 2 + this.r);
    }
    if (this.x * this.dir >= REF_W / 2 - this.r) {
      this.vx = 0;
      this.x = this.dir * (REF_W / 2 - this.r);
    }
  }
  updateState(ball, opponent) {
    this.state.x   = this.x  * this.dir;
    this.state.y   = this.y;
    this.state.vx  = this.vx * this.dir;
    this.state.vy  = this.vy;
    this.state.bx  = ball.x  * this.dir;
    this.state.by  = ball.y;
    this.state.bvx = ball.vx * this.dir;
    this.state.bvy = ball.vy;
    this.state.ox  = opponent.x  * (-this.dir);
    this.state.oy  = opponent.y;
    this.state.ovx = opponent.vx * (-this.dir);
    this.state.ovy = opponent.vy;
  }
  getObservation() {
    const s = this.state;
    const out = new Float32Array(12);
    out[0]  = s.x;   out[1]  = s.y;   out[2]  = s.vx;  out[3]  = s.vy;
    out[4]  = s.bx;  out[5]  = s.by;  out[6]  = s.bvx; out[7]  = s.bvy;
    out[8]  = s.ox;  out[9]  = s.oy;  out[10] = s.ovx; out[11] = s.ovy;
    for (let i = 0; i < 12; i++) out[i] /= 10.0;
    return out;
  }
}

class DelayScreen {
  constructor(life = INIT_DELAY_FRAMES) { this.life = life; }
  reset(life = INIT_DELAY_FRAMES) { this.life = life; }
  status() {
    if (this.life === 0) return true;
    this.life -= 1;
    return false;
  }
}

function newBall() {
  // Math.random() is per-worker, so each match gets fresh ball randomness.
  const vx = -20 + Math.random() * 40;
  const vy =  10 + Math.random() * 15;
  return new Particle(0, REF_W / 4, vx, vy, 0.5);
}

class World {
  constructor() { this.reset(); }
  reset() {
    this.ground    = new Wall(0, 0.75, REF_W, REF_U);
    this.fence     = new Wall(0, 0.75 + REF_WALL_HEIGHT / 2, REF_WALL_WIDTH, REF_WALL_HEIGHT - 1.5);
    this.fenceStub = new Particle(0, REF_WALL_HEIGHT, 0, 0, REF_WALL_WIDTH / 2);
    this.ball = newBall();
    this.agent_left  = new Agent(-1, -REF_W / 4, 1.5);
    this.agent_right = new Agent( 1,  REF_W / 4, 1.5);
    this.agent_left.updateState(this.ball, this.agent_right);
    this.agent_right.updateState(this.ball, this.agent_left);
    this.delayScreen = new DelayScreen();
    this.t = 0;
    this.done = false;
    this.lastResult = 0;
  }
  newMatch() {
    this.ball = newBall();
    this.delayScreen.reset();
  }
}

function stepWorld(world, leftAction, rightAction) {
  if (world.done) return 0;
  world.t += 1;
  world.agent_left.setAction(leftAction);
  world.agent_right.setAction(rightAction);
  world.agent_left.update();
  world.agent_right.update();
  if (world.delayScreen.status()) {
    world.ball.applyAcceleration(0, GRAVITY);
    world.ball.limitSpeed(0, MAX_BALL_SPEED);
    world.ball.move();
  }
  if (world.ball.isColliding(world.agent_left))  world.ball.bounce(world.agent_left);
  if (world.ball.isColliding(world.agent_right)) world.ball.bounce(world.agent_right);
  if (world.ball.isColliding(world.fenceStub))   world.ball.bounce(world.fenceStub);
  const result = world.ball.checkEdges();
  world.lastResult = result;
  if (result !== 0) {
    if (result < 0) world.agent_left.life -= 1;
    else            world.agent_right.life -= 1;
    world.newMatch();
  }
  world.agent_left.updateState(world.ball, world.agent_right);
  world.agent_right.updateState(world.ball, world.agent_left);
  if (world.t >= MAX_EPISODE_STEPS) world.done = true;
  if (world.agent_left.life <= 0 || world.agent_right.life <= 0) world.done = true;
  return result;
}

function getObs(world, side) {
  const isRight = (side === 'right') || (side === 1) || (side === +1);
  const a = isRight ? world.agent_right : world.agent_left;
  return a.getObservation();
}

// --- BaselinePolicy (verbatim from baseline.js) -----------------------------
const BASELINE_WEIGHT_FLAT = [
   7.5719,  4.4285,  2.2716, -0.3598, -7.8189, -2.5422, -3.2034,  0.3935,  1.2202, -0.4900, -0.0316,  0.5221,  0.7026,  0.4179, -2.1689,
   1.6460,-13.3639,  1.5151,  1.1175, -5.3561,  5.0442,  0.8451,  0.3987, -2.9501, -3.7811, -5.8994,  6.4167,  2.5014,  7.3380, -2.9887,
   2.4586, 13.4191,  2.7395, -3.9708,  1.6548, -2.7554, -1.5345, -6.4708,  9.2426, -0.7392,  0.4452,  1.8828, -2.6277,-10.8510, -3.2353,
  -4.4653, -3.1153, -1.3707,  7.3180, 16.0902,  1.4686,  7.0391,  1.7765, -1.1550,  2.6697, -8.8877,  1.1958, -3.2839, -5.4425,  1.6809,
   7.6812, -2.4732,  1.7380,  0.3781,  0.8718,  2.5886,  1.6911,  1.2953, -9.0052, -4.6038, -6.7447, -2.5528,  0.4391, -4.9278, -3.6695,
  -4.8673, -1.6035,  1.5011, -5.6124,  4.9747,  1.8998,  3.0359,  6.2983, -4.8568, -2.1888, -4.1143, -3.9874, -0.0459,  4.7134,  2.8952,
  -9.3627, -4.6850,  0.3601, -1.3699,  9.7294, 11.5596,  0.1918,  3.0783,  0.0329, -0.1362, -0.1188, -0.7579,  0.3278, -0.9770, -0.9377,
];
const BASELINE_BIAS = [ 2.2935, -2.0353, -1.7786, 5.4567, -3.6368, 3.4996, -0.0685 ];
const N_GAME_INPUT = 8;
const N_GAME_OUTPUT = 3;
const N_RECURRENT_STATE = 4;
const N_OUTPUT = N_GAME_OUTPUT + N_RECURRENT_STATE;
const N_INPUT = N_GAME_INPUT + N_OUTPUT;

const BASELINE_TRIPLE_TO_ACTION = (function () {
  const table = new Int32Array(8);
  for (let i = 0; i < ACTION_TABLE.length; i++) {
    const [f, b, j] = ACTION_TABLE[i];
    table[(f << 2) | (b << 1) | j] = i;
  }
  return table;
})();

class BaselineAgent {
  constructor() {
    this.outputState = new Float32Array(N_OUTPUT);
    this.inputState  = new Float32Array(N_INPUT);
  }
  // obs is already-normalized (divided by 10) Float32Array length 12.
  // BaselinePolicy uses ONLY first 8 components.
  async act(obs) {
    for (let i = 0; i < N_GAME_INPUT; i++) this.inputState[i] = obs[i];
    for (let i = 0; i < N_OUTPUT; i++) this.inputState[N_GAME_INPUT + i] = this.outputState[i];
    for (let r = 0; r < N_OUTPUT; r++) {
      let acc = BASELINE_BIAS[r];
      const rowBase = r * N_INPUT;
      for (let c = 0; c < N_INPUT; c++) {
        acc += BASELINE_WEIGHT_FLAT[rowBase + c] * this.inputState[c];
      }
      this.outputState[r] = Math.tanh(acc);
    }
    const f = this.outputState[0] > 0.75 ? 1 : 0;
    const b = this.outputState[1] > 0.75 ? 1 : 0;
    const j = this.outputState[2] > 0.75 ? 1 : 0;
    const action = BASELINE_TRIPLE_TO_ACTION[(f << 2) | (b << 1) | j];
    return { action };
  }
}

// --- ONNX agent (minimal — no caching, lifetime is one match) ----------------
class WorkerONNXAgent {
  constructor(modelUrl) {
    this.modelUrl = modelUrl;
    this.session = null;
    this.inputName = 'observation';
    this.outputName = 'q_values';
  }
  async load() {
    this.session = await ort.InferenceSession.create(this.modelUrl, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    const ins = this.session.inputNames;
    const outs = this.session.outputNames;
    if (ins.length > 0)  this.inputName  = ins[0];
    if (outs.length > 0) this.outputName = outs[0];
  }
  async act(obs) {
    const tensor = new ort.Tensor('float32', obs, [1, 12]);
    const feeds = {};
    feeds[this.inputName] = tensor;
    const out = await this.session.run(feeds);
    const q = out[this.outputName].data;
    let best = 0, bestVal = q[0];
    for (let i = 1; i < q.length; i++) {
      if (q[i] > bestVal) { bestVal = q[i]; best = i; }
    }
    return { action: best, qs: q };
  }
}

// Build the right policy for a side. variantKey 'baseline' -> BaselineAgent
// (no model fetch). Anything else -> ONNX session over the given URL.
async function buildAgent(variantKey, modelUrl) {
  if (variantKey === 'baseline' || !modelUrl) return new BaselineAgent();
  const a = new WorkerONNXAgent(modelUrl);
  await a.load();
  return a;
}

// One ONNX session may be shared across both sides if both pick the same
// variant — saves a session.create() and matches the master's behavior.
async function buildAgents(leftKey, rightKey, leftUrl, rightUrl) {
  if (leftKey === 'baseline' || rightKey === 'baseline' ||
      leftKey !== rightKey || !leftUrl || leftUrl !== rightUrl) {
    const [l, r] = await Promise.all([
      buildAgent(leftKey, leftUrl),
      buildAgent(rightKey, rightUrl),
    ]);
    return { left: l, right: r };
  }
  // Same ONNX url on both sides: one shared session, two callers — fine, the
  // session is stateless.
  const shared = await buildAgent(leftKey, leftUrl);
  return { left: shared, right: shared };
}

// --- match loop -------------------------------------------------------------
async function runMatch(msg) {
  const { leftVariantKey, rightVariantKey, leftModelUrl, rightModelUrl, maxSteps, forceNoop, traceBall } = msg;
  const { left, right } = await buildAgents(
    leftVariantKey, rightVariantKey, leftModelUrl, rightModelUrl,
  );

  const world = new World();
  const cap = (typeof maxSteps === 'number' && maxSteps > 0) ? maxSteps : (MAX_EPISODE_STEPS + 100);
  let stepCount = 0;
  const lActHist = [0,0,0,0,0,0];
  const rActHist = [0,0,0,0,0,0];
  let firstObsL = null, firstObsR = null, firstQL = null, firstQR = null;
  const ballTrace = traceBall ? [] : null;
  while (!world.done && stepCount < cap) {
    let aL, aR;
    if (forceNoop) { aL = 0; aR = 0; }
    else {
      const obsL = getObs(world, -1);
      const obsR = getObs(world, +1);
      const [resL, resR] = await Promise.all([left.act(obsL), right.act(obsR)]);
      aL = resL.action || 0;
      aR = resR.action || 0;
      if (stepCount === 50) {
        firstObsL = Array.from(obsL); firstObsR = Array.from(obsR);
        firstQL = resL.qs ? Array.from(resL.qs) : null;
        firstQR = resR.qs ? Array.from(resR.qs) : null;
      }
    }
    lActHist[aL]++; rActHist[aR]++;
    stepWorld(world, ACTION_TABLE[aL], ACTION_TABLE[aR]);
    if (ballTrace) {
      ballTrace.push({ step: stepCount, x: world.ball.x, y: world.ball.y, vx: world.ball.vx, vy: world.ball.vy });
    }
    stepCount++;
  }
  return {
    leftLives:  world.agent_left.life,
    rightLives: world.agent_right.life,
    steps: stepCount,
    debug: {
      lActHist, rActHist,
      firstObsL, firstObsR, firstQL, firstQR,
      leftVariant: leftVariantKey, rightVariant: rightVariantKey,
      ballTrace,
    },
  };
}

self.onmessage = async (e) => {
  try {
    const result = await runMatch(e.data || {});
    self.postMessage(result);
  } catch (err) {
    // postMessage an error-shaped result so the master can record / log it.
    self.postMessage({
      error: (err && err.message) ? err.message : String(err),
      leftLives: 0, rightLives: 0, steps: 0,
    });
  } finally {
    // Free the WASM session before tearing down the worker so any
    // session-internal cleanup runs (best-effort; close() will reap regardless).
    self.close();
  }
};
