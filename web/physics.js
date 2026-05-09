// Port of slimevolleygym/slimevolley.py physics. Apache 2.0, hardmaru.
// Numbers stay in the source's reference units; rendering scales to pixels separately.
// Globals exposed: REF_W, REF_H, REF_U, REF_WALL_WIDTH, REF_WALL_HEIGHT, ACTION_TABLE,
//                  Particle, Agent, Wall, DelayScreen, World, stepWorld, getObs, resetWorld.

const REF_W = 24 * 2;
const REF_H = REF_W;
const REF_U = 1.5;                  // ground thickness
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

// MultiBinary [forward, backward, jump]; indexed by argmax of 6 Q-values.
const ACTION_TABLE = [
  [0, 0, 0], // 0 NOOP
  [1, 0, 0], // 1 forward
  [1, 0, 1], // 2 forward+jump
  [0, 0, 1], // 3 jump
  [0, 1, 1], // 4 backward+jump
  [0, 1, 0], // 5 backward
];

class Particle {
  constructor(x, y, vx, vy, r) {
    this.x = x; this.y = y;
    this.prev_x = x; this.prev_y = y;
    this.vx = vx; this.vy = vy;
    this.r = r;
  }
  move() {
    this.prev_x = this.x;
    this.prev_y = this.y;
    this.x += this.vx * TIMESTEP;
    this.y += this.vy * TIMESTEP;
  }
  applyAcceleration(ax, ay) {
    this.vx += ax * TIMESTEP;
    this.vy += ay * TIMESTEP;
  }
  // Returns -1 if ball landed on left side, +1 right side, 0 otherwise.
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
    // Fence side collisions: only when previously on the other side and below fence top.
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
    abx /= abd;
    aby /= abd;
    const nx = abx, ny = aby;
    abx *= NUDGE;
    aby *= NUDGE;
    while (this.isColliding(p)) {
      this.x += abx;
      this.y += aby;
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
  constructor(x, y, w, h) {
    this.x = x; this.y = y;
    this.w = w; this.h = h;
  }
}

// Slime/agent. dir = -1 for left player, +1 for right player.
class Agent {
  constructor(dir, x, y) {
    this.dir = dir;
    this.x = x; this.y = y;
    this.r = 1.5;
    this.vx = 0; this.vy = 0;
    this.desired_vx = 0;
    this.desired_vy = 0;
    this.life = MAXLIVES;
    // mirrored observation slots
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
    this.desired_vx = 0;
    this.desired_vy = 0;
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
    // Stay in own half. Multiplying by dir folds both sides into the same comparison.
    if (this.x * this.dir <= REF_WALL_WIDTH / 2 + this.r) {
      this.vx = 0;
      this.x = this.dir * (REF_WALL_WIDTH / 2 + this.r);
    }
    if (this.x * this.dir >= REF_W / 2 - this.r) {
      this.vx = 0;
      this.x = this.dir * (REF_W / 2 - this.r);
    }
  }
  // Builds the relative obs so the agent always sees itself on the +x side
  // and the opponent on the -x side. Matches Python RelativeState exactly.
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
  // Returns Float32Array length 12, divided by scaleFactor=10.
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
  // Returns true once the delay expired; otherwise decrements and returns false.
  // Matches Python semantics: ball stays still while delay is non-zero.
  status() {
    if (this.life === 0) return true;
    this.life -= 1;
    return false;
  }
}

// World holds the full game state. Pure data + a few helpers; stepWorld() mutates.
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
    this.lastResult = 0; // -1 ball hit ground left, +1 right, 0 otherwise
  }
  newMatch() {
    this.ball = newBall();
    this.delayScreen.reset();
  }
}

function newBall() {
  // Python uses np_random.uniform; here we just use Math.random — ranges identical.
  const vx = -20 + Math.random() * 40;
  const vy =  10 + Math.random() * 15;
  return new Particle(0, REF_W / 4, vx, vy, 0.5);
}

// Steps one simulation tick. leftAction / rightAction are MultiBinary [f,b,j].
// Returns -1 if left player scored, +1 if right player scored, 0 otherwise.
function stepWorld(world, leftAction, rightAction) {
  if (world.done) return 0;
  world.t += 1;

  world.agent_left.setAction(leftAction);
  world.agent_right.setAction(rightAction);

  world.agent_left.update();
  world.agent_right.update();

  // Ball moves only after the start delay expires (status() returns true).
  if (world.delayScreen.status()) {
    world.ball.applyAcceleration(0, GRAVITY);
    world.ball.limitSpeed(0, MAX_BALL_SPEED);
    world.ball.move();
  }

  if (world.ball.isColliding(world.agent_left))  world.ball.bounce(world.agent_left);
  if (world.ball.isColliding(world.agent_right)) world.ball.bounce(world.agent_right);
  if (world.ball.isColliding(world.fenceStub))   world.ball.bounce(world.fenceStub);

  // Python returns reward as -checkEdges() so right-agent (the env's "self") wins => +1.
  // Here we keep the raw checkEdges sign: -1 = left side floor (right scores),
  //                                       +1 = right side floor (left scores).
  // The caller (game.js) handles life decrement using whichever convention it wants.
  const result = world.ball.checkEdges();
  world.lastResult = result;

  if (result !== 0) {
    if (result < 0) {
      // Ball hit left floor: right player scored.
      world.agent_left.life -= 1;
    } else {
      // Ball hit right floor: left player scored.
      world.agent_right.life -= 1;
    }
    world.newMatch();
  }

  world.agent_left.updateState(world.ball, world.agent_right);
  world.agent_right.updateState(world.ball, world.agent_left);

  if (world.t >= MAX_EPISODE_STEPS) world.done = true;
  if (world.agent_left.life <= 0 || world.agent_right.life <= 0) world.done = true;

  return result;
}

// Returns a Float32Array(12) obs from the perspective of one agent.
// 'side' may be 'right' / 'left' (string) or +1 / -1 (numeric Agent.dir).
// Mirroring is already baked into Agent.updateState, so feeding a left-side
// AI getObs(world, -1) gives it the same +x-self/-x-opponent canonical view
// the model was trained on.
function getObs(world, side) {
  const isRight = (side === 'right') || (side === 1) || (side === +1);
  const a = isRight ? world.agent_right : world.agent_left;
  return a.getObservation();
}

// Reset helper (kept so other files don't need to know about World internals).
function resetWorld(world) { world.reset(); }
