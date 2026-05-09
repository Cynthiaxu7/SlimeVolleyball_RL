// Port of slimevolleygym's BaselinePolicy (slimevolley.py, ~120-param tanh RNN).
// Apache 2.0, hardmaru. The exact forward pass is:
//
//   inputState[0:8]  = obs[0:8]                 // self & ball state only
//   inputState[8:15] = previous outputState      // ALL 7 outputs feed back, not just the 4
//   outputState      = tanh(W @ inputState + b)  // W is 7x15, b is 7
//   forward = outputState[0] > 0.75
//   backward = outputState[1] > 0.75
//   jump     = outputState[2] > 0.75
//
// The remaining 4 entries of outputState are the "recurrent state" — but the
// Python source actually feeds back ALL 7 outputs (see _setInputState), so the
// distinction is cosmetic; we mirror that exactly.
//
// Weights and biases are taken verbatim from BaselinePolicy.__init__ in
// slimevolleygym/slimevolley.py (a 105-element flat list reshaped to (7, 15),
// plus a 7-element bias).
//
// Globals exposed: BaselineAgent.

// Flat 105-element weight list (row-major) from BaselinePolicy. Reshaped to 7x15.
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
const N_OUTPUT = N_GAME_OUTPUT + N_RECURRENT_STATE;        // 7
const N_INPUT = N_GAME_INPUT + N_OUTPUT;                   // 15

// Reverse map [forward, backward, jump] -> Discrete(6) index, matching
// physics.js ACTION_TABLE. Any non-canonical triple falls back to NOOP (0).
// Index = forward<<2 | backward<<1 | jump (0..7); fill the 6 valid rows from
// ACTION_TABLE and leave [1,1,0]=6 and [1,1,1]=7 as 0 (NOOP).
const BASELINE_TRIPLE_TO_ACTION = (function () {
  const table = new Int32Array(8); // default 0 (NOOP) for unmapped triples
  for (let i = 0; i < ACTION_TABLE.length; i++) {
    const [f, b, j] = ACTION_TABLE[i];
    table[(f << 2) | (b << 1) | j] = i;
  }
  return table;
})();

class BaselineAgent {
  constructor(name = 'Baseline') {
    this.name = name;
    this.modelUrl = null;
    this.session = null;
    this.lastQ = new Float32Array(6);
    this.lastAction = 0;
    // Mirror ONNXAgent's lifecycle flags so game.js's existing "ready / unavailable / busy"
    // checks treat us identically.
    this.ready = true;
    this.unavailable = false;
    this.busy = false;

    // RNN state. outputState carries across frames; inputState is rebuilt each act().
    this.outputState = new Float32Array(N_OUTPUT);
    this.inputState  = new Float32Array(N_INPUT);
  }

  // No async load needed; resolve immediately so callers awaiting load() don't break.
  async load() { this.ready = true; }

  // obs: Float32Array length 12 (already /10 by Agent.getObservation in physics.js).
  // Baseline uses ONLY the first 8 components (self + ball, no opponent).
  // Returns: { action: int 0..5, qs: Float32Array length 6 } — qs is a one-hot
  //          on the chosen action so the existing Q-chart still has something to plot.
  async act(obs) {
    // 1) Build inputState: first 8 from obs, last 7 are previous outputState.
    for (let i = 0; i < N_GAME_INPUT; i++) this.inputState[i] = obs[i];
    for (let i = 0; i < N_OUTPUT; i++) this.inputState[N_GAME_INPUT + i] = this.outputState[i];

    // 2) Forward: outputState = tanh(W @ inputState + b). W is row-major 7x15.
    for (let r = 0; r < N_OUTPUT; r++) {
      let acc = BASELINE_BIAS[r];
      const rowBase = r * N_INPUT;
      for (let c = 0; c < N_INPUT; c++) {
        acc += BASELINE_WEIGHT_FLAT[rowBase + c] * this.inputState[c];
      }
      this.outputState[r] = Math.tanh(acc);
    }

    // 3) Threshold the first 3 outputs to get [forward, backward, jump].
    const f = this.outputState[0] > 0.75 ? 1 : 0;
    const b = this.outputState[1] > 0.75 ? 1 : 0;
    const j = this.outputState[2] > 0.75 ? 1 : 0;

    // 4) Map binary triple -> Discrete(6) via reversed ACTION_TABLE; non-canonical
    //    triples ([1,1,0] / [1,1,1]) fall back to NOOP.
    const action = BASELINE_TRIPLE_TO_ACTION[(f << 2) | (b << 1) | j];

    // 5) One-hot Q-vector for the chart (the policy doesn't actually output Q's).
    const qs = new Float32Array(6);
    qs[action] = 1.0;
    this.lastQ = qs;
    this.lastAction = action;
    return { action, qs };
  }

  // Clear hidden state. Called on slot (re)selection and on game restart.
  reset() {
    this.outputState.fill(0);
    this.inputState.fill(0);
    this.lastQ = new Float32Array(6);
    this.lastAction = 0;
  }
}
