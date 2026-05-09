// Main game module: input, render, loop, scoring, restart, and multi-agent
// orchestration. Two slots (left, right). Each slot is either keyboard input
// or one of three ONNXAgent variants. The selectors in #mode-bar drive what
// each slot is at any moment.

(function () {
  const CANVAS_W = 1200;
  const CANVAS_H = 500;
  const FACTOR   = CANVAS_W / REF_W; // pixels per reference unit

  // World->screen helpers. y is flipped because canvas y grows downward.
  function toX(x) { return (x + REF_W / 2) * FACTOR; }
  function toY(y) { return CANVAS_H - y * FACTOR; }
  function toP(r) { return r * FACTOR; }

  // Columbia University home court palette.
  // Columbia Blue (#75AADB / #B9D9EB) is the school's signature color since
  // 1852; pairs with white and a deep navy backdrop for arena feel. Slimes
  // are blue (home) vs crimson (away) for instant left/right legibility.
  const COLOR_BG          = '#0a1929';   // arena navy backdrop
  const COLOR_GROUND      = '#75AADB';   // Columbia Blue court
  const COLOR_GROUND_LINE = '#ffffff';   // court boundary lines
  const COLOR_FENCE       = '#ffffff';   // white net
  const COLOR_BALL        = '#fafafa';   // volleyball body (drawn with seams)
  const COLOR_BALL_SEAM   = '#1B3A6B';   // navy seam lines
  const COLOR_LEFT        = '#3A7BD5';   // home: Columbia Blue (slightly deeper than court for contrast)
  const COLOR_LEFT_TRIM   = '#FFFFFF';   // home jersey trim: white chest stripe + dome rim
  const COLOR_RIGHT       = '#FDB515';   // away: California Gold (UC Berkeley Bears)
  const COLOR_RIGHT_TRIM  = '#003262';   // away jersey trim: Berkeley Blue
  const COLOR_BENCH       = '#3A4554';   // stadium bench (warm concrete gray)
  const COLOR_BENCH_EDGE  = 'rgba(255,255,255,0.10)';  // top-edge highlight
  const COLOR_EYE_WHITE   = '#ffffff';
  const COLOR_EYE_PUPIL   = '#000000';
  const COLOR_LIFE        = '#FFD11A';   // gold accent
  const COLOR_BRAND_TEXT  = 'rgba(255,255,255,0.18)';  // faded "COLUMBIA" wordmark
  const COLOR_BRAND_CROWN = 'rgba(255,255,255,0.22)';  // king's crown silhouette

  const canvas = document.getElementById('game');
  const ctx    = canvas.getContext('2d');
  const livesEl = document.getElementById('lives');
  const statusEl = document.getElementById('status');
  const fpsEl  = document.getElementById('fps');
  const bannerEl = document.getElementById('banner');
  const restartBtn = document.getElementById('restart');
  const loadingEl = document.getElementById('loading');
  const leftSelect = document.getElementById('left-select');
  const rightSelect = document.getElementById('right-select');
  const speedInput = document.getElementById('speed');
  const speedOut = document.getElementById('speed-out');
  const qchartCanvas = document.getElementById('qchart');

  // Ladder mode wiring (added to main #mode-bar and a dedicated #ladder-panel).
  const ladderPanel  = document.getElementById('ladder-panel');
  const ladderTbody  = document.querySelector('#ladder-table tbody');
  const ladderStatus = document.getElementById('ladder-status');
  const playNextBtn  = document.getElementById('play-next');
  const queueMeBtn   = document.getElementById('queue-me');
  const autoLoopBtn  = document.getElementById('auto-loop');
  const fastSim100Btn  = document.getElementById('fast-sim-100');
  const fastSim1000Btn = document.getElementById('fast-sim-1000');
  const fastSimStopBtn = document.getElementById('fast-sim-stop');
  const fastSimProgress = document.getElementById('fast-sim-progress');
  const fastSimPlaceholder = document.getElementById('fast-sim-placeholder');

  // Replay mode wiring (parallel to ladder; loads JSONs from web/replays/).
  const replayPanel    = document.getElementById('replay-panel');
  const replaySelect   = document.getElementById('replay-select');
  const replayPlayBtn  = document.getElementById('replay-play');
  const replayPauseBtn = document.getElementById('replay-pause');
  const replayRestartBtn = document.getElementById('replay-restart');
  const replayScrubber = document.getElementById('replay-scrubber');
  const replayFrameInfo = document.getElementById('replay-frame-info');
  const replayOutcome  = document.getElementById('replay-outcome');

  const world = new World();
  const qchart = new QChart(qchartCanvas, 200);
  const ladder = new EloLadder();
  // Replay player (Replay mode draws frames from a recorded JSON onto the
  // same canvas; physics is bypassed entirely). Constructed eagerly so the
  // mode-switch handler can call into it without null-checks.
  const replayPlayer = new ReplayPlayer(canvas, qchart);
  replayPlayer.setQSideGetter(() => getQForSide());
  // Index of available replays (filename -> metadata). Loaded once on first
  // entry into Replay mode; null until then.
  let replayIndex = null;
  let replayIndexLoading = false;

  // AI variant registry. modelUrl is fetched lazily by ONNXAgent.load().
  // 'baseline' is special: pure-JS BaselineAgent (no ONNX, no model file).
  const AI_VARIANTS = {
    plain:          { name: 'Plain DQN',       url: 'model_plain.onnx' },
    dueling:        { name: 'Dueling',         url: 'model.onnx' },
    duel_double:    { name: 'Duel+Double',     url: 'model_duel_double.onnx' },
    duel_nstep:     { name: 'Duel+Nstep',      url: 'model_duel_nstep.onnx' },
    v1:             { name: 'V1 cold',         url: 'model_v1.onnx' },
    v1_selfplay:    { name: 'V1 selfplay',     url: 'model_v1_selfplay.onnx' },
    v1_purepool:    { name: 'V1 sp purepool',  url: 'model_v1_purepool.onnx' },
    v1_selfplay_seeded: { name: 'V1 sp seeded', url: 'model_v1_selfplay_seeded.onnx' },
    v1_sp_fixed:    { name: 'V1 sp ⭐fixed',    url: 'model_v1_selfplay_fixed.onnx' },
    ppo:            { name: 'PPO',             url: 'model_ppo.onnx' },
    ppo_fixed:      { name: 'PPO ⭐fixed',      url: 'model_ppo_fixed.onnx' },
    ppo_rescue:     { name: 'PPO 🔥rescue',     url: 'model_ppo_rescue.onnx' },
    rainbow:        { name: 'Rainbow 20M sp',  url: 'model_rainbow.onnx' },
    rainbow_5m:     { name: 'Rainbow 5M',      url: 'model_rainbow_5m.onnx' },
    rainbow_5m_sp:  { name: 'Rainbow 5M sp',   url: 'model_rainbow_5m_sp.onnx' },
    rainbow_sp_fixed: { name: 'Rainbow sp ⭐fixed', url: 'model_rainbow_sp_fixed.onnx' },
    rainbow_full:   { name: 'Rainbow full (C51+Noisy)', url: 'model_rainbow_full.onnx' },
    rainbow_eps:    { name: 'Rainbow C51 (no Noisy)',   url: 'model_rainbow_eps.onnx' },
    baseline:       { name: 'Baseline',        url: null },
  };
  // value (dropdown key) -> ONNXAgent instance (constructed on first use).
  // Note: BaselineAgent is NOT cached here — each slot gets its own instance so
  // the recurrent hidden state is per-slot (left and right baselines are independent).
  const agents = {};

  // Slot state. Each slot has:
  //   kind: 'human' | variant key
  //   agent: ONNXAgent | null   (may be shared across slots when both sides use the
  //                              same variant -- per-slot busy/lastQ avoid races)
  //   pendingAction: int        (last AI argmax; reused while inference in flight)
  //   busy: bool                (per-slot inference-in-flight guard)
  //   lastQ: Float32Array(6)    (per-slot snapshot for the debug overlay)
  //   lastAction: int           (per-slot mirror of pendingAction at last act())
  function makeSlot(initialKind) {
    return {
      kind: initialKind, agent: null, pendingAction: 0,
      busy: false, lastQ: new Float32Array(6), lastAction: 0,
    };
  }
  const slots = {
    left:  makeSlot('human'),
    right: makeSlot('dueling'),
  };

  const keys = new Set();
  let showDebug = false;
  let lastFrameTs = performance.now();
  let fpsAvg = 0;
  let bannerUntil = 0;
  let gameOver = false;

  // Ladder mode state.
  //   mode:           'free' (current single-match mode) or 'ladder'.
  //   currentMatch:   { leftId, rightId } during a ladder match (else null).
  //   autoLoop:       true while the auto-loop is scheduling next matches.
  //   loopTimer:      pending setTimeout id for the inter-match delay (so we
  //                   can cancel it cleanly when the user toggles off / mode
  //                   switches).
  //   gameOverHandled:guard so a single game-over only triggers one
  //                   recordResult/auto-schedule pass.
  let mode = 'free';
  let currentMatch = null;
  let autoLoop = false;
  let loopTimer = null;
  let gameOverHandled = false;

  // Background fast-sim state. running flag is checked inside the headless loop
  // so Stop / mode switch cancels mid-batch. completed/target drive the progress
  // span. reloads is incremented every time the auto-reload step fires (shown
  // in the progress text so the user can see why a long run pauses briefly).
  // ladderAgentCache memoizes ONNX session instances per variantKey so a
  // 1000-match run doesn't re-instantiate sessions; baseline gets fresh
  // instances per match because it's stateful (RNN hidden state).
  let fastSim = { running: false, completed: 0, target: 0, reloads: 0 };
  const ladderAgentCache = {};

  // Mid-run auto-reload cadence. WASM heap pressure / session-internal tensor
  // accumulation can crash agents after a few hundred matches; periodically
  // tearing down and re-creating sessions is the primary defense (see agent.js
  // comment about v1.20.1 having no public Tensor.dispose()). 100 matches is
  // small enough to keep memory bounded for 1000-match runs and large enough
  // that the ~few-hundred-ms reload cost is negligible per match.
  const RELOAD_EVERY_MATCHES = 100;

  // Physics is fixed at 30Hz to match slimevolleygym's TIMESTEP=1/30s. Render
  // continues at RAF rate (smooth visuals on 60/120Hz displays); the accumulator
  // pattern decouples the two so game speed is wall-clock invariant.
  // Speed multiplier semantics: real elapsed ms is multiplied by speedMul before
  // draining into the accumulator. So 1x = native 30Hz, 2x = 60Hz, 0.5x = 15Hz.
  // Per-frame step count is capped at MAX_STEPS_PER_FRAME so a long stall (or a
  // huge speedMul on a slow GPU) does not produce a one-shot burst that locks
  // the UI.
  const PHYSICS_DT_MS = 1000 / 30;
  const MAX_STEPS_PER_FRAME = 8;
  let physicsAccum = 0;
  let speedMul = 1.0;

  // --- input -------------------------------------------------------------
  window.addEventListener('keydown', (e) => {
    const k = e.key.toLowerCase();
    keys.add(k);
    if (k === 'q')      showDebug = !showDebug;
    if (k === 'r')      doRestart();
    if (k === ' ')      e.preventDefault(); // stop page-scroll
    if (k === 'arrowup' || k === 'arrowdown' || k === 'arrowleft' || k === 'arrowright') {
      e.preventDefault();
    }
  });
  window.addEventListener('keyup', (e) => {
    keys.delete(e.key.toLowerCase());
  });

  // Left / P1 controls: A/D move, W or Space jump (also arrow keys).
  function getLeftHumanAction() {
    // Left agent dir=-1. Forward (toward fence) = world +x = D / right key.
    // Backward (away from fence) = world -x = A / left key.
    const left  = keys.has('a') || keys.has('arrowleft');
    const right = keys.has('d') || keys.has('arrowright');
    const jump  = keys.has('w') || keys.has('arrowup') || keys.has(' ');
    let forward = 0, backward = 0;
    if (right && !left) forward = 1;
    if (left && !right) backward = 1;
    return [forward, backward, jump ? 1 : 0];
  }

  // Right / P2 controls (used when right slot is "Human"): J/L move, I jump.
  function getRightHumanAction() {
    // Right agent dir=+1. Forward (toward fence) = world -x = J / numpad4.
    // Backward (away from fence) = world +x = L / numpad6.
    const lkey = keys.has('j') || keys.has('4');
    const rkey = keys.has('l') || keys.has('6');
    const jump = keys.has('i') || keys.has('8') || keys.has('k');
    let forward = 0, backward = 0;
    if (lkey && !rkey) forward = 1;
    if (rkey && !lkey) backward = 1;
    return [forward, backward, jump ? 1 : 0];
  }

  // --- selectors / lazy AI loading --------------------------------------
  function disableOption(selectEl, value, suffix = '(unavailable)') {
    for (const opt of selectEl.options) {
      if (opt.value === value) {
        if (!opt.textContent.endsWith(suffix)) {
          opt.textContent = `${opt.textContent} ${suffix}`;
        }
        opt.disabled = true;
      }
    }
  }

  function getOrCreateAgent(variantKey) {
    // Baseline: always a fresh per-slot instance (recurrent hidden state must
    // not be shared across left/right slots).
    if (variantKey === 'baseline') return new BaselineAgent('Baseline');
    if (agents[variantKey]) return agents[variantKey];
    const meta = AI_VARIANTS[variantKey];
    if (!meta) return null;
    agents[variantKey] = new ONNXAgent(meta.name, meta.url);
    return agents[variantKey];
  }

  async function ensureLoaded(variantKey) {
    const agent = getOrCreateAgent(variantKey);
    if (!agent) return null;
    if (agent.ready) return agent;
    if (agent.unavailable) {
      ladder.markVariantUnavailable(variantKey);
      renderLadderTable();
      return null;
    }
    try {
      await agent.load();
      return agent.ready ? agent : null;
    } catch {
      // Mark unavailable in BOTH dropdowns so the user cannot reselect.
      disableOption(leftSelect, variantKey);
      disableOption(rightSelect, variantKey);
      ladder.markVariantUnavailable(variantKey);
      renderLadderTable();
      return null;
    }
  }

  async function applySlotSelection(side, variantKey) {
    const slot = slots[side];
    slot.kind = variantKey;
    slot.pendingAction = 0;
    slot.busy = false;
    slot.lastQ = new Float32Array(6);
    slot.lastAction = 0;

    if (variantKey === 'human') {
      slot.agent = null;
      updateLivesHud();
      return;
    }
    const agent = await ensureLoaded(variantKey);
    if (!agent) {
      // Drop back to "human" silently (the option is now disabled).
      const sel = (side === 'left') ? leftSelect : rightSelect;
      sel.value = 'human';
      slot.kind = 'human';
      slot.agent = null;
      updateLivesHud();
      return;
    }
    slot.agent = agent;
    // Stateful agents (e.g. BaselineAgent's RNN) need a fresh hidden state on
    // slot (re)selection. Defensive: ONNXAgent has no reset() and that's fine.
    if (typeof agent.reset === 'function') agent.reset();
    updateLivesHud();
  }

  leftSelect.addEventListener('change', (e) => {
    if (mode === 'ladder') return;          // dropdowns ignored in ladder mode
    applySlotSelection('left', e.target.value);
  });
  rightSelect.addEventListener('change', (e) => {
    if (mode === 'ladder') return;
    applySlotSelection('right', e.target.value);
  });

  // --- ladder mode -------------------------------------------------------
  // Ladder mode replaces the per-side dropdowns with EloLadder-driven
  // matchmaking. Slot configuration is set per-match via applyLadderMatch();
  // game.js's existing physics/render path is reused unchanged.

  // Map a ladder roster id -> { variantKey, isHuman } for slot assignment.
  function rosterEntry(id) { return ladder.byId.get(id); }

  // Apply a scheduled ladder match: configure left/right slots from the two
  // roster ids and reset the world. Both sides may share an underlying
  // ONNXAgent instance (deterministic policy, see elo.js header) — the per-
  // slot busy/lastQ guards already isolate inference state.
  async function applyLadderMatch(leftId, rightId) {
    const lEntry = rosterEntry(leftId);
    const rEntry = rosterEntry(rightId);
    if (!lEntry || !rEntry) return false;
    currentMatch = { leftId, rightId };
    // Configure slots. Note: applySlotSelection awaits ensureLoaded so models
    // are fetched lazily on first use. If a variant 404s here, the ladder
    // marks it unavailable and we abort and try a different pair.
    await applySlotSelection('left',  lEntry.variantKey);
    await applySlotSelection('right', rEntry.variantKey);
    // If either side fell back to 'human' (model unavailable), bail and
    // schedule a fresh pair instead of forcing the user into an unintended
    // human game.
    if (slots.left.kind !== lEntry.variantKey && !lEntry.isHuman) return false;
    if (slots.right.kind !== rEntry.variantKey && !rEntry.isHuman) return false;
    doRestart();
    // Set status AFTER doRestart (which clears statusEl).
    statusEl.textContent = `Ladder: ${lEntry.name} vs ${rEntry.name}`;
    gameOverHandled = false;
    return true;
  }

  // Pick a pair from the ladder and launch the match. Returns true if a match
  // was actually scheduled (false on empty pool or repeated unavailable picks).
  async function startNextLadderMatch() {
    for (let attempt = 0; attempt < 5; attempt++) {
      const m = ladder.nextMatch();
      if (!m) {
        statusEl.textContent = 'Ladder: no eligible players';
        return false;
      }
      const ok = await applyLadderMatch(m.leftId, m.rightId);
      if (ok) return true;
    }
    statusEl.textContent = 'Ladder: could not schedule match';
    return false;
  }

  // Called from the physics step when game.done flips true. Computes ELO
  // outcome from final lives and (optionally) schedules the next match.
  function onLadderGameOver() {
    if (gameOverHandled || !currentMatch) return;
    gameOverHandled = true;
    const { leftId, rightId } = currentMatch;
    const lLife = world.agent_left.life;
    const rLife = world.agent_right.life;
    // ELO outcome: leftLife > rightLife => left wins (S=1), tie => 0.5,
    // leftLife < rightLife => left loses (S=0). Maps directly to slimevolley
    // semantics (right's training reward = leftLivesLost - rightLivesLost,
    // i.e. right wins iff rLife > lLife) but stated from the left side.
    ladder.recordResult(leftId, rightId, lLife, rLife);
    renderLadderTable();
    currentMatch = null;
    if (autoLoop && mode === 'ladder') {
      // Small delay so the user can read the score / banner.
      if (loopTimer !== null) clearTimeout(loopTimer);
      loopTimer = setTimeout(() => {
        loopTimer = null;
        if (autoLoop && mode === 'ladder') startNextLadderMatch();
      }, 800);
    }
  }

  // Render the ladder table sorted by descending ELO.
  function renderLadderTable() {
    if (!ladderTbody) return;
    const rows = ladder.getRoster();
    const html = rows.map((r, i) => {
      const cls = r.unavailable ? 'unavailable' : (r.isHuman ? 'human' : '');
      const tag = r.unavailable ? ' <span class="tag">(unavailable)</span>' : '';
      return `<tr class="${cls}">`
        + `<td>${i + 1}</td>`
        + `<td>${r.name}${tag}</td>`
        + `<td>${r.eloRating.toFixed(0)}</td>`
        + `<td>${r.gamesPlayed}</td>`
        + `<td>${r.wins}-${r.draws}-${r.losses}</td>`
        + `</tr>`;
    }).join('');
    ladderTbody.innerHTML = html;
  }

  function setMode(newMode) {
    if (newMode === mode) return;
    // Cancel any in-flight background sim before switching modes; the next loop
    // iteration will see fastSim.running=false and bail out, then the awaited
    // setBackgroundUiState(false) restores the canvas. We don't await it here
    // (setMode is sync and called from the radio change handler).
    if (fastSim.running) fastSim.running = false;
    // Pause replay playback when leaving Replay mode (otherwise its RAF would
    // fight with the main tick() drawing the live World).
    if (mode === 'replay') {
      replayPlayer.pause();
    }
    mode = newMode;
    // Clear any ladder timers / autoloop when leaving ladder mode.
    if (mode !== 'ladder') {
      autoLoop = false;
      if (loopTimer !== null) { clearTimeout(loopTimer); loopTimer = null; }
      autoLoopBtn.textContent = 'Auto-loop: off';
      currentMatch = null;
    }
    // Show/hide UI based on the new mode.
    // Free play  : canvas + qchart visible; ladder/replay panels hidden.
    // Ladder     : canvas + ladder panel visible; qchart hidden (Q lines are
    //              meaningless during 100-1000 match runs, kept hidden so the
    //              ladder table has more vertical space).
    // Replay     : canvas + replay panel + qchart all visible (qchart shows
    //              the recorded Q values for whichever side has them).
    if (mode === 'ladder') {
      ladderPanel.classList.remove('hidden');
      replayPanel.classList.add('hidden');
      qchartCanvas.classList.add('hidden');
      leftSelect.disabled = true;
      rightSelect.disabled = true;
      renderLadderTable();
      statusEl.textContent = 'Ladder ready - click "Play next match"';
    } else if (mode === 'replay') {
      ladderPanel.classList.add('hidden');
      replayPanel.classList.remove('hidden');
      qchartCanvas.classList.remove('hidden');
      // Free-play dropdowns make no sense in Replay (replay decides who's
      // who via meta.left_variant / meta.right_variant). Disable to convey
      // that, but leave them visible.
      leftSelect.disabled = true;
      rightSelect.disabled = true;
      // Banner gets reused for free-play "Right scores" messages, but in
      // Replay mode the meta + scrubber + canvas HUD already convey state,
      // so clear it on entry to avoid leftover text.
      bannerEl.textContent = '';
      statusEl.textContent = 'Replay ready - pick a recording';
      // Lazy-load index.json on first entry. Subsequent entries reuse the
      // cached list (it's small enough that we don't need eviction).
      ensureReplayIndex();
    } else {
      ladderPanel.classList.add('hidden');
      replayPanel.classList.add('hidden');
      qchartCanvas.classList.remove('hidden');
      leftSelect.disabled = false;
      rightSelect.disabled = false;
      statusEl.textContent = '';
    }
  }

  // Mode-toggle radio.
  document.querySelectorAll('input[name="mode"]').forEach((el) => {
    el.addEventListener('change', (e) => {
      if (e.target.checked) setMode(e.target.value);
    });
  });

  // Strategy radio.
  document.querySelectorAll('input[name="strategy"]').forEach((el) => {
    el.addEventListener('change', (e) => {
      if (e.target.checked) ladder.setStrategy(e.target.value);
    });
  });

  // Ladder buttons.
  playNextBtn.addEventListener('click', () => {
    if (mode !== 'ladder') return;
    if (loopTimer !== null) { clearTimeout(loopTimer); loopTimer = null; }
    startNextLadderMatch();
  });
  queueMeBtn.addEventListener('click', () => {
    if (mode !== 'ladder') return;
    ladder.queueHuman();
    statusEl.textContent = 'Ladder: human queued for next match';
    // If auto-loop is off and no match in flight, kick off the human match
    // immediately. If a match is in flight, the queue flag is consumed when
    // the next nextMatch() runs (inside auto-loop or Play-next).
    if (!currentMatch && !autoLoop) startNextLadderMatch();
  });
  autoLoopBtn.addEventListener('click', () => {
    if (mode !== 'ladder') return;
    autoLoop = !autoLoop;
    autoLoopBtn.textContent = autoLoop ? 'Auto-loop: on' : 'Auto-loop: off';
    if (autoLoop && !currentMatch) startNextLadderMatch();
    if (!autoLoop && loopTimer !== null) {
      clearTimeout(loopTimer); loopTimer = null;
    }
  });

  // --- background fast-sim ----------------------------------------------
  // Runs many AI-vs-AI matches with NO rendering and NO RAF. Each step is a
  // pair of (await agent.act(obs), stepWorld). ONNX inference is the
  // bottleneck (~1-5ms each), so we Promise.all() the L+R inferences to let
  // the WASM EP overlap them. Yielding via setTimeout(0) every ~10 matches
  // keeps the browser responsive (DOM events still fire while we await).

  // Per-match agent factory. ONNX agents are cached by variantKey across
  // matches (deterministic policy, stateless session.run). Baseline gets a
  // fresh instance per slot per match because it has recurrent state that
  // would leak between matches if shared.
  async function getOrCreateLadderAgent(playerId) {
    const entry = ladder.byId.get(playerId);
    if (!entry || entry.isHuman) return null;
    const variantKey = entry.variantKey;
    if (variantKey === 'baseline') return new BaselineAgent('Baseline');
    if (ladderAgentCache[variantKey]) return ladderAgentCache[variantKey];
    const agent = await ensureLoaded(variantKey);
    if (!agent) return null;
    ladderAgentCache[variantKey] = agent;
    return agent;
  }

  function ladderActAsync(agent, obs) {
    if (agent && typeof agent.act === 'function') {
      return agent.act(obs).catch(() => ({ action: 0, qs: null }));
    }
    return Promise.resolve({ action: 0, qs: null });
  }

  // Worker isolation: spin up a fresh Web Worker per match so onnxruntime-web's
  // WASM heap (kernel caches, intermediate activations) cannot accumulate
  // across matches — the worker dies between matches and takes its WASM heap
  // with it. The worker re-implements physics + baseline inline (see
  // match_worker.js) and only depends on ort.min.js from the CDN. Toggle to
  // false to A/B against the in-thread headless loop (kept as fallback below).
  const ENABLE_WORKER_ISOLATION = true;

  function runOneMatchInWorker(match) {
    return new Promise((resolve, reject) => {
      const worker = new Worker('match_worker.js');
      const timeout = setTimeout(() => {
        worker.terminate();
        reject(new Error('worker timeout'));
      }, 60_000);  // a single match should never take 60s
      worker.onmessage = (e) => {
        clearTimeout(timeout);
        worker.terminate();  // defensive: worker also self.close()'s
        const data = e.data || {};
        // Match the runOneMatchHeadless return shape.
        resolve({
          leftLives:  data.leftLives  | 0,
          rightLives: data.rightLives | 0,
          skipped:    false,
          steps:      data.steps | 0,
        });
      };
      worker.onerror = (e) => {
        clearTimeout(timeout);
        worker.terminate();
        reject(e);
      };
      const left  = ladder.byId.get(match.leftId);
      const right = ladder.byId.get(match.rightId);
      const variantUrl = (key) => (AI_VARIANTS[key] && AI_VARIANTS[key].url) || null;
      worker.postMessage({
        leftVariantKey:  left ? left.variantKey  : null,
        rightVariantKey: right ? right.variantKey : null,
        leftModelUrl:    left  ? variantUrl(left.variantKey)  : null,
        rightModelUrl:   right ? variantUrl(right.variantKey) : null,
        seed:    Math.floor(Math.random() * 1e9),
        maxSteps: 3100,
      });
    });
  }

  // Headless one-match runner. Returns final lives so the caller can hand
  // them to ladder.recordResult. Loop bound MAX_STEPS = MAX_EPISODE_STEPS +
  // safety margin so a misbehaving agent never wedges the run.
  // Kept as a fallback for ENABLE_WORKER_ISOLATION=false A/B testing; the
  // active fast-sim path now goes through runOneMatchInWorker.
  async function runOneMatchHeadless(leftId, rightId) {
    const leftAgent  = await getOrCreateLadderAgent(leftId);
    const rightAgent = await getOrCreateLadderAgent(rightId);
    // Both must exist for an AI-vs-AI headless run; the matchmaker only
    // returns AI pairs in this code path (Queue-me is consumed in foreground).
    if (!leftAgent || !rightAgent) {
      return { leftLives: 0, rightLives: 0, skipped: true };
    }
    // Stateful agents need fresh hidden state per match.
    if (typeof leftAgent.reset  === 'function') leftAgent.reset();
    if (typeof rightAgent.reset === 'function') rightAgent.reset();

    resetWorld(world);
    const MAX_STEPS = 3000 + 100;
    let stepCount = 0;
    while (!world.done && stepCount < MAX_STEPS && fastSim.running) {
      const obsL = getObs(world, -1);
      const obsR = getObs(world, +1);
      // Parallel inference: WASM EP can schedule both runs without one
      // blocking the other on the JS event loop.
      const [resL, resR] = await Promise.all([
        ladderActAsync(leftAgent,  obsL),
        ladderActAsync(rightAgent, obsR),
      ]);
      const aL = resL.action || 0;
      const aR = resR.action || 0;
      stepWorld(world, ACTION_TABLE[aL], ACTION_TABLE[aR]);
      stepCount++;
    }
    return {
      leftLives:  world.agent_left.life,
      rightLives: world.agent_right.life,
      skipped: false,
    };
  }

  // Show / hide canvas + placeholder, toggle ladder buttons and Stop button.
  function setBackgroundUiState(on) {
    if (on) {
      canvas.classList.add('hidden');
      qchartCanvas.classList.add('hidden');
      fastSimPlaceholder.classList.remove('hidden');
      fastSim100Btn.disabled  = true;
      fastSim1000Btn.disabled = true;
      fastSimStopBtn.classList.remove('hidden');
      playNextBtn.disabled = true;
      queueMeBtn.disabled  = true;
      autoLoopBtn.disabled = true;
    } else {
      canvas.classList.remove('hidden');
      // Keep qchart hidden in ladder mode (parity with setMode), but show it
      // back if we somehow exited fast-sim while in free play.
      if (mode !== 'ladder') qchartCanvas.classList.remove('hidden');
      fastSimPlaceholder.classList.add('hidden');
      fastSim100Btn.disabled  = false;
      fastSim1000Btn.disabled = false;
      fastSimStopBtn.classList.add('hidden');
      playNextBtn.disabled = false;
      queueMeBtn.disabled  = false;
      autoLoopBtn.disabled = false;
      fastSimProgress.textContent = '';
    }
  }

  // Optional `override` short-circuits the default "running X/Y (N reloads)"
  // text — used during the auto-reload pause to show "reloading models...".
  // The reload count is suppressed when zero so short runs (target < cadence)
  // don't show a pointless "(0 reloads)" suffix.
  function updateFastSimProgress(override) {
    if (override) {
      fastSimProgress.textContent = override;
      return;
    }
    const suffix = fastSim.reloads > 0 ? ` (${fastSim.reloads} reload${fastSim.reloads === 1 ? '' : 's'})` : '';
    fastSimProgress.textContent = `running ${fastSim.completed}/${fastSim.target}${suffix}`;
  }

  // Reusable reload step. Called by the manual "Reload models" button AND by
  // the mid-fast-sim auto-reload cadence. Steps:
  //   1. reload() every cached ONNXAgent (free WASM session, clear loadPromise)
  //   2. clear ladderAgentCache so getOrCreateLadderAgent constructs fresh
  //      ONNXAgent instances next time (the old refs in ladderAgentCache point
  //      at agents whose session=null after reload; safer to start fresh)
  //   3. re-enable any dropdown options that prior failures had marked
  //      "(unavailable)" — see disableOption()
  //   4. clear the ladder roster's per-variant unavailable flags
  //   5. (optional) pre-warm: walk the ladder roster, compute the unique set
  //      of variant keys actually in play, and ensureLoaded() each so the
  //      first match after reload doesn't pay the .onnx fetch cost on its
  //      hot path. prewarm=false skips this for the manual button (which has
  //      no immediate next match).
  async function reloadAllAgents({ prewarm = false } = {}) {
    // 1. release sessions in both caches.
    const reloadPromises = [];
    for (const k of Object.keys(agents)) {
      const a = agents[k];
      if (a && typeof a.reload === 'function') {
        const p = a.reload();
        if (p && typeof p.then === 'function') reloadPromises.push(p.catch(() => {}));
      }
    }
    for (const k of Object.keys(ladderAgentCache)) {
      const a = ladderAgentCache[k];
      if (a && typeof a.reload === 'function') {
        const p = a.reload();
        if (p && typeof p.then === 'function') reloadPromises.push(p.catch(() => {}));
      }
      // 2. always clear the cache key — even if the same agent ref is also in
      //    `agents`, the next getOrCreateLadderAgent() will resolve through
      //    ensureLoaded(variantKey) which returns the (now-reset) shared agent.
      delete ladderAgentCache[k];
    }
    await Promise.all(reloadPromises);

    // 3. re-enable dropdown options that ensureLoaded() may have disabled.
    for (const sel of [leftSelect, rightSelect]) {
      for (const opt of sel.options) {
        opt.disabled = false;
        opt.textContent = opt.textContent.replace(/\s*\(unavailable\)\s*$/, '');
      }
    }

    // 4. reset per-variant unavailable flags on the roster.
    if (typeof ladder.markVariantAvailable === 'function') {
      for (const v of Object.keys(AI_VARIANTS)) ladder.markVariantAvailable(v);
    } else {
      for (const r of ladder.roster) r.unavailable = false;
    }

    // 5. pre-warm only the variants that are actually in the current ladder
    //    roster (skip 'human' and 'baseline'; baseline has no .onnx file).
    //    Errors here are swallowed — ensureLoaded already marks failed
    //    variants unavailable internally and the next match will pick someone
    //    else. We do this serially: the WASM EP is single-threaded and
    //    parallel session creates can spike memory worse than the leak we're
    //    fighting.
    if (prewarm) {
      const variantSet = new Set();
      for (const r of ladder.roster) {
        if (r.isHuman) continue;
        if (r.variantKey === 'baseline') continue;
        variantSet.add(r.variantKey);
      }
      for (const v of variantSet) {
        if (!fastSim.running) break;  // user hit Stop mid-prewarm
        try { await ensureLoaded(v); } catch (e) { /* ignore */ }
      }
    }
  }

  async function runFastSim(targetMatches) {
    if (fastSim.running) return;
    // Pause normal ladder auto-loop / timers so they don't race the headless
    // path (which mutates `world` directly, just like the rendered path).
    if (loopTimer !== null) { clearTimeout(loopTimer); loopTimer = null; }
    autoLoop = false;
    autoLoopBtn.textContent = 'Auto-loop: off';
    currentMatch = null;
    gameOverHandled = true;   // suppress onLadderGameOver if a leftover RAF tick fires

    fastSim.running = true;
    fastSim.target = targetMatches;
    fastSim.completed = 0;
    fastSim.reloads = 0;
    setBackgroundUiState(true);
    updateFastSimProgress();

    while (fastSim.running && fastSim.completed < targetMatches) {
      // Auto-reload cadence: every RELOAD_EVERY_MATCHES *completed* matches,
      // tear down sessions and re-create them. Guarded on completed > 0 so
      // we don't reload at the start of the run, and on the modulo so a
      // target < RELOAD_EVERY_MATCHES (e.g. 50-match Run) never triggers.
      // The reload itself respects fastSim.running (prewarm step bails out
      // mid-loop if the user hits Stop), and after the reload we re-check
      // running before continuing — so Stop mid-reload cancels cleanly.
      if (fastSim.completed > 0 && fastSim.completed % RELOAD_EVERY_MATCHES === 0) {
        updateFastSimProgress('reloading models...');
        await reloadAllAgents({ prewarm: true });
        if (!fastSim.running) break;
        fastSim.reloads++;
        updateFastSimProgress();
        // Give the browser a beat to GC / paint the updated status before
        // we slam the WASM EP again with the next match's session.run().
        await new Promise((r) => setTimeout(r, 50));
        if (!fastSim.running) break;
      }

      const match = ladder.nextMatch();
      if (!match) break;
      let result;
      try {
        result = ENABLE_WORKER_ISOLATION
          ? await runOneMatchInWorker(match)
          : await runOneMatchHeadless(match.leftId, match.rightId);
      } catch (err) {
        console.warn('[fast-sim] match worker failed, skipping:', err && err.message ? err.message : err);
        result = { leftLives: 0, rightLives: 0, skipped: true };
      }
      // Only record if we actually ran the match (not aborted mid-loop and
      // not skipped due to unresolved agent).
      if (!result.skipped && fastSim.running) {
        ladder.recordResult(match.leftId, match.rightId, result.leftLives, result.rightLives);
      }
      fastSim.completed++;
      // Yield every 10 matches so the browser stays responsive (Stop button,
      // mode switch, etc.). renderLadderTable touches the DOM so we batch it
      // here too.
      if (fastSim.completed % 10 === 0) {
        renderLadderTable();
        updateFastSimProgress();
        await new Promise((r) => setTimeout(r, 0));
      }
    }
    renderLadderTable();
    fastSim.running = false;
    setBackgroundUiState(false);
    // After a fast-sim run we leave the world in a "done" state from the
    // last match. Reset it so a subsequent click on Play-next / Restart
    // starts cleanly.
    resetWorld(world);
    gameOver = false;
    gameOverHandled = false;
    updateLivesHud();
    const reloadNote = fastSim.reloads > 0
      ? ` (${fastSim.reloads} reload${fastSim.reloads === 1 ? '' : 's'})`
      : '';
    statusEl.textContent = `Ladder: completed ${fastSim.completed} background matches${reloadNote}`;
  }

  // --- replay mode -------------------------------------------------------
  // Replay mode reads pre-recorded match JSONs from web/replays/ and plays
  // them back on the canvas. Physics runs entirely in the Python recorder
  // (scripts/record_replays.py); this side just paints frames + pushes
  // recorded Q-values to the chart. The whole point is to verify whether
  // the Python ladder matches what the user sees in the web UI.

  // Populate the dropdown from web/replays/index.json.
  async function ensureReplayIndex() {
    if (replayIndex || replayIndexLoading) return;
    replayIndexLoading = true;
    try {
      const resp = await fetch('replays/index.json', { cache: 'no-store' });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      replayIndex = data;
      // Reset and repopulate.
      while (replaySelect.options.length > 1) replaySelect.remove(1);
      const list = Array.isArray(data && data.replays) ? data.replays : [];
      for (const r of list) {
        const opt = document.createElement('option');
        opt.value = r.file;
        opt.textContent = `${r.right} vs ${r.left} - ${r.outcome} - seed ${r.seed}`;
        replaySelect.appendChild(opt);
      }
      if (list.length === 0) {
        statusEl.textContent = 'Replay: index.json has no entries (run scripts/record_replays.py)';
      } else {
        statusEl.textContent = `Replay ready - ${list.length} recordings loaded`;
      }
    } catch (err) {
      console.warn('Replay: failed to load replays/index.json:', err);
      statusEl.textContent = 'Replay: could not load index.json (run scripts/record_replays.py?)';
    } finally {
      replayIndexLoading = false;
    }
  }

  function updateReplayUiForFrame(idx, total) {
    if (!total) {
      replayFrameInfo.textContent = 'frame 0 / 0';
      replayScrubber.max = '0';
      replayScrubber.value = '0';
      return;
    }
    replayScrubber.max = String(Math.max(0, total - 1));
    replayScrubber.value = String(idx);
    replayFrameInfo.textContent = `frame ${idx + 1} / ${total}`;
    // Refresh top HUD so it reflects the replay's current lives, not the
    // stale `world` state from whatever match was last active in free-play.
    if (replayPlayer.frames && replayPlayer.frames[idx] && replayPlayer.meta) {
      const lives = replayPlayer.frames[idx].lives || [0, 0];
      const lv = replayPlayer.meta.left_variant || 'left';
      const rv = replayPlayer.meta.right_variant || 'right';
      livesEl.textContent = `${lv}: ${lives[0]}  vs  ${rv}: ${lives[1]}`;
    }
  }
  replayPlayer.onFrameAdvance = updateReplayUiForFrame;

  replaySelect.addEventListener('change', async () => {
    const fname = replaySelect.value;
    if (!fname) return;
    try {
      const meta = await replayPlayer.loadFromUrl(`replays/${fname}`);
      // Sync the player's playback speed to whatever the global speed slider
      // currently shows (otherwise it would default to 1.0x even if the user
      // had moved the slider before picking a replay).
      replayPlayer.setSpeed(speedMul);
      const o = meta.outcome || {};
      replayOutcome.textContent =
        `${o.winner || '?'} wins (${o.left_lives ?? '?'}-${o.right_lives ?? '?'}, ${o.steps ?? '?'} steps)`;
      updateReplayUiForFrame(0, replayPlayer.frames.length);
      statusEl.textContent = `Replay loaded: ${fname}`;
    } catch (err) {
      console.warn('Replay load failed:', err);
      statusEl.textContent = `Replay load failed: ${err.message || err}`;
    }
  });

  replayPlayBtn.addEventListener('click', () => {
    if (mode !== 'replay') return;
    replayPlayer.play();
  });
  replayPauseBtn.addEventListener('click', () => {
    if (mode !== 'replay') return;
    replayPlayer.pause();
  });
  replayRestartBtn.addEventListener('click', () => {
    if (mode !== 'replay') return;
    replayPlayer.restart();
  });
  // Scrubber: read live during input (drag), pause and seek on each event.
  replayScrubber.addEventListener('input', () => {
    if (mode !== 'replay') return;
    replayPlayer.pause();
    const idx = parseInt(replayScrubber.value, 10) || 0;
    replayPlayer.seek(idx);
  });

  // Replay speed: re-use the existing global speed slider for parity with
  // free-play mode. The replay player respects setSpeed() independently.
  // We re-parse from speedInput.value here (rather than reading the cached
  // `speedMul`) because the existing free-play listener (declared further
  // down with addEventListener) assigns speedMul AFTER this one fires —
  // listener order is registration order. Reading the input directly gets
  // the freshly-typed value regardless.
  speedInput.addEventListener('input', () => {
    if (mode === 'replay') {
      const v = parseFloat(speedInput.value) || 1.0;
      replayPlayer.setSpeed(v);
    }
  });

  fastSim100Btn.addEventListener('click', () => {
    if (mode !== 'ladder' || fastSim.running) return;
    runFastSim(100);
  });
  fastSim1000Btn.addEventListener('click', () => {
    if (mode !== 'ladder' || fastSim.running) return;
    runFastSim(1000);
  });
  fastSimStopBtn.addEventListener('click', () => {
    fastSim.running = false;
  });

  // Reload-models button: recovery path for the case where repeated fast-sim
  // runs leak WASM heap or trip a cached load failure that left agents marked
  // unavailable. The actual reload work lives in reloadAllAgents() so the
  // mid-fast-sim auto-reload cadence can call the same code path; this
  // handler:
  //   1. cancels any in-flight fast-sim (so the loop stops calling
  //      reloadAllAgents itself),
  //   2. delegates the heavy lifting (release sessions, clear caches,
  //      re-enable dropdowns, clear roster unavailable flags) to
  //      reloadAllAgents({ prewarm: false }) — no need to pre-warm here, the
  //      user will pick the next match manually,
  //   3. updates the status text and re-renders the table.
  const reloadModelsBtn = document.getElementById('reload-models');
  if (reloadModelsBtn) {
    reloadModelsBtn.addEventListener('click', async () => {
      if (fastSim.running) fastSim.running = false;
      await reloadAllAgents({ prewarm: false });
      renderLadderTable();
      ladderStatus.textContent = 'Models reloaded - try a match.';
    });
  }

  // Hard refresh: page reload. Sometimes the WASM heap is so leaky that
  // session.release() can't recover — e.g., after thousands of fast-sim
  // matches. ELO state is persisted to localStorage so this only loses the
  // in-flight match queue, not progress.
  const hardRefreshBtn = document.createElement('button');
  hardRefreshBtn.id = 'hard-refresh';
  hardRefreshBtn.textContent = 'Hard refresh';
  hardRefreshBtn.title = 'Page reload (last-resort recovery; ELO progress is saved across reloads)';
  if (reloadModelsBtn && reloadModelsBtn.parentNode) {
    reloadModelsBtn.parentNode.insertBefore(hardRefreshBtn, reloadModelsBtn.nextSibling);
  }
  hardRefreshBtn.addEventListener('click', () => {
    if (fastSim.running) fastSim.running = false;
    // ELO already auto-persists on every recordResult; nothing else to save.
    window.location.reload();
  });

  // Reset ladder ratings (clears localStorage too).
  const resetEloBtn = document.createElement('button');
  resetEloBtn.id = 'reset-elo';
  resetEloBtn.textContent = 'Reset ELO';
  resetEloBtn.title = 'Wipe ELO history and start fresh at 1500 each';
  if (hardRefreshBtn.parentNode) {
    hardRefreshBtn.parentNode.insertBefore(resetEloBtn, hardRefreshBtn.nextSibling);
  }
  resetEloBtn.addEventListener('click', () => {
    if (fastSim.running) fastSim.running = false;
    if (typeof ladder.resetAllRatings === 'function') ladder.resetAllRatings();
    renderLadderTable();
    ladderStatus.textContent = 'ELO reset to 1500 for everyone.';
  });

  // Speed slider.
  speedInput.addEventListener('input', () => {
    speedMul = parseFloat(speedInput.value) || 1.0;
    speedOut.textContent = `${speedMul.toFixed(2)}x`;
    physicsAccum = 0; // avoid releasing a stale burst on speed change
  });

  // "Show Q for" radio.
  function getQForSide() {
    const checked = document.querySelector('input[name="qfor"]:checked');
    return checked ? checked.value : 'right';
  }
  // In Replay mode, switching this radio mid-replay should refresh the chart
  // with the other side's recorded Q-values. We re-seek to the current
  // frame: ReplayPlayer.seek() clears + re-pushes Q history for the new
  // side, then renders. Cheap because only the chart history is recomputed.
  document.querySelectorAll('input[name="qfor"]').forEach((el) => {
    el.addEventListener('change', () => {
      if (mode === 'replay' && replayPlayer.frames.length > 0) {
        replayPlayer.seek(replayPlayer.frameIdx);
      }
    });
  });

  // Chart mode (Q vs advantage = Q - mean Q). Dueling DQN's V dominates Q
  // magnitude so the 6 action lines overlap on the raw-Q view; advantage
  // mode amplifies the action-differentiating A-stream signal.
  document.querySelectorAll('input[name="qmode"]').forEach((el) => {
    el.addEventListener('change', (e) => {
      qchart.setMode(e.target.value);
    });
  });

  restartBtn.addEventListener('click', doRestart);

  function setBanner(text, ms = 1500) {
    bannerEl.textContent = text;
    bannerUntil = performance.now() + ms;
  }

  function slotLabel(slot) {
    if (slot.kind === 'human') return 'Human';
    return AI_VARIANTS[slot.kind] ? AI_VARIANTS[slot.kind].name : slot.kind;
  }

  function updateLivesHud() {
    const ll = slotLabel(slots.left);
    const rl = slotLabel(slots.right);
    livesEl.textContent = `${ll}: ${world.agent_left.life}  vs  ${rl}: ${world.agent_right.life}`;
  }

  function doRestart() {
    resetWorld(world);
    gameOver = false;
    bannerEl.textContent = '';
    statusEl.textContent = '';
    for (const s of [slots.left, slots.right]) {
      s.pendingAction = 0;
      s.lastAction = 0;
      s.lastQ = new Float32Array(6);
      // Leave s.busy alone: any in-flight inference is harmless on completion.
      // Stateful agents (Baseline RNN) must clear hidden state on a new game.
      if (s.agent && typeof s.agent.reset === 'function') s.agent.reset();
    }
    physicsAccum = 0;
    qchart.clear();
    updateLivesHud();
  }

  // --- rendering ---------------------------------------------------------
  function drawBackground() {
    ctx.fillStyle = COLOR_BG;
    ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);
  }
  function drawGround() {
    const g = world.ground;
    const x = toX(g.x - g.w / 2);
    const w = toP(g.w);
    const yTop = toY(g.y + g.h / 2);
    const h = toP(g.h);
    // Columbia-blue court with a thin white boundary line on top (the "service line").
    ctx.fillStyle = COLOR_GROUND;
    ctx.fillRect(x, yTop, w, h);
    ctx.fillStyle = COLOR_GROUND_LINE;
    ctx.fillRect(x, yTop, w, Math.max(2, FACTOR * 0.08));
  }

  // Pre-computed crowd: two banks of small slimes seated above the court
  // (the upper portion of the canvas, behind the wordmark). Left half is
  // home (Columbia Blue), right half is away (Cal Gold). Positions are
  // generated once at module load via a seeded PRNG so the layout is stable
  // across frames; per-slime phase is used by drawCrowd for the bobbing
  // cheer animation.
  const CROWD_SLIMES = (function () {
    let s = 1337;
    const rand = () => {
      s = (s * 1103515245 + 12345) & 0x7fffffff;
      return s / 0x7fffffff;
    };
    const slimes = [];
    const rows = 2;
    const slimeR = 11;
    const spacingX = 26;
    const spacingY = 22;
    const yBase = 18;
    for (let row = 0; row < rows; row++) {
      // Stagger every other row by half a column for stadium-bench feel.
      const offsetX = (row % 2) * (spacingX * 0.5);
      const y = yBase + row * spacingY;
      let col = 0;
      while (true) {
        const x = col * spacingX + offsetX + spacingX * 0.5;
        if (x > CANVAS_W - 6) break;
        const isHome = x < CANVAS_W / 2;
        slimes.push({
          x,
          y,
          r: slimeR + (rand() - 0.5) * 1.5,
          color: isHome ? COLOR_LEFT      : COLOR_RIGHT,
          trim:  isHome ? COLOR_LEFT_TRIM : COLOR_RIGHT_TRIM,
          phase: rand() * Math.PI * 2,
          // Frequency jitter so they don't all bob in lockstep.
          freq: 1.2 + rand() * 1.2,
        });
        col++;
      }
    }
    return slimes;
  })();

  function drawCrowd(tSec) {
    // Bleachers FIRST (behind the slimes). One horizontal bench per row of
    // crowd, spanning the full canvas width. Slime base y == row y, so the
    // bench sits flush under each row; when a slime bobs upward it briefly
    // lifts off the bench, which reads naturally as cheering.
    const rowYs = [];
    const seen = new Set();
    for (const sl of CROWD_SLIMES) {
      if (!seen.has(sl.y)) { seen.add(sl.y); rowYs.push(sl.y); }
    }
    rowYs.sort((a, b) => a - b);
    const benchH = 6;
    for (const ry of rowYs) {
      ctx.fillStyle = COLOR_BENCH;
      ctx.fillRect(0, ry, CANVAS_W, benchH);
      ctx.fillStyle = COLOR_BENCH_EDGE;
      ctx.fillRect(0, ry, CANVAS_W, 1);
    }

    // Ball position in screen pixels — used for the crowd's pupil tracking
    // (everyone in the stands follows the ball, like real volleyball fans).
    const ballPxX = toX(world.ball.x);
    const ballPxY = toY(world.ball.y);

    // Slimes ON TOP of benches.
    for (const sl of CROWD_SLIMES) {
      // Half-rectified sine: each slime "jumps" briefly then sits.
      const bob = Math.max(0, Math.sin(tSec * sl.freq + sl.phase)) * 4;
      const cy = sl.y - bob;
      // Body dome.
      ctx.fillStyle = sl.color;
      ctx.beginPath();
      ctx.arc(sl.x, cy, sl.r, Math.PI, 0, false);
      ctx.closePath();
      ctx.fill();
      // Thin rim outline only (chest stripe was unreadable at this scale).
      if (sl.trim) {
        ctx.strokeStyle = sl.trim;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.arc(sl.x, cy, sl.r, Math.PI, 0, false);
        ctx.stroke();
      }

      // Two big round eyes (sclera) — the cute fan look. Positioned in the
      // upper third of the dome with a comfortable horizontal gap.
      const eyeR = sl.r * 0.34;
      const pupilR = sl.r * 0.18;
      const eyeY = cy - sl.r * 0.46;
      const eyeXOffset = sl.r * 0.38;
      const eyeLeftX  = sl.x - eyeXOffset;
      const eyeRightX = sl.x + eyeXOffset;

      ctx.fillStyle = COLOR_EYE_WHITE;
      ctx.beginPath();
      ctx.arc(eyeLeftX,  eyeY, eyeR, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(eyeRightX, eyeY, eyeR, 0, Math.PI * 2);
      ctx.fill();

      // Pupils track the ball (limited to inside the eye white). Two pupils
      // get computed independently so eyes can wall-eye slightly when the
      // ball is close — adds personality.
      ctx.fillStyle = COLOR_EYE_PUPIL;
      const maxOffset = (eyeR - pupilR) * 0.9;
      for (const ex of [eyeLeftX, eyeRightX]) {
        let dx = ballPxX - ex;
        let dy = ballPxY - eyeY;
        const d = Math.hypot(dx, dy) || 1;
        dx /= d; dy /= d;
        ctx.beginPath();
        ctx.arc(ex + dx * maxOffset, eyeY + dy * maxOffset, pupilR, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    // Faint dividing line between home and away sections (vertical aisle).
    ctx.strokeStyle = 'rgba(255,255,255,0.08)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(CANVAS_W / 2, 8);
    ctx.lineTo(CANVAS_W / 2, rowYs[rowYs.length - 1] + benchH);
    ctx.stroke();
  }

  // Big faded Columbia wordmark + king's crown silhouette painted on the
  // court between drawGround and drawFence — sits "behind" the play but in
  // front of the surface, like sponsor branding on a real arena floor.
  function drawCourtBranding() {
    const cx = CANVAS_W / 2;
    const groundTopPx = toY(world.ground.y + world.ground.h / 2);
    // King's crown silhouette ABOVE the court surface (in the lower part of
    // the playable air) so slimes don't overdraw it. We center it horizontally
    // and place it ~40% of the way down the canvas. Drawn as a chunky sans
    // bold "♔" — that's the Unicode king-of-chess glyph; Columbia's actual
    // king's crown is similar in silhouette and renders consistently across
    // browsers without needing a custom SVG asset.
    ctx.save();
    ctx.fillStyle = COLOR_BRAND_CROWN;
    ctx.font = '700 110px ui-sans-serif, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const crownY = groundTopPx - 110;
    ctx.fillText('♔', cx, crownY);
    // Stylized "COLUMBIA" wordmark beneath the crown, on the court.
    ctx.fillStyle = COLOR_BRAND_TEXT;
    ctx.font = '700 38px "Times New Roman", Georgia, serif';
    ctx.fillText('COLUMBIA', cx, crownY + 80);
    ctx.font = '600 14px ui-sans-serif, system-ui, sans-serif';
    ctx.fillText('LIONS — ROAR LIONS ROAR', cx, crownY + 108);
    ctx.restore();
  }
  function drawFence() {
    const f = world.fence;
    const x = toX(f.x - f.w / 2);
    const w = toP(f.w);
    const yTop = toY(f.y + f.h / 2);
    const h = toP(f.h);
    ctx.fillStyle = COLOR_FENCE;
    ctx.fillRect(x, yTop, w, h);
    // stub on top
    const s = world.fenceStub;
    ctx.beginPath();
    ctx.arc(toX(s.x), toY(s.y), toP(s.r), 0, Math.PI * 2);
    ctx.fill();
  }
  function drawBall() {
    const b = world.ball;
    const cx = toX(b.x), cy = toY(b.y), r = toP(b.r);
    // Body: white with a soft radial gradient so it reads as a 3D sphere
    // rather than a flat disc. Light source sits at upper-left of the ball.
    const grad = ctx.createRadialGradient(
      cx - r * 0.35, cy - r * 0.35, r * 0.1,
      cx, cy, r,
    );
    grad.addColorStop(0.0, '#ffffff');
    grad.addColorStop(0.6, COLOR_BALL);
    grad.addColorStop(1.0, '#d0d4dc');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.fill();

    // Subtle dark outline for definition against a light court.
    ctx.lineWidth = Math.max(1, r * 0.07);
    ctx.strokeStyle = 'rgba(20, 35, 70, 0.55)';
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();

    // Three "panel seam" arcs (volleyball look). Each seam is a quadratic
    // curve from one rim point to its diametric opposite, with the control
    // point pulled perpendicularly to bow the line — collectively they form
    // a pseudo-tri-panel pattern. Real volleyballs have 6 panels but the
    // 3-curve approximation reads as "volleyball" at gameplay scale.
    ctx.lineWidth = Math.max(1, r * 0.09);
    ctx.strokeStyle = COLOR_BALL_SEAM;
    ctx.lineCap = 'round';
    for (let k = 0; k < 3; k++) {
      const ang = k * (Math.PI * 2 / 3) + Math.PI / 6;  // rotate so seams aren't axis-aligned
      const x1 = cx + r * Math.cos(ang);
      const y1 = cy + r * Math.sin(ang);
      const x2 = cx + r * Math.cos(ang + Math.PI);
      const y2 = cy + r * Math.sin(ang + Math.PI);
      const perp = ang + Math.PI / 2;
      const cpx = cx + r * 0.55 * Math.cos(perp);
      const cpy = cy + r * 0.55 * Math.sin(perp);
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.quadraticCurveTo(cpx, cpy, x2, y2);
      ctx.stroke();
    }
  }
  function drawSlime(agent, color, trim) {
    const cx = toX(agent.x);
    const cy = toY(agent.y);
    const r  = toP(agent.r);
    ctx.fillStyle = color;
    ctx.beginPath();
    // Top-half dome resting on the ground line. In canvas, angles increase
    // clockwise on screen (because y flips). To pass through the *top* half
    // (canvas -y direction = 12 o'clock = 3π/2), we sweep π -> 3π/2 -> 2π,
    // i.e. anticlockwise=false (the default clockwise spec direction).
    ctx.arc(cx, cy, r, Math.PI, 0, false);
    ctx.closePath();
    ctx.fill();
    // Optional jersey trim (used by the away slime — Berkeley Blue accents
    // on top of California Gold body, just like Cal Bears uniforms). Two
    // pieces: a thick rim outline + a horizontal mid-body stripe ("shoulder
    // band") which is the most readable jersey indicator at this scale.
    if (trim) {
      ctx.strokeStyle = trim;
      ctx.lineWidth = Math.max(2.5, r * 0.13);
      ctx.beginPath();
      ctx.arc(cx, cy, r, Math.PI, 0, false);
      ctx.stroke();
      // Mid-body horizontal stripe (chest stripe). We draw the stripe by
      // clipping a rectangle to the dome shape so it doesn't bleed past
      // the slime's outline.
      ctx.save();
      ctx.beginPath();
      ctx.arc(cx, cy, r, Math.PI, 0, false);
      ctx.closePath();
      ctx.clip();
      ctx.fillStyle = trim;
      const stripeY = cy - r * 0.45;
      const stripeH = Math.max(3, r * 0.18);
      ctx.fillRect(cx - r, stripeY, r * 2, stripeH);
      ctx.restore();
    }

    // Eye, oriented toward the fence (matches the Python angle).
    const angle = agent.dir === 1 ? Math.PI * 120 / 180 : Math.PI * 60 / 180;
    const eyeBaseX = agent.x + 0.6 * agent.r * Math.cos(angle);
    const eyeBaseY = agent.y + 0.6 * agent.r * Math.sin(angle);
    const eyePxX = toX(eyeBaseX);
    const eyePxY = toY(eyeBaseY);
    const eyeR  = r * 0.3;
    ctx.fillStyle = COLOR_EYE_WHITE;
    ctx.beginPath();
    ctx.arc(eyePxX, eyePxY, eyeR, 0, Math.PI * 2);
    ctx.fill();

    // Pupil tracks the ball.
    let dx = world.ball.x - eyeBaseX;
    let dy = world.ball.y - eyeBaseY;
    const d = Math.hypot(dx, dy) || 1;
    dx /= d; dy /= d;
    const pupilOffset = 0.15 * agent.r;
    ctx.fillStyle = COLOR_EYE_PUPIL;
    ctx.beginPath();
    ctx.arc(toX(eyeBaseX + dx * pupilOffset),
            toY(eyeBaseY + dy * pupilOffset),
            r * 0.1, 0, Math.PI * 2);
    ctx.fill();
  }

  function drawLives() {
    // Coins along the top edge of each side, just above ground.
    ctx.fillStyle = COLOR_LIFE;
    for (let i = 1; i < world.agent_left.life; i++) {
      const wx = -REF_W / 2 + i * 2;
      ctx.beginPath();
      ctx.arc(toX(wx), toY(REF_H - 2), toP(0.5), 0, Math.PI * 2);
      ctx.fill();
    }
    for (let i = 1; i < world.agent_right.life; i++) {
      const wx = REF_W / 2 - i * 2;
      ctx.beginPath();
      ctx.arc(toX(wx), toY(REF_H - 2), toP(0.5), 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // Overlay shows the Q-vector for whichever side the radio selects, if it's an AI.
  function drawDebug() {
    if (!showDebug) return;
    const labels = ['NOOP', 'FWD', 'FWD+J', 'JUMP', 'BCK+J', 'BACK'];
    const side = getQForSide();
    const slot = slots[side];
    if (!slot.agent) return;
    const q = slot.lastQ;
    const x0 = (side === 'left') ? 20 : (CANVAS_W - 200);
    const y0 = 20;
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    ctx.fillRect(x0 - 8, y0 - 6, 200, 6 + labels.length * 18 + 4);
    ctx.font = '12px monospace';
    let qmin = Infinity, qmax = -Infinity;
    for (let i = 0; i < q.length; i++) { if (q[i] < qmin) qmin = q[i]; if (q[i] > qmax) qmax = q[i]; }
    const span = Math.max(0.001, qmax - qmin);
    for (let i = 0; i < labels.length; i++) {
      const y = y0 + i * 18;
      const norm = (q[i] - qmin) / span;
      ctx.fillStyle = (i === slot.lastAction) ? '#7dd87d' : '#9aa6c2';
      ctx.fillRect(x0 + 60, y - 2, Math.round(norm * 100), 12);
      ctx.fillStyle = '#e6e8ef';
      ctx.fillText(`${labels[i].padEnd(6)} ${q[i].toFixed(2)}`, x0, y + 8);
    }
  }

  function render() {
    const tSec = performance.now() / 1000;
    drawBackground();
    drawCrowd(tSec);               // stadium seating across the upper canvas
    drawGround();
    drawCourtBranding();           // sponsor-style branding on the court surface
    drawFence();
    drawSlime(world.agent_left,  COLOR_LEFT,  COLOR_LEFT_TRIM);
    drawSlime(world.agent_right, COLOR_RIGHT, COLOR_RIGHT_TRIM);
    drawBall();
    drawLives();
    drawDebug();
  }

  // --- per-frame helpers -------------------------------------------------

  // Stable key for the chart's per-agent buffer.
  function chartKey(side, slot) {
    if (slot.kind === 'human') return null;
    return `${side}:${AI_VARIANTS[slot.kind].name}`;
  }

  // Kick off async inference for a slot if its own previous call has resolved.
  // We do not await: pendingAction is reused across frames so the physics step
  // stays synchronous and frame pacing remains stable. The busy flag lives on
  // the slot (not the shared agent) so two slots that pick the same variant
  // can each have an in-flight inference using their own observation.
  function maybeQueueAI(side, slot) {
    if (!slot.agent || !slot.agent.ready || slot.busy) return;
    slot.busy = true;
    // Mirror obs for left-side AI by passing -1 (dir of left agent). getObs in
    // physics.js routes -1/+1 through Agent.updateState's mirroring so the
    // model always sees the canonical "self on +x" view it was trained on.
    const sideArg = (side === 'left') ? -1 : +1;
    const obs = getObs(world, sideArg);
    slot.agent.act(obs).then(({ action, qs }) => {
      slot.pendingAction = action;
      slot.lastAction = action;
      // Defensive copy: the shared agent overwrites lastQ on every call.
      const snap = new Float32Array(qs.length);
      for (let i = 0; i < qs.length; i++) snap[i] = qs[i];
      slot.lastQ = snap;
      const key = chartKey(side, slot);
      if (key) qchart.push(key, snap, action);
      slot.busy = false;
    }).catch(() => { slot.busy = false; });
  }

  function getActionFor(side, slot) {
    if (slot.kind === 'human') {
      return (side === 'left') ? getLeftHumanAction() : getRightHumanAction();
    }
    return ACTION_TABLE[slot.pendingAction] || ACTION_TABLE[0];
  }

  function physicsStep() {
    // Ask each AI for an action (non-blocking).
    maybeQueueAI('left',  slots.left);
    maybeQueueAI('right', slots.right);

    const leftAction  = getActionFor('left',  slots.left);
    const rightAction = getActionFor('right', slots.right);

    const prevLeftLife  = world.agent_left.life;
    const prevRightLife = world.agent_right.life;
    stepWorld(world, leftAction, rightAction);
    if (world.agent_left.life !== prevLeftLife)  setBanner('Right scores');
    if (world.agent_right.life !== prevRightLife) setBanner('Left scores');
    updateLivesHud();

    if (world.done) {
      gameOver = true;
      if (world.agent_right.life <= 0) bannerEl.textContent = 'Left wins';
      else if (world.agent_left.life <= 0) bannerEl.textContent = 'Right wins';
      else bannerEl.textContent = 'Time up';
      if (mode === 'ladder') {
        onLadderGameOver();
        // statusEl is updated by onLadderGameOver / startNextLadderMatch.
      } else {
        statusEl.textContent = 'Press R or Restart';
      }
    }
  }

  // --- main loop ---------------------------------------------------------
  function tick() {
    const now = performance.now();
    const dt = now - lastFrameTs;
    lastFrameTs = now;
    fpsAvg = fpsAvg ? fpsAvg * 0.9 + (1000 / dt) * 0.1 : 1000 / dt;
    fpsEl.textContent = `FPS ${fpsAvg.toFixed(0)}`;

    // While a background fast-sim is running, the headless loop owns `world`;
    // skip physics + render entirely so we don't double-step or paint a stale
    // frame over the placeholder. We keep RAF alive so the moment fast-sim
    // ends, normal rendering resumes without a re-bootstrap.
    if (fastSim.running) {
      requestAnimationFrame(tick);
      return;
    }

    // Replay mode: ReplayPlayer owns the canvas + qchart and drives its own
    // RAF loop for frame advance. We must NOT step physics or call render()
    // (which paints `world` state) here, otherwise we'd flicker between the
    // replayed frame and the stale live World on every tick. Q-chart is
    // also drawn by replayPlayer's seek/push hooks, so we just keep RAF
    // alive and bail.
    if (mode === 'replay') {
      // Still draw the Q chart (with whichever side is selected) so toggling
      // "Show Q for" updates without waiting for a new frame.
      qchart.draw(replayPlayer.currentAgentName);
      requestAnimationFrame(tick);
      return;
    }

    if (!gameOver) {
      // Scale wall-clock dt by speed multiplier before draining into the
      // accumulator. Cap accumulator so re-focusing the tab doesn't fast-
      // forward the game.
      physicsAccum += dt * speedMul;
      const accumCap = 250 * Math.max(1, speedMul);
      if (physicsAccum > accumCap) physicsAccum = PHYSICS_DT_MS;
      let stepsThisFrame = 0;
      while (physicsAccum >= PHYSICS_DT_MS && stepsThisFrame < MAX_STEPS_PER_FRAME && !gameOver) {
        physicsAccum -= PHYSICS_DT_MS;
        stepsThisFrame++;
        physicsStep();
      }
    }

    if (bannerEl.textContent && performance.now() > bannerUntil && !gameOver) {
      bannerEl.textContent = '';
    }

    render();

    // Q-chart shows whichever side the radio picks. If that side is human, fall
    // back to showing the other side if it's an AI; otherwise show empty.
    const showSide = getQForSide();
    const otherSide = showSide === 'left' ? 'right' : 'left';
    let key = chartKey(showSide, slots[showSide]);
    if (!key) key = chartKey(otherSide, slots[otherSide]);
    qchart.draw(key);

    requestAnimationFrame(tick);
  }

  // --- boot --------------------------------------------------------------
  async function boot() {
    speedMul = parseFloat(speedInput.value) || 1.0;
    speedOut.textContent = `${speedMul.toFixed(2)}x`;
    updateLivesHud();
    try {
      // Pin wasm artifact paths to the same CDN bundle as ort.min.js so the
      // runtime doesn't try to fetch *.wasm relative to the page (which 404s).
      if (typeof ort !== 'undefined' && ort.env && ort.env.wasm) {
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.25.1/dist/';
      }
      // Eager-load the slot we boot with (right=Dueling by default). All other
      // variants stay lazy and only fetch when explicitly selected.
      await applySlotSelection('right', rightSelect.value);
      loadingEl.classList.add('hidden');
    } catch (err) {
      console.error('Boot error:', err);
      loadingEl.textContent = 'Failed to load default model. Right slot fell back to Human.';
    }
    requestAnimationFrame(tick);
  }

  boot();
})();
