// ELO ladder / tournament mode for the slime volleyball arena.
// Globals exposed: EloLadder.
//
// Roster design:
//   Each AI variant fields TWO entries (e.g. "Dueling A" and "Dueling B"). Both
//   entries share the same underlying ONNXAgent / BaselineAgent instance via
//   game.js's getOrCreateAgent(variantKey) cache — the policy is deterministic
//   given a fixed observation, so cloning the entry does not change behavior.
//   ELO between the two clones still drifts thanks to:
//     1. env stochasticity — newBall() draws (vx, vy) uniformly each match,
//        so identical policies can win or lose any single match.
//     2. side asymmetry — the agent on the left vs. right sees a mirrored
//        observation, and the ball's initial velocity distribution is not
//        symmetric in side (e.g. positive vy bias). Even a perfect mirror
//        policy can have side-dependent expected returns over short windows.
//     3. opponent diversity — across many matches each clone faces different
//        opponents in different orders; small per-match deltas integrate.
//   The "two entries per variant" trick lets the user see the variance band
//   instead of a single point estimate; the spread between A and B is a
//   rough proxy for sampling noise around each variant's true ELO.
//
// Notation:
//   * 13 entries: 12 AI (6 variants x 2) + 1 Human.
//   * default ELO = 1500, K = 32.
//   * Human is only matched when explicitly queued via queueHuman().

class EloLadder {
  constructor(options = {}) {
    this.K = options.K || 32;
    this.defaultRating = options.defaultRating || 1500;
    this.strategy = 'random';            // 'random' | 'closest'
    this.humanQueued = false;
    this.recentMatches = [];             // last few "leftId|rightId" strings, for dedup
    this.recentMaxLen = 3;

    // Roster: 16 AI (6 base variants + 2 extra Rainbow training configs, 2 clones each) + 1 Human.
    this.roster = [
      { id: 'plain_a',          name: 'Plain DQN A',      variantKey: 'plain',         isHuman: false },
      { id: 'plain_b',          name: 'Plain DQN B',      variantKey: 'plain',         isHuman: false },
      { id: 'dueling_a',        name: 'Dueling A',        variantKey: 'dueling',       isHuman: false },
      { id: 'dueling_b',        name: 'Dueling B',        variantKey: 'dueling',       isHuman: false },
      { id: 'duel_double_a',    name: 'Duel+Double A',    variantKey: 'duel_double',   isHuman: false },
      { id: 'duel_double_b',    name: 'Duel+Double B',    variantKey: 'duel_double',   isHuman: false },
      { id: 'duel_nstep_a',     name: 'Duel+Nstep A',     variantKey: 'duel_nstep',    isHuman: false },
      { id: 'duel_nstep_b',     name: 'Duel+Nstep B',     variantKey: 'duel_nstep',    isHuman: false },
      { id: 'v1_a',             name: 'V1 cold A',        variantKey: 'v1',            isHuman: false },
      { id: 'v1_b',             name: 'V1 cold B',        variantKey: 'v1',            isHuman: false },
      { id: 'v1_selfplay_a',    name: 'V1 selfplay A',    variantKey: 'v1_selfplay',   isHuman: false },
      { id: 'v1_selfplay_b',    name: 'V1 selfplay B',    variantKey: 'v1_selfplay',   isHuman: false },
      { id: 'v1_purepool_a',    name: 'V1 purepool A',    variantKey: 'v1_purepool',   isHuman: false },
      { id: 'v1_purepool_b',    name: 'V1 purepool B',    variantKey: 'v1_purepool',   isHuman: false },
      { id: 'v1_sp_seeded_a',   name: 'V1 sp seeded A',   variantKey: 'v1_selfplay_seeded', isHuman: false },
      { id: 'v1_sp_seeded_b',   name: 'V1 sp seeded B',   variantKey: 'v1_selfplay_seeded', isHuman: false },
      { id: 'v1_sp_fixed_a',    name: 'V1 sp fixed A',    variantKey: 'v1_sp_fixed',    isHuman: false },
      { id: 'v1_sp_fixed_b',    name: 'V1 sp fixed B',    variantKey: 'v1_sp_fixed',    isHuman: false },
      { id: 'rainbow_a',        name: 'Rainbow 20M sp A', variantKey: 'rainbow',       isHuman: false },
      { id: 'rainbow_b',        name: 'Rainbow 20M sp B', variantKey: 'rainbow',       isHuman: false },
      { id: 'rainbow_5m_a',     name: 'Rainbow 5M A',     variantKey: 'rainbow_5m',    isHuman: false },
      { id: 'rainbow_5m_b',     name: 'Rainbow 5M B',     variantKey: 'rainbow_5m',    isHuman: false },
      { id: 'rainbow_5m_sp_a',  name: 'Rainbow 5M sp A',  variantKey: 'rainbow_5m_sp', isHuman: false },
      { id: 'rainbow_5m_sp_b',  name: 'Rainbow 5M sp B',  variantKey: 'rainbow_5m_sp', isHuman: false },
      { id: 'rainbow_sp_fixed_a', name: 'Rainbow sp fixed A', variantKey: 'rainbow_sp_fixed', isHuman: false },
      { id: 'rainbow_sp_fixed_b', name: 'Rainbow sp fixed B', variantKey: 'rainbow_sp_fixed', isHuman: false },
      { id: 'rainbow_full_a',   name: 'Rainbow full A',   variantKey: 'rainbow_full', isHuman: false },
      { id: 'rainbow_full_b',   name: 'Rainbow full B',   variantKey: 'rainbow_full', isHuman: false },
      { id: 'rainbow_eps_a',    name: 'Rainbow eps A',    variantKey: 'rainbow_eps',  isHuman: false },
      { id: 'rainbow_eps_b',    name: 'Rainbow eps B',    variantKey: 'rainbow_eps',  isHuman: false },
      { id: 'ppo_a',            name: 'PPO A',            variantKey: 'ppo',           isHuman: false },
      { id: 'ppo_b',            name: 'PPO B',            variantKey: 'ppo',           isHuman: false },
      { id: 'ppo_fixed_a',      name: 'PPO fixed A',      variantKey: 'ppo_fixed',      isHuman: false },
      { id: 'ppo_fixed_b',      name: 'PPO fixed B',      variantKey: 'ppo_fixed',      isHuman: false },
      { id: 'ppo_rescue_a',     name: 'PPO rescue A',     variantKey: 'ppo_rescue',     isHuman: false },
      { id: 'ppo_rescue_b',     name: 'PPO rescue B',     variantKey: 'ppo_rescue',     isHuman: false },
      { id: 'baseline_a',       name: 'Baseline A',       variantKey: 'baseline',      isHuman: false },
      { id: 'baseline_b',       name: 'Baseline B',       variantKey: 'baseline',      isHuman: false },
      { id: 'human',            name: 'Human',            variantKey: 'human',         isHuman: true  },
    ].map((r) => Object.assign(r, {
      eloRating: this.defaultRating,
      gamesPlayed: 0,
      wins: 0, draws: 0, losses: 0,
      history: [this.defaultRating],
      unavailable: false,
    }));

    this.byId = new Map(this.roster.map((r) => [r.id, r]));

    // Persist ELO + history across page reloads (since the only reliable WASM
    // recovery path is window.location.reload(), we keep ladder state in
    // localStorage so a refresh doesn't wipe progress).
    this._storageKey = 'slime_volley_elo_v1';
    this._loadFromStorage();
  }

  _loadFromStorage() {
    try {
      const raw = localStorage.getItem(this._storageKey);
      if (!raw) return;
      const data = JSON.parse(raw);
      if (!data || !Array.isArray(data.roster)) return;
      for (const saved of data.roster) {
        const r = this.byId.get(saved.id);
        if (!r) continue;
        if (typeof saved.eloRating === 'number') r.eloRating = saved.eloRating;
        if (typeof saved.gamesPlayed === 'number') r.gamesPlayed = saved.gamesPlayed;
        if (typeof saved.wins === 'number') r.wins = saved.wins;
        if (typeof saved.draws === 'number') r.draws = saved.draws;
        if (typeof saved.losses === 'number') r.losses = saved.losses;
        if (Array.isArray(saved.history)) r.history = saved.history.slice(-200);
      }
    } catch (e) { /* ignore corrupt storage */ }
  }

  _saveToStorage() {
    try {
      const slim = this.roster.map((r) => ({
        id: r.id, eloRating: r.eloRating, gamesPlayed: r.gamesPlayed,
        wins: r.wins, draws: r.draws, losses: r.losses,
        history: r.history.slice(-50),
      }));
      localStorage.setItem(this._storageKey, JSON.stringify({ roster: slim }));
    } catch (e) { /* quota exceeded etc — non-fatal */ }
  }

  resetAllRatings() {
    for (const r of this.roster) {
      r.eloRating = this.defaultRating;
      r.gamesPlayed = 0;
      r.wins = 0; r.draws = 0; r.losses = 0;
      r.history = [this.defaultRating];
    }
    try { localStorage.removeItem(this._storageKey); } catch (e) {}
  }

  // Mark a variant unavailable (e.g. model 404). Both A/B clones get tagged so
  // matchmaking skips them. Called by game.js after ensureLoaded() fails.
  markVariantUnavailable(variantKey) {
    for (const r of this.roster) {
      if (r.variantKey === variantKey) r.unavailable = true;
    }
  }

  // Counterpart to markVariantUnavailable: clear the unavailable flag so the
  // matchmaker considers the variant again. Called by game.js after the user
  // clicks "Reload models" to recover from a stuck state.
  markVariantAvailable(variantKey) {
    for (const r of this.roster) {
      if (r.variantKey === variantKey) r.unavailable = false;
    }
  }

  setStrategy(s) {
    if (s === 'random' || s === 'closest') this.strategy = s;
  }

  queueHuman() { this.humanQueued = true; }
  isHumanQueued() { return this.humanQueued; }

  // Pool of selectable AI players (excludes Human and unavailable variants).
  _aiPool() {
    return this.roster.filter((r) => !r.isHuman && !r.unavailable);
  }

  _humanEntry() { return this.byId.get('human'); }

  _matchKey(a, b) {
    // Order-independent so A-vs-B and B-vs-A count as the same recent pair.
    return a < b ? `${a}|${b}` : `${b}|${a}`;
  }

  _wasRecent(a, b) {
    const k = this._matchKey(a, b);
    return this.recentMatches.indexOf(k) !== -1;
  }

  _pushRecent(a, b) {
    this.recentMatches.push(this._matchKey(a, b));
    while (this.recentMatches.length > this.recentMaxLen) this.recentMatches.shift();
  }

  // Returns { leftId, rightId } or null if there's no valid pair.
  // Side assignment is randomized so a player isn't permanently the "left" slot.
  nextMatch() {
    const pool = this._aiPool();
    if (pool.length < 2 && !this.humanQueued) return null;

    let a = null, b = null;

    if (this.humanQueued) {
      // Human + closest-ELO available opponent (excluding human).
      this.humanQueued = false;
      const human = this._humanEntry();
      if (pool.length === 0) return null;
      // Sort pool by |ELO gap to human|, prefer non-recent, fall back to closest.
      const sorted = pool.slice().sort((x, y) =>
        Math.abs(x.eloRating - human.eloRating) - Math.abs(y.eloRating - human.eloRating));
      let pick = sorted.find((p) => !this._wasRecent(human.id, p.id));
      if (!pick) pick = sorted[0];
      a = human; b = pick;
    } else if (this.strategy === 'closest') {
      // Find the pair with smallest ELO gap that isn't a recent rematch.
      let bestGap = Infinity;
      for (let i = 0; i < pool.length; i++) {
        for (let j = i + 1; j < pool.length; j++) {
          const gap = Math.abs(pool[i].eloRating - pool[j].eloRating);
          if (this._wasRecent(pool[i].id, pool[j].id)) continue;
          if (gap < bestGap) {
            bestGap = gap; a = pool[i]; b = pool[j];
          }
        }
      }
      if (!a) {
        // Everyone in the pool was "recent" — fall back to overall closest.
        let bestGap2 = Infinity;
        for (let i = 0; i < pool.length; i++) {
          for (let j = i + 1; j < pool.length; j++) {
            const gap = Math.abs(pool[i].eloRating - pool[j].eloRating);
            if (gap < bestGap2) { bestGap2 = gap; a = pool[i]; b = pool[j]; }
          }
        }
      }
    } else {
      // Random pair. Try a few times to avoid recent dupes; give up if all
      // candidates are recent (small pool case).
      for (let attempt = 0; attempt < 10; attempt++) {
        const i = Math.floor(Math.random() * pool.length);
        let j = Math.floor(Math.random() * (pool.length - 1));
        if (j >= i) j += 1;
        const x = pool[i], y = pool[j];
        if (!this._wasRecent(x.id, y.id)) { a = x; b = y; break; }
        a = x; b = y; // remember last fallback
      }
    }

    if (!a || !b) return null;

    // Randomize side assignment.
    const swap = Math.random() < 0.5;
    const leftId  = swap ? b.id : a.id;
    const rightId = swap ? a.id : b.id;
    this._pushRecent(leftId, rightId);
    return { leftId, rightId };
  }

  // Convert final game state to ELO outcome and update both players.
  //   leftLife > rightLife  => left wins (S_left = 1)
  //   leftLife < rightLife  => right wins (S_left = 0)
  //   leftLife == rightLife => draw (S_left = 0.5)
  // The K=32 standard formula:
  //   E_a = 1 / (1 + 10^((R_b - R_a) / 400))
  //   R'_a = R_a + K * (S_a - E_a)
  recordResult(leftId, rightId, leftLife, rightLife) {
    const left  = this.byId.get(leftId);
    const right = this.byId.get(rightId);
    if (!left || !right) return null;

    let sLeft;
    if      (leftLife >  rightLife) sLeft = 1.0;
    else if (leftLife <  rightLife) sLeft = 0.0;
    else                            sLeft = 0.5;
    const sRight = 1.0 - sLeft;

    const eLeft  = 1.0 / (1.0 + Math.pow(10, (right.eloRating - left.eloRating) / 400));
    const eRight = 1.0 - eLeft;

    left.eloRating  = left.eloRating  + this.K * (sLeft  - eLeft);
    right.eloRating = right.eloRating + this.K * (sRight - eRight);

    for (const [r, s] of [[left, sLeft], [right, sRight]]) {
      r.gamesPlayed += 1;
      if      (s === 1.0) r.wins   += 1;
      else if (s === 0.5) r.draws  += 1;
      else                r.losses += 1;
      r.history.push(r.eloRating);
      if (r.history.length > 200) r.history.shift();
    }
    this._saveToStorage();
    return { sLeft, leftElo: left.eloRating, rightElo: right.eloRating };
  }

  // Snapshot sorted by descending ELO. Stable: ties resolved by name.
  getRoster() {
    return this.roster.slice().sort((a, b) => {
      if (b.eloRating !== a.eloRating) return b.eloRating - a.eloRating;
      return a.name.localeCompare(b.name);
    });
  }
}
