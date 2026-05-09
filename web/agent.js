// ONNX inference wrapper. One instance per checkpoint variant. Sessions are
// created lazily by load() so the page does not fetch model_v1.onnx /
// model_rainbow.onnx unless the user actually selects them.
// Globals: ONNXAgent.

class ONNXAgent {
  constructor(name, modelUrl) {
    this.name = name;
    this.modelUrl = modelUrl;
    this.session = null;
    this.lastQ = new Float32Array(6);  // exposed for the chart / debug overlay
    this.lastAction = 0;
    this.inputName = 'observation';
    this.outputName = 'q_values';
    this.ready = false;
    this.unavailable = false;          // set on 404 / load error
    this.loadPromise = null;
  }

  // Lazy: returns a cached promise after the first call.
  async load() {
    if (this.loadPromise) return this.loadPromise;
    this.loadPromise = (async () => {
      try {
        // WASM-only EP: this is a 12->...->6 MLP, GPU upload would dominate.
        this.session = await ort.InferenceSession.create(this.modelUrl, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
        });
        const ins = this.session.inputNames;
        const outs = this.session.outputNames;
        if (ins.length > 0)  this.inputName  = ins[0];
        if (outs.length > 0) this.outputName = outs[0];
        this.ready = true;
      } catch (err) {
        this.unavailable = true;
        this.ready = false;
        console.warn(`Could not load ${this.modelUrl}:`, err && err.message ? err.message : err);
        throw err;
      }
    })();
    return this.loadPromise;
  }

  // obs: Float32Array length 12 (already divided by 10 by Agent.getObservation).
  // Returns: { action: int 0..5, qs: Float32Array length 6 }.
  async act(obs) {
    if (!this.ready) {
      return { action: 0, qs: this.lastQ };
    }
    // GLOBAL mutex (across ALL ONNXAgent instances). onnxruntime-web 1.25.1
    // WASM EP has a singleton inference state — concurrent session.run() calls
    // on DIFFERENT sessions in the same WASM instance still throw
    // "Session already started" / "Session mismatch". Serialize ALL ONNX runs
    // through the same chain. .catch wrapper prevents one failed call from
    // poisoning the queue. (No effect on baseline / human slots — they don't
    // touch this chain.)
    const myTurn = ONNXAgent._globalInflight.then(() => this._actUnsafe(obs));
    ONNXAgent._globalInflight = myTurn.then(() => undefined, () => undefined);
    return myTurn;
  }

  async _actUnsafe(obs) {
    const tensor = new ort.Tensor('float32', obs, [1, 12]);
    const feeds = {};
    feeds[this.inputName] = tensor;
    let q;
    try {
      const out = await this.session.run(feeds);
      q = out[this.outputName].data; // Float32Array length 6
      // NOTE on disposal in onnxruntime-web v1.20.1:
      //   * Tensor / OnnxValue does NOT expose a public .dispose() method on
      //     this version (the public type in onnxruntime-common has no
      //     dispose). Output tensors are GC'd when no JS reference remains;
      //     we just drop `out` at end-of-scope. The `typeof === 'function'`
      //     guard below is therefore a no-op on v1.20.1 but kept defensively
      //     in case a future runtime adds it.
      //   * InferenceSession DOES expose .release() (async) — see reload().
      //   * No public WASM heap query; mid-run reload (game.js) is the
      //     primary defense against accumulating session/tensor heap.
      for (const k of Object.keys(out)) {
        if (out[k] && typeof out[k].dispose === 'function') {
          try { out[k].dispose(); } catch (e) { /* ignore */ }
        }
      }
    } catch (err) {
      console.warn(`[ONNXAgent ${this.name}] inference error:`, err);
      // Keep agent alive — fall back to last known Q so the action history
      // doesn't suddenly collapse to NOOP. ladderActAsync also has its own
      // safety NOOP fallback.
      return { action: this.lastAction, qs: this.lastQ };
    } finally {
      // No-op on v1.20.1 (see comment above), kept for forward-compat.
      if (typeof tensor.dispose === 'function') {
        try { tensor.dispose(); } catch (e) { /* ignore */ }
      }
    }
    this.lastQ = q;

    let best = 0;
    let bestVal = q[0];
    for (let i = 1; i < q.length; i++) {
      if (q[i] > bestVal) { bestVal = q[i]; best = i; }
    }
    this.lastAction = best;
    return { action: best, qs: q };
  }

  // Drop any prior session so it gets garbage-collected, then clear the
  // memoized load promise so the next load() call actually fetches again.
  // Safe to call multiple times; safe to call before load() ever resolved.
  // Async because InferenceSession.release() in onnxruntime-web v1.20.1
  // returns Promise<void> (it tears down the WASM session resources). Callers
  // that don't care about ordering can fire-and-forget; the mid-run
  // auto-reload path in game.js awaits this so the next load() doesn't race
  // a still-tearing-down session for the same model.
  async reload() {
    if (this.session && typeof this.session.release === 'function') {
      try { await this.session.release(); } catch (e) { /* ignore */ }
    }
    this.session = null;
    this.ready = false;
    this.unavailable = false;
    this.loadPromise = null;
  }
}

// Class-level mutex chain shared across all ONNXAgent instances.
// See comment in act() for why this must be global.
ONNXAgent._globalInflight = Promise.resolve();
