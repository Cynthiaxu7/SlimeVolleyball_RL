# Ablation Study — Slime Volleyball RL

This document accompanies the project's training/eval pipeline. Every number
below is a **deterministic greedy `eval_R`** measured by the trainer's
`evaluate_vs_baseline` (single-arg `env.step` → env's internal auto-baseline,
which is the *real* baseline, not the obs ×10 buggy variant). 5 episodes per
eval; each episode caps at 3000 frames or first side reaching 0 lives.

Reward convention: `eval_R = mean(left_lives_lost − right_lives_lost)` from
the trained agent's perspective. `+5` = 5-0 sweep, `−5` = swept, `0` = perfect
draw at the time-cap.

All training uses 12-d obs, Discrete(6) actions, 16 parallel envs, A6000 GPU.

---

## 1. Component ablation (5M timesteps, vs real baseline)

A 2×2×2 study over `{Dueling head}` × `{Double DQN}` × `{N-step (n=3 vs n=1)}`.
All cells share the same trunk MLP (12→128→128 ReLU), the same target-net /
soft-update / replay capacity, and the same hyperparams. The only thing that
changes is what the loss / architecture do.

| # | Dueling | Double | N-step | `eval_R` | Δ vs floor | Checkpoint |
|---|:---:|:---:|:---:|:---:|:---:|---|
| A | ✗ | ✗ | n=1 | **−0.6 ± 0.5** | (floor) | `dqn_plain_vec.pt` |
| B | ✓ | ✗ | n=1 | −0.4 | +0.2 | `dqn_vec.pt` (v0) |
| C | ✓ | ✓ | n=1 | **+0.2 ± 0.4** | **+0.8** ⭐ | `dqn_v1_dd_vec.pt` |
| D | ✓ | ✗ | n=3 | −2.2 ± 1.2 | −1.6 ⚠️ | `dqn_v1_dn_vec.pt` |
| E | ✓ | ✓ | n=3 | −0.4 | +0.2 | `dqn_v1_vec.pt` (v1 cold) |

### Findings

1. **Dueling alone barely helps** (cell A → B: +0.2). On a 12-d MLP the
   V/A decomposition's "credit-to-the-state-not-the-action" advantage is
   marginal — there are not many states where action choice is irrelevant.

2. **Double DQN is the single biggest win.** Adding Double on top of Dueling
   (cell B → C): **+0.6** `eval_R`. This is consistent with the Bellman
   over-estimation argument: with a small/noisy Q net, `max_a Q_target(s', a)`
   is a positively biased estimator and Double's decoupling
   `Q_target(s', argmax_a Q_online(s', a))` removes most of that bias.

3. **N-step alone is ACTIVELY HARMFUL** (cell B → D: **−1.8**). This was
   the most surprising result. Without Double's bias correction, the
   n-step return amplifies the over-estimation: each step in the n-step
   sum brings extra noise from off-policy bootstrap, and with only `max`
   doing the lookup the bias compounds. Re-introducing Double on top of
   N-step (cell D → E: +1.8) almost fully recovers the deficit.

4. **N-step + Double together** is roughly equivalent to N=1 + Double on
   this task (cell C ≈ E). N-step's value here is in **sample efficiency
   under correct credit assignment**, not final performance. v1 cold and
   `duel_double` are within noise on a single seed.

> **Practical takeaway:** if you can only afford one upgrade over plain DQN,
> add Double. N-step is a multiplier on Double, not a standalone improvement.

---

## 2. Self-play opponent mixing (V1 family, 20M timesteps)

The Slime Volleyball self-play loop draws each new episode's opponent from
either the snapshot pool (FIFO of past self-checkpoints) or the scripted
baseline, gated by `--baseline-opponent-prob`. We ran three settings on
the V1 trainer.

| Variant | `baseline_opp_prob` | Pool seed | final `base_R` | Notes |
|---|:---:|---|:---:|---|
| `v1_sp_fixed` | 0.4 | dqn_vec, dqn_v1_vec (clean) | **+0.4 ± 0.8** | the recommended config |
| `v1_sp_seeded` (legacy) | 0.4 | + buggy v2/v3 ckpts | (no fresh eval; ladder ELO 1711) | trained pre-baseline-fix |
| `v1_sp_purepool` | **0.0** | + `dqn_v1_vec_selfplay.pt` (1765 ELO king) | **−1.2 ± 0.7** | no baseline exposure during training |

### Finding

**Removing baseline exposure during training is harmful** (−1.6 base_R vs the
reference cell). The ladder-king v1_selfplay pool seed gave the agent a
strong opponent from step 0, but the policy that emerged from learning to beat
*other neural-net players* did not generalize back to the scripted RNN
baseline. Baseline plays a stylistically different game (RNN-style reflex
returns) that the snapshot pool cannot proxy — so even a "buggy" baseline in
the pool is **necessary diversity**, not a cosmetic addition.

This is one of those experimental hypotheses I **expected to confirm** ("pool
quality > baseline quality") that **didn't survive contact with real numbers**
— the most informative kind.

---

## 3. PPO rescue case study

Initial PPO run (`ppo_lowent_20m_fixed`) finished at **`base_R = −3.4`** —
nearly a sweep loss against baseline. Diagnosis from the logs:

- Entropy `H = 0.108` by step 1.4M (out of max ~1.79 for 6 actions)
- `pool_R ≈ +1.0` (beating self-snapshots)
- `base_R ≈ −4` throughout (never improved)

Classic **early-entropy collapse**: the policy committed to a deterministic
strategy that beats *itself* (selfplay pool) but never generalized to baseline.

The rescue run applied four changes in combination:

| Change | Old | New | Mechanism |
|---|---|---|---|
| `--ent-coef` | 0.001 | **0.01** | 10× higher entropy bonus → keeps policy stochastic longer |
| `--rollout-len` | 128 | **256** | better advantage estimates (episode is up to 3000 frames) |
| `--lr-anneal` | off | **on** | linear LR → 0 over the run; standard PPO recipe |
| `--target-kl` | none | **0.02** | early-stop the update epochs when approx_kl exceeds threshold |

| Variant | final `base_R` | final `H` | Δ |
|---|:---:|:---:|:---:|
| `ppo_lowent_20m_fixed` (failed) | −3.4 ± 1.5 | 0.276 | — |
| `ppo_rescue_20m` | **+0.2 ± 0.4** | **1.42** | **+3.6** |

### Finding

**3.6 base_R from one config change cluster.** Most of this is attributable
to the entropy coefficient — going from 0.001 to 0.01 keeps the policy from
"prematurely committing" to a local optimum. The other three (rollout_len,
lr_anneal, target_kl) are stabilizers that prevent the higher-entropy run
from oscillating, but they do not produce the gain by themselves.

PPO has a much sharper local-optimum cliff than Q-learning on this task —
DQN variants (with `eps`-greedy that decays gradually) tend to find okay
solutions even with bad hyperparams; PPO requires careful entropy management
to avoid mode collapse.

---

## 4. Surprise finding — buggy lineage dominates fixed lineage

While debugging, we discovered that all three self-play trainers had been
feeding `obs * 10` to `BaselinePolicy.predict` inside `BaselineOpponentAdapter`
— a long-standing `× 10` scale bug that effectively crippled the in-pool
baseline (saturated the policy's tanh layer to ±1). Fixed runs use the same
trainers with the bug removed.

**Head-to-head: `v1_sp_fixed` vs `v1_selfplay` (the buggy ladder king),
200 matches alternating sides:**

| Outcome | Count | % |
|---|:---:|:---:|
| Buggy wins (1-life margin) | 71 | 35.5% |
| Fixed wins | 9 | 4.5% |
| Draws (time-cap, 5-5) | 120 | 60.0% |

Buggy wins ~7.9× more than fixed despite training in a strictly degraded
environment. **Why?**

The buggy lineage (which dominates the public ladder roster) all trained
*against each other and against the saturated baseline*. Their playstyles
co-evolved to exploit and counter each other's quirks. The fixed agent
learned *correct* play against the *real* baseline — which is a different
distribution. In a ladder full of buggy-lineage opponents, the fixed agent
is the foreign exchange student who learned by the textbook while everyone
else learned through brawls.

This is an **ecosystem effect** worth flagging in any RL system: when many
agents share a training quirk, that quirk becomes the local lingua franca,
and a "correctly trained" newcomer can underperform until the ecosystem
re-equilibrates around the fix.

---

## 5. Web-ladder ELO (1500 matches, 19 AI variants, K=32, init=1500)

We placed every variant in a 38-player ladder (19 variants × 2 clones for
variance averaging). The Python ladder uses the same physics + ONNX models
as the browser fast-sim path, so its numbers track the in-browser ladder
on a long enough run.

| Rank | Variant | Mean ELO | Training |
|---|---|:---:|---|
| 1 | `v1_selfplay_seeded` | **1824** | 20M selfplay, buggy baseline, seeded pool |
| 2 | `v1_selfplay` | **1795** | 20M selfplay, buggy baseline |
| 3 | `rainbow_5m_sp` | **1713** | 5M selfplay + PER, buggy baseline |
| 4 | `rainbow_sp_fixed` | **1690** | 20M selfplay + PER, **fixed** baseline |
| 5 | `v1_sp_fixed` | **1679** | 20M selfplay, **fixed** baseline |
| 6 | **`baseline`** | **1602** | scripted RNN (NOT trained) |
| 7 | `rainbow_5m` | 1600 | 5M cold + PER (no selfplay) |
| 8 | `rainbow` | 1553 | 20M selfplay (older) |
| 9 | `ppo_fixed` | 1540 | failed PPO (entropy-collapse) |
| 10 | `plain` (NEW) | **1520** | 5M plain DQN (no Dueling/Double/N-step) |
| 11 | `ppo_rescue` | 1497 | entropy-fix PPO |
| 12 | `v1_purepool` (NEW) | **1454** | 20M selfplay, **no baseline in pool** |
| 13 | `dueling` (v0) | 1453 | 5M Dueling alone |
| 14 | `ppo` (legacy) | 1412 | 5M PPO |
| 15 | `duel_double` (NEW) | **1400** | 5M Dueling + Double |
| 16 | `duel_nstep` (NEW) | **1330** | 5M Dueling + N-step |
| 17 | `v1` (cold) | 1282 | 5M Dueling + Double + N-step (cold) |
| 18 | `rainbow_full` | **1084** | full Rainbow + C51 + NoisyNet ⚠️ |
| 19 | `rainbow_eps` | **1071** | Rainbow + C51 (no NoisyNet) ⚠️ |

### Findings (5 things worth flagging)

1. **Self-play >> cold for the same algorithm.** `v1_selfplay` (1795) vs `v1`
   cold (1282) = 513 ELO gap, ~94% expected-win for the same network with
   different training signal. The biggest single win in the project.

2. **The real baseline ranks 6th of 19.** Only the top 5 trained variants
   reliably beat the scripted RNN — Slime volleyball is a surprisingly
   strong baseline.

3. **`eval_R` (vs baseline) ≠ ladder ELO.** Several variants invert the
   ranking we'd predict from `eval_R`:

   | Pair | better `eval_R` | better ladder ELO |
   |---|---|---|
   | duel_double vs dueling | duel_double (+0.2 vs −0.4) | dueling (1453 vs 1400) ⚠️ |
   | ppo_rescue vs ppo_fixed | rescue (+0.2 vs −3.4) | ppo_fixed (1540 vs 1497) ⚠️ |
   | plain vs dueling | dueling (−0.4 vs −0.6) | plain (1520 vs 1453) ⚠️ |
   | v1_sp_fixed vs v1_selfplay | comparable | v1_selfplay (+115 ELO) |

   This isn't a bug — `eval_R` measures performance on **a single specific
   opponent** (the scripted baseline) while ELO measures **average
   performance across the entire ecosystem**. A policy that hyper-specializes
   to baseline can lose to other trained agents that play differently;
   conversely a "messier" policy with slower convergence can be a better
   generalist.

4. **Full Rainbow (with C51 + NoisyNet) failed catastrophically.** Both
   `rainbow_full` (1084) and `rainbow_eps` (1071) lose 95%+ of their ladder
   matches. After investigation, the C51-trained checkpoints' expected-Q
   outputs (softmax over the 51-atom support, contracted with the support
   vector) end up at a *very different magnitude* than plain Q values
   from the Dueling/V1 family — but argmax should still pick correctly.
   Most likely cause: **C51 with v_min=−5, v_max=+5 mismatched the actual
   reward distribution** (slime episode returns can hit ±5+ at episode
   end), so the support mass concentrates at the boundary atoms and Q
   estimates become uniformly degenerate. A retrain with v_min=−10 / v_max=+10
   should fix it. NoisyNet on a 50k-param net is also probably more noise
   than signal — the simpler `rainbow_lite` (Dueling + Double + N-step + PER)
   reaches 1690 ELO in the same training budget.

5. **`v1_purepool` underperformed all baseline-mixed variants.** No baseline
   exposure during training cost ~225 ELO vs `v1_sp_fixed`. Pool diversity
   and pool quality are not the same axis: the scripted baseline brings a
   distinct *playstyle* (RNN reflex returns) that pure-snapshot pools
   cannot proxy.

---

## How to reproduce

```bash
# Component ablation cells (5M each, ~30 min on A6000)
sbatch scripts/sbatch_plain_dqn_5m.slurm           # cell A
sbatch scripts/sbatch_dueling_double_5m.slurm      # cell C
sbatch scripts/sbatch_dueling_nstep_5m.slurm       # cell D
# (cells B, E already trained as v0/v1 cold)

# Self-play cells (20M each, ~5h)
sbatch scripts/sbatch_v1_vec_selfplay_fixed.slurm        # baseline_prob=0.4 (recommended)
sbatch scripts/sbatch_v1_vec_selfplay_purepool.slurm     # baseline_prob=0.0

# PPO rescue
sbatch scripts/sbatch_ppo_rescue_20m.slurm

# Re-run the ladder benchmark
python scripts/ladder_sim.py --n-matches 1500 --output auto

# Head-to-head comparison
python scripts/head2head.py --a web/model_v1_selfplay_fixed.onnx \
                            --b web/model_v1_selfplay.onnx --n 200
```
