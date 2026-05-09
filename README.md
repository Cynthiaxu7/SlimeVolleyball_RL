# Slime Volleyball — Deep RL

Final project for INDENG 1/242B (Deep Learning), UC Berkeley, Spring 2026.

We train deep RL agents on the [slimevolleygym](https://github.com/hardmaru/slimevolleygym)
environment using a Rainbow-style DQN family (Dueling, Double, N-step, PER,
distributional C51, NoisyNet) and PPO with self-play opponent pools, then
deploy every variant in a browser-based demo + ELO ladder via ONNX Runtime
Web. Trained on Berkeley's `a6k` SLURM partition (NVIDIA A6000 GPUs).

## Repository layout

```
train/        DQN / Rainbow / PPO trainers + ONNX export utilities
scripts/      SLURM sbatch templates, ladder simulator, replay recorder,
              head-to-head evaluator, plotting / screenshot scripts
web/          Static browser demo (HTML + JS + ONNX runtime web). Includes
              free-play, ELO ladder, and replay modes. ONNX models live here.
docs/         ABLATIONS.md (the full ablation write-up)
report/       LaTeX source + slime_v_report.pdf (ICLR-style course report)
```

## Quick start

### Web demo (no training required)

```bash
cd web && python -m http.server 8000
# open http://localhost:8000
```

The dropdowns expose every trained variant. Switch to "Ladder" mode and
run 100 / 1000 background matches to populate the ELO table.

### Training

Conda env: `slime_rl` with key deps `torch`, `gym==0.21`, `numpy<2`,
`slimevolleygym`, `onnxruntime`, `onnxruntime-web`.

```bash
# Single-GPU SLURM jobs (see scripts/sbatch_*.slurm for variants)
sbatch scripts/sbatch_v1_vec_selfplay_fixed.slurm
sbatch scripts/sbatch_rainbow_lite_sp_20m_fixed.slurm
sbatch scripts/sbatch_ppo_rescue_20m.slurm

# After training, export to ONNX for the web demo
python train/export_onnx.py \
    --checkpoint checkpoints/dqn_v1_vec_selfplay_fixed.pt \
    --output     web/model_v1_selfplay_fixed.onnx
```

### Ablation study

The full ablation write-up — including a 2x2x2 component grid, the
PPO entropy-collapse rescue, and the buggy-vs-fixed ecosystem effect —
is in [`docs/ABLATIONS.md`](docs/ABLATIONS.md).

To regenerate the 1500-match ELO ladder:

```bash
python scripts/ladder_sim.py --n-matches 1500 --output auto
```

## Acknowledgments

- Environment: [`slimevolleygym`](https://github.com/hardmaru/slimevolleygym)
  by hardmaru, Apache 2.0.
- Inference: ONNX Runtime Web (Microsoft, MIT).

## License

Course project. No license decision yet — please ask before reuse.
