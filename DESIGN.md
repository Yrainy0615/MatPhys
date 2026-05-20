# Design

This repository estimates per-edge physical parameters of a deformable
object from a monocular RGB video, fits them through a differentiable
spring–mass simulator, and renders the resulting motion with 3D Gaussian
Splatting. It is built on top of the PhysTwin baseline; the main change
is the addition of a semantic / material-aware feed-forward predictor
trained per scene.

## Pipeline overview

```
            video                 +---------------------+
              |                   |   semantic chain    |
              v                   |  Cupid → DINO →     |
   +--------------------+  -----> |  VLM material prior |---+
   |   preprocessing    |         +---------------------+   |
   |  (segmentation,    |                                   |
   |   tracking, GS)    |                                   v
   +--------------------+                       +---------------------+
              |                                 |  per-case predictor |
              |                                 |  (VideoMAE motion + |
              |                                 |  material codebook  |
              |                                 |  → log-k, collision)|
              v                                 +----------+----------+
        spring graph                                       |
        + GS scaffold                                      v
              |                                 +---------------------+
              +-------------------------------> | differentiable warp |
                                                | spring-mass sim     |
                                                +----------+----------+
                                                           |
                                                           v
                                                   loss = track + chamfer
                                                          (+ render, prior,
                                                           smoothness)
```

Everything below describes the components that make up that diagram.

## Preprocessing

Two preprocessing pipelines coexist; both produce the same per-case
artifacts (`final_data.pkl`, `metadata.json`, `calibrate.pkl`, `split.json`,
Gaussian scaffolds, GT object/controller tracks) so the trainer is agnostic
to which one produced them.

1. **Multi-view PhysTwin baseline** — orchestrated by `process_data.py`.
   Stages live in `data_process/`: GroundedSAM2 (`segment.py`,
   `segment_util_image.py`), super-resolution + shape prior
   (`image_upscale.py`, `shape_prior.py`), Co-Tracker dense tracking
   (`dense_track.py`), point-cloud/mask processing
   (`data_process_pcd.py`, `data_process_mask.py`,
   `data_process_track.py`), alignment (`align.py`) and structure
   sampling (`data_process_sample.py`).

2. **Single-video pipeline** — orchestrated by
   `scripts/preprocess/single_video.sh`. Stages:
   1. Frame extraction (`data_process/prepare_single_video.py`).
   2. GroundedSAM2 video segmentation for object + hand
      (`data_process/segment_util_video.py`).
   3. Cupid 3D reconstruction from frame 0
      (`semantic/run_cupid_case.py`).
   4. Co-Tracker single-camera dense tracking
      (`data_process/dense_track_single.py`).
   5. Lifting 2D tracks to 3D via ray-cast against the Cupid mesh
      (`data_process/lift_tracks_single.py`).
   6. DINOv2 feature lifting onto visible Gaussians
      (`semantic/lift_dino_to_gaussian.py`).
   7. Writing the final dataset files
      (`semantic/prepare_final_data_case.py`).

The single-video pipeline runs across two conda environments
(`phystwin` and `cupid`); the multi-view one stays inside `phystwin`.

## Semantic / material chain

Downstream of preprocessing, the predictor consumes part-level
semantic and material signals:

- **DINOv2 features** lifted onto the Cupid Gaussians
  (`semantic/lift_dino_to_gaussian.py`), then propagated across
  symmetric parts (`semantic/gs_symmetry.py`).
- **Per-part material distribution** from a VLM
  (`semantic/extract_material_parts.py`, with per-case mappings in
  `semantic/case_to_material_*.json`) over 10 material classes.
- **GPT-4o numeric physics prior** (`semantic/query_physics_prior_gpt.py`)
  produces a per-part `(μ, σ, conf)` over `log_k`, persisted to
  `results/<case>/gpt_physics_prior.json`. The current trainer uses it
  only as a weak regularizer.

`semantic/material_param_dataset.py` packages all of this into a graph
sample per case: structure points, edge list, per-edge geometry
features, per-part material distribution, optional teacher `log_k` from
first-order per-edge optimization, and the GS scaffold.

## Predictor

`semantic/train_model_video_material_simple.py` is the final training
entry point. The model (`semantic/models.py`) is a compact, per-case
predictor with a frozen video backbone:

| Block | Source |
|---|---|
| Frozen VideoMAE → `z_motion` (128-d) | `train_models.load_video_frames` + VideoMAE in trainer |
| Material codebook (10×32) → per-part embedding | `MaterialCodebook` |
| Per-edge geometry MLP → `z_geo_enc` | `LocalGeometryEncoder` |
| Edge stiffness head → per-edge `log_k` | `EdgeLevelMaterialPhysics` / `PartLevelMaterialPhysics` |
| Global decoder → collision/friction/damping | `GlobalPhysicsDecoder` |

The decoder predicts a residual on `log_k` around the GPT prior, with a
soft clamp (`logk_soft_clamp`) so the network cannot drift arbitrarily
far from the LLM prior. Global params (collision elasticity / friction
for floor and object, drag/dashpot damping, collision distance) are
predicted by `GlobalPhysicsDecoder` from a global motion + material
pool.

## Simulator and rendering

`qqtt/` is the warp-based differentiable spring-mass simulator
inherited from PhysTwin:

- `qqtt/model/diff_simulator/spring_mass_warp.py` — `SpringMassSystemWarp`,
  a warp graph that integrates a mass-spring system with collision,
  drag and dashpot damping.
- `qqtt/engine/trainer_warp.py` — `InvPhyTrainerWarp`, the per-case
  rollout / loss container. The new trainer calls into it to render
  predicted parameters and compute track + chamfer losses, and to
  expose hooks for the new smoothness regularizer
  (`cfg.acc_weight`).
- `qqtt/engine/cma_optimize_warp.py` — CMA-ES baseline optimizer over
  spring stiffnesses (used to produce first-order teacher targets).

Rendering uses the local 3D Gaussian Splatting fork under
`gaussian_splatting/`. Per-case Gaussians are produced once
(`gs_train.py`, `gs_render.py`, `scripts/preprocess/gs_train.sh`); during physics training,
predicted node positions deform the Gaussians and an L1 render loss is
optionally applied via `gaussian_render_l1_loss`
(`semantic/train_models.py`).

## Losses

The trainer combines four families of losses:

1. **`sim_loss` (main signal)** — Warp rollout track L2 + chamfer
   between predicted and GT object point cloud over the full sequence.
2. **`render_loss` (optional)** — masked L1 between rendered and GT
   frame, weighted by `--lambda_render` (off by default; the
   smooth_fitall recipe sets it to 0).
3. **`phys_prior_loss`** — soft prior against the GPT
   `(μ, σ, conf)` per part (`compute_part_physics_prior_loss`) and a
   barrier on global collision/damping bounds
   (`compute_global_physics_prior_loss`).
4. **`acc_smooth` regularizer** — warp-side jerk smoothness on
   simulated trajectories (`cfg.acc_weight`, `--lambda_acc_smooth`).
   This is the anti-vibration term introduced after observing that
   per-scene fits would converge to oscillating equilibria.

The reference recipe `scripts/ours/train_all.sh` uses
`λ_track = λ_geo = 1`, `λ_render = 0`, `λ_phys_prior = 1e-3`,
`λ_acc_smooth = 1e-2`, with `--fit_all_frames` so the trainer fits
train + test frames jointly (per-case fitting setup, no held-out
generalization).

## Per-case fitting vs. per-edge optimization

Two reference points exist for every case:

- **Per-edge first-order optimization** — direct gradient descent on
  per-edge `log_k` and global params, no semantic model. Produces
  `experiments_optimization/<case>/`. Treated as the strong baseline
  for chamfer/track numbers in the paper and as a *teacher* signal
  when `--teacher_reg_weight > 0`.
- **Per-case predictor (this repo's "ours")** — same simulator, same
  losses, but parameters come from the network outputs. Run with
  `run_per_case_smooth_fitall.sh` (all 22 cases) or
  `run_ours_single_case.sh` (one case).

Baseline comparisons should be against the first-order per-edge
optimization, not against the cotracker pseudo-GT or against the
zero-shot mean prior.

## Repository layout

```
data_process/      preprocessing (multi-view + single-video)
semantic/          dataset, model, training & eval, semantic chain
qqtt/              warp differentiable spring-mass simulator
gaussian_splatting/ local 3DGS fork (training + rendering)
scripts/           pipeline wrappers (preproc + training + viz)
configs/, env_install/  config files and conda env setup
evaluate_*.py, evaluate.sh   evaluation (chamfer, track, render)
process_data.py    multi-view preprocessing orchestrator
```

Checkpoints live under `checkpoints/`; preprocessed cases under
`data/`; per-case optimization references under
`experiments_optimization/`; trained 3DGS scaffolds under
`gaussian_output/`; numeric and rendered results under `results/`.
