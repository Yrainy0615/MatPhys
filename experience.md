# Current Experiment Notes

## Scope

This note summarizes the current observations from reviewing and testing the
`semantic/train_models.py` pipeline against the existing PhysTwin optimization
baseline.

## Code-Level Conclusions

- The main repository training entry still follows the original PhysTwin
  per-scene fitting pipeline.
- The semantic/material-aware branch is currently implemented in
  `semantic/train_models.py` as a separate predictor-based training path.
- Before the recent fix, global collision parameters predicted by the model
  were not effectively trained, because gradients were only routed back through
  spring stiffness.
- This has now been partially corrected:
  - Warp gradients for `collide_elas`, `collide_fric`,
    `collide_object_elas`, and `collide_object_fric` are now routed back to the
    model.
  - `collision_dist`, `drag_damping`, and `dashpot_damping` still do not have
    direct simulator gradients in the current implementation, so they are only
    weakly supervised with teacher regression.

## Single-Case Fitting Observation

### Setting

- Case: `single_lift_zebra`
- Model: `edge_level`
- Tested both with and without render loss

### Result 1: With render loss

Observed behavior:

- Total loss stayed around `0.15`
- Render term dominated the objective
- Track and geometry terms were much smaller

Interpretation:

- The current mask-projection render loss is the main optimization bottleneck
  in single-case fitting.
- In this setup, the predictor is not obviously failing because of lack of
  capacity alone; it is mostly being slowed down by an overly dominant render
  objective.

### Result 2: Without render loss (`--lambda_render 0`)

Observed behavior:

- Training quickly dropped to around `2.1e-4` by about epoch 29
- Example decomposition:
  - `track ~= 8.0e-5`
  - `geo ~= 1.3e-4`

Interpretation:

- The model does have nontrivial single-case fitting ability on physics losses.
- The major failure mode in the earlier setting was indeed the render term.
- However, the predictor still underfits relative to per-scene optimization.

### Result 3: Teacher warm-start + part-level KL consistency

Observed behavior on `single_lift_zebra`:

- Test total loss: about `2.5e-5`
- Track loss: about `1.0e-5`
- Geometry loss: about `1.4e-5`

Interpretation:

- This is a large improvement over the earlier physics-only no-warm-start run.
- The current design now reaches roughly the same scale as the zero-order
  optimization baseline on this single-case fitting test.
- It still does not match the first-order per-scene optimization oracle, but
  the remaining gap is now much smaller and looks like a refinement problem,
  not a complete optimization failure.

## Baseline Comparison

For the same case (`single_lift_zebra`), existing logs show:

- CMA zero-order optimization best error: about `2.45e-5`
- First-order PhysTwin optimization best loss: about `8.88e-6`

Compared with the predictor:

- Physics-only single-case predictor fit is around `2.1e-4`
- This is much better than the render-dominated `0.15` regime
- But still clearly worse than both optimization baselines
- With teacher warm-start and part-level KL consistency, the predictor improves
  to about `2.5e-5`, which is close to the zero-order baseline and still above
  the first-order baseline

## Current Interpretation

The current evidence suggests:

- The predictor is not completely broken.
- The render loss design is currently too strong and too coarse for this stage.
- With warm-start and part-level KL consistency, the structured predictor can
  approach zero-order optimization quality on a single case.
- The remaining gap is now mainly versus first-order per-scene optimization.

This means the current bottleneck is likely a combination of:

- optimization difficulty
- structured predictor capacity limits
- weak supervision for some global parameters
- lack of a good warm-start for per-edge stiffness prediction

## Implication For Generalization Experiments

Since the predictor has not yet matched strong single-case fitting, weak
performance or weak differences in train/test experiments should not yet be
interpreted as evidence about generalization quality alone.

At this stage:

- single-case capacity must be improved first
- then train/test split conclusions will be more meaningful

## Implemented Next Step

A teacher warm-start mechanism has been added to `semantic/train_models.py`.

Design:

- It applies a weak per-edge teacher loss on `teacher_logk`
- It is intended only for early training
- The weight decays linearly to zero after a user-specified number of epochs

Relevant flags:

- `--lambda_teacher_warmstart`
- `--teacher_warmstart_epochs`

In addition, a part-level semantic consistency loss has been added.

Design:

- Use `part_features` from the semantic/material preprocessing pipeline
- Predict a part-level material distribution from semantic part features
- Apply KL divergence against the VLM material distribution

This is intended to make part semantic features more material-aware and more
stable across optimization, without forcing raw DINO features to match exactly.

Relevant flags:

- `--lambda_part_kl`
- `--part_kl_start_epoch`

Suggested first trial:

```bash
python semantic/train_models.py \
  --case_name single_lift_zebra \
  --model_type edge_level \
  --epochs 100 \
  --batch_size 1 \
  --lambda_render 0 \
  --lambda_teacher_warmstart 0.1 \
  --teacher_warmstart_epochs 20 \
  --lambda_part_kl 0.02 \
  --part_kl_start_epoch 20 \
  --save_dir checkpoints/single_case_single_lift_zebra_warmstart
```

## Recommended Immediate Next Experiments

1. Run the same single-case fitting experiment with teacher warm-start.
2. Compare convergence speed and final physics-only loss against the
   no-warm-start run.
3. Move on to multi-case train/test generalization, now that the single-case
   fitting ceiling is no longer obviously the main bottleneck.
4. Prefer stronger splits such as leave-one-identity-out, not only random 8:2.
