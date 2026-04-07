# Single-View Material Physics Pipeline - Implementation Status

**Date**: 2025-03-09
**Branch**: `phys_space`

---

## Pipeline Overview

```
Step 1: Cupid        -> gaussians + pose
Step 2: DINO Lift    -> visible gaussian features
Step 3: (in Step 2)  -> feature visualization
Step 4: Symmetry     -> propagate to invisible gaussians
Step 5: Material     -> part segmentation + material distribution
```

---

## Implemented Files

### Step 1: Cupid Runner
- `semantic/run_cupid_case.py` - Run Cupid reconstruction
- `scripts/run_cupid.sh` - Batch script

**Output**: `results/[case]/cupid/`
- `gaussians.pt`, `pose.json`, `input_masked.png`

### Step 2 & 3: DINO Feature Lifting
- `semantic/lift_dino_to_gaussian.py` - Lift DINOv2 features to visible gaussians
- `scripts/run_lift_dino.sh` - Batch script

**Output**: `results/[case]/gaussian_vis/`
- `gaussians_vis.pt` - Gaussians with `feat_sem`, `feat_valid`, `feat_visible`
- `visible_mask.pt`, `dino_feat_vis.pt`
- `feature_render_pca.png` - PCA visualization

### Step 4: Symmetry Propagation
- `semantic/gs_symmetry.py` - Propagate features via symmetry
- `scripts/run_sym.sh` - Batch script

**Output**: `results/[case]/gaussian_vis/`
- `gaussians_vis_sym.pt` - With `feat_sym_conf` added
- `symmetry_plane.json` - Estimated symmetry plane
- `symmetry_match.pt` - Correspondence info
- `feature_render_sym_pca.png`

### Step 5: Material & Part Extraction
- `semantic/extract_material_parts.py` - SAM + CLIP based extraction
- `scripts/run_material.sh` - Batch script

**Output**: `results/[case]/train/`
- `part_masks.pt` - Part assignments `[N]`, soft probs `[N, K]`
- `part_features.pt` - Aggregated semantic features `[K, D]`
- `material_distribution.pt` - CLIP probabilities `[K, M]`
- `material_embeddings.pt` - Mixture embeddings `[K, embed_dim]`
- `train_ready.pt` - Combined data for training
- `parts_viz.png`

---

## Usage

```bash
# Full pipeline for single case
bash scripts/run_cupid.sh single_lift_cloth
bash scripts/run_lift_dino.sh single_lift_cloth
bash scripts/run_sym.sh single_lift_cloth
bash scripts/run_material.sh single_lift_cloth

# Or batch all cases
bash scripts/run_cupid.sh
bash scripts/run_lift_dino.sh
bash scripts/run_sym.sh
bash scripts/run_material.sh
```

### Environment Variables

```bash
# Step 5 options
SKIP_SAM=1 bash scripts/run_material.sh    # Use object mask as single part
SKIP_CLIP=1 bash scripts/run_material.sh   # Use uniform material distribution
SAM_CHECKPOINT=/path/to/sam.pth bash scripts/run_material.sh
```

---

## Data Format

### Gaussian Dict (standardized)
```python
{
    "xyz": [N, 3],
    "scale": [N, 3],
    "rotation": [N, 4],
    "opacity": [N, 1],
    "color": [N, 1, 3] or [N, 3],
    "aabb": [...],
    "init_params": {...},

    # Semantic fields (added by pipeline)
    "feat_sem": [N, D],        # D=1024 for dinov2_vitl14
    "feat_visible": [N, 1],    # 1.0 if visible from input view
    "feat_valid": [N, 1],      # confidence of feature validity
    "feat_sym_conf": [N, 1],   # symmetry propagation confidence
}
```

### train_ready.pt
```python
{
    # Gaussian data
    "xyz": [N, 3],
    "feat_sem": [N, D],
    "feat_valid": [N, 1],
    "feat_sym_conf": [N, 1],

    # Part assignments
    "part_assignments": [N],      # int, -1 for unassigned
    "part_probs": [N, K],         # soft assignment
    "num_parts": K,

    # Part-level features
    "part_features": [K, D],
    "part_counts": [K],

    # Material info
    "material_distributions": [K, M],   # M=20 default materials
    "material_embeddings": [K, embed_dim],
    "material_names": [M],
}
```

---

## Default Materials (20 classes)

```python
[
    "fabric", "cloth", "leather", "rubber", "plastic",
    "metal", "wood", "paper", "foam", "cotton",
    "silk", "wool", "nylon", "polyester", "denim",
    "canvas", "felt", "velvet", "fur", "rope",
]
```

---

## Dependencies

- **Step 1**: Cupid environment (`conda run -n cupid`)
- **Step 2**: DINOv2 (`torch.hub.load("facebookresearch/dinov2")`)
- **Step 4**: scipy, numpy
- **Step 5**:
  - SAM (`segment_anything`) - optional, fallback to single part
  - CLIP (`pip install git+https://github.com/openai/CLIP.git`) - optional, fallback to uniform

---

## TODO / Debug Notes

### Pending
- [ ] Test full pipeline end-to-end
- [ ] Verify SAM checkpoint loading paths
- [ ] Test CLIP material classification accuracy
- [ ] Connect `train_ready.pt` to downstream `train_paramnet_konly.py`

### Known Issues
- SAM checkpoint path may need adjustment per machine
- CLIP temperature scaling (100) may need tuning

### Next Steps
1. Run pipeline on test cases
2. Visualize part segmentation quality
3. Check material distribution reasonableness
4. Integrate with physics training loop

---

## Related Docs

- `SINGLE_VIEW_MATERIAL_PHYSICS_PLAN.md` - Original pipeline design
- `semantic/METHOD_SUMMARY_AND_COMPARISON.md` - Training method comparison
