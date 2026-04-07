"""
Material physics models (Position-Invariant).

Architecture:
    Part-level material codebook + Local geometry → Physics parameters

Key design:
    - Learnable material codebook [M, mat_dim]: M material type codes
    - VLM-precomputed material_dist [K, M]: per-part material distribution
    - z_mat per part = material_dist @ codebook, shared by all edges in that part
    - Local variation from z_geo (topology, curvature, etc.) per edge

Models:
    - EdgeLevelMaterialPhysics: (z_geo_enc, z_sem, z_mat) → per-edge stiffness
    - PartLevelMaterialPhysics: z_mat only → per-part stiffness (no geometry)
"""

from typing import Optional

import torch
import torch.nn as nn


# ============================================================================
# Components
# ============================================================================

class LocalGeometryEncoder(nn.Module):
    """Encode position-invariant local geometry features (rest length, curvature, etc.)."""

    def __init__(self, input_dim: int = 11, output_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z_geo: torch.Tensor) -> torch.Tensor:
        return self.net(z_geo)


class MaterialCodebook(nn.Module):
    """
    Learnable material codebook [M, material_dim].

    M material type codes (e.g. fabric, leather, rubber, ...).
    Per-part material embedding: z_mat = material_dist [K, M] @ codes [M, D] → [K, D]
    """

    def __init__(self, num_materials: int = 10, material_dim: int = 64):
        super().__init__()
        self.codes = nn.Parameter(torch.randn(num_materials, material_dim) * 0.01)

    def forward(self, material_dist: torch.Tensor) -> torch.Tensor:
        """
        Args:
            material_dist: [K, M] per-part material distribution from VLM
        Returns:
            z_mat: [K, material_dim] per-part material embedding
        """
        return material_dist @ self.codes


class GlobalPhysicsDecoder(nn.Module):
    """Predict scene-level collision/friction/damping parameters from global semantic + material features."""

    def __init__(self, dino_dim: int = 1024, material_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dino_dim + material_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 7),  # 4 collision + 3 new params
        )

    def forward(self, z_sem_global: torch.Tensor, z_mat_global: torch.Tensor) -> dict:
        """
        Args:
            z_sem_global: [1, dino_dim] mean-pooled DINO semantic features
            z_mat_global: [1, material_dim] mean-pooled material embedding
        """
        out = self.net(torch.cat([z_sem_global, z_mat_global], dim=-1))  # [1, 7]
        # collision_dist: sigmoid → [0.01, 0.05]
        collision_dist = torch.sigmoid(out[:, 4]) * 0.04 + 0.01
        # dashpot_damping: sigmoid → [0, 200]
        dashpot_damping = torch.sigmoid(out[:, 5]) * 200.0
        # drag_damping: sigmoid → [0, 20]
        drag_damping = torch.sigmoid(out[:, 6]) * 20.0
        return {
            "collide_elas": out[:, 0],
            "collide_fric": out[:, 1],
            "collide_object_elas": out[:, 2],
            "collide_object_fric": out[:, 3],
            "collision_dist": collision_dist,
            "dashpot_damping": dashpot_damping,
            "drag_damping": drag_damping,
        }


class PhysicsDecoder(nn.Module):
    """Decode (z_geo_enc, z_sem, z_mat) → log(k) spring stiffness."""

    def __init__(self, geo_dim: int = 32, sem_dim: int = 1024, material_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(geo_dim + sem_dim + material_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_geo_enc: torch.Tensor, z_sem: torch.Tensor, z_mat: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_geo_enc, z_sem, z_mat], dim=-1))


class ControllerSpringDecoder(nn.Module):
    """
    Predict controller-object spring stiffness from the object endpoint's features.

    Each controller spring connects a hand/controller point to an object point.
    Stiffness depends on what material/region the hand is grasping.

    Input per controller spring:
        z_sem_point [C, sem_dim]  : DINO semantic feature of the object endpoint
        z_mat_point [C, mat_dim]  : material embedding of the object endpoint's part
        rest_length [C, 1]        : rest length of the spring

    Output:
        log_k [C, 1] : predicted log stiffness
    """

    def __init__(self, sem_dim: int = 1024, material_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sem_dim + material_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        z_sem_point: torch.Tensor,   # [C, sem_dim]
        z_mat_point: torch.Tensor,   # [C, mat_dim]
        rest_length: torch.Tensor,   # [C, 1]
    ) -> torch.Tensor:
        return self.net(torch.cat([z_sem_point, z_mat_point, rest_length], dim=-1))


# ============================================================================
# Models
# ============================================================================

class EdgeLevelMaterialPhysics(nn.Module):
    """
    Edge-level material physics model.

    Per part (computed once per case):
        material_dist [K, M] @ codebook [M, D] → z_mat_parts [K, D]

    Object springs (per edge):
        z_geo [E, 10]  → GeoEncoder → z_geo_enc [E, 32]
        z_sem [E, sem_dim]                                (per-edge DINO features)
        z_mat = z_mat_parts[edge_part_idx]       [E, D]   (same part = same z_mat)
        (z_geo_enc, z_sem, z_mat) → Decoder → log_k       [E, 1]

    Controller springs (per controller-object connection):
        z_sem_point [C, sem_dim]  : DINO feature of the object endpoint
        z_mat_point [C, mat_dim]  : material embedding of the object endpoint's part
        rest_length [C, 1]        : rest length
        → ControllerSpringDecoder → ctrl_log_k [C, 1]
    """

    def __init__(
        self,
        num_materials: int = 10,
        material_dim: int = 64,
        geo_input_dim: int = 11,
        geo_output_dim: int = 32,
        geo_hidden: int = 64,
        sem_dim: int = 1024,
        decoder_hidden: int = 64,
        # Global collision/friction
        dino_dim: int = 1024,
        global_hidden: int = 128,
    ):
        super().__init__()

        self.codebook = MaterialCodebook(num_materials, material_dim)

        self.geo_encoder = LocalGeometryEncoder(
            input_dim=geo_input_dim,
            output_dim=geo_output_dim,
            hidden_dim=geo_hidden,
        )

        self.decoder = PhysicsDecoder(
            geo_dim=geo_output_dim,
            sem_dim=sem_dim,
            material_dim=material_dim,
            hidden_dim=decoder_hidden,
        )

        self.ctrl_decoder = ControllerSpringDecoder(
            sem_dim=sem_dim,
            material_dim=material_dim,
            hidden_dim=decoder_hidden,
        )

        self.global_decoder = GlobalPhysicsDecoder(
            dino_dim=dino_dim,
            material_dim=material_dim,
            hidden_dim=global_hidden,
        )

        self.part_sem_projector = nn.Sequential(
            nn.LayerNorm(sem_dim),
            nn.Linear(sem_dim, material_dim),
            nn.SiLU(),
            nn.Linear(material_dim, material_dim),
        )
        self.part_material_head = nn.Linear(material_dim, num_materials)

    def forward(
        self,
        z_geo: torch.Tensor,              # [E, 10]
        z_sem: torch.Tensor,              # [E, sem_dim] per-edge DINO features
        material_dist: torch.Tensor,       # [K, M] per-part material distribution
        edge_part_idx: torch.Tensor,       # [E] part index per edge
        z_sem_global: torch.Tensor,        # [1, dino_dim]
        # Controller spring inputs (optional)
        ctrl_sem: Optional[torch.Tensor] = None,       # [C, sem_dim]
        ctrl_rest_length: Optional[torch.Tensor] = None,  # [C, 1]
        ctrl_part_idx: Optional[torch.Tensor] = None,  # [C] part index of object endpoint
    ) -> dict:
        # Part-level material embedding (computed once, broadcast to edges)
        z_mat_parts = self.codebook(material_dist)     # [K, mat_dim]
        z_mat = z_mat_parts[edge_part_idx]             # [E, mat_dim]

        # Per-edge geometry
        z_geo_enc = self.geo_encoder(z_geo)            # [E, 32]

        log_k = self.decoder(z_geo_enc, z_sem, z_mat)  # [E, 1]

        # Global material embedding (mean-pooled across parts)
        z_mat_global = z_mat_parts.mean(dim=0, keepdim=True)  # [1, mat_dim]

        result = {"log_k": log_k}

        # Controller springs
        if ctrl_sem is not None and ctrl_part_idx is not None:
            ctrl_z_mat = z_mat_parts[ctrl_part_idx]    # [C, mat_dim]
            ctrl_log_k = self.ctrl_decoder(ctrl_sem, ctrl_z_mat, ctrl_rest_length)
            result["ctrl_log_k"] = ctrl_log_k          # [C, 1]

        # Global collision/friction (always predicted)
        result.update(self.global_decoder(z_sem_global, z_mat_global))

        return result

    def predict_part_material_logits(self, part_features: torch.Tensor) -> torch.Tensor:
        part_latent = self.part_sem_projector(part_features)
        return self.part_material_head(part_latent)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_breakdown(self) -> dict:
        breakdown = {
            "codebook": sum(p.numel() for p in self.codebook.parameters()),
            "geo_encoder": sum(p.numel() for p in self.geo_encoder.parameters()),
            "decoder": sum(p.numel() for p in self.decoder.parameters()),
            "ctrl_decoder": sum(p.numel() for p in self.ctrl_decoder.parameters()),
            "global_decoder": sum(p.numel() for p in self.global_decoder.parameters()),
            "part_sem_projector": sum(p.numel() for p in self.part_sem_projector.parameters()),
            "part_material_head": sum(p.numel() for p in self.part_material_head.parameters()),
        }
        breakdown["total"] = self.count_parameters()
        return breakdown


class PartLevelMaterialPhysics(nn.Module):
    """
    Part-level material physics model (no per-edge variation).

    material_dist [K, M] @ codebook [M, D] → z_mat [K, D] → head → log_k [K, 1]
    Broadcast to edges via edge_part_idx.
    """

    def __init__(self, num_materials: int = 10, material_dim: int = 64, decoder_hidden: int = 64):
        super().__init__()

        self.codebook = MaterialCodebook(num_materials, material_dim)
        self.physics_head = nn.Sequential(
            nn.Linear(material_dim, decoder_hidden),
            nn.SiLU(),
            nn.Linear(decoder_hidden, decoder_hidden),
            nn.SiLU(),
            nn.Linear(decoder_hidden, 1),
        )
        self.part_sem_projector = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, material_dim),
            nn.SiLU(),
            nn.Linear(material_dim, material_dim),
        )
        self.part_material_head = nn.Linear(material_dim, num_materials)

    def forward(
        self,
        material_dist: torch.Tensor,    # [K, M]
        edge_part_idx: torch.Tensor,    # [E]
    ) -> torch.Tensor:
        z_mat = self.codebook(material_dist)           # [K, mat_dim]
        log_k_per_part = self.physics_head(z_mat)      # [K, 1]
        return log_k_per_part[edge_part_idx]           # [E, 1]

    def predict_part_material_logits(self, part_features: torch.Tensor) -> torch.Tensor:
        part_latent = self.part_sem_projector(part_features)
        return self.part_material_head(part_latent)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Factory & Utilities
# ============================================================================

def create_model(model_type: str = "edge_level", **kwargs) -> nn.Module:
    if model_type == "edge_level":
        return EdgeLevelMaterialPhysics(**kwargs)
    elif model_type == "part_level":
        return PartLevelMaterialPhysics(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def print_model_summary(model: nn.Module):
    print("\n" + "=" * 60)
    print("Model Summary")
    print("=" * 60)
    print(f"  Model: {model.__class__.__name__}")
    if hasattr(model, "get_parameter_breakdown"):
        for name, count in model.get_parameter_breakdown().items():
            print(f"  {name:20s}: {count:>10,} params")
    else:
        total = model.count_parameters() if hasattr(model, "count_parameters") else \
                sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  {'total':20s}: {total:>10,} params")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    print("Testing Models\n")

    E, K, M, D, C = 1000, 8, 10, 1024, 50
    z_geo = torch.randn(E, 11)
    z_sem = torch.randn(E, D)
    material_dist = torch.softmax(torch.randn(K, M), dim=1)
    edge_part_idx = torch.randint(0, K, (E,))
    z_sem_global = torch.randn(1, D)

    # Controller spring features
    ctrl_sem = torch.randn(C, D)
    ctrl_rest_length = torch.rand(C, 1) * 0.1
    ctrl_part_idx = torch.randint(0, K, (C,))

    print("1. EdgeLevelMaterialPhysics")
    m1 = EdgeLevelMaterialPhysics(num_materials=M, sem_dim=D, dino_dim=D)
    print_model_summary(m1)
    out = m1(z_geo, z_sem, material_dist, edge_part_idx, z_sem_global,
             ctrl_sem=ctrl_sem, ctrl_rest_length=ctrl_rest_length, ctrl_part_idx=ctrl_part_idx)
    print(f"   log_k: {out['log_k'].shape}")
    print(f"   ctrl_log_k: {out['ctrl_log_k'].shape}")
    print(f"   collide_elas: {out['collide_elas'].shape}")
    print(f"   collide_fric: {out['collide_fric'].shape}")
    print(f"   collision_dist: {out['collision_dist'].shape} val={out['collision_dist'].item():.4f}")
    print(f"   dashpot_damping: {out['dashpot_damping'].shape} val={out['dashpot_damping'].item():.2f}")
    print(f"   drag_damping: {out['drag_damping'].shape} val={out['drag_damping'].item():.2f}\n")

    print("2. PartLevelMaterialPhysics")
    m2 = PartLevelMaterialPhysics(num_materials=M)
    print_model_summary(m2)
    log_k = m2(material_dist, edge_part_idx)
    print(f"   log_k: {log_k.shape}\n")

    print("All tests passed!")
