"""
Material physics models (position invariant).

The material prior is represented at part level and can be encoded in two ways:
    - `codebook`: material_dist @ learnable codes
    - `direct_mlp`: material_dist -> MLP -> prior latent

Both edge-level and part-level physics models consume the same encoded prior.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_K_MIN: float = math.log(1e3)
LOG_K_MAX: float = math.log(1e5)


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
    def __init__(self, num_materials: int = 10, material_dim: int = 64):
        super().__init__()
        self.codes = nn.Parameter(torch.randn(num_materials, material_dim) * 0.01)

    def forward(self, material_dist: torch.Tensor) -> torch.Tensor:
        return material_dist @ self.codes


class DirectMaterialEncoder(nn.Module):
    def __init__(self, num_materials: int = 10, material_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_materials, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, material_dim),
        )

    def forward(self, material_dist: torch.Tensor) -> torch.Tensor:
        return self.net(material_dist)


class MaterialPriorEncoder(nn.Module):
    def __init__(
        self,
        num_materials: int = 10,
        material_dim: int = 64,
        encoder_type: str = "codebook",
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        if encoder_type == "codebook":
            self.codebook = MaterialCodebook(num_materials, material_dim)
        elif encoder_type == "direct_mlp":
            self.direct_mlp = DirectMaterialEncoder(
                num_materials=num_materials,
                material_dim=material_dim,
                hidden_dim=hidden_dim,
            )
        else:
            raise ValueError(f"Unknown prior encoder type: {encoder_type}")

    def forward(self, material_dist: torch.Tensor) -> torch.Tensor:
        if self.encoder_type == "codebook":
            return self.codebook(material_dist)
        return self.direct_mlp(material_dist)


class PartFeaturePriorHead(nn.Module):
    """Predict VLM-style material prior logits from part-level semantic features."""

    def __init__(self, sem_dim: int = 1024, hidden_dim: int = 128, num_materials: int = 10):
        super().__init__()
        self.projector = nn.Sequential(
            nn.LayerNorm(sem_dim),
            nn.Linear(sem_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.head = nn.Linear(hidden_dim, num_materials)

    def forward(self, part_features: torch.Tensor) -> torch.Tensor:
        return self.head(self.projector(part_features))


class ContinuousMaterialPhysicsPrior(nn.Module):
    """Predict a continuous log-stiffness prior from VLM material distributions."""

    def __init__(self, num_materials: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(num_materials, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.log_sigma_head = nn.Linear(hidden_dim, 1)

    def forward(self, material_dist: torch.Tensor) -> dict:
        h = self.backbone(material_dist)
        mu = self.mu_head(h)
        log_sigma = self.log_sigma_head(h).clamp(min=-2.5, max=2.0)
        return {"mu": mu, "log_sigma": log_sigma}


class PartSetAttentionPool(nn.Module):
    """Pool part latents with a global-semantic-conditioned attention query."""

    def __init__(self, query_dim: int = 1024, value_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.query_proj = nn.Sequential(
            nn.LayerNorm(query_dim),
            nn.Linear(query_dim, hidden_dim),
            nn.SiLU(),
        )
        self.key_proj = nn.Sequential(
            nn.LayerNorm(value_dim),
            nn.Linear(value_dim, hidden_dim),
            nn.SiLU(),
        )
        self.value_proj = nn.Sequential(
            nn.LayerNorm(value_dim),
            nn.Linear(value_dim, hidden_dim),
            nn.SiLU(),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim + value_dim, value_dim),
            nn.LayerNorm(value_dim),
            nn.SiLU(),
        )
        self.scale = hidden_dim ** -0.5

    def forward(self, z_sem_global: torch.Tensor, z_mat_parts: torch.Tensor) -> torch.Tensor:
        if z_mat_parts.ndim != 2 or z_mat_parts.shape[0] == 0:
            raise ValueError("z_mat_parts must have shape [K, D] with K > 0")
        q = self.query_proj(z_sem_global)
        k = self.key_proj(z_mat_parts)
        v = self.value_proj(z_mat_parts)
        attn = torch.softmax((q @ k.transpose(0, 1)) * self.scale, dim=-1)
        pooled = attn @ v
        mean_feat = z_mat_parts.mean(dim=0, keepdim=True)
        return self.out_proj(torch.cat([pooled, mean_feat], dim=-1))


class GlobalPhysicsDecoder(nn.Module):
    """Predict scene-level collision/friction/damping from global semantic + pooled material + geometry features."""

    GEO_STATS_DIM = 6  # [mean_rest_length, std_rest_length, log_num_nodes, log_num_edges, bbox_diag, mean_degree]
    GEO_STATS_ENC_DIM = 32

    def __init__(self, dino_dim: int = 1024, material_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.pool = PartSetAttentionPool(query_dim=dino_dim, value_dim=material_dim, hidden_dim=hidden_dim)
        self.geo_stats_encoder = nn.Sequential(
            nn.Linear(self.GEO_STATS_DIM, 32),
            nn.SiLU(),
            nn.Linear(32, self.GEO_STATS_ENC_DIM),
        )
        backbone_in = dino_dim + material_dim + material_dim + self.GEO_STATS_ENC_DIM
        self.shared = nn.Sequential(
            nn.Linear(backbone_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.contact_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 4),
        )
        self.dynamics_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

    def forward(self, z_sem_global: torch.Tensor, z_mat_parts: torch.Tensor, geo_stats: Optional[torch.Tensor] = None) -> dict:
        z_mat_attn = self.pool(z_sem_global, z_mat_parts)
        z_mat_mean = z_mat_parts.mean(dim=0, keepdim=True)
        if geo_stats is not None:
            z_geo_g = self.geo_stats_encoder(geo_stats.view(1, -1).to(z_sem_global.device))
        else:
            z_geo_g = torch.zeros(1, self.GEO_STATS_ENC_DIM, device=z_sem_global.device, dtype=z_sem_global.dtype)
        h = self.shared(torch.cat([z_sem_global, z_mat_attn, z_mat_mean, z_geo_g], dim=-1))

        contact = self.contact_head(h)
        dynamics = self.dynamics_head(h)
        collide_elas = torch.sigmoid(contact[:, 0])
        collide_fric = torch.sigmoid(contact[:, 1]) * 2.0
        collide_object_elas = torch.sigmoid(contact[:, 2])
        collide_object_fric = torch.sigmoid(contact[:, 3]) * 2.0
        collision_dist = torch.sigmoid(dynamics[:, 0]) * 0.04 + 0.01
        dashpot_damping = torch.sigmoid(dynamics[:, 1]) * 200.0
        drag_damping = torch.sigmoid(dynamics[:, 2]) * 20.0
        return {
            "collide_elas": collide_elas,
            "collide_fric": collide_fric,
            "collide_object_elas": collide_object_elas,
            "collide_object_fric": collide_object_fric,
            "collision_dist": collision_dist,
            "dashpot_damping": dashpot_damping,
            "drag_damping": drag_damping,
        }


class PhysicsDecoder(nn.Module):
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


class DeeperPartStiffnessHead(nn.Module):
    def __init__(self, sem_dim: int = 1024, material_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sem_dim + material_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, part_features: torch.Tensor, z_mat: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([part_features, z_mat], dim=-1))


class ControllerSpringDecoder(nn.Module):
    def __init__(self, sem_dim: int = 1024, material_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sem_dim + material_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_sem_point: torch.Tensor, z_mat_point: torch.Tensor, rest_length: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_sem_point, z_mat_point, rest_length], dim=-1))


class EdgeLevelMaterialPhysics(nn.Module):
    def __init__(
        self,
        num_materials: int = 10,
        material_dim: int = 64,
        prior_encoder_type: str = "codebook",
        prior_hidden_dim: int = 128,
        geo_input_dim: int = 11,
        geo_output_dim: int = 32,
        geo_hidden: int = 64,
        sem_dim: int = 1024,
        decoder_hidden: int = 64,
        dino_dim: int = 1024,
        global_hidden: int = 256,
    ):
        super().__init__()
        self.prior_encoder_type = prior_encoder_type
        self.prior_encoder = MaterialPriorEncoder(
            num_materials=num_materials,
            material_dim=material_dim,
            encoder_type=prior_encoder_type,
            hidden_dim=prior_hidden_dim,
        )
        self.codebook = getattr(self.prior_encoder, "codebook", None)
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
            hidden_dim=max(128, decoder_hidden),
        )
        self.global_decoder = GlobalPhysicsDecoder(
            dino_dim=dino_dim,
            material_dim=material_dim,
            hidden_dim=global_hidden,
        )
        self.part_prior_head = PartFeaturePriorHead(
            sem_dim=sem_dim,
            hidden_dim=material_dim,
            num_materials=num_materials,
        )
        self.phys_prior_head = ContinuousMaterialPhysicsPrior(
            num_materials=num_materials,
            hidden_dim=max(64, material_dim),
        )

    def predict_part_material_logits(self, part_features: torch.Tensor) -> torch.Tensor:
        return self.part_prior_head(part_features)

    def predict_part_material_dist(self, part_features: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.predict_part_material_logits(part_features), dim=-1)

    def predict_part_physics_prior(self, material_dist: torch.Tensor) -> dict:
        return self.phys_prior_head(material_dist)

    def select_material_prior(self, material_dist: torch.Tensor, part_features: Optional[torch.Tensor] = None, prior_source: str = "vlm") -> torch.Tensor:
        if prior_source == "vlm":
            return material_dist
        if prior_source == "distilled":
            if part_features is None:
                raise ValueError("part_features are required when prior_source='distilled'")
            return self.predict_part_material_dist(part_features)
        raise ValueError(f"Unknown prior source: {prior_source}")

    def encode_material_prior(self, material_dist: torch.Tensor, part_features: Optional[torch.Tensor] = None, prior_source: str = "vlm") -> torch.Tensor:
        material_prior = self.select_material_prior(material_dist, part_features, prior_source)
        return self.prior_encoder(material_prior)

    def forward(
        self,
        z_geo: torch.Tensor,
        z_sem: torch.Tensor,
        material_dist: torch.Tensor,
        edge_part_idx: torch.Tensor,
        z_sem_global: torch.Tensor,
        part_features: Optional[torch.Tensor] = None,
        prior_source: str = "vlm",
        ctrl_sem: Optional[torch.Tensor] = None,
        ctrl_rest_length: Optional[torch.Tensor] = None,
        ctrl_part_idx: Optional[torch.Tensor] = None,
        geo_stats: Optional[torch.Tensor] = None,
    ) -> dict:
        z_mat_parts = self.encode_material_prior(material_dist, part_features, prior_source)
        z_mat = z_mat_parts[edge_part_idx]
        z_geo_enc = self.geo_encoder(z_geo)

        raw_log_k = self.decoder(z_geo_enc, z_sem, z_mat)
        log_k = torch.sigmoid(raw_log_k) * (LOG_K_MAX - LOG_K_MIN) + LOG_K_MIN

        result = {"log_k": log_k}
        if ctrl_sem is not None and ctrl_part_idx is not None and ctrl_rest_length is not None and ctrl_part_idx.numel() > 0:
            ctrl_z_mat = z_mat_parts[ctrl_part_idx]
            raw_ctrl_log_k = self.ctrl_decoder(ctrl_sem, ctrl_z_mat, ctrl_rest_length)
            result["ctrl_log_k"] = torch.sigmoid(raw_ctrl_log_k) * (LOG_K_MAX - LOG_K_MIN) + LOG_K_MIN

        result.update(self.global_decoder(z_sem_global, z_mat_parts, geo_stats=geo_stats))
        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_breakdown(self) -> dict:
        breakdown = {
            "prior_encoder": sum(p.numel() for p in self.prior_encoder.parameters()),
            "geo_encoder": sum(p.numel() for p in self.geo_encoder.parameters()),
            "decoder": sum(p.numel() for p in self.decoder.parameters()),
            "ctrl_decoder": sum(p.numel() for p in self.ctrl_decoder.parameters()),
            "global_decoder": sum(p.numel() for p in self.global_decoder.parameters()),
            "part_prior_head": sum(p.numel() for p in self.part_prior_head.parameters()),
        }
        breakdown["total"] = self.count_parameters()
        return breakdown


class PartLevelMaterialPhysics(nn.Module):
    """Part-level material physics model with global prediction and controller-spring prediction."""

    def __init__(
        self,
        num_materials: int = 10,
        material_dim: int = 64,
        prior_encoder_type: str = "codebook",
        prior_hidden_dim: int = 128,
        sem_dim: int = 1024,
        decoder_hidden: int = 256,
        global_hidden: int = 256,
    ):
        super().__init__()
        self.prior_encoder_type = prior_encoder_type
        self.prior_encoder = MaterialPriorEncoder(
            num_materials=num_materials,
            material_dim=material_dim,
            encoder_type=prior_encoder_type,
            hidden_dim=prior_hidden_dim,
        )
        self.codebook = getattr(self.prior_encoder, "codebook", None)
        self.physics_head = DeeperPartStiffnessHead(
            sem_dim=sem_dim,
            material_dim=material_dim,
            hidden_dim=decoder_hidden,
        )
        self.ctrl_decoder = ControllerSpringDecoder(
            sem_dim=sem_dim,
            material_dim=material_dim,
            hidden_dim=max(128, decoder_hidden // 2),
        )
        self.global_decoder = GlobalPhysicsDecoder(
            dino_dim=sem_dim,
            material_dim=material_dim,
            hidden_dim=global_hidden,
        )
        self.part_prior_head = PartFeaturePriorHead(
            sem_dim=sem_dim,
            hidden_dim=material_dim,
            num_materials=num_materials,
        )
        self.phys_prior_head = ContinuousMaterialPhysicsPrior(
            num_materials=num_materials,
            hidden_dim=max(64, material_dim),
        )

    def predict_part_material_logits(self, part_features: torch.Tensor) -> torch.Tensor:
        return self.part_prior_head(part_features)

    def predict_part_material_dist(self, part_features: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.predict_part_material_logits(part_features), dim=-1)

    def predict_part_physics_prior(self, material_dist: torch.Tensor) -> dict:
        return self.phys_prior_head(material_dist)

    def select_material_prior(self, material_dist: torch.Tensor, part_features: Optional[torch.Tensor] = None, prior_source: str = "vlm") -> torch.Tensor:
        if prior_source == "vlm":
            return material_dist
        if prior_source == "distilled":
            if part_features is None:
                raise ValueError("part_features are required when prior_source='distilled'")
            return self.predict_part_material_dist(part_features)
        raise ValueError(f"Unknown prior source: {prior_source}")

    def encode_material_prior(self, material_dist: torch.Tensor, part_features: Optional[torch.Tensor] = None, prior_source: str = "vlm") -> torch.Tensor:
        material_prior = self.select_material_prior(material_dist, part_features, prior_source)
        return self.prior_encoder(material_prior)

    def forward(
        self,
        part_features: torch.Tensor,
        material_dist: torch.Tensor,
        edge_part_idx: torch.Tensor,
        z_sem_global: torch.Tensor,
        prior_source: str = "vlm",
        ctrl_sem: Optional[torch.Tensor] = None,
        ctrl_rest_length: Optional[torch.Tensor] = None,
        ctrl_part_idx: Optional[torch.Tensor] = None,
        geo_stats: Optional[torch.Tensor] = None,
    ) -> dict:
        z_mat_parts = self.encode_material_prior(material_dist, part_features, prior_source)
        raw_log_k_per_part = self.physics_head(part_features, z_mat_parts)
        log_k_per_part = torch.sigmoid(raw_log_k_per_part) * (LOG_K_MAX - LOG_K_MIN) + LOG_K_MIN

        result = {"log_k": log_k_per_part[edge_part_idx]}
        if ctrl_sem is not None and ctrl_part_idx is not None and ctrl_rest_length is not None and ctrl_part_idx.numel() > 0:
            ctrl_z_mat = z_mat_parts[ctrl_part_idx]
            raw_ctrl_log_k = self.ctrl_decoder(ctrl_sem, ctrl_z_mat, ctrl_rest_length)
            result["ctrl_log_k"] = torch.sigmoid(raw_ctrl_log_k) * (LOG_K_MAX - LOG_K_MIN) + LOG_K_MIN

        result.update(self.global_decoder(z_sem_global, z_mat_parts, geo_stats=geo_stats))
        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_parameter_breakdown(self) -> dict:
        breakdown = {
            "prior_encoder": sum(p.numel() for p in self.prior_encoder.parameters()),
            "physics_head": sum(p.numel() for p in self.physics_head.parameters()),
            "ctrl_decoder": sum(p.numel() for p in self.ctrl_decoder.parameters()),
            "global_decoder": sum(p.numel() for p in self.global_decoder.parameters()),
            "part_prior_head": sum(p.numel() for p in self.part_prior_head.parameters()),
        }
        breakdown["total"] = self.count_parameters()
        return breakdown


def create_model(model_type: str = "edge_level", **kwargs) -> nn.Module:
    if model_type == "edge_level":
        return EdgeLevelMaterialPhysics(**kwargs)
    if model_type == "part_level":
        return PartLevelMaterialPhysics(**kwargs)
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
        total = model.count_parameters() if hasattr(model, "count_parameters") else sum(p.numel() for p in model.parameters() if p.requires_grad)
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
    part_features = torch.randn(K, D)
    ctrl_sem = torch.randn(C, D)
    ctrl_rest_length = torch.rand(C, 1) * 0.1
    ctrl_part_idx = torch.randint(0, K, (C,))

    print("1. EdgeLevelMaterialPhysics")
    m1 = EdgeLevelMaterialPhysics(num_materials=M, sem_dim=D, dino_dim=D)
    print_model_summary(m1)
    out = m1(
        z_geo,
        z_sem,
        material_dist,
        edge_part_idx,
        z_sem_global,
        part_features=part_features,
        ctrl_sem=ctrl_sem,
        ctrl_rest_length=ctrl_rest_length,
        ctrl_part_idx=ctrl_part_idx,
    )
    print(f"   log_k: {out['log_k'].shape}")
    print(f"   ctrl_log_k: {out['ctrl_log_k'].shape}")
    print(f"   collide_elas: {out['collide_elas'].shape}")
    print(f"   collision_dist: {out['collision_dist'].shape} val={out['collision_dist'].item():.4f}")

    print("2. PartLevelMaterialPhysics")
    m2 = PartLevelMaterialPhysics(num_materials=M, sem_dim=D)
    print_model_summary(m2)
    out2 = m2(
        part_features,
        material_dist,
        edge_part_idx,
        z_sem_global,
        ctrl_sem=ctrl_sem,
        ctrl_rest_length=ctrl_rest_length,
        ctrl_part_idx=ctrl_part_idx,
    )
    print(f"   log_k: {out2['log_k'].shape}")
    print(f"   ctrl_log_k: {out2['ctrl_log_k'].shape}")
    print(f"   drag_damping: {out2['drag_damping'].shape} val={out2['drag_damping'].item():.2f}\n")

    print("All tests passed!")
