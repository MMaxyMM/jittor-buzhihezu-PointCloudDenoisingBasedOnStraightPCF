"""Jittor implementation of the coupled and distance-aware StraightPCF stages.

The architecture is adapted from the MIT-licensed StraightPCF reference
implementation. See NOTICE for attribution and the paper citation.
"""

import os
from typing import Dict, List

import jittor as jt
import numpy as np
from jittor import nn

from ..data.asset import Asset
from .feature import Decoder, FeatureExtraction
from .spec import ModelSpec
from .vm import VelocityModule, patch_based_denoise


def _charbonnier_vector_loss(diff, eps: float = 1e-12):
    """Smooth L1-like loss for a 3D vector residual."""
    return jt.sqrt((diff ** 2).sum(dim=-1) + eps).mean()


def _charbonnier_scalar_loss(diff, eps: float = 1e-12):
    """Smooth L1-like loss for a scalar residual."""
    return jt.sqrt(diff ** 2 + eps).mean()


def _velocity_config(cfg: Dict) -> Dict:
    return {
        "frame_knn": cfg["frame_knn"],
        "num_train_points": cfg["num_train_points"],
        "dsm_sigma": cfg["dsm_sigma"],
        "feat_embedding_dim": cfg["velocity_embedding_dim"],
        "decoder_hidden_dim": cfg["decoder_hidden_dim"],
        "edge_aggregation": cfg.get("edge_aggregation", "mean"),
    }


def _build_velocity_modules(cfg: Dict, transform_config: Dict, checkpoint_key: str):
    checkpoint = cfg.get(checkpoint_key)
    if checkpoint and not os.path.isfile(checkpoint):
        raise FileNotFoundError(
            f"required initialization checkpoint does not exist: {checkpoint}. "
            f"Update {checkpoint_key} after completing the previous stage."
        )
    modules = nn.ModuleList()
    for _ in range(cfg["num_modules"]):
        module = VelocityModule(_velocity_config(cfg), transform_config)
        if checkpoint:
            module.load(checkpoint)
        modules.append(module)
    return modules


class _StraightPCFBase(ModelSpec):
    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        cfg = self.model_config
        self.frame_knn = cfg["frame_knn"]
        self.num_train_points = cfg["num_train_points"]
        self.dsm_sigma = cfg["dsm_sigma"]
        self.num_modules = cfg["num_modules"]
        self.total_iterations = cfg["total_iterations"]
        self.patch_size = cfg.get("patch_size", 1000)
        self.seed_k = cfg.get("seed_k", 6)
        self.seed_k_alpha = cfg.get("seed_k_alpha", 1)
        self.residual_alpha = float(cfg.get("residual_alpha", 1.0))

    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        pc_noisy_batch = batch["pc_noisy"]
        assert pc_noisy_batch.ndim == 3
        # predict_rounds: 多轮迭代降噪，将上一轮输出重新作为输入
        num_rounds = int(self.model_config.get("predict_rounds", 1))
        fusion = self.model_config.get("fusion_mode", "best")
        results = []
        for pc_noisy in pc_noisy_batch:
            pc_current = pc_noisy
            for _ in range(num_rounds):
                denoised = patch_based_denoise(
                    self, pc_current, self.patch_size, self.seed_k, self.seed_k_alpha,
                    fusion=fusion,
                )
                if denoised is None:
                    # patch 推理失败时回退上一步结果，保证输出完整
                    break
                pc_current = denoised
            # 与本地 alpha oracle 一致：在所有 patch/round 完成后，
            # 以原始 noisy 点云为基准统一缩放最终位移。
            pc_current = pc_noisy + self.residual_alpha * (pc_current - pc_noisy)
            denoised = pc_current.detach().numpy().astype(np.float32, copy=False)
            if denoised.shape != tuple(pc_noisy.shape):
                raise RuntimeError(
                    f"denoised shape {denoised.shape} does not match input {tuple(pc_noisy.shape)}"
                )
            results.append({"pc_denoised": denoised})
        return results

    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        results = []
        for asset in batch:
            if self.is_predict():
                results.append({"pc_noisy": asset.sampled_vertices_noisy})
                continue
            if asset.meta is None:
                raise RuntimeError(
                    "StraightPCF training requires train_cvm_network: true"
                )
            required = ("pc_noisy", "pc_clean", "seed_points_t", "original_time_step")
            missing = [key for key in required if key not in asset.meta]
            if missing:
                raise RuntimeError(f"StraightPCF patch metadata is missing: {missing}")
            results.append({key: asset.meta[key] for key in required})
        return results

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch["pc_noisy"].shape[-2]
        loss = self.get_supervised_loss(
            pc_clean=batch["pc_clean"].reshape(-1, patch_size, 3),
            pc_noisy=batch["pc_noisy"].reshape(-1, patch_size, 3),
            seed_points_t=batch["seed_points_t"].reshape(-1, 1, 3),
            original_time_step=batch["original_time_step"].reshape(-1),
        )
        return {"loss": loss}

    def execute(self, **kwargs) -> Dict:
        return self.training_step(**kwargs)


class CoupledVelocityModule(_StraightPCFBase):
    """VelocityModule stack with intermediate trajectory consistency."""

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        self.velocity_nets = _build_velocity_modules(
            self.model_config, transform_config, "init_velocity_ckpt"
        )
        self.consistency_weight = self.model_config.get("consistency_weight", 10.0)

    def get_supervised_loss(
        self, pc_clean, pc_noisy, seed_points_t, original_time_step
    ):
        batch_size, num_points, dims = pc_noisy.shape
        grad_target = pc_clean - pc_noisy
        time = original_time_step.reshape(batch_size, 1, 1)
        pc_current = time * pc_clean + (1.0 - time) * pc_noisy - seed_points_t

        direction_loss = jt.array(0.0)
        consistency_loss = jt.array(0.0)
        for module_idx, velocity_net in enumerate(self.velocity_nets):
            feat = velocity_net.encoder(pc_current)
            feature_dim = feat.shape[-1]
            pred_dir = velocity_net.decoder(
                c=feat.reshape(-1, feature_dim)
            ).reshape(batch_size, num_points, dims)
            direction_loss = direction_loss + _charbonnier_vector_loss(
                pred_dir - grad_target
            )

            pc_current = pc_current + ((1.0 - time) / self.num_modules) * pred_dir
            if module_idx < self.num_modules - 1:
                next_step = (
                    original_time_step * (self.num_modules - module_idx - 1)
                    + module_idx + 1
                ) / self.num_modules
                next_step = next_step.reshape(batch_size, 1, 1)
                target = (
                    next_step * pc_clean
                    + (1.0 - next_step) * pc_noisy
                    - seed_points_t
                )
                consistency_loss = consistency_loss + _charbonnier_vector_loss(
                    target - pc_current
                )

        return (
            direction_loss + self.consistency_weight * consistency_loss
        ) / self.dsm_sigma

    def denoise_langevin_dynamics(self, pcl_noisy):
        batch_size, num_points, dims = pcl_noisy.shape
        with jt.no_grad():
            pc_next = pcl_noisy.clone()
            for _ in range(self.total_iterations):
                for velocity_net in self.velocity_nets:
                    feat = velocity_net.encoder(pc_next)
                    feature_dim = feat.shape[-1]
                    pred_dir = velocity_net.decoder(
                        c=feat.reshape(-1, feature_dim)
                    ).reshape(batch_size, num_points, dims)
                    pc_next = pc_next + pred_dir / (
                        self.total_iterations * self.num_modules
                    )
        return pc_next, None


class StraightPCFModule(_StraightPCFBase):
    """Full coupled velocity stack plus learned DistanceModule."""

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)
        cfg = self.model_config
        cvm_checkpoint = cfg.get("init_cvm_ckpt")
        if not cvm_checkpoint or not os.path.isfile(cvm_checkpoint):
            raise FileNotFoundError(
                f"required CVM checkpoint does not exist: {cvm_checkpoint}. "
                "Train CoupledVelocityModule first and update init_cvm_ckpt."
            )
        cvm_config = dict(cfg)
        cvm_config["init_velocity_ckpt"] = None
        cvm = CoupledVelocityModule(cvm_config, transform_config)
        cvm.load(cvm_checkpoint)
        self.velocity_nets = cvm.velocity_nets

        embedding_dim = cfg.get("distance_embedding_dim", 128)
        self.distance_encoder = FeatureExtraction(
            k=self.frame_knn,
            input_dim=3,
            embedding_dim=embedding_dim,
            distance_estimation=True,
            aggregation=cfg.get("edge_aggregation", "mean"),
        )
        self.distance_decoder = Decoder(
            z_dim=embedding_dim,
            dim=3,
            out_dim=1,
            hidden_size=cfg["decoder_hidden_dim"],
        )
        self.finetune_weight = cfg.get("finetune_weight", 200.0)
        self._keep_velocity_eval_with_grad()

    def _keep_velocity_eval_with_grad(self):
        """Freeze BN/dropout state while retaining endpoint gradients.

        Jittor's Module.eval() also calls stop_grad() on all parameters, unlike
        PyTorch. Set only the module mode flags here, then explicitly keep the
        velocity parameters trainable.
        """
        for module in self.velocity_nets.modules():
            module.is_train = False
        for name, parameter in self.velocity_nets.named_parameters():
            # BN running statistics are state, not trainable affine parameters.
            # Jittor exposes them through named_parameters(), so keep them
            # stopped while allowing weights/biases to receive endpoint grads.
            if name.endswith("running_mean") or name.endswith("running_var"):
                parameter.stop_grad()
            else:
                parameter.start_grad()

    def train(self):
        # DummySystem calls model.train() at every epoch. Restore eval mode for
        # velocity BN/dropout after the recursive train-mode switch.
        super().train()
        self._keep_velocity_eval_with_grad()
        return self

    def _predict_distance(self, pc):
        batch_size, num_points, _ = pc.shape
        feat = self.distance_encoder(pc)
        return self.distance_decoder(
            c=feat.reshape(-1, feat.shape[-1]), B=batch_size, N=num_points
        ).reshape(batch_size, 1, 1)

    def get_supervised_loss(
        self, pc_clean, pc_noisy, seed_points_t, original_time_step
    ):
        batch_size, num_points, dims = pc_noisy.shape
        time = original_time_step.reshape(batch_size, 1, 1)
        pc_current = time * pc_clean + (1.0 - time) * pc_noisy
        distance_target = 1.0 - original_time_step

        pc_clean = pc_clean - seed_points_t
        pc_current = pc_current - seed_points_t
        pred_distance = self._predict_distance(pc_current)
        distance_loss = _charbonnier_scalar_loss(
            pred_distance.reshape(batch_size) - distance_target
        )

        for velocity_net in self.velocity_nets:
            feat = velocity_net.encoder(pc_current)
            pred_dir = velocity_net.decoder(
                c=feat.reshape(-1, feat.shape[-1])
            ).reshape(batch_size, num_points, dims)
            pc_current = pc_current + pred_distance * pred_dir / self.num_modules

        endpoint_loss = _charbonnier_vector_loss(pc_clean - pc_current)
        return (
            distance_loss + self.finetune_weight * endpoint_loss
        ) / self.dsm_sigma

    def denoise_langevin_dynamics(self, pcl_noisy):
        batch_size, num_points, dims = pcl_noisy.shape
        with jt.no_grad():
            pc_next = pcl_noisy.clone()
            pred_distance = self._predict_distance(pc_next)
            for _ in range(self.total_iterations):
                for velocity_net in self.velocity_nets:
                    feat = velocity_net.encoder(pc_next)
                    pred_dir = velocity_net.decoder(
                        c=feat.reshape(-1, feat.shape[-1])
                    ).reshape(batch_size, num_points, dims)
                    pc_next = pc_next + pred_distance * pred_dir / (
                        self.total_iterations * self.num_modules
                    )
        return pc_next, None
