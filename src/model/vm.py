from math import ceil
from typing import Dict, List

import jittor as jt
import numpy as np

from .feature import FeatureExtraction, Decoder
from .spec import ModelSpec

from ..data.asset import Asset

def get_random_indices(n, m):
    assert m < n
    idx = np.random.permutation(n)[:m]
    return jt.array(idx).int32()

class VelocityModule(ModelSpec):

    def __init__(self, model_config, transform_config):
        super().__init__(model_config, transform_config)

        cfg = self.model_config
        # geometry
        self.frame_knn = cfg['frame_knn']
        self.num_train_points = cfg['num_train_points']

        # score-matching
        self.dsm_sigma = cfg['dsm_sigma']

        # networks
        self.encoder = FeatureExtraction(
            k=self.frame_knn,
            input_dim=3,
            embedding_dim=cfg['feat_embedding_dim'],
            aggregation=cfg.get('edge_aggregation', 'mean'),
        )

        self.decoder = Decoder(
            z_dim=self.encoder.embedding_dim,
            dim=3,
            out_dim=3,
            hidden_size=cfg['decoder_hidden_dim'],
        )

    def get_supervised_loss(self, pc_noisy, pc_mix, pc_clean):
        """
        pcl_noisy: (B, N, 3)
        pcl_clean: (B, N, 3)
        """
        B, N_noisy, d = pc_mix.shape

        pnt_idx = get_random_indices(N_noisy, self.num_train_points)

        # Feature extraction
        feat = self.encoder(pc_mix)  # (B, N, F)
        F_dim = feat.shape[2]

        # gather
        feat = feat[:, pnt_idx, :]
        pc_noisy = pc_noisy[:, pnt_idx, :]
        pc_mix = pc_mix[:, pnt_idx, :]
        pc_clean = pc_clean[:, pnt_idx, :]

        # target
        grad_dir_t_target = pc_clean - pc_noisy

        # decoder
        pred_dir = self.decoder(
            c=feat.reshape(-1, F_dim)
        ).reshape(B, len(pnt_idx), d) # type: ignore

        # 拉普拉斯噪声的 MLE 对应 L1 损失；使用 Charbonnier（平滑 L1）
        # 既保留对重尾离群噪声的鲁棒性，又避免 L1 在零点处不可导
        diff = pred_dir - grad_dir_t_target
        loss = (jt.sqrt((diff ** 2.0).sum(dim=-1) + 1e-12) / self.dsm_sigma).mean()

        return loss

    def denoise_langevin_dynamics(self, pcl_noisy, num_steps: int=4):
        """
        pcl_noisy: (B, N, 3)
        """
        B, N, d = pcl_noisy.shape
        with jt.no_grad():
            pcl_next = pcl_noisy.clone()
            for it in range(num_steps):
                feat = self.encoder(pcl_next)  # (B, N, F)
                F_dim = feat.shape[2]

                pred_dir = self.decoder(
                    c=feat.reshape(-1, F_dim)
                ).reshape(B, N, d)

                pcl_next = pcl_next + (1.0 / num_steps) * pred_dir
        return pcl_next, None

    def training_step(self, batch: Dict) -> Dict:
        patch_size = batch['pc_noisy'].shape[-2]
        pc_noisy = batch['pc_noisy'].reshape(-1, patch_size, 3)
        pc_mix = batch['pc_mix'].reshape(-1, patch_size, 3)
        pc_clean = batch['pc_clean'].reshape(-1, patch_size, 3)
        loss = self.get_supervised_loss(
            pc_noisy=pc_noisy,
            pc_mix=pc_mix,
            pc_clean=pc_clean,
        )
        return {"loss": loss}

    def execute(self, **kwargs) -> Dict: # type: ignore
        return self.training_step(**kwargs)

    @jt.no_grad()
    def predict_step(self, batch: Dict) -> List[Dict]:
        pc_noisy_batch = batch['pc_noisy']
        assert pc_noisy_batch.ndim == 3

        # predict_rounds: 将上一轮输出重新当作输入迭代降噪，
        # 对拉普拉斯重尾残留的离群点有效；>1 时需验证细节不过度收缩
        num_steps = int(self.model_config.get('predict_rounds', 1))
        fusion = self.model_config.get('fusion_mode', 'best')
        residual_alpha = float(self.model_config.get('residual_alpha', 1.0))
        res = []
        for i, pc_noisy in enumerate(pc_noisy_batch):
            pc_next = pc_noisy
            for it in range(num_steps):
                pc_out = patch_based_denoise(
                    model=self,
                    pcl_noisy=pc_next,
                    patch_size=1000,
                    seed_k=6,
                    seed_k_alpha=1,
                    fusion=fusion,
                )
                if pc_out is None:
                    # patch 推理失败时回退上一步结果，保证输出完整
                    break
                pc_next = pc_out
            # 对完整点云的最终残差统一缩放；alpha=1 保持原始推理。
            pc_next = pc_noisy + residual_alpha * (pc_next - pc_noisy)
            pc_denoised = pc_next.detach().numpy().astype(np.float32, copy=False)
            res.append({"pc_denoised": pc_denoised})
        return res

    def process_fn(self, batch: List[Asset]) -> List[Dict]:
        res = []
        for b in batch:
            if not self.is_predict():
                assert b.meta is not None
                res.append({
                    "pc_noisy": b.meta['pc_noisy'], # (num_patches, patch_size, 3)
                    "pc_clean": b.meta['pc_clean'],
                    "pc_mix": b.meta['pc_mix'],
                })
            else:
                d = {
                    "pc_noisy": b.sampled_vertices_noisy, # (N, 3)
                }
                if b.sampled_vertices is not None:
                    d["pc_clean"] = b.sampled_vertices
                res.append(d)
        return res

def farthest_point_sampling(pcls, num_pnts):
    """
    pcls: (B, N, 3)
    return:
        sampled: (B, num_pnts, 3)
        indices: (B, num_pnts)
    """
    B, N, _ = pcls.shape
    sampled = []
    indices = []
    for b in range(B):
        pts = pcls[b]  # (N, 3)
        selected = []
        dist = jt.ones((N,)) * 1e10
        farthest = 0
        for i in range(num_pnts):
            selected.append(farthest)
            centroid = pts[farthest]  # (3,)
            d = ((pts - centroid) ** 2).sum(dim=1)
            dist = jt.minimum(dist, d)
            farthest, _ = jt.argmax(dist, dim=-1)
            farthest = farthest.item()
        idx = jt.array(selected).int32()
        sampled.append(pts[idx][None, ...])
        indices.append(idx[None, ...])
    sampled = jt.concat(sampled, dim=0)
    indices = jt.concat(indices, dim=0)
    return sampled, indices

def knn_points(x, y, k):
    """
    x: (B, P, 3)
    y: (B, N, 3)
    return:
        dist: (B, P, k)
        idx:  (B, P, k)
        nn:   (B, P, k, 3)
    """
    dist = ((x.unsqueeze(2) - y.unsqueeze(1)) ** 2).sum(-1)
    dist_k, idx = jt.topk(dist, k=k, dim=-1, largest=False)
    B = x.shape[0]
    nn = []
    for b in range(B):
        nn.append(y[b][idx[b]])
    nn = jt.stack(nn, dim=0)
    return dist_k, idx, nn

def patch_based_denoise(model: VelocityModule, pcl_noisy, patch_size=1000, seed_k=6, seed_k_alpha=1, fusion='best') -> jt.Var:
    """
    pcl_noisy: (N, 3)
    fusion: 'weighted' = 多 patch 加权平均（抗离群，但略有平滑收缩）；
            'best'     = 原版 starter code 的单最佳 patch（保边缘，P2S 可能更好）
    """
    assert len(pcl_noisy.shape) == 2

    N, d = pcl_noisy.shape
    num_patches = int(seed_k * N / patch_size)
    pcl_noisy = pcl_noisy.unsqueeze(0)  # (1, N, 3)

    seed_pnts, seed_idx = farthest_point_sampling(pcl_noisy, num_patches)
    patch_dists, point_idxs, patches = knn_points(seed_pnts, pcl_noisy, patch_size)

    patches = patches[0]              # (P, M, 3)
    patch_dists = patch_dists[0]      # (P, M)
    point_idxs = point_idxs[0]        # (P, M)

    seed_expand = seed_pnts.squeeze().unsqueeze(1).broadcast(patches.shape)
    patches = patches - seed_expand

    patch_dists = patch_dists / (patch_dists[:, -1:].broadcast(patch_dists.shape) + 1e-8)

    i = 0
    patch_step = int(ceil(N / (seed_k_alpha * patch_size)))
    assert patch_step > 0
    patches_denoised = []
    while i < num_patches:
        curr = patches[i:i+patch_step]
        try:
            out, _ = model.denoise_langevin_dynamics(curr)
        except Exception as e:
            print("Denoise error:", e)
            return None
        patches_denoised.append(out)
        i += patch_step

    patches_denoised = jt.concat(patches_denoised, dim=0)
    patches_denoised = patches_denoised + seed_expand
    pcl_original = pcl_noisy.squeeze(0)

    if fusion == 'best':
        # starter code 原版策略：每点只取权重最大（最近）patch 的预测。
        # 按 patch 循环（P≈6N/1000，远小于原版的逐点 N 次循环）保持语义一致
        best_w = jt.full((N,), -1e10)
        best_pred = pcl_original.clone()
        for p in range(num_patches):
            idx_p = point_idxs[p]                    # (M,)
            w_p = jt.exp(-patch_dists[p])            # (M,)
            better = (w_p > best_w[idx_p]).unsqueeze(1)  # (M, 1)
            best_w[idx_p] = jt.maximum(w_p, best_w[idx_p])
            best_pred[idx_p] = jt.where(
                better.broadcast((idx_p.shape[0], 3)),
                patches_denoised[p],
                best_pred[idx_p],
            )
        return best_pred

    # 'weighted'：按 exp(-dist) 权重对所有覆盖该点的 patch 预测做加权融合
    flat_idx = point_idxs.reshape(-1)                                    # (P*M,)
    flat_w = jt.exp(-patch_dists).reshape(-1, 1)                         # (P*M, 1)
    flat_pred = patches_denoised.reshape(-1, 3) * flat_w                 # (P*M, 3)
    num_flat = flat_idx.shape[0]

    pred_sum = jt.zeros((N, 3)).scatter_(
        0, flat_idx.unsqueeze(1).broadcast((num_flat, 3)), flat_pred, reduce='add'
    )
    weight_sum = jt.zeros((N, 1)).scatter_(
        0, flat_idx.unsqueeze(1).broadcast((num_flat, 1)), flat_w, reduce='add'
    )

    covered = (weight_sum > 1e-12).broadcast((N, 3))
    pcl_fused = pred_sum / (weight_sum + 1e-12)
    pcl_out = jt.where(covered, pcl_fused, pcl_original)
    return pcl_out
