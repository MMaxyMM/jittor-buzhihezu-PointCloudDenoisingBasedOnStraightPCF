from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from scipy.spatial import cKDTree
from typing import Dict, List, Optional, Tuple, Union

import multiprocessing as mp
import numpy as np
import os
from pathlib import Path

from .asset import Asset
from .spec import ConfigSpec
from .utils import (
    random_euler_rotation,
    sample_mixed_noise,
    sample_noise_scale,
    sample_vertex_groups,
    validate_noise_mixture,
    validate_noise_scale_sampling,
)


_NOISE_DEBUG_COUNTS: Dict[int, int] = {}
_PATCH_DEBUG_COUNTS: Dict[int, int] = {}
_DEBUG_BASE_SEED_ENV = "PCD_DEBUG_BASE_SEED"


def _debug_enabled() -> bool:
    return os.environ.get("PCD_DEBUG_NOISE") == "1"


def _remember_debug_base_seed() -> None:
    if not _debug_enabled() or _DEBUG_BASE_SEED_ENV in os.environ:
        return
    try:
        import jittor as jt

        os.environ[_DEBUG_BASE_SEED_ENV] = str(int(jt.get_seed()))
    except Exception:
        # PID/process identity still make the diagnostic useful if the Jittor
        # seed cannot be queried in an unusual runtime.
        pass


def _debug_worker_label() -> str:
    try:
        import jittor as jt

        current_seed = int(jt.get_seed())
        base_seed_text = os.environ.get(_DEBUG_BASE_SEED_ENV)
        if base_seed_text is not None:
            base_seed = int(base_seed_text)
            if current_seed == base_seed:
                return "main"
            for worker_id in range(256):
                worker_seed = (base_seed ^ (worker_id * 1167)) ^ 1234
                if current_seed == worker_seed:
                    return str(worker_id)
    except Exception:
        pass
    identity = getattr(mp.current_process(), "_identity", ())
    return f"mp:{identity[0]}" if identity else "unknown"


def _debug_path(path: Optional[str]) -> str:
    if not path:
        return "unknown"
    parts = Path(path).parts
    return "/".join(parts[-4:])


def _debug_noise(
    asset: Asset, noise_type: str, scale: float, pc: np.ndarray
) -> None:
    if not _debug_enabled():
        return
    pid = os.getpid()
    count = _NOISE_DEBUG_COUNTS.get(pid, 0) + 1
    _NOISE_DEBUG_COUNTS[pid] = count
    if count > 10:
        return
    sample0 = ",".join(f"{float(value):.6g}" for value in pc[0])
    print(
        f"[PCD_DEBUG_NOISE] pid={pid} worker={_debug_worker_label()} "
        f"count={count} type={noise_type} scale={scale:.8f} "
        f"path={_debug_path(asset.path)} "
        f"mesh_sample0={sample0}",
        flush=True,
    )


def _debug_patch(asset: Asset, seed_idx: np.ndarray) -> None:
    if not _debug_enabled():
        return
    pid = os.getpid()
    count = _PATCH_DEBUG_COUNTS.get(pid, 0) + 1
    _PATCH_DEBUG_COUNTS[pid] = count
    if count > 10:
        return
    seeds = ",".join(str(int(value)) for value in seed_idx[:4])
    print(
        f"[PCD_DEBUG_PATCH] pid={pid} worker={_debug_worker_label()} "
        f"count={count} seed_idx={seeds} path={_debug_path(asset.path)}",
        flush=True,
    )


@dataclass(frozen=True)
class Augment(ConfigSpec):

    @classmethod
    @abstractmethod
    def parse(cls, **kwags) -> 'Augment':
        pass

    @abstractmethod
    def apply(self, asset: Asset, **kwargs):
        pass

@dataclass(frozen=True)
class AugmentSample(Augment):

    num_samples: int # total number of vertices on the face to be sampled

    num_vertex_samples: int=0 # number of vertices to be chosen

    @classmethod
    def parse(cls, **kwargs) -> 'AugmentSample':
        cls.check_keys(kwargs)
        return AugmentSample(**kwargs)

    def apply(self, asset: Asset, **kwargs):
        if asset.vertices is not None and asset.faces is not None:
            sampled_vertices, sampled_normals, sampled_vertex_groups, hidden_states = sample_vertex_groups(
                vertices=asset.vertices,
                faces=asset.faces,
                num_samples=self.num_samples,
                num_vertex_samples=self.num_vertex_samples,
            )
            asset.sampled_vertices = sampled_vertices
            return

        # Cached clean point clouds have already been sampled from the mesh.
        # Preserve original OBJ vertices, then fill from the surface pool.
        pc = asset.sampled_vertices
        if pc is None:
            raise ValueError("sample requires either a mesh or a cached clean point cloud")
        if pc.ndim != 2 or pc.shape[1] != 3:
            raise ValueError(f"cached clean point cloud must have shape (N, 3), got {pc.shape}")

        cached_vertices = asset.cached_vertices
        if cached_vertices is not None:
            num_vertices = min(
                self.num_vertex_samples, self.num_samples, cached_vertices.shape[0]
            )
            vertex_indices = np.random.permutation(cached_vertices.shape[0])[:num_vertices]
            selected_vertices = cached_vertices[vertex_indices]
        else:
            num_vertices = 0
            selected_vertices = np.empty((0, 3), dtype=np.float32)

        num_surface = self.num_samples - num_vertices
        replace = pc.shape[0] < num_surface
        surface_indices = np.random.choice(pc.shape[0], size=num_surface, replace=replace)
        sampled = np.concatenate([selected_vertices, pc[surface_indices]], axis=0)
        asset.sampled_vertices = sampled.astype(np.float32, copy=False)

@dataclass(frozen=True)
class AugmentNormalizePC(Augment):

    @classmethod
    def parse(cls, **kwargs) -> 'AugmentNormalizePC':
        cls.check_keys(kwargs)
        return AugmentNormalizePC(**kwargs)

    def apply(self, asset: Asset, **kwargs):
        pc = asset.sampled_vertices
        assert pc is not None, "sampled_vertices is None, cannot apply AugmentNormalizePC"
        p_max = pc.max(axis=0)
        p_min = pc.min(axis=0)
        center = (p_max + p_min) / 2
        pc = pc - center
        scale = np.sqrt((pc**2).sum(axis=1).max()).max()
        asset.sampled_vertices = pc / scale

@dataclass(frozen=True)
class AugmentAddNoise(Augment):

    noise_std_min: float=0.005

    noise_std_max: float=0.020

    noise_type: str="laplace"

    noise_sampling: str="uniform"

    noise_mid_min: Optional[float]=None

    noise_mid_max: Optional[float]=None

    noise_low_prob: Optional[float]=None

    noise_mid_prob: Optional[float]=None

    noise_high_prob: Optional[float]=None

    noise_mixture: Optional[list]=None

    @classmethod
    def parse(cls, **kwargs) -> 'AugmentAddNoise':
        cls.check_keys(kwargs)
        if kwargs.get("noise_mixture") is not None:
            kwargs["noise_mixture"] = validate_noise_mixture(
                kwargs["noise_mixture"]
            )
        else:
            validate_noise_scale_sampling(
                noise_min=kwargs.get("noise_std_min", 0.005),
                noise_max=kwargs.get("noise_std_max", 0.020),
                sampling=kwargs.get("noise_sampling", "uniform"),
                mid_min=kwargs.get("noise_mid_min"),
                mid_max=kwargs.get("noise_mid_max"),
                low_prob=kwargs.get("noise_low_prob"),
                mid_prob=kwargs.get("noise_mid_prob"),
                high_prob=kwargs.get("noise_high_prob"),
            )
        _remember_debug_base_seed()
        return AugmentAddNoise(**kwargs)

    def apply(self, asset: Asset, **kwargs):
        pc = asset.sampled_vertices
        assert pc is not None, "sampled_vertices is None, cannot apply AugmentAddNoise"
        if self.noise_mixture is not None:
            noise, noise_type, noise_std = sample_mixed_noise(
                pc.shape, self.noise_mixture
            )
        else:
            noise_type = self.noise_type
            noise_std = sample_noise_scale(
                noise_min=self.noise_std_min,
                noise_max=self.noise_std_max,
                sampling=self.noise_sampling,
                mid_min=self.noise_mid_min,
                mid_max=self.noise_mid_max,
                low_prob=self.noise_low_prob,
                mid_prob=self.noise_mid_prob,
                high_prob=self.noise_high_prob,
                validate=False,
            )
            if noise_type == "laplace":
                # 与官方 starter code 完全一致：配置值直接作为 laplace 的尺度参数 b。
                noise = np.random.laplace(0.0, noise_std, size=pc.shape)
            elif noise_type == "gaussian":
                noise = np.random.normal(0.0, noise_std, size=pc.shape)
            else:
                raise ValueError(f"unsupported noise_type: {noise_type}")
        _debug_noise(
            asset=asset, noise_type=noise_type, scale=noise_std, pc=pc
        )
        asset.sampled_vertices_noisy = (pc + noise).astype(np.float32, copy=False)

@dataclass(frozen=True)
class AugmentLinear(Augment):

    scale: Tuple[float, float]=(1.0, 1.0)

    rotate_x_range: Tuple[float, float]=(0.0, 0.0)

    rotate_y_range: Tuple[float, float]=(0.0, 0.0)

    rotate_z_range: Tuple[float, float]=(0.0, 0.0)

    scale_p: float=0.0

    rotate_p: float=0.0

    @classmethod
    def parse(cls, **kwargs) -> 'AugmentLinear':
        cls.check_keys(kwargs)
        return AugmentLinear(**kwargs)

    def apply(self, asset: Asset, **kwargs):
        trans_vertex = np.eye(4, dtype=np.float32)
        if np.random.rand() < self.rotate_p:
            r = random_euler_rotation(
                1,
                x_range=self.rotate_x_range,
                y_range=self.rotate_y_range,
                z_range=self.rotate_z_range,
            )[0]
            trans_vertex = r @ trans_vertex
        if np.random.rand() < self.scale_p:
            scale = np.zeros((4, 4), dtype=np.float32)
            scale[0, 0] = np.random.uniform(self.scale[0], self.scale[1])
            scale[1, 1] = np.random.uniform(self.scale[0], self.scale[1])
            scale[2, 2] = np.random.uniform(self.scale[0], self.scale[1])
            scale[3, 3] = 1.0
            trans_vertex = scale @ trans_vertex
        asset.transform(trans_vertex)

@dataclass(frozen=True)
class AugmentPatch(Augment):

    patch_size: int

    num_patches: int

    train_cvm_network: bool

    @classmethod
    def parse(cls, **kwargs) -> 'AugmentPatch':
        cls.check_keys(kwargs)
        return AugmentPatch(**kwargs)

    def apply(self, asset: Asset, **kwargs):
        pc = asset.sampled_vertices
        pc_noisy = asset.sampled_vertices_noisy

        assert pc is not None
        assert pc_noisy is not None

        N = pc_noisy.shape[0]

        seed_idx = np.random.permutation(N)[:self.num_patches]   # (P,)
        _debug_patch(asset=asset, seed_idx=seed_idx)
        seed_points = pc_noisy[seed_idx]                         # (P, 3)

        tree = cKDTree(pc_noisy)
        _, nn_idx = tree.query(seed_points, k=self.patch_size)   # (P, M)

        pat_A = pc_noisy[nn_idx].astype(np.float32, copy=False)  # (P, M, 3)
        pat_B = pc[nn_idx].astype(np.float32, copy=False)        # (P, M, 3)

        if self.train_cvm_network:
            # CVM and DistanceModule share one interpolation time per patch.
            t = np.random.uniform(1e-8, 1.0, size=(self.num_patches,)).astype(np.float32)
            seed_points_t = (
                t[:, None] * pc[seed_idx] +
                (1.0 - t[:, None]) * pc_noisy[seed_idx]
            ).astype(np.float32, copy=False)
            if asset.meta is None:
                asset.meta = {}
            asset.meta['pc_noisy'] = pat_A
            asset.meta['pc_clean'] = pat_B
            asset.meta['seed_points_t'] = seed_points_t[:, None, :]
            asset.meta['original_time_step'] = t
            return

        l1, l2 = 1e-8, 1.0
        # Use one interpolation time for every point in the same patch. This
        # preserves local geometry seen by the dynamic KNN graph.
        t = np.random.uniform(
            l1, l2, size=(self.num_patches, 1, 1)
        ).astype(np.float32)

        pat_t = t * pat_B + (1 - t) * pat_A
        seed_points_t = (
            t * pc[seed_idx][:, None, :] +
            (1 - t) * pc_noisy[seed_idx][:, None, :]
        )

        pat_A = pat_A - seed_points_t
        pat_B = pat_B - seed_points_t
        pat_t = pat_t - seed_points_t

        if asset.meta is None:
            asset.meta = {}
        asset.meta['pc_noisy'] = pat_A
        asset.meta['pc_clean'] = pat_B
        asset.meta['pc_mix'] = pat_t

def get_augments(*args) -> List[Augment]:
    MAP = {
        "sample": AugmentSample,
        "normalize_pc": AugmentNormalizePC,
        "add_noise": AugmentAddNoise,
        "linear": AugmentLinear,
        "patch": AugmentPatch,
    }
    MAP: Dict[str, type[Augment]]
    augments = []
    for (i, config) in enumerate(args):
        __target__ = config.get('__target__')
        assert __target__ is not None, f"do not find `__target__` in augment of position {i}"
        c = deepcopy(config)
        del c['__target__']
        augments.append(MAP[__target__].parse(**c))
    return augments