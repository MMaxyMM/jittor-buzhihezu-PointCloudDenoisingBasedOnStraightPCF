#!/usr/bin/env python3
"""Evaluate exp6 alpha/soup candidates and exp6+exp2 ensembles on local200.

The script caches six actual model predictions:

1. exp6 best with alpha=1.00 (other exp6 alphas are derived exactly);
2. exp6 late Top-2 soup with alpha=1.05;
3. exp6 late Top-3 soup with alpha=1.05;
4. exp6 late Top-4 soup with alpha=1.05;
5. exp6 late Top-5 soup with alpha=1.05;
6. exp2 best with alpha=1.10 (other exp2 alphas are derived exactly).

All requested alpha variants and ensemble weights are evaluated from those
caches.  The final result file uses one line per candidate:

    model name：score，cd_pred，cd_noisy，p2s_pred，p2s_noisy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial import cKDTree


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_local_test_models import (  # noqa: E402
    discover_samples,
    load_normalized_mesh,
    normalize_reference,
    score_metric,
    validate_cloud,
)


@dataclass(frozen=True)
class PredictionSpec:
    cache_name: str
    label: str
    model_config: Path
    checkpoint: Path
    residual_alpha: float


class MetricAccumulator:
    def __init__(self, label: str):
        self.label = label
        self.cd_pred: List[float] = []
        self.cd_noisy: List[float] = []
        self.cd_score: List[float] = []
        self.p2s_pred: List[float] = []
        self.p2s_noisy: List[float] = []
        self.p2s_score: List[float] = []

    def append(
        self,
        cd_pred: float,
        cd_noisy: float,
        p2s_pred: float,
        p2s_noisy: float,
    ) -> None:
        self.cd_pred.append(cd_pred)
        self.cd_noisy.append(cd_noisy)
        self.cd_score.append(score_metric(cd_pred, cd_noisy))
        self.p2s_pred.append(p2s_pred)
        self.p2s_noisy.append(p2s_noisy)
        self.p2s_score.append(score_metric(p2s_pred, p2s_noisy))

    def summarize(self) -> dict:
        if not self.cd_pred:
            raise RuntimeError(f"candidate has no metrics: {self.label}")
        cd_score = float(np.mean(self.cd_score))
        p2s_score = float(np.mean(self.p2s_score))
        return {
            "label": self.label,
            "samples": len(self.cd_pred),
            "score": 0.5 * cd_score + 0.5 * p2s_score,
            "cd_score": cd_score,
            "p2s_score": p2s_score,
            "cd_pred": float(np.mean(self.cd_pred)),
            "cd_noisy": float(np.mean(self.cd_noisy)),
            "p2s_pred": float(np.mean(self.p2s_pred)),
            "p2s_noisy": float(np.mean(self.p2s_noisy)),
        }


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_yaml(path: Path) -> dict:
    value = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(value, dict):
        raise ValueError(f"YAML top-level value must be a mapping: {path}")
    return value


def checkpoint_identity(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def sample_digest(samples: Sequence[Tuple[str, Path]]) -> str:
    payload = "\n".join(key for key, _ in samples).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".npy", delete=False
        ) as file:
            temporary = file.name
            np.save(file, value.astype(np.float32, copy=False), allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def prediction_path(cache_root: Path, spec: PredictionSpec, key: str) -> Path:
    return cache_root / spec.cache_name / key / "denoised.npy"


def expected_cache_manifest(
    spec: PredictionSpec,
    samples: Sequence[Tuple[str, Path]],
    seed: int,
) -> dict:
    return {
        "label": spec.label,
        "model_config": checkpoint_identity(spec.model_config),
        "checkpoint": checkpoint_identity(spec.checkpoint),
        "residual_alpha": spec.residual_alpha,
        "fusion_mode": "best",
        "predict_rounds": 1,
        "seed": seed,
        "sample_count": len(samples),
        "sample_digest": sample_digest(samples),
    }


def prepare_cache_manifest(
    cache_root: Path,
    spec: PredictionSpec,
    samples: Sequence[Tuple[str, Path]],
    seed: int,
) -> None:
    model_cache = cache_root / spec.cache_name
    manifest_path = model_cache / "cache_manifest.json"
    expected = expected_cache_manifest(spec, samples, seed)
    if manifest_path.is_file():
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        if actual != expected:
            raise RuntimeError(
                f"prediction cache does not match current request: {model_cache}. "
                "Use a different --cache-dir; the script will not delete old caches."
            )
        return
    if model_cache.exists() and any(model_cache.iterdir()):
        raise RuntimeError(
            f"non-empty cache has no manifest: {model_cache}. "
            "Use a different --cache-dir."
        )
    model_cache.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(expected, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def cache_file_is_valid(path: Path, expected_points: int) -> bool:
    if not path.is_file():
        return False
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        return (
            value.shape == (expected_points, 3)
            and value.dtype == np.float32
            and np.isfinite(value).all()
        )
    except (OSError, ValueError):
        return False


def build_model(spec: PredictionSpec, transform_config: Path):
    import jittor as jt
    from src.model.parse import get_model

    model_config = load_yaml(spec.model_config)
    model_config["residual_alpha"] = spec.residual_alpha
    model_config["fusion_mode"] = "best"
    model_config["predict_rounds"] = 1
    model = get_model(
        model_config=model_config,
        transform_config=load_yaml(transform_config),
    )

    state = jt.load(str(spec.checkpoint))
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint is not a state dict: {spec.checkpoint}")
    model_state = model.state_dict()
    missing = sorted(set(model_state) - set(state))
    extra = sorted(set(state) - set(model_state))
    if missing or extra:
        raise RuntimeError(
            f"checkpoint/model key mismatch for {spec.label}: "
            f"missing={missing}, extra={extra}"
        )
    for key, model_value in model_state.items():
        state_value = state[key]
        if tuple(model_value.shape) != tuple(state_value.shape):
            raise RuntimeError(
                f"checkpoint/model shape mismatch for {spec.label}, {key}: "
                f"{tuple(state_value.shape)} vs {tuple(model_value.shape)}"
            )
    del state, model_state
    model.load(str(spec.checkpoint))
    model.set_predict(True)
    model.eval()
    return model


def ensure_prediction_cache(
    spec: PredictionSpec,
    samples: Sequence[Tuple[str, Path]],
    cache_root: Path,
    transform_config: Path,
    seed: int,
) -> None:
    import jittor as jt

    prepare_cache_manifest(cache_root, spec, samples, seed)
    missing: List[Tuple[str, Path, int]] = []
    for key, model_dir in samples:
        noisy_path = model_dir / "noisy.npy"
        noisy = np.load(noisy_path, mmap_mode="r", allow_pickle=False)
        if noisy.ndim != 2 or noisy.shape[1] != 3:
            raise ValueError(f"invalid noisy cloud shape: {noisy_path}: {noisy.shape}")
        output_path = prediction_path(cache_root, spec, key)
        if not cache_file_is_valid(output_path, noisy.shape[0]):
            missing.append((key, model_dir, noisy.shape[0]))

    if not missing:
        print(f"[{spec.label}] prediction cache complete; skip inference")
        return

    print(f"[{spec.label}] missing cached predictions: {len(missing)}/{len(samples)}")
    model = build_model(spec, transform_config)
    start = time.time()
    for index, (key, model_dir, expected_points) in enumerate(missing, start=1):
        noisy_path = model_dir / "noisy.npy"
        noisy = validate_cloud(
            np.load(noisy_path, allow_pickle=False), noisy_path, expected_points
        )
        with jt.no_grad():
            output = model.predict_step({"pc_noisy": jt.array(noisy[None, ...])})
        prediction = output[0]["pc_denoised"]
        if not isinstance(prediction, np.ndarray):
            prediction = prediction.numpy()
        prediction = validate_cloud(
            np.asarray(prediction),
            Path(f"<prediction:{spec.label}:{key}>"),
            expected_points,
        )
        atomic_save_npy(prediction_path(cache_root, spec, key), prediction)
        del output, prediction
        elapsed = time.time() - start
        remaining = elapsed / index * (len(missing) - index)
        print(
            f"\r[{spec.label}] inference {index}/{len(missing)} "
            f"elapsed={elapsed / 60:.1f}m remaining={remaining / 60:.1f}m",
            end="",
            flush=True,
        )
    print()
    del model
    jt.gc()


def load_reference(key: str, model_dir: Path, mesh_root: Path) -> dict:
    noisy_path = model_dir / "noisy.npy"
    clean_path = model_dir / "clean.npy"
    normalization_path = model_dir / "normalization.npz"
    mesh_path = mesh_root / key / "models/model_normalized.obj"
    noisy = validate_cloud(np.load(noisy_path, allow_pickle=False), noisy_path)
    clean = validate_cloud(np.load(clean_path, allow_pickle=False), clean_path)
    if noisy.shape != clean.shape:
        raise ValueError(f"clean/noisy shape mismatch: {key}")
    clean_norm, center, scale = normalize_reference(clean)
    mesh_vertices, mesh_faces = load_normalized_mesh(
        mesh_path, normalization_path
    )
    mesh_norm = (mesh_vertices - center) / scale
    clean_tree = cKDTree(clean_norm)
    return {
        "noisy": noisy,
        "center": center,
        "scale": scale,
        "clean_norm": clean_norm,
        "clean_tree": clean_tree,
        "mesh_vertices": mesh_norm.astype(np.float32),
        "mesh_faces": mesh_faces,
    }


def metrics_for_cloud(prediction: np.ndarray, reference: dict) -> Tuple[float, float]:
    import point_cloud_utils as pcu

    prediction = validate_cloud(
        np.asarray(prediction),
        Path("<candidate prediction>"),
        reference["noisy"].shape[0],
    )
    pred_norm = (
        prediction.astype(np.float64) - reference["center"]
    ) / reference["scale"]
    pred_to_clean = reference["clean_tree"].query(
        pred_norm, k=1, workers=1
    )[0]
    pred_tree = cKDTree(pred_norm)
    clean_to_pred = pred_tree.query(
        reference["clean_norm"], k=1, workers=1
    )[0]
    cd = float(np.mean(pred_to_clean**2) + np.mean(clean_to_pred**2))
    distances, _, _ = pcu.closest_points_on_mesh(
        pred_norm.astype(np.float32),
        reference["mesh_vertices"],
        reference["mesh_faces"],
    )
    p2s = float(np.mean(np.asarray(distances, dtype=np.float64) ** 2))
    return cd, p2s


def alpha_label(alpha: float) -> str:
    value = int(round(alpha * 100))
    return f"checkpoint_selection_b_exp6_b32_alpha{value:03d}_straightpcf"


def ensemble_label(weight: float, exp2_alpha: float) -> str:
    exp2_alpha_value = int(round(exp2_alpha * 100))
    return (
        "checkpoint_selection_b_exp6_b32_alpha105_straightpcf"
        f"+exp2_alpha{exp2_alpha_value:03d}_ensemble("
        f"exp6_weight={weight:.2f},exp2_weight={1.0-weight:.2f})"
    )


def unique_floats(values: Iterable[float], name: str) -> List[float]:
    result = []
    for value in values:
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{name} contains a non-finite value")
        if value not in result:
            result.append(value)
    return result


def evaluate_all(
    samples: Sequence[Tuple[str, Path]],
    mesh_root: Path,
    cache_root: Path,
    specs: Dict[str, PredictionSpec],
    alphas: Sequence[float],
    exp2_alphas: Sequence[float],
    ensemble_weights: Sequence[float],
) -> List[dict]:
    accumulators: Dict[str, MetricAccumulator] = {}
    for alpha in alphas:
        label = alpha_label(alpha)
        accumulators[label] = MetricAccumulator(label)
    for name in ("soup_top2", "soup_top3", "soup_top4", "soup_top5"):
        label = specs[name].label
        accumulators[label] = MetricAccumulator(label)
    for exp2_alpha in exp2_alphas:
        for weight in ensemble_weights:
            label = ensemble_label(weight, exp2_alpha)
            accumulators[label] = MetricAccumulator(label)

    start = time.time()
    for index, (key, model_dir) in enumerate(samples, start=1):
        reference = load_reference(key, model_dir, mesh_root)
        noisy = reference["noisy"]
        cd_noisy, p2s_noisy = metrics_for_cloud(noisy, reference)

        exp6_base = validate_cloud(
            np.load(
                prediction_path(cache_root, specs["exp6"], key),
                allow_pickle=False,
            ),
            Path(f"<cache:exp6:{key}>"),
            noisy.shape[0],
        )
        exp6_predictions = {}
        for alpha in alphas:
            prediction = noisy + alpha * (exp6_base - noisy)
            prediction = prediction.astype(np.float32, copy=False)
            exp6_predictions[alpha] = prediction
            cd_pred, p2s_pred = metrics_for_cloud(prediction, reference)
            accumulators[alpha_label(alpha)].append(
                cd_pred, cd_noisy, p2s_pred, p2s_noisy
            )

        for name in ("soup_top2", "soup_top3", "soup_top4", "soup_top5"):
            prediction = np.load(
                prediction_path(cache_root, specs[name], key),
                allow_pickle=False,
            )
            cd_pred, p2s_pred = metrics_for_cloud(prediction, reference)
            accumulators[specs[name].label].append(
                cd_pred, cd_noisy, p2s_pred, p2s_noisy
            )

        exp2_cached = validate_cloud(
            np.load(
                prediction_path(cache_root, specs["exp2"], key),
                allow_pickle=False,
            ),
            Path(f"<cache:exp2:{key}>"),
            noisy.shape[0],
        )
        exp6_alpha105 = exp6_predictions[1.05]
        for exp2_alpha in exp2_alphas:
            exp2_prediction = (
                noisy
                + (exp2_alpha / specs["exp2"].residual_alpha)
                * (exp2_cached - noisy)
            ).astype(np.float32, copy=False)
            for weight in ensemble_weights:
                prediction = (
                    weight * exp6_alpha105 + (1.0 - weight) * exp2_prediction
                ).astype(np.float32, copy=False)
                cd_pred, p2s_pred = metrics_for_cloud(prediction, reference)
                accumulators[ensemble_label(weight, exp2_alpha)].append(
                    cd_pred, cd_noisy, p2s_pred, p2s_noisy
                )

        elapsed = time.time() - start
        remaining = elapsed / index * (len(samples) - index)
        print(
            f"\r[metrics] {index}/{len(samples)} "
            f"elapsed={elapsed / 60:.1f}m remaining={remaining / 60:.1f}m",
            end="",
            flush=True,
        )
    print()
    return [accumulator.summarize() for accumulator in accumulators.values()]


def write_results(results: Sequence[dict], result_file: Path, overwrite: bool) -> None:
    if result_file.exists() and not overwrite:
        raise FileExistsError(
            f"result file already exists: {result_file}; "
            "use --overwrite-result to replace it"
        )
    result_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=result_file.parent,
            prefix=result_file.name + ".",
            suffix=".tmp",
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as file:
            temporary = file.name
            for result in results:
                file.write(
                    f"{result['label']}：{result['score']:.4f}，"
                    f"{result['cd_pred']:.8f}，{result['cd_noisy']:.8f}，"
                    f"{result['p2s_pred']:.8f}，{result['p2s_noisy']:.8f}\n"
                )
        os.replace(temporary, result_file)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="dataset_train_pcd_disk/local_test")
    parser.add_argument("--mesh-root", default="dataset_train/local_test")
    parser.add_argument(
        "--datalist", default="dataset_train/local_test/datalist.txt"
    )
    parser.add_argument(
        "--transform-config", default="configs/transform/predict.yaml"
    )
    parser.add_argument(
        "--exp6-model-config",
        default="configs/model/straightpcf_b_exp6_b32_alpha100.yaml",
    )
    parser.add_argument(
        "--exp2-model-config",
        default="configs/model/straightpcf_maxagg_endpoint.yaml",
    )
    parser.add_argument(
        "--exp6-checkpoint",
        default=(
            "checkpoint_selection_b_exp6_b32_alpha105_straightpcf/"
            "best_checkpoint.pkl"
        ),
    )
    parser.add_argument(
        "--soup-top2-checkpoint",
        default=(
            "checkpoint_selection_b_exp6_b32_alpha105_straightpcf_soup_top2/"
            "best_checkpoint.pkl"
        ),
    )
    parser.add_argument(
        "--soup-top3-checkpoint",
        default=(
            "checkpoint_selection_b_exp6_b32_alpha105_straightpcf_soup_top3/"
            "best_checkpoint.pkl"
        ),
    )
    parser.add_argument(
        "--soup-top4-checkpoint",
        default=(
            "checkpoint_selection_b_exp6_b32_alpha105_straightpcf_soup_top4/"
            "best_checkpoint.pkl"
        ),
    )
    parser.add_argument(
        "--soup-top5-checkpoint",
        default=(
            "checkpoint_selection_b_exp6_b32_alpha105_straightpcf_soup_top5/"
            "best_checkpoint.pkl"
        ),
    )
    parser.add_argument(
        "--exp2-checkpoint",
        default="checkpoint_selection_b_exp2_batch_straightpcf/best_checkpoint.pkl",
    )
    parser.add_argument(
        "--exp2-alphas", nargs="+", type=float, default=(1.00, 1.05, 1.10)
    )
    parser.add_argument(
        "--alphas", nargs="+", type=float, default=(1.00, 1.05, 1.07, 1.10)
    )
    parser.add_argument(
        "--ensemble-weights",
        nargs="+",
        type=float,
        default=(0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95),
        help="exp6 weights; exp2 receives 1-weight",
    )
    parser.add_argument(
        "--cache-dir", default="outputs/local_test_b200_prediction_cache"
    )
    parser.add_argument("--result-file", default="result.txt")
    parser.add_argument("--overwrite-result", action="store_true")
    parser.add_argument("--use-cuda", type=int, choices=(0, 1), default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    alphas = unique_floats(args.alphas, "--alphas")
    required_alphas = {1.00, 1.05, 1.07, 1.10}
    if set(alphas) != required_alphas:
        raise SystemExit(f"--alphas must contain exactly {sorted(required_alphas)}")
    ensemble_weights = unique_floats(
        args.ensemble_weights, "--ensemble-weights"
    )
    if not ensemble_weights or any(not 0.0 <= value <= 1.0 for value in ensemble_weights):
        raise SystemExit("--ensemble-weights must contain values in [0, 1]")
    exp2_alphas = unique_floats(args.exp2_alphas, "--exp2-alphas")
    required_exp2_alphas = {1.00, 1.05, 1.10}
    if set(exp2_alphas) != required_exp2_alphas:
        raise SystemExit(
            f"--exp2-alphas must contain exactly {sorted(required_exp2_alphas)}"
        )

    data_root = resolve_path(args.data_root)
    mesh_root = resolve_path(args.mesh_root)
    datalist = resolve_path(args.datalist)
    transform_config = resolve_path(args.transform_config)
    exp6_model_config = resolve_path(args.exp6_model_config)
    exp2_model_config = resolve_path(args.exp2_model_config)
    cache_root = resolve_path(args.cache_dir)
    result_file = resolve_path(args.result_file)
    if result_file.exists() and not args.overwrite_result and not args.check_only:
        raise FileExistsError(
            f"result file already exists: {result_file}; "
            "use --overwrite-result to replace it"
        )
    specs = {
        "exp6": PredictionSpec(
            "exp6_alpha100",
            "checkpoint_selection_b_exp6_b32_alpha100_straightpcf",
            exp6_model_config,
            resolve_path(args.exp6_checkpoint),
            1.00,
        ),
        "soup_top2": PredictionSpec(
            "exp6_soup_top2_alpha105",
            "checkpoint_selection_b_exp6_b32_alpha105_straightpcf_soup_top2",
            exp6_model_config,
            resolve_path(args.soup_top2_checkpoint),
            1.05,
        ),
        "soup_top3": PredictionSpec(
            "exp6_soup_top3_alpha105",
            "checkpoint_selection_b_exp6_b32_alpha105_straightpcf_soup_top3",
            exp6_model_config,
            resolve_path(args.soup_top3_checkpoint),
            1.05,
        ),
        "soup_top4": PredictionSpec(
            "exp6_soup_top4_alpha105",
            "checkpoint_selection_b_exp6_b32_alpha105_straightpcf_soup_top4",
            exp6_model_config,
            resolve_path(args.soup_top4_checkpoint),
            1.05,
        ),
        "soup_top5": PredictionSpec(
            "exp6_soup_top5_alpha105",
            "checkpoint_selection_b_exp6_b32_alpha105_straightpcf_soup_top5",
            exp6_model_config,
            resolve_path(args.soup_top5_checkpoint),
            1.05,
        ),
        "exp2": PredictionSpec(
            "exp2_alpha110",
            "checkpoint_selection_b_exp2_batch_straightpcf",
            exp2_model_config,
            resolve_path(args.exp2_checkpoint),
            1.10,
        ),
    }

    for path in (
        data_root,
        mesh_root,
        datalist,
        transform_config,
        exp6_model_config,
        exp2_model_config,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    for spec in specs.values():
        if not spec.checkpoint.is_file():
            raise FileNotFoundError(
                f"required checkpoint does not exist: {spec.checkpoint}"
            )
        load_yaml(spec.model_config)
    transform = load_yaml(transform_config)
    if transform.get("predict_transform", {}).get("augments", None) != []:
        raise ValueError("predict_transform.augments must be empty")

    samples = discover_samples(data_root, args.limit, datalist)
    print(f"local200 samples: {len(samples)}")
    print(f"result file: {result_file}")
    print(f"prediction cache: {cache_root}")
    print(f"exp6 alphas: {alphas}")
    print(f"exp2 ensemble alphas: {exp2_alphas}")
    print(f"ensemble exp6 weights: {ensemble_weights}")
    for spec in specs.values():
        print(
            f"candidate source: {spec.label} | alpha={spec.residual_alpha:g} | "
            f"{spec.checkpoint}"
        )
    if args.check_only:
        print("check-only complete; no model was loaded and no output was written")
        return 0

    import jittor as jt

    jt.flags.use_cuda = args.use_cuda
    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)
    for name in (
        "exp6",
        "soup_top2",
        "soup_top3",
        "soup_top4",
        "soup_top5",
        "exp2",
    ):
        ensure_prediction_cache(
            specs[name], samples, cache_root, transform_config, args.seed
        )

    results = evaluate_all(
        samples,
        mesh_root,
        cache_root,
        specs,
        alphas,
        exp2_alphas,
        ensemble_weights,
    )
    write_results(results, result_file, args.overwrite_result)
    print(f"results written: {result_file}")
    for result in results:
        print(
            f"{result['label']}: score={result['score']:.4f}, "
            f"CD_score={result['cd_score']:.4f}, "
            f"P2S_score={result['p2s_score']:.4f}, "
            f"cd_pred={result['cd_pred']:.8f}, "
            f"cd_noisy={result['cd_noisy']:.8f}, "
            f"p2s_pred={result['p2s_pred']:.8f}, "
            f"p2s_noisy={result['p2s_noisy']:.8f}"
        )
    ensemble_results = [
        result for result in results if "_ensemble(" in result["label"]
    ]
    for exp2_alpha in exp2_alphas:
        marker = f"+exp2_alpha{int(round(exp2_alpha * 100)):03d}_ensemble("
        alpha_results = [
            result for result in ensemble_results if marker in result["label"]
        ]
        best = max(alpha_results, key=lambda result: result["score"])
        print(
            f"best local ensemble for exp2 alpha={exp2_alpha:.2f}: "
            f"{best['label']} score={best['score']:.4f}"
        )
    best = max(ensemble_results, key=lambda result: result["score"])
    print(f"best local ensemble overall: {best['label']} score={best['score']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
