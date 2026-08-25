#!/usr/bin/env python3
"""Conditionally evaluate independent TTA2 candidates for exp6 Soup Top-4.

The existing alpha=1.05 Soup Top-4 cache is reused for the unrotated branch.
Each rotated branch is inferred once with alpha=1.00, inverse-rotated, and
cached.  The two aligned base predictions are averaged before residual alpha
is applied, so alpha 1.03/1.04/1.05 can be evaluated without more inference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_b_local200_candidates import (  # noqa: E402
    MetricAccumulator,
    PredictionSpec,
    atomic_save_npy,
    build_model,
    cache_file_is_valid,
    checkpoint_identity,
    load_reference,
    load_yaml,
    metrics_for_cloud,
    resolve_path,
    sample_digest,
    validate_cloud,
)
from scripts.evaluate_local_test_models import discover_samples  # noqa: E402


@dataclass(frozen=True)
class RotationSpec:
    name: str
    matrix: np.ndarray


ROTATIONS: Dict[str, RotationSpec] = {
    "z90": RotationSpec(
        "z90",
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32),
    ),
    "x90": RotationSpec(
        "x90",
        np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32),
    ),
    "cycle_xyz": RotationSpec(
        "cycle_xyz",
        np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float32),
    ),
}


def validate_rotation(rotation: RotationSpec) -> None:
    matrix = rotation.matrix.astype(np.float64)
    if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-7):
        raise ValueError(f"rotation is not orthonormal: {rotation.name}")
    if not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-7):
        raise ValueError(f"rotation determinant is not +1: {rotation.name}")


def rotated_prediction_path(cache_root: Path, rotation: RotationSpec, key: str) -> Path:
    return cache_root / f"soup_top4_tta_{rotation.name}_alpha100" / key / "denoised.npy"


def expected_manifest(
    model_spec: PredictionSpec,
    rotation: RotationSpec,
    samples: Sequence[Tuple[str, Path]],
    seed: int,
) -> dict:
    return {
        "cache_version": 1,
        "model_label": model_spec.label,
        "model_config": checkpoint_identity(model_spec.model_config),
        "checkpoint": checkpoint_identity(model_spec.checkpoint),
        "model_residual_alpha": model_spec.residual_alpha,
        "rotation_name": rotation.name,
        "rotation_matrix": rotation.matrix.tolist(),
        "inverse_rotation": "row-vector output @ rotation_matrix",
        "fusion_mode": "best",
        "predict_rounds": 1,
        "seed": seed,
        "sample_count": len(samples),
        "sample_digest": sample_digest(samples),
    }


def prepare_manifest(
    cache_root: Path,
    model_spec: PredictionSpec,
    rotation: RotationSpec,
    samples: Sequence[Tuple[str, Path]],
    seed: int,
) -> None:
    model_cache = cache_root / f"soup_top4_tta_{rotation.name}_alpha100"
    manifest_path = model_cache / "cache_manifest.json"
    expected = expected_manifest(model_spec, rotation, samples, seed)
    if manifest_path.is_file():
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        if actual != expected:
            raise RuntimeError(
                f"TTA cache does not match current request: {model_cache}. "
                "Use a different --tta-cache-dir; old caches are not deleted."
            )
        return
    if model_cache.exists() and any(model_cache.iterdir()):
        raise RuntimeError(f"non-empty TTA cache has no manifest: {model_cache}")
    model_cache.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(expected, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def ensure_rotated_cache(
    model_spec: PredictionSpec,
    rotation: RotationSpec,
    samples: Sequence[Tuple[str, Path]],
    cache_root: Path,
    transform_config: Path,
    seed: int,
) -> None:
    import jittor as jt

    prepare_manifest(cache_root, model_spec, rotation, samples, seed)
    missing: List[Tuple[str, Path, int]] = []
    for key, model_dir in samples:
        noisy_path = model_dir / "noisy.npy"
        noisy = np.load(noisy_path, mmap_mode="r", allow_pickle=False)
        if noisy.ndim != 2 or noisy.shape[1] != 3:
            raise ValueError(f"invalid noisy cloud: {noisy_path}: {noisy.shape}")
        output_path = rotated_prediction_path(cache_root, rotation, key)
        if not cache_file_is_valid(output_path, noisy.shape[0]):
            missing.append((key, model_dir, noisy.shape[0]))

    if not missing:
        print(f"[TTA {rotation.name}] prediction cache complete; skip inference")
        return

    print(f"[TTA {rotation.name}] missing cached predictions: {len(missing)}/{len(samples)}")
    model = build_model(model_spec, transform_config)
    start = time.time()
    matrix = rotation.matrix
    for index, (key, model_dir, expected_points) in enumerate(missing, start=1):
        noisy_path = model_dir / "noisy.npy"
        noisy = validate_cloud(
            np.load(noisy_path, allow_pickle=False), noisy_path, expected_points
        )
        noisy_rotated = (noisy @ matrix.T).astype(np.float32, copy=False)
        with jt.no_grad():
            output = model.predict_step({"pc_noisy": jt.array(noisy_rotated[None])})
        prediction_rotated = output[0]["pc_denoised"]
        if not isinstance(prediction_rotated, np.ndarray):
            prediction_rotated = prediction_rotated.numpy()
        prediction_rotated = validate_cloud(
            np.asarray(prediction_rotated),
            Path(f"<TTA prediction:{rotation.name}:{key}>"),
            expected_points,
        )
        prediction_back = (prediction_rotated @ matrix).astype(np.float32, copy=False)
        atomic_save_npy(
            rotated_prediction_path(cache_root, rotation, key), prediction_back
        )
        del output, prediction_rotated, prediction_back
        elapsed = time.time() - start
        remaining = elapsed / index * (len(missing) - index)
        print(
            f"\r[TTA {rotation.name}] inference {index}/{len(missing)} "
            f"elapsed={elapsed / 60:.1f}m remaining={remaining / 60:.1f}m",
            end="",
            flush=True,
        )
    print()
    del model
    jt.gc()


def original_prediction_path(original_cache: Path, key: str) -> Path:
    return original_cache / key / "denoised.npy"


def tta_label(rotation: RotationSpec, alpha: float) -> str:
    alpha_value = int(round(alpha * 100))
    return (
        "checkpoint_selection_b_exp6_b32_alpha105_straightpcf_soup_top4"
        f"_tta2_{rotation.name}_alpha{alpha_value:03d}"
    )


def append_results(results: Sequence[dict], result_file: Path) -> None:
    """Append candidates without replacing any existing result content."""
    result_file.parent.mkdir(parents=True, exist_ok=True)
    needs_separator = False
    if result_file.is_file() and result_file.stat().st_size > 0:
        with result_file.open("rb") as file:
            file.seek(-1, 2)
            needs_separator = file.read(1) != b"\n"
    with result_file.open("a", encoding="utf-8") as file:
        if needs_separator:
            file.write("\n")
        for result in results:
            file.write(
                f"{result['label']}：{result['score']:.4f}，"
                f"{result['cd_pred']:.8f}，{result['cd_noisy']:.8f}，"
                f"{result['p2s_pred']:.8f}，{result['p2s_noisy']:.8f}\n"
            )


def evaluate_rotation(
    samples: Sequence[Tuple[str, Path]],
    mesh_root: Path,
    original_cache: Path,
    tta_cache: Path,
    rotation: RotationSpec,
    alphas: Sequence[float],
    include_baseline: bool = False,
) -> Tuple[List[dict], dict | None]:
    accumulators = {
        alpha: MetricAccumulator(tta_label(rotation, alpha)) for alpha in alphas
    }
    baseline = MetricAccumulator("soup_top4_alpha105_reference") if include_baseline else None
    start = time.time()
    for index, (key, model_dir) in enumerate(samples, start=1):
        reference = load_reference(key, model_dir, mesh_root)
        noisy = reference["noisy"]
        cd_noisy, p2s_noisy = metrics_for_cloud(noisy, reference)
        original_alpha105 = validate_cloud(
            np.load(original_prediction_path(original_cache, key), allow_pickle=False),
            Path(f"<original Soup Top-4 cache:{key}>"),
            noisy.shape[0],
        )
        rotated_base = validate_cloud(
            np.load(
                rotated_prediction_path(tta_cache, rotation, key),
                allow_pickle=False,
            ),
            Path(f"<TTA cache:{rotation.name}:{key}>"),
            noisy.shape[0],
        )
        original_base = noisy + (original_alpha105 - noisy) / 1.05
        tta_base = 0.5 * (original_base + rotated_base)

        if baseline is not None:
            cd_pred, p2s_pred = metrics_for_cloud(original_alpha105, reference)
            baseline.append(cd_pred, cd_noisy, p2s_pred, p2s_noisy)

        for alpha, accumulator in accumulators.items():
            prediction = (noisy + alpha * (tta_base - noisy)).astype(
                np.float32, copy=False
            )
            cd_pred, p2s_pred = metrics_for_cloud(prediction, reference)
            accumulator.append(cd_pred, cd_noisy, p2s_pred, p2s_noisy)

        elapsed = time.time() - start
        remaining = elapsed / index * (len(samples) - index)
        print(
            f"\r[TTA {rotation.name} metrics] {index}/{len(samples)} "
            f"elapsed={elapsed / 60:.1f}m remaining={remaining / 60:.1f}m",
            end="",
            flush=True,
        )
    print()
    return (
        [accumulators[alpha].summarize() for alpha in alphas],
        baseline.summarize() if baseline is not None else None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="dataset_train_pcd_disk/local_test")
    parser.add_argument("--mesh-root", default="dataset_train/local_test")
    parser.add_argument("--datalist", default="dataset_train/local_test/datalist.txt")
    parser.add_argument("--transform-config", default="configs/transform/predict.yaml")
    parser.add_argument(
        "--model-config", default="configs/model/straightpcf_b_exp6_b32_alpha100.yaml"
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoint_selection_b_exp6_b32_alpha105_straightpcf_soup_top4/"
            "best_checkpoint.pkl"
        ),
    )
    parser.add_argument(
        "--original-cache",
        default="outputs/local_test_b200_prediction_cache/exp6_soup_top4_alpha105",
    )
    parser.add_argument("--tta-cache-dir", default="outputs/local_test_b200_tta_cache")
    parser.add_argument("--result-file", default="result_tta.txt")
    parser.add_argument("--alphas", nargs="+", type=float, default=(1.03, 1.04, 1.05))
    parser.add_argument("--min-improvement", type=float, default=0.0)
    parser.add_argument("--force-all-rotations", action="store_true")
    parser.add_argument("--use-cuda", type=int, choices=(0, 1), default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    alphas = sorted(set(float(value) for value in args.alphas))
    if alphas != [1.03, 1.04, 1.05]:
        raise SystemExit("--alphas must contain exactly 1.03 1.04 1.05")
    if not np.isfinite(args.min_improvement) or args.min_improvement < 0:
        raise SystemExit("--min-improvement must be finite and non-negative")
    for rotation in ROTATIONS.values():
        validate_rotation(rotation)

    data_root = resolve_path(args.data_root)
    mesh_root = resolve_path(args.mesh_root)
    datalist = resolve_path(args.datalist)
    transform_config = resolve_path(args.transform_config)
    model_config = resolve_path(args.model_config)
    checkpoint = resolve_path(args.checkpoint)
    original_cache = resolve_path(args.original_cache)
    tta_cache = resolve_path(args.tta_cache_dir)
    result_file = resolve_path(args.result_file)
    for path in (
        data_root,
        mesh_root,
        datalist,
        transform_config,
        model_config,
        checkpoint,
        original_cache,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    transform = load_yaml(transform_config)
    if transform.get("predict_transform", {}).get("augments", None) != []:
        raise ValueError("predict_transform.augments must remain empty for TTA")
    load_yaml(model_config)

    samples = discover_samples(data_root, args.limit, datalist)
    for key, model_dir in samples:
        noisy = np.load(model_dir / "noisy.npy", mmap_mode="r", allow_pickle=False)
        if not cache_file_is_valid(original_prediction_path(original_cache, key), noisy.shape[0]):
            raise FileNotFoundError(f"missing original Soup Top-4 cache for {key}")
    print(f"samples: {len(samples)}")
    print(f"original Soup Top-4 cache: {original_cache}")
    print(f"TTA cache: {tta_cache}")
    print(f"result file: {result_file}")
    print("stage 1: z90 alpha=1.05; stage 2 is conditional; stage 3 scans 1.03/1.04/1.05")
    if args.check_only:
        print("check-only complete; no model loaded and no output written")
        return 0

    import jittor as jt

    jt.flags.use_cuda = args.use_cuda
    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)
    model_spec = PredictionSpec(
        "soup_top4_tta_base_alpha100",
        "checkpoint_selection_b_exp6_b32_alpha105_straightpcf_soup_top4",
        model_config,
        checkpoint,
        1.00,
    )

    z90 = ROTATIONS["z90"]
    ensure_rotated_cache(
        model_spec, z90, samples, tta_cache, transform_config, args.seed
    )
    z_results, baseline = evaluate_rotation(
        samples,
        mesh_root,
        original_cache,
        tta_cache,
        z90,
        [1.05],
        include_baseline=True,
    )
    assert baseline is not None
    results = list(z_results)
    z_score = z_results[0]["score"]
    threshold = baseline["score"] + args.min_improvement
    improved = z_score > threshold
    print(
        f"baseline Soup Top-4 alpha=1.05: {baseline['score']:.6f}; "
        f"z90 TTA2 alpha=1.05: {z_score:.6f}; delta={z_score-baseline['score']:+.6f}"
    )

    rotation_results = {"z90": z_results[0]}
    if improved or args.force_all_rotations:
        for name in ("x90", "cycle_xyz"):
            rotation = ROTATIONS[name]
            ensure_rotated_cache(
                model_spec,
                rotation,
                samples,
                tta_cache,
                transform_config,
                args.seed,
            )
            evaluated, _ = evaluate_rotation(
                samples,
                mesh_root,
                original_cache,
                tta_cache,
                rotation,
                [1.05],
            )
            rotation_results[name] = evaluated[0]
            results.extend(evaluated)
    else:
        print("z90 did not improve at alpha=1.05; skip x90/cycle_xyz")

    best_name = max(
        rotation_results,
        key=lambda name: rotation_results[name]["score"],
    )
    best_rotation = ROTATIONS[best_name]
    print(
        f"best tested TTA rotation at alpha=1.05: {best_name} "
        f"score={rotation_results[best_name]['score']:.6f}"
    )
    alpha_results, _ = evaluate_rotation(
        samples,
        mesh_root,
        original_cache,
        tta_cache,
        best_rotation,
        [1.03, 1.04],
    )
    results.extend(alpha_results)

    append_results(results, result_file)
    print(f"results appended: {result_file}")
    for result in results:
        print(
            f"{result['label']}: score={result['score']:.6f}, "
            f"CD_score={result['cd_score']:.6f}, "
            f"P2S_score={result['p2s_score']:.6f}, "
            f"cd_pred={result['cd_pred']:.8f}, "
            f"p2s_pred={result['p2s_pred']:.8f}"
        )
    best = max(results, key=lambda result: result["score"])
    print(f"best new TTA candidate: {best['label']} score={best['score']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
