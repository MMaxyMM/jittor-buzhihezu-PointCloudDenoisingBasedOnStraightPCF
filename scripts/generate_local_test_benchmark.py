#!/usr/bin/env python3
"""Generate a calibrated 50k-point benchmark from dataset_train/local_test."""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import point_cloud_utils as pcu
import trimesh
from scipy.spatial import cKDTree
from tqdm import tqdm

# 混合噪声采样与训练侧 AugmentAddNoise 共用同一实现，禁止在此另写一套。
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from src.data.utils import sample_mixed_noise, validate_noise_mixture


OBJ_RELATIVE_PATH = Path("models/model_normalized.obj")


def stable_seed(global_seed: int, key: str, stream: str) -> int:
    payload = f"{global_seed}:{stream}:{key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (
        2**32
    )


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".npy", delete=False
        ) as file:
            temporary = file.name
            np.save(file, array, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def atomic_save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".npz", delete=False
        ) as file:
            temporary = file.name
            np.savez(file, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def load_mesh(path: Path) -> trimesh.Trimesh:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        geometries = tuple(mesh.geometry.values())
        if not geometries:
            raise ValueError(f"mesh scene contains no geometry: {path}")
        mesh = trimesh.util.concatenate(geometries)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"unsupported mesh type: {type(mesh).__name__}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError(f"mesh has no vertices or faces: {path}")
    return mesh


def sample_clean_points(
    mesh: trimesh.Trimesh,
    num_points: int,
    num_vertex_samples: int,
    seed: int,
) -> np.ndarray:
    """Keep up to N original vertices, then fill by area-weighted sampling."""
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    vertex_count = min(num_vertex_samples, num_points, len(vertices))
    rng = np.random.RandomState(seed)
    vertex_indices = rng.permutation(len(vertices))[:vertex_count]
    selected_vertices = vertices[vertex_indices]

    # trimesh.sample uses NumPy's module-level RNG.
    previous_state = np.random.get_state()
    try:
        np.random.seed(seed ^ 0xA5A5A5A5)
        surface_points, _ = trimesh.sample.sample_surface(
            mesh, num_points - vertex_count
        )
    finally:
        np.random.set_state(previous_state)

    clean = np.concatenate(
        [selected_vertices, np.asarray(surface_points, dtype=np.float32)],
        axis=0,
    )
    rng.shuffle(clean)
    if clean.shape != (num_points, 3):
        raise ValueError(f"unexpected clean point shape: {clean.shape}")
    return clean.astype(np.float32, copy=False)


def normalize_clean_and_mesh(
    clean: np.ndarray, mesh: trimesh.Trimesh
) -> Tuple[np.ndarray, trimesh.Trimesh, np.ndarray, float]:
    """Apply the competition clean-reference unit-sphere transform."""
    center = (clean.max(axis=0) + clean.min(axis=0)) / 2.0
    centered = clean - center
    scale = float(np.sqrt((centered**2).sum(axis=1)).max())
    if scale < 1e-12:
        raise ValueError("cannot normalize a degenerate point cloud")

    normalized_mesh = mesh.copy()
    normalized_mesh.vertices = (
        np.asarray(mesh.vertices, dtype=np.float64) - center
    ) / scale
    return (
        (centered / scale).astype(np.float32, copy=False),
        normalized_mesh,
        center.astype(np.float32),
        scale,
    )


def add_laplace_noise(
    clean: np.ndarray,
    key: str,
    seed: int,
    std_min: float,
    std_max: float,
    noise_scale: float,
) -> Tuple[np.ndarray, float]:
    """Match AugmentAddNoise: Laplace b = configured_std / sqrt(2)."""
    rng = np.random.RandomState(stable_seed(seed, key, "noise"))
    effective_std = float(rng.uniform(std_min, std_max)) * noise_scale
    noise = rng.laplace(
        0.0,
        effective_std / math.sqrt(2.0),
        size=clean.shape,
    ).astype(np.float32)
    return (clean + noise).astype(np.float32, copy=False), effective_std


def add_mixed_noise(
    clean: np.ndarray,
    key: str,
    seed: int,
    mixture: list,
    noise_scale: float,
) -> Tuple[np.ndarray, float]:
    """混合噪声路径（--noise_mixture 启用）。

    成分的 min/max 语义与训练侧 AugmentAddNoise 完全一致：laplace 为 scale
    b、gaussian 为 sigma（注意这与本文件 add_laplace_noise 旧路径的
    "std、b=std/√2" 语义不同；noise_scale 标定乘子会吸收整体差异）。
    """
    rng = np.random.RandomState(stable_seed(seed, key, "noise"))
    noise, _, scale = sample_mixed_noise(clean.shape, mixture, rng=rng)
    noise = (noise * noise_scale).astype(np.float32)
    return (clean + noise).astype(np.float32, copy=False), scale * noise_scale


def chamfer_distance(a: np.ndarray, b: np.ndarray) -> float:
    tree_b = cKDTree(b)
    distance_a, _ = tree_b.query(a, k=1)
    tree_a = cKDTree(a)
    distance_b, _ = tree_a.query(b, k=1)
    return float((distance_a**2).mean() + (distance_b**2).mean())


def point_to_surface_distance(
    points: np.ndarray, mesh: trimesh.Trimesh
) -> float:
    distances, _, _ = pcu.closest_points_on_mesh(
        np.asarray(points, dtype=np.float32),
        np.asarray(mesh.vertices, dtype=np.float32),
        np.asarray(mesh.faces, dtype=np.int32),
    )
    return float((distances**2).mean())


def create_sample(task: Dict) -> Dict:
    entry = task["entry"]
    mesh_path = Path(task["dataset_dir"]) / entry / OBJ_RELATIVE_PATH
    mesh = load_mesh(mesh_path)
    raw_clean = sample_clean_points(
        mesh,
        task["num_points"],
        task["num_vertex_samples"],
        stable_seed(task["seed"], entry, "clean"),
    )
    clean, normalized_mesh, center, normalization_scale = (
        normalize_clean_and_mesh(raw_clean, mesh)
    )
    if task.get("noise_mixture"):
        noisy, effective_std = add_mixed_noise(
            clean,
            entry,
            task["seed"],
            task["noise_mixture"],
            task["noise_scale"],
        )
    else:
        noisy, effective_std = add_laplace_noise(
            clean,
            entry,
            task["seed"],
            task["noise_std_min"],
            task["noise_std_max"],
            task["noise_scale"],
        )
    cd = chamfer_distance(noisy, clean)
    p2s = point_to_surface_distance(noisy, normalized_mesh)

    if task["write"]:
        destination = Path(task["output_dir"]) / entry
        atomic_save_npy(destination / "clean.npy", clean)
        atomic_save_npy(destination / "noisy.npy", noisy)
        atomic_save_npz(
            destination / "normalization.npz",
            center=center,
            scale=np.float32(normalization_scale),
        )

    return {
        "key": entry,
        "noise_std": effective_std,
        "CD_noisy": cd,
        "P2S_noisy": p2s,
    }


def base_task(args, entry: str, noise_scale: float, write: bool) -> Dict:
    return {
        "entry": entry,
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
        "num_points": args.num_points,
        "num_vertex_samples": args.num_vertex_samples,
        "seed": args.seed,
        "noise_std_min": args.noise_std_min,
        "noise_std_max": args.noise_std_max,
        "noise_mixture": getattr(args, "noise_mixture", None),
        "noise_scale": noise_scale,
        "write": write,
    }


def run_tasks(tasks: Sequence[Dict], workers: int, description: str):
    if workers == 1:
        return [
            create_sample(task)
            for task in tqdm(tasks, desc=description)
        ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        return list(
            tqdm(
                executor.map(create_sample, tasks, chunksize=1),
                total=len(tasks),
                desc=description,
            )
        )


def summarize(rows: Sequence[Dict]) -> Dict:
    if not rows:
        raise RuntimeError("no samples were evaluated")
    return {
        "samples": len(rows),
        "mean_CD_noisy": float(
            np.mean([row["CD_noisy"] for row in rows])
        ),
        "mean_P2S_noisy": float(
            np.mean([row["P2S_noisy"] for row in rows])
        ),
    }


def load_entries(local_test_root: Path) -> List[str]:
    datalist = local_test_root / "datalist.txt"
    if not datalist.is_file():
        raise SystemExit(f"local holdout datalist does not exist: {datalist}")
    entries = [
        line.strip()
        for line in datalist.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not entries:
        raise SystemExit(f"local holdout datalist is empty: {datalist}")
    return sorted(entries)


def calibration_subset(
    entries: Sequence[str], limit: int, seed: int
) -> List[str]:
    """Round-robin categories so all categories enter calibration first."""
    by_category: Dict[str, List[str]] = defaultdict(list)
    for entry in entries:
        by_category[Path(entry).parts[1]].append(entry)
    rng = np.random.RandomState(seed)
    for category_entries in by_category.values():
        rng.shuffle(category_entries)

    selected = []
    categories = sorted(by_category)
    position = 0
    target = min(limit, len(entries))
    while len(selected) < target:
        made_progress = False
        for category in categories:
            values = by_category[category]
            if position < len(values):
                selected.append(values[position])
                made_progress = True
                if len(selected) == target:
                    break
        if not made_progress:
            break
        position += 1
    return selected


def calibration_error(metrics: Dict, args) -> float:
    return (
        math.log(metrics["mean_CD_noisy"] / args.target_cd) ** 2
        + math.log(metrics["mean_P2S_noisy"] / args.target_p2s) ** 2
    )


def evaluate_scale(entries, args, noise_scale):
    rows = run_tasks(
        [
            base_task(args, entry, noise_scale, write=False)
            for entry in entries
        ],
        args.workers,
        f"Calibrate scale={noise_scale:.5f}",
    )
    return summarize(rows)


def calibrate(entries, args):
    """Use target-derived estimates followed by residual-based refinement."""
    initial = evaluate_scale(entries, args, 1.0)
    scale_cd = math.sqrt(args.target_cd / initial["mean_CD_noisy"])
    scale_p2s = math.sqrt(
        args.target_p2s / initial["mean_P2S_noisy"]
    )
    center = math.sqrt(scale_cd * scale_p2s)
    candidates = sorted(
        {
            1.0,
            scale_cd,
            scale_p2s,
            center * 0.97,
            center,
            center * 1.03,
        }
    )

    records = []
    evaluated = {}

    def evaluate_candidate(scale):
        key = round(float(scale), 12)
        if key in evaluated:
            return
        metrics = (
            initial
            if abs(scale - 1.0) < 1e-12
            else evaluate_scale(entries, args, scale)
        )
        record = {
            "noise_scale": scale,
            **metrics,
            "relative_CD_error": (
                metrics["mean_CD_noisy"] / args.target_cd - 1.0
            ),
            "relative_P2S_error": (
                metrics["mean_P2S_noisy"] / args.target_p2s - 1.0
            ),
        }
        record["objective"] = calibration_error(metrics, args)
        records.append(record)
        evaluated[key] = record
        print(json.dumps(record, ensure_ascii=False), flush=True)

    for scale in candidates:
        evaluate_candidate(scale)

    preliminary = min(records, key=lambda row: row["objective"])
    corrected_cd = preliminary["noise_scale"] * math.sqrt(
        args.target_cd / preliminary["mean_CD_noisy"]
    )
    corrected_p2s = preliminary["noise_scale"] * math.sqrt(
        args.target_p2s / preliminary["mean_P2S_noisy"]
    )
    refined_center = math.sqrt(corrected_cd * corrected_p2s)
    refined_candidates = {
        corrected_cd,
        corrected_p2s,
        refined_center * 0.985,
        refined_center,
        refined_center * 1.015,
    }
    for scale in sorted(refined_candidates):
        evaluate_candidate(scale)

    best = min(records, key=lambda row: row["objective"])
    return float(best["noise_scale"]), records


def write_metrics(output_root: Path, rows: Sequence[Dict]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["key", "noise_std", "CD_noisy", "P2S_noisy"],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", default="dataset_train")
    parser.add_argument(
        "--local_test_dir", default="dataset_train/local_test"
    )
    parser.add_argument("--output_dir", default="dataset_local_test")
    parser.add_argument("--num_points", type=int, default=50000)
    parser.add_argument("--num_vertex_samples", type=int, default=1024)
    parser.add_argument("--noise_std_min", type=float, default=0.005)
    parser.add_argument("--noise_std_max", type=float, default=0.020)
    parser.add_argument(
        "--noise_mixture",
        type=str,
        default=None,
        help=(
            'JSON 混合噪声配置，设置后取代 --noise_std_min/max，例如：'
            '\'[{"type":"laplace","weight":0.6,"min":0.0075,"max":0.0125},'
            '{"type":"gaussian","weight":0.2,"min":0.0125,"max":0.020}]\'。'
            "laplace 的 min/max 为 scale b，gaussian 为 sigma。"
        ),
    )
    parser.add_argument("--target_cd", type=float, default=0.000246)
    parser.add_argument("--target_p2s", type=float, default=0.000196)
    parser.add_argument(
        "--calibration_limit",
        type=int,
        default=0,
        help="Calibration sample count; 0 uses the complete local_test.",
    )
    parser.add_argument("--noise_scale", type=float, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=min(os.cpu_count() or 1, 16),
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing benchmark output directory.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.num_points <= 0:
        raise SystemExit("--num_points must be positive")
    if not 0 <= args.num_vertex_samples <= args.num_points:
        raise SystemExit("invalid --num_vertex_samples")
    if not 0 < args.noise_std_min <= args.noise_std_max:
        raise SystemExit("invalid noise std range")
    if args.noise_mixture is not None:
        try:
            args.noise_mixture = validate_noise_mixture(
                json.loads(args.noise_mixture)
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise SystemExit(f"invalid --noise_mixture: {exc}") from exc
    if args.target_cd <= 0 or args.target_p2s <= 0:
        raise SystemExit("target metrics must be positive")
    if args.calibration_limit < 0 or args.workers <= 0:
        raise SystemExit("calibration_limit must be non-negative and workers positive")
    if args.noise_scale is not None and args.noise_scale <= 0:
        raise SystemExit("--noise_scale must be positive")
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    output_root = Path(args.output_dir)
    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"output directory is not empty: {output_root}; "
            "use --overwrite to replace generated files"
        )


def main() -> int:
    args = parse_args()
    validate_args(args)
    entries = load_entries(Path(args.local_test_dir))
    if args.limit is not None:
        entries = entries[: args.limit]

    calibration_records = []
    if args.noise_scale is None:
        calibration_limit = (
            len(entries)
            if args.calibration_limit == 0
            else args.calibration_limit
        )
        calibration_entries = calibration_subset(
            entries, calibration_limit, args.seed
        )
        noise_scale, calibration_records = calibrate(
            calibration_entries, args
        )
    else:
        noise_scale = args.noise_scale
    print(f"Selected noise_scale={noise_scale:.8f}", flush=True)

    rows = run_tasks(
        [
            base_task(args, entry, noise_scale, write=True)
            for entry in entries
        ],
        args.workers,
        "Generate benchmark",
    )
    final_metrics = summarize(rows)
    output_root = Path(args.output_dir)
    write_metrics(output_root, rows)
    report = {
        "dataset_dir": str(Path(args.dataset_dir).resolve()),
        "local_test_dir": str(Path(args.local_test_dir).resolve()),
        "output_dir": str(output_root.resolve()),
        "num_points": args.num_points,
        "num_vertex_samples": args.num_vertex_samples,
        "noise_type": "mixture" if args.noise_mixture else "laplace",
        "noise_mixture": args.noise_mixture,
        "noise_std_min_before_scale": args.noise_std_min,
        "noise_std_max_before_scale": args.noise_std_max,
        "noise_scale": noise_scale,
        "effective_noise_std_min": args.noise_std_min * noise_scale,
        "effective_noise_std_max": args.noise_std_max * noise_scale,
        "target_CD_noisy": args.target_cd,
        "target_P2S_noisy": args.target_p2s,
        "workers": args.workers,
        "calibration": calibration_records,
        "final": final_metrics,
    }
    (output_root / "generation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final_metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
