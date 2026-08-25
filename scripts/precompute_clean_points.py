#!/usr/bin/env python3
"""Build split clean-point caches or estimate noise in noisy point clouds.

Cache mode (default):
    python scripts/precompute_clean_points.py

Noise-estimation mode:
    python scripts/precompute_clean_points.py --mode estimate-noise \
        --noisy_input_dir dataset_test_noisy

The cache mode reads dataset_train/local_train/datalist.txt and
dataset_train/local_test/datalist.txt.  It writes:

    dataset_train_pcd_disk/
      local_train/shapenet/<synset>/<model>/clean.npy
      local_train/shapenet/<synset>/<model>/vertices.npy
      local_test/shapenet/<synset>/<model>/clean.npy
      local_test/shapenet/<synset>/<model>/noisy.npy
      local_test/shapenet/<synset>/<model>/normalization.npz

Training caches default to 200,000 surface samples so every epoch can draw a
different 32,768-point subset and add fresh noise.  Local-test clouds default
to one deterministic 50,000-point clean/noisy pair shared by every checkpoint.
All point arrays are float32.
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

# 与训练侧共用混合噪声校验逻辑（src/data/__init__.py 为空，此导入很轻）。
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from src.data.utils import validate_noise_mixture


OBJ_RELATIVE_PATH = Path("models/model_normalized.obj")
OFFICIAL_NOISE_STD_MIN = 0.005
OFFICIAL_NOISE_STD_MAX = 0.020


def stable_seed(global_seed: int, split: str, relative_path: Path) -> int:
    key = f"{global_seed}:{split}:{relative_path.as_posix()}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little") % (2**32)


def load_mesh(path: Path) -> Any:
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError(
            "cache mode requires trimesh; install project requirements first"
        ) from exc

    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        geometries = tuple(mesh.geometry.values())
        if not geometries:
            raise ValueError("mesh scene contains no geometry")
        mesh = trimesh.util.concatenate(geometries)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"unsupported mesh type: {type(mesh).__name__}")
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise ValueError("mesh has no vertices or faces")
    return mesh


def atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[str] = None
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


def valid_existing_cache(
    clean_path: Path, vertices_path: Path, num_points: int
) -> bool:
    if not clean_path.is_file() or not vertices_path.is_file():
        return False
    try:
        clean = np.load(clean_path, mmap_mode="r", allow_pickle=False)
        vertices = np.load(vertices_path, mmap_mode="r", allow_pickle=False)
        return (
            clean.shape == (num_points, 3)
            and clean.dtype == np.float32
            and vertices.ndim == 2
            and vertices.shape[1] == 3
            and vertices.dtype == np.float32
        )
    except (OSError, ValueError):
        return False


CacheTask = Tuple[str, str, str, str, int, int, bool]


def cache_one(task: CacheTask) -> Tuple[str, str]:
    (
        dataset_root_s,
        output_root_s,
        split,
        relative_s,
        num_points,
        seed,
        overwrite,
    ) = task
    dataset_root = Path(dataset_root_s)
    output_root = Path(output_root_s)
    relative = Path(relative_s)
    # Prefer the split tree itself.  This works whether local_train/local_test
    # contains real directories or persistent relative symlinks.  The fallback
    # keeps compatibility with datalists that only reference the original tree.
    source = dataset_root / split / relative / OBJ_RELATIVE_PATH
    if not source.is_file():
        source = dataset_root / relative / OBJ_RELATIVE_PATH
    destination_root = output_root / split / relative
    clean_path = destination_root / "clean.npy"
    vertices_path = destination_root / "vertices.npy"
    label = f"{split}/{relative.as_posix()}"

    if not overwrite and valid_existing_cache(
        clean_path, vertices_path, num_points
    ):
        return "skipped", label

    try:
        if not source.is_file():
            raise FileNotFoundError(source)
        import trimesh

        mesh = load_mesh(source)
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"unexpected mesh vertex shape: {vertices.shape}")
        if not np.isfinite(vertices).all():
            raise ValueError("mesh vertices contain non-finite values")

        # trimesh.sample uses NumPy's module-level random state.  Each worker
        # handles one mesh at a time, so a deterministic per-model seed is safe.
        np.random.seed(stable_seed(seed, split, relative))
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        points = np.asarray(points, dtype=np.float32)
        if points.shape != (num_points, 3):
            raise ValueError(f"unexpected sampled shape: {points.shape}")
        if not np.isfinite(points).all():
            raise ValueError("sampled points contain non-finite values")

        atomic_save_npy(clean_path, points)
        atomic_save_npy(vertices_path, vertices)
        return "written", label
    except Exception as exc:
        return "failed", f"{label}: {type(exc).__name__}: {exc}"


def load_split_entries(dataset_root: Path, split: str) -> List[Path]:
    split_root = dataset_root / split
    datalist = split_root / "datalist.txt"
    if datalist.is_file():
        entries = [
            Path(line.strip())
            for line in datalist.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        entries = [
            path.parent.parent.relative_to(split_root)
            for path in sorted(
                split_root.glob("shapenet/*/*/models/model_normalized.obj")
            )
        ]
    if not entries:
        raise SystemExit(
            f"no mesh entries found for split '{split}' under {split_root}"
        )
    invalid = [
        entry
        for entry in entries
        if len(entry.parts) != 3 or entry.parts[0] != "shapenet"
    ]
    if invalid:
        raise SystemExit(
            f"invalid datalist entry in {datalist}: {invalid[0].as_posix()}"
        )
    return sorted(set(entries), key=lambda item: item.as_posix())


def run_cache_tasks(
    tasks: Sequence[CacheTask], workers: int
) -> Tuple[Dict[str, int], List[str]]:
    counts = {"written": 0, "skipped": 0, "failed": 0}
    failures: List[str] = []
    if workers == 1:
        results: Iterable[Tuple[str, str]] = map(cache_one, tasks)
        for status, message in tqdm(
            results, total=len(tasks), desc="Sampling meshes"
        ):
            counts[status] += 1
            if status == "failed":
                failures.append(message)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = executor.map(cache_one, tasks, chunksize=1)
            for status, message in tqdm(
                results, total=len(tasks), desc="Sampling meshes"
            ):
                counts[status] += 1
                if status == "failed":
                    failures.append(message)
    return counts, failures


def fixed_test_is_complete(
    output_root: Path, entries: Sequence[str], num_points: int
) -> bool:
    for entry in entries:
        sample_root = output_root / entry
        clean_path = sample_root / "clean.npy"
        noisy_path = sample_root / "noisy.npy"
        normalization_path = sample_root / "normalization.npz"
        if not (
            clean_path.is_file()
            and noisy_path.is_file()
            and normalization_path.is_file()
        ):
            return False
        try:
            clean = np.load(clean_path, mmap_mode="r", allow_pickle=False)
            noisy = np.load(noisy_path, mmap_mode="r", allow_pickle=False)
            with np.load(normalization_path, allow_pickle=False) as transform:
                valid_transform = (
                    "center" in transform
                    and "scale" in transform
                    and transform["center"].shape == (3,)
                    and np.asarray(transform["scale"]).size == 1
                )
            if not (
                clean.shape == (num_points, 3)
                and noisy.shape == clean.shape
                and clean.dtype == np.float32
                and noisy.dtype == np.float32
                and valid_transform
            ):
                return False
        except (OSError, ValueError):
            return False
    return True


def generate_fixed_local_test(
    dataset_root: Path,
    output_root: Path,
    entries: Sequence[str],
    num_points: int,
    args: argparse.Namespace,
) -> Dict:
    """Generate one deterministic noisy benchmark shared by all checkpoints."""
    if fixed_test_is_complete(
        output_root, entries, num_points
    ) and not args.overwrite:
        return {
            "status": "skipped",
            "samples": len(entries),
            "reason": "all fixed local-test files are already valid",
        }

    if output_root.exists() and any(output_root.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"incomplete local-test output exists: {output_root}; "
            "use --overwrite to regenerate it consistently"
        )

    try:
        import generate_local_test_benchmark as benchmark
    except ImportError as exc:
        raise SystemExit(
            "local_test generation requires "
            "scripts/generate_local_test_benchmark.py and its dependencies"
        ) from exc

    benchmark_args = SimpleNamespace(
        dataset_dir=str(dataset_root),
        output_dir=str(output_root),
        num_points=num_points,
        num_vertex_samples=args.num_vertex_samples,
        seed=args.seed,
        noise_std_min=args.noise_std_min,
        noise_std_max=args.noise_std_max,
        noise_mixture=args.noise_mixture,
        target_cd=args.target_cd,
        target_p2s=args.target_p2s,
        calibration_limit=args.calibration_limit,
        noise_scale=args.noise_scale,
        workers=args.workers,
    )

    calibration_records = []
    if args.noise_scale is None:
        calibration_count = (
            len(entries)
            if args.calibration_limit == 0
            else min(args.calibration_limit, len(entries))
        )
        calibration_entries = benchmark.calibration_subset(
            entries, calibration_count, args.seed
        )
        noise_scale, calibration_records = benchmark.calibrate(
            calibration_entries, benchmark_args
        )
    else:
        noise_scale = args.noise_scale
    print(f"Selected fixed local-test noise_scale={noise_scale:.8f}")

    rows = benchmark.run_tasks(
        [
            benchmark.base_task(
                benchmark_args, entry, noise_scale, write=True
            )
            for entry in entries
        ],
        args.workers,
        "Generate fixed local_test",
    )
    metrics = benchmark.summarize(rows)
    benchmark.write_metrics(output_root, rows)

    # vertices.npy belongs to the dynamic training cache, not the fixed test.
    for entry in entries:
        obsolete_vertices = output_root / entry / "vertices.npy"
        if obsolete_vertices.is_file():
            obsolete_vertices.unlink()

    report = {
        "status": "written",
        "samples": len(entries),
        "num_points": num_points,
        "dtype": "float32",
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
        "calibration": calibration_records,
        "final": metrics,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "generation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def run_cache_mode(args: argparse.Namespace) -> int:
    dataset_root = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"input directory does not exist: {dataset_root}")

    point_counts = {
        "local_train": args.train_num_points,
        "local_test": args.test_num_points,
    }
    if args.num_points is not None:
        # Backward-compatible override for the old command line.
        point_counts = {split: args.num_points for split in args.splits}

    tasks: List[CacheTask] = []
    split_counts: Dict[str, int] = {}
    for split in args.splits:
        entries = load_split_entries(dataset_root, split)
        if args.limit is not None:
            entries = entries[: args.limit]
        split_counts[split] = len(entries)
        if split == "local_test":
            continue
        num_points = point_counts.get(split, args.train_num_points)
        tasks.extend(
            (
                str(dataset_root),
                str(output_root),
                split,
                entry.as_posix(),
                num_points,
                args.seed,
                args.overwrite,
            )
            for entry in entries
        )

    counts, failures = run_cache_tasks(tasks, args.workers)
    local_test_result = None
    if "local_test" in args.splits:
        local_test_entries = load_split_entries(dataset_root, "local_test")
        if args.limit is not None:
            local_test_entries = local_test_entries[: args.limit]
        local_test_result = generate_fixed_local_test(
            dataset_root=dataset_root,
            output_root=output_root / "local_test",
            entries=[entry.as_posix() for entry in local_test_entries],
            num_points=point_counts["local_test"],
            args=args,
        )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "splits": split_counts,
        "surface_points": {
            split: point_counts.get(split, args.train_num_points)
            for split in args.splits
        },
        "seed": args.seed,
        "dtype": "float32",
        "layout": {
            "local_train": ["clean.npy", "vertices.npy"],
            "local_test": [
                "clean.npy",
                "noisy.npy",
                "normalization.npz",
            ],
        },
        "result": counts,
    }
    if local_test_result is not None:
        manifest["local_test_benchmark"] = local_test_result
    (output_root / "precompute_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(counts, ensure_ascii=False))
    for failure in failures:
        print(f"ERROR: {failure}")
    return 1 if failures else 0


def estimate_noise_file(task: Tuple[str, int]) -> Dict:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError(
            "estimate-noise mode requires scipy; install project "
            "requirements first"
        ) from exc

    path_s, k = task
    path = Path(path_s)
    points = np.load(path, allow_pickle=False)
    shape = tuple(points.shape)
    dtype = str(points.dtype)
    if shape != (50000, 3):
        raise ValueError(f"{path}: expected (50000, 3), got {shape}")
    if points.dtype != np.float32:
        raise ValueError(f"{path}: expected float32, got {points.dtype}")
    if not np.isfinite(points).all():
        raise ValueError(f"{path}: contains non-finite values")

    points64 = points.astype(np.float64)
    tree = cKDTree(points64)
    _, indices = tree.query(points64, k=k + 1)
    neighbors = points64[indices[:, 1:]]
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    covariance = np.einsum("nki,nkj->nij", centered, centered) / k
    minimum_eigenvalue = np.maximum(
        np.linalg.eigvalsh(covariance)[:, 0], 0.0
    )
    estimate = float(np.sqrt(np.median(minimum_eigenvalue)))
    return {
        "path": path_s,
        "estimated_std": estimate,
        "shape": list(shape),
        "dtype": dtype,
    }


def run_noise_estimation(args: argparse.Namespace) -> int:
    input_root = Path(args.noisy_input_dir).resolve()
    files = sorted(input_root.glob(args.noisy_pattern))
    if args.noise_limit is not None:
        files = files[: args.noise_limit]
    if not files:
        raise SystemExit(
            f"no noisy files found: {input_root / args.noisy_pattern}"
        )
    tasks = [(str(path), args.k) for path in files]
    if args.workers == 1:
        rows = [
            estimate_noise_file(task)
            for task in tqdm(tasks, desc="Estimating noise")
        ]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = list(
                tqdm(
                    executor.map(
                        estimate_noise_file, tasks, chunksize=1
                    ),
                    total=len(tasks),
                    desc="Estimating noise",
                )
            )

    estimates = np.asarray(
        [row["estimated_std"] for row in rows], dtype=np.float64
    )
    summary = {
        "files": len(rows),
        "expected_shape": [50000, 3],
        "expected_dtype": "float32",
        "pca_k": args.k,
        "estimated_std_median": float(np.median(estimates)),
        "estimated_std_mean": float(estimates.mean()),
        "estimated_std_min": float(estimates.min()),
        "estimated_std_max": float(estimates.max()),
        "competition_generation_std_range": [
            OFFICIAL_NOISE_STD_MIN,
            OFFICIAL_NOISE_STD_MAX,
        ],
        "note": (
            "Local-PCA normal residual is an indirect geometry-dependent "
            "estimate, not the ground-truth noise parameter."
        ),
    }
    report_path = Path(args.noise_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({"summary": summary, "samples": rows}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"report: {report_path.resolve()}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("cache", "estimate-noise"),
        default="cache",
    )
    parser.add_argument("--input_dir", default="dataset_train")
    parser.add_argument(
        "--output_dir", default="dataset_train_pcd_disk"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=("local_train", "local_test"),
    )
    parser.add_argument("--train_num_points", type=int, default=200000)
    parser.add_argument("--test_num_points", type=int, default=50000)
    parser.add_argument("--num_vertex_samples", type=int, default=1024)
    parser.add_argument("--noise_std_min", type=float, default=0.005)
    parser.add_argument("--noise_std_max", type=float, default=0.020)
    parser.add_argument(
        "--noise_mixture",
        type=str,
        default=None,
        help=(
            "local_test 的 JSON 混合噪声配置（取代 --noise_std_min/max），"
            '如 \'[{"type":"laplace","weight":0.6,"min":0.0075,"max":0.0125},'
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
        help="Models used to calibrate fixed test noise; 0 uses all.",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=None,
        help="Use a known fixed test noise scale and skip calibration.",
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=None,
        help="Set one point count for every split (old CLI compatibility).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(os.cpu_count() or 1, 16),
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cache at most this many models per split (for testing).",
    )

    parser.add_argument(
        "--noisy_input_dir", default="dataset_test_noisy"
    )
    parser.add_argument(
        "--noisy_pattern", default="shapenet/*/*/noisy.npy"
    )
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--noise_limit", type=int, default=None)
    parser.add_argument(
        "--noise_report",
        default="noise_level_report.json",
    )
    args = parser.parse_args()
    positive_values = {
        "--train_num_points": args.train_num_points,
        "--test_num_points": args.test_num_points,
        "--workers": args.workers,
        "--k": args.k,
    }
    if args.num_points is not None:
        positive_values["--num_points"] = args.num_points
    if args.limit is not None:
        positive_values["--limit"] = args.limit
    if args.noise_limit is not None:
        positive_values["--noise_limit"] = args.noise_limit
    for name, value in positive_values.items():
        if value <= 0:
            raise SystemExit(f"{name} must be positive")
    effective_test_points = (
        args.test_num_points
        if args.num_points is None
        else args.num_points
    )
    if not 0 <= args.num_vertex_samples <= effective_test_points:
        raise SystemExit(
            "--num_vertex_samples must be between 0 and test_num_points"
        )
    if not 0 < args.noise_std_min <= args.noise_std_max:
        raise SystemExit("invalid fixed-test noise std range")
    if args.noise_mixture is not None:
        try:
            args.noise_mixture = validate_noise_mixture(
                json.loads(args.noise_mixture)
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise SystemExit(f"invalid --noise_mixture: {exc}") from exc
    if args.target_cd <= 0 or args.target_p2s <= 0:
        raise SystemExit("target CD/P2S must be positive")
    if args.calibration_limit < 0:
        raise SystemExit("--calibration_limit must be non-negative")
    if args.noise_scale is not None and args.noise_scale <= 0:
        raise SystemExit("--noise_scale must be positive")
    return args


def main() -> int:
    args = parse_args()
    if args.mode == "estimate-noise":
        return run_noise_estimation(args)
    return run_cache_mode(args)


if __name__ == "__main__":
    raise SystemExit(main())
