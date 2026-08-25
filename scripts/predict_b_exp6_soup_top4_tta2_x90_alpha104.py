#!/usr/bin/env python3
"""Official B-test prediction for exp6 Soup Top-4 + X90 TTA2 + alpha=1.04.

This is independent from run.py and the ordinary prediction configs.  For
each official noisy.npy it runs the same Soup Top-4 model on the original
cloud and on a fixed +90 degree X-axis rotation, inverse-rotates the second
prediction, averages the aligned alpha=1.00 predictions, and finally applies
residual alpha=1.04 relative to the untouched official noisy cloud.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Sequence

import numpy as np
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model.parse import get_model  # noqa: E402


X90 = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float32,
)


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_yaml(path: Path) -> dict:
    value = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(value, dict):
        raise ValueError(f"YAML top level must be a mapping: {path}")
    return value


def file_identity(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def read_items(datalist: Path) -> List[str]:
    items: List[str] = []
    seen = set()
    for line_number, raw in enumerate(
        datalist.read_text(encoding="utf-8").splitlines(), start=1
    ):
        item = raw.strip().replace("\\", "/").strip("/")
        if not item:
            continue
        parts = Path(item).parts
        if item.startswith("/") or ".." in parts or not item.startswith("shapenet/"):
            raise ValueError(f"unsafe datalist entry at line {line_number}: {raw!r}")
        if item in seen:
            raise ValueError(f"duplicate datalist entry: {item}")
        seen.add(item)
        items.append(item)
    if not items:
        raise ValueError(f"empty datalist: {datalist}")
    return items


def items_digest(items: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()


def validate_cloud(value: np.ndarray, path: Path) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != 2 or value.shape[1] != 3:
        raise ValueError(f"point cloud must have shape (N,3): {path}: {value.shape}")
    if value.dtype != np.float32:
        value = value.astype(np.float32)
    if not np.isfinite(value).all():
        raise ValueError(f"point cloud contains NaN/Inf: {path}")
    return value


def output_is_valid(path: Path, expected_shape: tuple) -> bool:
    if not path.is_file():
        return False
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        return (
            value.shape == expected_shape
            and value.dtype == np.float32
            and np.isfinite(value).all()
        )
    except (OSError, ValueError):
        return False


def atomic_save(path: Path, value: np.ndarray) -> None:
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


def expected_manifest(
    model_config: Path,
    transform_config: Path,
    checkpoint: Path,
    data_root: Path,
    datalist: Path,
    items: Sequence[str],
    seed: int,
    alpha: float,
) -> dict:
    return {
        "prediction": "exp6_soup_top4_tta2_x90_alpha104",
        "model_config": file_identity(model_config),
        "transform_config": file_identity(transform_config),
        "checkpoint": file_identity(checkpoint),
        "data_root": str(data_root),
        "datalist": file_identity(datalist),
        "sample_count": len(items),
        "sample_digest": items_digest(items),
        "rotation_name": "x90",
        "rotation_matrix": X90.tolist(),
        "tta_branches": 2,
        "base_residual_alpha": 1.0,
        "final_residual_alpha": alpha,
        "fusion_mode": "best",
        "predict_rounds": 1,
        "seed": seed,
        "official_noisy_is_not_augmented": True,
    }


def prepare_output(output_root: Path, manifest: dict) -> None:
    manifest_path = output_root / "prediction_manifest.json"
    if manifest_path.is_file():
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        if actual != manifest:
            raise RuntimeError(
                f"output manifest mismatch: {output_root}. "
                "Use a new --output-root; existing output is never deleted."
            )
        return
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"non-empty output has no manifest: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_model(model_config_path: Path, transform_config_path: Path, checkpoint: Path):
    import jittor as jt

    model_config = load_yaml(model_config_path)
    model_config["residual_alpha"] = 1.0
    model_config["fusion_mode"] = "best"
    model_config["predict_rounds"] = 1
    transform_config = load_yaml(transform_config_path)
    if transform_config.get("predict_transform", {}).get("augments", None) != []:
        raise ValueError("predict_transform.augments must be empty")
    model = get_model(model_config=model_config, transform_config=transform_config)

    state = jt.load(str(checkpoint))
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint is not a state dict: {checkpoint}")
    model_state = model.state_dict()
    if set(state) != set(model_state):
        missing = sorted(set(model_state) - set(state))
        extra = sorted(set(state) - set(model_state))
        raise RuntimeError(f"checkpoint key mismatch: missing={missing}, extra={extra}")
    for key, model_value in model_state.items():
        if tuple(state[key].shape) != tuple(model_value.shape):
            raise RuntimeError(
                f"checkpoint shape mismatch for {key}: "
                f"{tuple(state[key].shape)} vs {tuple(model_value.shape)}"
            )
    del state, model_state
    model.load(str(checkpoint))
    model.set_predict(True)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="dataset_test_noisy")
    parser.add_argument("--datalist", default="datalist/test_b.txt")
    parser.add_argument(
        "--model-config", default="configs/model/straightpcf_b_exp6_b32_alpha100.yaml"
    )
    parser.add_argument("--transform-config", default="configs/transform/predict.yaml")
    parser.add_argument(
        "--checkpoint",
        default=(
            "checkpoint_selection_b_exp6_b32_alpha105_straightpcf_soup_top4/"
            "best_checkpoint.pkl"
        ),
    )
    parser.add_argument(
        "--output-root",
        default=(
            "results_b_exp6_soup_top4_tta2_x90_alpha104/"
            "dataset_test_noisy"
        ),
    )
    parser.add_argument("--alpha", type=float, default=1.04)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--use-cuda", type=int, choices=(0, 1), default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if not np.isclose(args.alpha, 1.04, atol=1e-12):
        raise SystemExit("this fixed candidate requires --alpha 1.04")
    if not np.allclose(X90 @ X90.T, np.eye(3), atol=1e-7):
        raise RuntimeError("X90 matrix is not orthonormal")
    if not np.isclose(np.linalg.det(X90), 1.0, atol=1e-7):
        raise RuntimeError("X90 determinant is not +1")

    data_root = resolve_path(args.data_root)
    datalist = resolve_path(args.datalist)
    model_config = resolve_path(args.model_config)
    transform_config = resolve_path(args.transform_config)
    checkpoint = resolve_path(args.checkpoint)
    output_root = resolve_path(args.output_root)
    for path in (data_root, datalist, model_config, transform_config, checkpoint):
        if not path.exists():
            raise FileNotFoundError(path)

    items_all = read_items(datalist)
    discovered = {
        str(path.parent.relative_to(data_root)).replace(os.sep, "/")
        for path in data_root.glob("shapenet/*/*/noisy.npy")
    }
    if set(items_all) != discovered:
        missing = sorted(discovered - set(items_all))
        extra = sorted(set(items_all) - discovered)
        raise RuntimeError(
            f"datalist/input mismatch: missing_from_list={missing}, extra_in_list={extra}"
        )
    items = items_all[: args.limit] if args.limit is not None else items_all
    for item in items:
        noisy_path = data_root / item / "noisy.npy"
        if not noisy_path.is_file():
            raise FileNotFoundError(noisy_path)
        noisy = np.load(noisy_path, mmap_mode="r", allow_pickle=False)
        if noisy.ndim != 2 or noisy.shape[1] != 3:
            raise ValueError(f"invalid noisy shape: {noisy_path}: {noisy.shape}")

    manifest = expected_manifest(
        model_config,
        transform_config,
        checkpoint,
        data_root,
        datalist,
        items,
        args.seed,
        args.alpha,
    )
    print(f"official samples: {len(items)}")
    print(f"checkpoint: {checkpoint}")
    print(f"output: {output_root}")
    print("candidate: Soup Top-4 + X90 TTA2 + alpha=1.04 + rounds=1 + best fusion")
    if args.check_only:
        load_yaml(model_config)
        transform = load_yaml(transform_config)
        if transform.get("predict_transform", {}).get("augments", None) != []:
            raise ValueError("predict_transform.augments must be empty")
        print("check-only complete; no model loaded and no output written")
        return 0

    prepare_output(output_root, manifest)
    pending = []
    for item in items:
        noisy_path = data_root / item / "noisy.npy"
        noisy = np.load(noisy_path, mmap_mode="r", allow_pickle=False)
        output_path = output_root / item / "denoised.npy"
        if not output_is_valid(output_path, noisy.shape):
            pending.append(item)
    if not pending:
        print("all official predictions already exist and are valid; skip inference")
        return 0

    import jittor as jt

    jt.flags.use_cuda = args.use_cuda
    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model = build_model(model_config, transform_config, checkpoint)
    start = time.time()
    for index, item in enumerate(pending, start=1):
        noisy_path = data_root / item / "noisy.npy"
        noisy = validate_cloud(np.load(noisy_path, allow_pickle=False), noisy_path)
        noisy_x90 = (noisy @ X90.T).astype(np.float32, copy=False)
        batch = np.stack((noisy, noisy_x90), axis=0)
        with jt.no_grad():
            outputs = model.predict_step({"pc_noisy": jt.array(batch)})
        original_base = validate_cloud(
            np.asarray(outputs[0]["pc_denoised"]), Path(f"<original:{item}>")
        )
        rotated_base = validate_cloud(
            np.asarray(outputs[1]["pc_denoised"]), Path(f"<x90:{item}>")
        )
        rotated_back = rotated_base @ X90
        tta_base = 0.5 * (original_base + rotated_back)
        prediction = noisy + args.alpha * (tta_base - noisy)
        prediction = validate_cloud(
            prediction.astype(np.float32, copy=False), Path(f"<final:{item}>")
        )
        if prediction.shape != noisy.shape:
            raise RuntimeError(
                f"prediction shape mismatch for {item}: {prediction.shape} vs {noisy.shape}"
            )
        atomic_save(output_root / item / "denoised.npy", prediction)
        del outputs, original_base, rotated_base, rotated_back, tta_base, prediction
        elapsed = time.time() - start
        remaining = elapsed / index * (len(pending) - index)
        print(
            f"\r[predict X90 TTA2] {index}/{len(pending)} "
            f"elapsed={elapsed / 60:.1f}m remaining={remaining / 60:.1f}m",
            end="",
            flush=True,
        )
    print()
    del model
    jt.gc()
    print("official TTA prediction complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
