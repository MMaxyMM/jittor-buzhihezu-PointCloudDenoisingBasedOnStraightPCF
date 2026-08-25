#!/usr/bin/env python
"""
Select the best checkpoint using local validation metrics.

This script evaluates checkpoints on the project's validate_dataset and ranks
them by validation loss, CD, or a loss/CD/P2S composite. Lower is better. The
composite uses weighted ranks so metrics with different numeric scales remain
comparable.

Example:
    python select_best_checkpoint.py --ckpt_dir experiments/vm --copy_best

Useful quick test:
    python select_best_checkpoint.py \
        --ckpt_dir experiments/vm \
        --limit 3

Composite selection (default weights loss:CD:P2S = 1:2:2):
    python select_best_checkpoint.py \
        --metric composite \
        --ckpt_dir experiments/vm \
        --task_template configs/task/train_vm_cached.yaml \
        --mesh_dir dataset_train \
        --output_dir checkpoint_selection_composite \
        --copy_best
"""

import argparse
import contextlib
import csv
import json
import random
import re
import shutil
import time
import traceback
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.config_override import apply_overrides


MODEL_TARGET_FALLBACK = {
    "vm": "VelocityModule",
    "residual_diffusion": "ResidualDiffusionModule",
    "direct_residual": "ResidualDiffusionModule",
}


@dataclass
class CheckpointResult:
    checkpoint: str
    epoch: Optional[int]
    score: Optional[float]
    status: str
    error: str = ""
    loss: Optional[float] = None
    cd: Optional[float] = None
    p2s: Optional[float] = None


def checkpoint_epoch(path: Path) -> Optional[int]:
    match = re.search(r"(\d+)(?=\.pkl$)", path.name)
    return int(match.group(1)) if match else None


def natural_key(path: Path) -> Tuple[int, str]:
    epoch = checkpoint_epoch(path)
    return (epoch if epoch is not None else -1, path.name)


def iter_checkpoints(
    ckpt_dir: Path,
    pattern: str,
    start_epoch: Optional[int],
    end_epoch: Optional[int],
) -> List[Path]:
    checkpoints = sorted(ckpt_dir.glob(pattern), key=natural_key)
    selected = []
    for ckpt in checkpoints:
        epoch = checkpoint_epoch(ckpt)
        if start_epoch is not None and (epoch is None or epoch < start_epoch):
            continue
        if end_epoch is not None and (epoch is None or epoch > end_epoch):
            continue
        selected.append(ckpt)
    return selected


def load_yaml(path: Path) -> Dict:
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise SystemExit("omegaconf is required. Install it with: pip install omegaconf") from exc

    if not path.exists():
        raise SystemExit(f"Config file does not exist: {path}")
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(cfg, dict):
        raise SystemExit(f"Config file must contain a mapping: {path}")
    return cfg


def config_path(config_dir: str, name: str) -> Path:
    path = Path(config_dir) / name
    if path.suffix != ".yaml":
        path = path.with_suffix(".yaml")
    return path


def ensure_model_target(model_config: Dict, component_name: str) -> Dict:
    cfg = deepcopy(model_config)
    if "__target__" not in cfg:
        fallback = MODEL_TARGET_FALLBACK.get(component_name)
        if fallback is None:
            raise SystemExit(
                f"configs/model/{component_name}.yaml has no __target__, "
                "and no fallback target is known for it."
            )
        cfg["__target__"] = fallback
    return cfg


def ensure_system_target(system_config: Dict, component_name: str) -> Dict:
    cfg = deepcopy(system_config)
    cfg.setdefault("__target__", component_name)
    return cfg


def item_value(value) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def write_results(results: Iterable[CheckpointResult], output_dir: Path) -> None:
    rows = [asdict(item) for item in results]
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "checkpoint_ranking.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "checkpoint",
                "epoch",
                "score",
                "status",
                "error",
                "loss",
                "cd",
                "p2s",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    json_path = output_dir / "checkpoint_ranking.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)


def load_existing_results(output_dir: Path) -> Dict[str, CheckpointResult]:
    json_path = output_dir / "checkpoint_ranking.json"
    if not json_path.exists():
        return {}

    with json_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    results = {}
    for row in rows:
        item = CheckpointResult(**row)
        results[item.checkpoint] = item
    return results


def build_validation_context(args) -> Dict:
    import jittor as jt
    import numpy as np

    from src.data.dataset import DatasetConfig
    from src.data.transform import Transform
    from src.model.parse import get_model

    jt.flags.use_cuda = args.use_cuda
    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    task = load_yaml(Path(args.task_template))
    components = task.get("components")
    if not isinstance(components, dict):
        raise SystemExit(f"{args.task_template} must contain a components mapping.")

    for name in ["data", "transform", "model", "system"]:
        if name not in components:
            raise SystemExit(f"{args.task_template} is missing components.{name}.")

    data_path = Path(args.data_config) if args.data_config else config_path("configs/data", components["data"])
    transform_path = config_path("configs/transform", components["transform"])
    model_path = config_path("configs/model", components["model"])
    system_path = config_path("configs/system", components["system"])

    data_config = load_yaml(data_path)
    validate_config = data_config.get("validate_dataset")
    if validate_config is None:
        raise SystemExit(f"No validate_dataset found in {data_path}")

    transform_config = load_yaml(transform_path)
    model_config = ensure_model_target(load_yaml(model_path), components["model"])
    model_config = apply_overrides(model_config, args.model_override)
    system_config = ensure_system_target(load_yaml(system_path), components["system"])

    validate_dataset_config = DatasetConfig.parse(**validate_config).split_by_cls()
    for dataset_config in validate_dataset_config.values():
        dataset_config.num_workers = args.validation_workers

    # get_model mutates model_config by deleting __target__, so always pass a copy.
    temp_model = get_model(
        model_config=deepcopy(model_config),
        transform_config=deepcopy(transform_config),
    )
    validate_transform = temp_model.get_validate_transform()
    if validate_transform is None:
        validate_transform = Transform.parse(**transform_config.get("validate_transform", {}))

    return {
        "seed": args.seed,
        "task": task,
        "model_config": model_config,
        "transform_config": transform_config,
        "system_config": system_config,
        "validate_dataset_config": validate_dataset_config,
        "validate_transform": validate_transform,
    }


def evaluate_checkpoint(ckpt_path: Path, context: Dict, log_path: Path) -> Tuple[Optional[float], Dict[str, float]]:
    import jittor as jt
    import numpy as np

    from src.data.dataset import PCDatasetModule
    from src.model.parse import get_model
    from src.system.parse import get_system

    jt.set_global_seed(context["seed"])
    np.random.seed(context["seed"])
    random.seed(context["seed"])

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            # get_model and get_system mutate config dicts, so pass fresh copies.
            model = get_model(
                model_config=deepcopy(context["model_config"]),
                transform_config=deepcopy(context["transform_config"]),
            )
            model.load(str(ckpt_path))
            model.set_predict(False)
            model.eval()

            dataset_module = PCDatasetModule(
                process_fn=model._process_fn,
                train_dataset_config=None,
                validate_dataset_config=context["validate_dataset_config"],
                predict_dataset_config=None,
                train_transform=None,
                validate_transform=context["validate_transform"],
                predict_transform=None,
            )

            system = get_system(
                dataset_module=dataset_module,
                model=model,
                optimizer_config=None,
                loss_config=context["task"].get("loss"),
                trainer_config=None,
                writer=None,
                **deepcopy(context["system_config"]),
            )

            validate_dataloader = dataset_module.validate_dataloader()
            if validate_dataloader is None:
                raise RuntimeError("validate_dataloader is None")

            losses: List[float] = []
            system.on_validation_epoch_start()
            loaders = validate_dataloader if isinstance(validate_dataloader, dict) else {"validate": validate_dataloader}

            with jt.no_grad():
                for loader_name, dataloader in loaders.items():
                    for batch in dataloader:
                        system.on_validation_batch_start()
                        loss = system.validation_step(batch)
                        loss_float = item_value(loss)
                        losses.append(loss_float)
                        print(f"{loader_name}: loss={loss_float:.8f}")
                        system.on_validation_batch_end()

            system.on_validation_epoch_end()
            val_loss = mean(losses)
            if val_loss is None:
                raise RuntimeError(
                    "validation set produced no batches; check data_config, "
                    "datalist paths, and cached dataset availability"
                )
            metrics = {
                name: value
                for name, values in system._validation_loss.items()
                for value in [mean([float(v) for v in values])]
                if value is not None
            }
            if val_loss is not None:
                metrics["val/loss_mean"] = val_loss

            print(json.dumps(metrics, indent=2, ensure_ascii=False))
            if hasattr(jt, "gc"):
                jt.gc()

    return metrics.get("val/loss_mean"), metrics


def _chamfer_distance(a, b) -> float:
    """与 evaluate.py 一致的 CD：双向最近点平方距离均值（点云已归一化）。"""
    from scipy.spatial import cKDTree

    tree_b = cKDTree(b)
    dist_a2b, _ = tree_b.query(a, k=1)
    tree_a = cKDTree(a)
    dist_b2a, _ = tree_a.query(b, k=1)
    return float((dist_a2b ** 2).mean() + (dist_b2a ** 2).mean())


def _normalize_unit_sphere(pc):
    p_max = pc.max(axis=0)
    p_min = pc.min(axis=0)
    center = (p_max + p_min) / 2
    centered = pc - center
    scale = np.sqrt((centered ** 2).sum(axis=1).max()).max()
    if scale < 1e-12:
        raise ValueError("cannot normalize a degenerate point cloud")
    return centered / scale, center, scale


def _mesh_path_for_asset(asset_path: str, args) -> Path:
    """Map a cached clean.npy path back to its source ShapeNet OBJ."""
    parts = Path(asset_path).parts
    try:
        shapenet_index = parts.index("shapenet")
    except ValueError as exc:
        raise RuntimeError(
            f"cannot derive ShapeNet mesh path from validation asset: {asset_path}"
        ) from exc
    model_relative_path = Path(*parts[shapenet_index:-1])
    return Path(args.mesh_dir) / model_relative_path / args.mesh_data_name


def _point_to_surface_distance(points, vertices, faces) -> float:
    """Exact mean squared point-to-triangle distance using PCU's BVH."""
    try:
        import point_cloud_utils as pcu
    except ImportError as exc:
        raise RuntimeError(
            "P2S requires point-cloud-utils. Install it with: "
            "pip install point-cloud-utils"
        ) from exc

    distances, _, _ = pcu.closest_points_on_mesh(
        np.asarray(points, dtype=np.float32),
        np.asarray(vertices, dtype=np.float32),
        np.asarray(faces, dtype=np.int32),
    )
    return float((distances ** 2).mean())


def _geometry_sample_from_validate_transform(
    lazy_asset,
    context: Dict,
    args,
    sample_index: int,
):
    """Build deterministic clean/noisy geometry using the task's validation transform."""
    asset = lazy_asset.load()
    source_vertices = asset.vertices
    source_faces = asset.faces
    center = None
    scale = None
    transform = context["validate_transform"]
    for augment in transform.augments or []:
        if augment.__class__.__name__ == "AugmentNormalizePC":
            if asset.sampled_vertices is None:
                raise RuntimeError(
                    "normalize_pc appeared before sample in validate transform"
                )
            _, center, scale = _normalize_unit_sphere(
                asset.sampled_vertices
            )
        augment.apply(asset)

    clean = asset.sampled_vertices
    noisy = asset.sampled_vertices_noisy
    if clean is None or noisy is None:
        raise RuntimeError(
            "validate transform must produce sampled_vertices and "
            "sampled_vertices_noisy for geometry selection"
        )
    if center is None or scale is None:
        raise RuntimeError(
            "validate transform must include normalize_pc for P2S alignment"
        )
    if clean.shape != noisy.shape:
        raise RuntimeError(
            f"clean/noisy shape mismatch: {clean.shape} versus {noisy.shape}"
        )
    if clean.shape[0] < args.cd_points:
        return None

    clean = clean.astype(np.float32, copy=False)
    noisy = noisy.astype(np.float32, copy=False)
    if clean.shape[0] > args.cd_points:
        rs = np.random.RandomState(
            context["seed"] + 1000003 + sample_index
        )
        indices = rs.choice(clean.shape[0], args.cd_points, replace=False)
        clean = clean[indices]
        noisy = noisy[indices]

    normalized_vertices = None
    normalized_faces = None
    if source_vertices is not None and source_faces is not None:
        normalized_vertices = (
            (np.asarray(source_vertices) - center) / scale
        ).astype(np.float32, copy=False)
        normalized_faces = np.asarray(source_faces, dtype=np.int32)
    return (
        asset,
        clean,
        noisy,
        center,
        scale,
        normalized_vertices,
        normalized_faces,
    )


def evaluate_checkpoint_geometry(
    ckpt_path: Path,
    context: Dict,
    log_path: Path,
    args,
    compute_p2s: bool = False,
) -> Tuple[Optional[float], Dict[str, float]]:
    """Compute deterministic validation CD and, optionally, exact mesh P2S.

    与 loss 模式不同，该指标与竞赛评分直接对齐。噪声按固定种子生成，
    所有 checkpoint 面对完全相同的含噪输入，排名可比。
    """
    import jittor as jt
    import numpy as np

    from src.model.parse import get_model

    np.random.seed(context["seed"])
    random.seed(context["seed"])

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            model = get_model(
                model_config=deepcopy(context["model_config"]),
                transform_config=deepcopy(context["transform_config"]),
            )
            model.load(str(ckpt_path))
            model.set_predict(True)
            model.eval()

            cds: List[float] = []
            noisy_cds: List[float] = []
            cd_scores: List[float] = []
            p2ss: List[float] = []
            noisy_p2ss: List[float] = []
            p2s_scores: List[float] = []
            inference_seconds: List[float] = []
            displacement_p95: List[float] = []
            sample_index = 0
            for cls, ds_config in context["validate_dataset_config"].items():
                for lazy_asset in ds_config.datapath.get_data():
                    if args.cd_limit is not None and sample_index >= args.cd_limit:
                        break
                    mesh_vertices = None
                    mesh_faces = None
                    if args.geometry_from_validate_transform:
                        sample = _geometry_sample_from_validate_transform(
                            lazy_asset, context, args, sample_index
                        )
                        if sample is None:
                            continue
                        (
                            asset,
                            clean,
                            noisy,
                            center,
                            scale,
                            mesh_vertices,
                            mesh_faces,
                        ) = sample
                    else:
                        asset = lazy_asset.load()
                        pool = asset.sampled_vertices
                        if pool is None or pool.shape[0] < args.cd_points:
                            continue

                        rs = np.random.RandomState(
                            context["seed"] + sample_index
                        )
                        clean = pool[
                            rs.choice(
                                pool.shape[0], args.cd_points, replace=False
                            )
                        ]
                        clean, center, scale = _normalize_unit_sphere(
                            clean.astype(np.float32)
                        )
                        noise_std = rs.uniform(
                            args.noise_std_min, args.noise_std_max
                        )
                        noise = rs.laplace(
                            0.0,
                            noise_std / np.sqrt(2.0),
                            size=clean.shape,
                        )
                        noisy = (clean + noise).astype(np.float32)

                    started_at = time.perf_counter()
                    pred = model.predict_step({"pc_noisy": jt.array(noisy[None])})
                    denoised = pred[0]["pc_denoised"]
                    inference_seconds.append(time.perf_counter() - started_at)
                    cd = _chamfer_distance(denoised, clean)
                    cd_noisy = _chamfer_distance(noisy, clean)
                    cds.append(cd)
                    noisy_cds.append(cd_noisy)
                    cd_scores.append(
                        float(np.clip(100.0 * (1.0 - cd / cd_noisy), 0.0, 100.0))
                    )
                    if compute_p2s:
                        if mesh_vertices is None or mesh_faces is None:
                            try:
                                import point_cloud_utils as pcu
                            except ImportError as exc:
                                raise RuntimeError(
                                    "P2S requires point-cloud-utils"
                                ) from exc
                            mesh_path = _mesh_path_for_asset(asset.path, args)
                            if not mesh_path.is_file():
                                raise FileNotFoundError(
                                    "validation mesh does not exist: "
                                    f"{mesh_path}"
                                )
                            mesh_vertices, mesh_faces = pcu.load_mesh_vf(
                                str(mesh_path)
                            )
                            mesh_vertices = (
                                np.asarray(mesh_vertices, dtype=np.float32)
                                - center
                            ) / scale
                            mesh_faces = np.asarray(
                                mesh_faces, dtype=np.int32
                            )
                        p2s = _point_to_surface_distance(
                            denoised, mesh_vertices, mesh_faces
                        )
                        noisy_p2s = _point_to_surface_distance(
                            noisy, mesh_vertices, mesh_faces
                        )
                        p2ss.append(p2s)
                        noisy_p2ss.append(noisy_p2s)
                        if noisy_p2s < 1e-15:
                            p2s_score = 100.0 if p2s < 1e-15 else 0.0
                        else:
                            p2s_score = float(
                                np.clip(
                                    100.0 * (1.0 - p2s / noisy_p2s),
                                    0.0,
                                    100.0,
                                )
                            )
                        p2s_scores.append(p2s_score)
                    displacement_p95.append(
                        float(np.percentile(np.linalg.norm(denoised - noisy, axis=1), 95))
                    )
                    p2s_text = (
                        f" p2s={p2ss[-1]:.8f} "
                        f"p2s_noisy={noisy_p2ss[-1]:.8f}"
                        if compute_p2s
                        else ""
                    )
                    print(
                        f"sample {sample_index} ({asset.path}): "
                        f"cd={cd:.8f} cd_noisy={cd_noisy:.8f}"
                        f"{p2s_text} improve={100*(1-cd/cd_noisy):.2f}"
                    )
                    sample_index += 1

            mean_cd = mean(cds)
            if mean_cd is None:
                raise RuntimeError(
                    "geometry validation produced no samples; check "
                    "cd_points, cd_limit, datalist paths, and cached dataset"
                )
            metrics = {"val/cd_mean": mean_cd} if mean_cd is not None else {}
            if noisy_cds:
                metrics.update({
                    "val/noisy_cd_mean": mean(noisy_cds),
                    "val/cd_score_mean": mean(cd_scores),
                    "val/inference_seconds_mean": mean(inference_seconds),
                    "val/displacement_p95_mean": mean(displacement_p95),
                    "model/parameters": float(
                        sum(int(np.prod(parameter.shape)) for parameter in model.parameters())
                    ),
                })
            if p2ss:
                metrics.update({
                    "val/p2s_mean": mean(p2ss),
                    "val/noisy_p2s_mean": mean(noisy_p2ss),
                    "val/p2s_score_mean": mean(p2s_scores),
                })
            print(json.dumps(metrics, indent=2, ensure_ascii=False))
            if hasattr(jt, "gc"):
                jt.gc()

    return metrics.get("val/cd_mean"), metrics


def _average_ranks(values: List[Tuple[int, float]]) -> Dict[int, float]:
    """Return one-based average ranks; lower metric values rank first."""
    ordered = sorted(values, key=lambda item: item[1])
    ranks: Dict[int, float] = {}
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[ordered[position][0]] = average_rank
        start = end
    return ranks


def apply_composite_scores(
    results: List[CheckpointResult],
    loss_weight: float,
    cd_weight: float,
    p2s_weight: float,
) -> None:
    """Set the weighted-rank composite score in place; lower is better."""
    valid = [
        (index, item)
        for index, item in enumerate(results)
        if item.status == "ok"
        and item.loss is not None
        and item.cd is not None
        and item.p2s is not None
    ]
    if not valid:
        return

    loss_ranks = _average_ranks([(index, item.loss) for index, item in valid])
    cd_ranks = _average_ranks([(index, item.cd) for index, item in valid])
    p2s_ranks = _average_ranks([(index, item.p2s) for index, item in valid])
    total_weight = loss_weight + cd_weight + p2s_weight
    for index, item in valid:
        item.score = (
            loss_weight * loss_ranks[index]
            + cd_weight * cd_ranks[index]
            + p2s_weight * p2s_ranks[index]
        ) / total_weight


def rank_results(results: List[CheckpointResult]) -> List[CheckpointResult]:
    ok = [item for item in results if item.status == "ok" and item.score is not None]
    bad = [item for item in results if item.status != "ok" or item.score is None]
    return sorted(ok, key=lambda item: item.score) + bad


def run_selection(args, checkpoints: List[Path], existing: Dict[str, CheckpointResult]) -> List[CheckpointResult]:
    output_dir = Path(args.output_dir)
    context = build_validation_context(args)
    results: List[CheckpointResult] = []

    for index, ckpt in enumerate(checkpoints, start=1):
        ckpt_abs = ckpt.resolve()
        ckpt_key = str(ckpt)

        existing_item = existing.get(ckpt_key)
        can_resume = (
            existing_item is not None
            and existing_item.status == "ok"
            and (
                args.metric != "composite"
                or (
                    existing_item.loss is not None
                    and existing_item.cd is not None
                    and existing_item.p2s is not None
                )
            )
        )
        if can_resume:
            print(f"[{index}/{len(checkpoints)}] skip existing: {ckpt}")
            results.append(existing_item)
            continue

        epoch = checkpoint_epoch(ckpt)
        log_path = output_dir / "logs" / f"{ckpt.stem}_{args.metric}.log"

        print(f"[{index}/{len(checkpoints)}] validate {args.metric}: {ckpt}")
        try:
            if args.metric == "cd":
                score, _ = evaluate_checkpoint_geometry(
                    ckpt_abs, context, log_path, args
                )
                result = CheckpointResult(
                    ckpt_key, epoch, score, "ok", cd=score
                )
            elif args.metric == "composite":
                loss_log_path = (
                    output_dir / "logs" / f"{ckpt.stem}_loss.log"
                )
                geometry_log_path = (
                    output_dir / "logs" / f"{ckpt.stem}_cd_p2s.log"
                )
                loss, _ = evaluate_checkpoint(
                    ckpt_abs, context, loss_log_path
                )
                cd, geometry_metrics = evaluate_checkpoint_geometry(
                    ckpt_abs,
                    context,
                    geometry_log_path,
                    args,
                    compute_p2s=True,
                )
                p2s = geometry_metrics.get("val/p2s_mean")
                if loss is None or cd is None or p2s is None:
                    raise RuntimeError(
                        "incomplete composite metrics: "
                        f"loss={loss}, cd={cd}, p2s={p2s}"
                    )
                result = CheckpointResult(
                    ckpt_key,
                    epoch,
                    None,
                    "ok",
                    loss=loss,
                    cd=cd,
                    p2s=p2s,
                )
            else:
                score, _ = evaluate_checkpoint(
                    ckpt_abs, context, log_path
                )
                result = CheckpointResult(
                    ckpt_key, epoch, score, "ok", loss=score
                )
        except Exception as exc:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8", errors="replace") as log_file:
                log_file.write("\n\nException traceback:\n")
                traceback.print_exc(file=log_file)
            error = f"validation failed, see {log_path}: {exc}"
            print(f"  {error}")
            results.append(CheckpointResult(ckpt_key, epoch, None, "validate_failed", error))
            write_results(rank_results(results), output_dir)
            continue

        if args.metric == "composite":
            results.append(result)
            apply_composite_scores(
                results,
                args.loss_weight,
                args.cd_weight,
                args.p2s_weight,
            )
            print(
                f"  loss={result.loss:.8f} cd={result.cd:.8f} "
                f"p2s={result.p2s:.8f}"
            )
        elif result.score is None:
            error = f"could not compute validation {args.metric}, see {log_path}"
            print(f"  {error}")
            results.append(CheckpointResult(ckpt_key, epoch, None, "parse_failed", error))
        else:
            print(f"  {args.metric}: {result.score:.8f}")
            results.append(result)

        write_results(rank_results(results), output_dir)

    if args.metric == "composite":
        apply_composite_scores(
            results,
            args.loss_weight,
            args.cd_weight,
            args.p2s_weight,
        )
    return rank_results(results)


def run_two_stage_composite(
    args,
    checkpoints: List[Path],
    existing: Dict[str, CheckpointResult],
) -> List[CheckpointResult]:
    """Quick-CD prefilter all checkpoints, then fully score only the Top-K."""
    output_dir = Path(args.output_dir)
    prefilter_checkpoints = (
        checkpoints[-args.prefilter_last_n:]
        if args.prefilter_last_n > 0
        else checkpoints
    )
    prefilter_args = deepcopy(args)
    prefilter_args.metric = "cd"
    prefilter_args.cd_points = args.prefilter_cd_points
    prefilter_args.cd_limit = args.prefilter_cd_limit
    prefilter_args.output_dir = str(output_dir / "prefilter")

    prefilter_output = Path(prefilter_args.output_dir)
    prefilter_existing = (
        load_existing_results(prefilter_output) if args.resume else {}
    )
    print(
        "\nTwo-stage composite selection: "
        f"prefilter the last {len(prefilter_checkpoints)} of "
        f"{len(checkpoints)} checkpoints with "
        f"{args.prefilter_cd_points} points, "
        f"{args.prefilter_cd_limit} validation samples."
    )
    prefilter_ranked = run_selection(
        prefilter_args, prefilter_checkpoints, prefilter_existing
    )
    write_results(prefilter_ranked, prefilter_output)

    valid_prefilter = [
        item
        for item in prefilter_ranked
        if item.status == "ok" and item.score is not None
    ]
    if not valid_prefilter:
        return []

    top_k = min(args.prefilter_top_k, len(valid_prefilter))
    rank_by_checkpoint = {
        item.checkpoint: index
        for index, item in enumerate(valid_prefilter)
    }
    candidates = [
        checkpoint
        for checkpoint in checkpoints
        if str(checkpoint) in rank_by_checkpoint
        and rank_by_checkpoint[str(checkpoint)] < top_k
    ]
    candidates.sort(key=lambda checkpoint: rank_by_checkpoint[str(checkpoint)])

    print(
        f"\nPrefilter complete. Running full loss/CD/P2S on "
        f"{len(candidates)} candidates:"
    )
    for checkpoint in candidates:
        print(f"  {checkpoint}")
    return run_selection(args, candidates, existing)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select the best checkpoint by local validation metrics."
    )
    parser.add_argument(
        "--metric",
        choices=["loss", "cd", "composite"],
        default="loss",
        help=(
            "loss=验证损失; cd=真实 Chamfer Distance; "
            "composite=loss/CD/精确P2S加权排名"
        ),
    )
    parser.add_argument("--cd_points", type=int, default=32768, help="CD 模式每个模型采样点数")
    parser.add_argument("--cd_limit", type=int, default=None, help="CD/composite 模式最多评估多少个验证模型")
    parser.add_argument("--noise_std_min", type=float, default=0.005, help="CD/composite 模式加噪 std 下限")
    parser.add_argument("--noise_std_max", type=float, default=0.020, help="CD/composite 模式加噪 std 上限")
    parser.add_argument(
        "--geometry_from_validate_transform",
        action="store_true",
        help=(
            "Use the task validate transform for raw-OBJ clean/noisy geometry; "
            "the default cached-data selection path remains unchanged."
        ),
    )
    parser.add_argument(
        "--mesh_dir",
        default="dataset_train",
        help="Composite 模式的原始 OBJ 数据集根目录",
    )
    parser.add_argument(
        "--mesh_data_name",
        default="models/model_normalized.obj",
        help="每个模型目录下的 OBJ 相对路径",
    )
    parser.add_argument("--loss_weight", type=float, default=1.0)
    parser.add_argument("--cd_weight", type=float, default=2.0)
    parser.add_argument("--p2s_weight", type=float, default=2.0)
    parser.add_argument(
        "--prefilter_top_k",
        type=int,
        default=0,
        help=(
            "Composite 模式先用快速CD筛选的候选数；0关闭两阶段筛选，"
            "保持原来的全checkpoint完整评测"
        ),
    )
    parser.add_argument(
        "--prefilter_cd_points",
        type=int,
        default=8192,
        help="两阶段筛选第一阶段每个模型使用的点数",
    )
    parser.add_argument(
        "--prefilter_cd_limit",
        type=int,
        default=20,
        help="两阶段筛选第一阶段每个checkpoint最多评测的验证模型数",
    )
    parser.add_argument(
        "--prefilter_last_n",
        type=int,
        default=70,
        help="两阶段筛选仅初筛按epoch排序后的最后N个checkpoint；0表示全部",
    )
    parser.add_argument("--ckpt_dir", default="experiments/vm", help="Directory containing checkpoint_*.pkl files.")
    parser.add_argument("--pattern", default="checkpoint_*.pkl", help="Checkpoint filename pattern.")
    parser.add_argument("--task_template", default="configs/task/train_vm.yaml", help="Training task yaml.")
    parser.add_argument("--data_config", default="", help="Optional data yaml. Defaults to the task component data config.")
    parser.add_argument("--output_dir", default="checkpoint_selection", help="Directory for logs and rankings.")
    parser.add_argument("--use_cuda", type=int, default=1, help="Jittor CUDA flag.")
    parser.add_argument(
        "--validation_workers",
        type=int,
        default=0,
        help=(
            "Validation DataLoader workers. Default 0 keeps dynamic sampling "
            "and noise deterministic across checkpoints."
        ),
    )
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--model_override",
        action="append",
        default=[],
        help="Override model config values, e.g. num_inference_steps=8",
    )
    parser.add_argument("--start_epoch", type=int, default=None, help="Only evaluate checkpoints with epoch >= this value.")
    parser.add_argument("--end_epoch", type=int, default=None, help="Only evaluate checkpoints with epoch <= this value.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate at most this many checkpoints after filtering.")
    parser.add_argument("--resume", action="store_true", help="Skip checkpoints already marked ok in checkpoint_ranking.json.")
    parser.add_argument("--copy_best", action="store_true", help="Copy the best checkpoint to output_dir/best_checkpoint.pkl.")
    args = parser.parse_args()

    weights = (args.loss_weight, args.cd_weight, args.p2s_weight)
    if args.validation_workers < 0:
        raise SystemExit("validation_workers must be non-negative.")
    if args.prefilter_top_k < 0:
        raise SystemExit("prefilter_top_k must be non-negative.")
    if args.prefilter_cd_points <= 0:
        raise SystemExit("prefilter_cd_points must be positive.")
    if args.prefilter_cd_limit <= 0:
        raise SystemExit("prefilter_cd_limit must be positive.")
    if args.prefilter_last_n < 0:
        raise SystemExit("prefilter_last_n must be non-negative.")
    if any(weight < 0 for weight in weights):
        raise SystemExit("Composite weights must be non-negative.")
    if sum(weights) <= 0:
        raise SystemExit("At least one composite weight must be positive.")

    ckpt_dir = Path(args.ckpt_dir)
    output_dir = Path(args.output_dir)

    if not ckpt_dir.exists():
        raise SystemExit(f"Checkpoint directory does not exist: {ckpt_dir}")

    checkpoints = iter_checkpoints(ckpt_dir, args.pattern, args.start_epoch, args.end_epoch)
    if args.limit is not None:
        checkpoints = checkpoints[: args.limit]
    if not checkpoints:
        raise SystemExit(f"No checkpoints matched {ckpt_dir / args.pattern}")

    existing = load_existing_results(output_dir) if args.resume else {}
    if args.metric == "composite" and args.prefilter_top_k > 0:
        ranked = run_two_stage_composite(args, checkpoints, existing)
    else:
        ranked = run_selection(args, checkpoints, existing)
    write_results(ranked, output_dir)

    ok_ranked = [item for item in ranked if item.status == "ok" and item.score is not None]
    if not ok_ranked:
        print("No checkpoint was evaluated successfully.")
        return 1

    best = ok_ranked[0]
    print("\nBest checkpoint")
    print(f"  checkpoint: {best.checkpoint}")
    print(f"  epoch: {best.epoch}")
    print(f"  {args.metric}: {best.score:.8f}")
    if args.metric == "composite":
        print(
            f"  raw metrics: loss={best.loss:.8f}, "
            f"cd={best.cd:.8f}, p2s={best.p2s:.8f}"
        )
        print(
            "  weights: "
            f"loss={args.loss_weight:g}, "
            f"cd={args.cd_weight:g}, "
            f"p2s={args.p2s_weight:g}"
        )
    print(f"  ranking: {output_dir / 'checkpoint_ranking.csv'}")

    if args.copy_best:
        best_path = output_dir / "best_checkpoint.pkl"
        shutil.copy2(best.checkpoint, best_path)
        print(f"  copied best checkpoint to: {best_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
