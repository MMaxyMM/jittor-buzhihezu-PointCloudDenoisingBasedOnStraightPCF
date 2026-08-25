#!/usr/bin/env python3
"""Visualize denoising predictions for the first local_test sample per category.

For every ShapeNet category in the current local_test datalist, this script:
1. selects the lexicographically first model;
2. runs the selected checkpoint on its noisy.npy;
3. writes an interactive HTML containing Clean (green), Prediction (blue),
   and Noisy (red) point clouds.

Plotly is used only to generate standalone HTML files. Install it with:
    python -m pip install plotly
"""

from __future__ import annotations

import argparse
import html
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    value = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(value, dict):
        raise ValueError(f"配置文件顶层必须是 mapping: {path}")
    return value


def infer_model_config(checkpoint: Path) -> Path:
    """Infer VM/CVM/StraightPCF from the checkpoint path.

    VM is the conservative fallback because many historical VM directories
    are named only checkpoint_selection_L1_x.x.
    """
    lowered = checkpoint.as_posix().lower()
    if "straightpcf" in lowered:
        name = "straightpcf"
    elif "cvm" in lowered:
        name = "cvm"
    else:
        name = "vm"
    return PROJECT_ROOT / "configs" / "model" / f"{name}.yaml"


def resolve_model_config(value: str, checkpoint: Path) -> Path:
    if value == "auto":
        return infer_model_config(checkpoint)
    path = Path(value)
    if path.suffix not in {".yaml", ".yml"} and len(path.parts) == 1:
        path = Path("configs/model") / f"{value}.yaml"
    return resolve_path(str(path))


def validate_cloud(
    points: np.ndarray,
    path: Path,
    expected_n: int | None = None,
) -> np.ndarray:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"点云必须是 (N, 3)，实际为 {points.shape}: {path}")
    if expected_n is not None and points.shape[0] != expected_n:
        raise ValueError(
            f"点数 {points.shape[0]} 与 noisy 点数 {expected_n} 不一致: {path}"
        )
    if not np.isfinite(points).all():
        raise ValueError(f"点云含 NaN/Inf: {path}")
    return points.astype(np.float32, copy=False)


def discover_first_per_category(
    data_root: Path,
    datalist: Path | None,
) -> List[Tuple[str, Path]]:
    """Return the first valid model (sorted by model id) in each category."""
    category_models: Dict[str, List[Tuple[str, Path]]] = {}

    if datalist is not None:
        if not datalist.is_file():
            raise FileNotFoundError(f"local_test datalist 不存在: {datalist}")
        entries = sorted(
            line.strip().strip("/")
            for line in datalist.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        for entry in entries:
            parts = Path(entry).parts
            if len(parts) < 3 or parts[-3] != "shapenet":
                raise ValueError(
                    "datalist 条目应为 shapenet/<synset>/<model>，"
                    f"实际为: {entry}"
                )
            category = parts[-2]
            model_dir = data_root / Path(*parts[-3:])
            category_models.setdefault(category, []).append((entry, model_dir))
    else:
        shapenet_root = data_root / "shapenet"
        for category_dir in sorted(path for path in shapenet_root.iterdir() if path.is_dir()):
            values = []
            for noisy_path in sorted(category_dir.glob("*/noisy.npy")):
                model_dir = noisy_path.parent
                key = model_dir.relative_to(data_root).as_posix()
                values.append((key, model_dir))
            category_models[category_dir.name] = values

    selected: List[Tuple[str, Path]] = []
    for category in sorted(category_models):
        candidates = sorted(category_models[category], key=lambda item: item[0])
        valid = []
        for key, model_dir in candidates:
            noisy_path = model_dir / "noisy.npy"
            clean_path = model_dir / "clean.npy"
            if noisy_path.is_file() and clean_path.is_file():
                valid.append((key, model_dir))
        if not valid:
            raise FileNotFoundError(
                f"类别 {category} 在 {data_root} 中没有同时包含 "
                "clean.npy 和 noisy.npy 的有效样本"
            )
        selected.append(valid[0])

    if not selected:
        raise RuntimeError(f"没有在 {data_root} 中找到 local_test 样本")
    return selected


def display_subset(
    points: np.ndarray,
    max_points: int,
    seed: int,
) -> np.ndarray:
    if max_points <= 0 or points.shape[0] <= max_points:
        return points
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(points.shape[0], size=max_points, replace=False))
    return points[indices]


def make_trace(go, points, name, color, size, opacity):
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        name=name,
        marker={
            "size": size,
            "color": color,
            "opacity": opacity,
        },
        hovertemplate=(
            f"{name}<br>x=%{{x:.5f}}<br>y=%{{y:.5f}}"
            "<br>z=%{z:.5f}<extra></extra>"
        ),
    )


def write_visualization(
    output_path: Path,
    key: str,
    checkpoint: Path,
    model_config: Path,
    clean: np.ndarray,
    pred: np.ndarray,
    noisy: np.ndarray,
    max_display_points: int,
    point_size: float,
    opacity: float,
    seed: int,
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise SystemExit(
            "生成交互式 HTML 需要 Plotly，请先执行："
            "python -m pip install plotly"
        ) from exc

    clean_view = display_subset(clean, max_display_points, seed)
    pred_view = display_subset(pred, max_display_points, seed)
    noisy_view = display_subset(noisy, max_display_points, seed)

    traces = [
        make_trace(go, clean_view, "Clean", "#19a44b", point_size, opacity),
        make_trace(go, pred_view, "Prediction", "#1769e0", point_size, opacity),
        make_trace(go, noisy_view, "Noisy", "#e53935", point_size, opacity),
    ]
    buttons = [
        {"label": "全部", "method": "update", "args": [{"visible": [True, True, True]}]},
        {
            "label": "Clean + Pred",
            "method": "update",
            "args": [{"visible": [True, True, False]}],
        },
        {
            "label": "Noisy + Pred",
            "method": "update",
            "args": [{"visible": [False, True, True]}],
        },
        {"label": "仅 Clean", "method": "update", "args": [{"visible": [True, False, False]}]},
        {"label": "仅 Pred", "method": "update", "args": [{"visible": [False, True, False]}]},
        {"label": "仅 Noisy", "method": "update", "args": [{"visible": [False, False, True]}]},
    ]

    displayed = clean_view.shape[0]
    display_text = (
        f"每组显示 {displayed:,} 点"
        if displayed < clean.shape[0]
        else f"显示完整 {clean.shape[0]:,} 点"
    )
    figure = go.Figure(data=traces)
    figure.update_layout(
        title={
            "text": (
                f"{key}<br><sup>{display_text}；"
                "点击图例或上方按钮可开关点云</sup>"
            ),
            "x": 0.5,
        },
        template="plotly_white",
        margin={"l": 0, "r": 0, "t": 115, "b": 0},
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.02,
            "yanchor": "bottom",
        },
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 0.5,
                "xanchor": "center",
                "y": 1.13,
                "yanchor": "top",
                "buttons": buttons,
                "showactive": True,
            }
        ],
        scene={
            "aspectmode": "data",
            "xaxis": {"title": "X"},
            "yaxis": {"title": "Y"},
            "zaxis": {"title": "Z"},
            "camera": {"projection": {"type": "orthographic"}},
        },
        meta={
            "checkpoint": str(checkpoint),
            "model_config": str(model_config),
        },
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.write_html(
        str(output_path),
        include_plotlyjs="directory",
        full_html=True,
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
        },
    )


def write_index(
    output_dir: Path,
    checkpoint: Path,
    model_config: Path,
    rows: Sequence[Tuple[str, str, int]],
) -> None:
    cards = "\n".join(
        (
            "<li>"
            f"<a href=\"{html.escape(filename)}\">{html.escape(key)}</a>"
            f"<span>{point_count:,} points</span>"
            "</li>"
        )
        for key, filename, point_count in rows
    )
    content = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Local Test 点云可视化</title>
  <style>
    body {{ max-width: 980px; margin: 40px auto; padding: 0 24px;
            font-family: system-ui, sans-serif; color: #172033; }}
    code {{ background: #eef2f7; padding: 2px 6px; border-radius: 5px;
            overflow-wrap: anywhere; }}
    ul {{ list-style: none; padding: 0; display: grid;
          grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }}
    li {{ border: 1px solid #dce3ec; border-radius: 10px; padding: 14px;
          display: flex; flex-direction: column; gap: 6px; }}
    a {{ color: #1769e0; font-weight: 650; text-decoration: none; }}
    span {{ color: #687386; font-size: 13px; }}
    .legend b {{ margin-right: 16px; }}
  </style>
</head>
<body>
  <h1>Local Test 点云可视化</h1>
  <p>Checkpoint：<code>{html.escape(str(checkpoint))}</code></p>
  <p>模型配置：<code>{html.escape(str(model_config))}</code></p>
  <p class="legend">
    <b style="color:#19a44b">● Clean</b>
    <b style="color:#1769e0">● Prediction</b>
    <b style="color:#e53935">● Noisy</b>
  </p>
  <p>每个 ShapeNet 类别选择当前 local_test 列表中按模型 ID 排序后的第一个样本。</p>
  <ul>{cards}</ul>
</body>
</html>
"""
    (output_dir / "index.html").write_text(content, encoding="utf-8")


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "使用指定最佳 checkpoint 对 local_test 每一类别的第一个模型推理，"
            "并生成可交互 HTML 点云可视化"
        )
    )
    parser.add_argument("checkpoint", help="最佳模型 checkpoint 路径")
    parser.add_argument(
        "--model-config",
        default="auto",
        help=(
            "模型配置：auto、vm、cvm、straightpcf 或 YAML 路径；"
            "auto 根据 checkpoint 路径推断，无法识别时按 VM 处理"
        ),
    )
    parser.add_argument(
        "--data-root",
        default="dataset_train_pcd_disk/local_test",
        help="包含 local_test clean.npy/noisy.npy 的缓存根目录",
    )
    parser.add_argument(
        "--datalist",
        default="dataset_train/local_test/datalist.txt",
        help="当前 local_test 列表；传入空字符串则直接扫描 data-root",
    )
    parser.add_argument(
        "--transform-config",
        default="configs/transform/predict.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录；默认 visualizations/<checkpoint父目录名>",
    )
    parser.add_argument(
        "--fusion-mode",
        choices=("mix", "max"),
        default="max",
        help="mix=重叠 patch 距离加权；max=采用最大权重 patch",
    )
    parser.add_argument(
        "--max-display-points",
        type=int,
        default=50000,
        help="每组最多写入 HTML 的点数；0 表示全部，默认 50000",
    )
    parser.add_argument("--point-size", type=float, default=1.2)
    parser.add_argument("--opacity", type=float, default=0.72)
    parser.add_argument("--use-cuda", type=int, choices=(0, 1), default=1)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_display_points < 0:
        raise SystemExit("--max-display-points 不能小于 0")
    if args.point_size <= 0:
        raise SystemExit("--point-size 必须大于 0")
    if not 0 < args.opacity <= 1:
        raise SystemExit("--opacity 必须在 (0, 1] 范围内")

    try:
        import plotly.graph_objects  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "生成交互式 HTML 需要 Plotly，请先执行："
            "python -m pip install plotly"
        ) from exc

    import jittor as jt
    from src.model.parse import get_model

    checkpoint = resolve_path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint}")
    model_config_path = resolve_model_config(args.model_config, checkpoint)
    transform_config_path = resolve_path(args.transform_config)
    data_root = resolve_path(args.data_root)
    datalist = resolve_path(args.datalist) if args.datalist else None
    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "visualizations" / checkpoint.parent.name
    )

    jt.flags.use_cuda = args.use_cuda
    jt.set_global_seed(args.seed)
    np.random.seed(args.seed)

    model_config = deepcopy(load_yaml(model_config_path))
    model_config["fusion_mode"] = {"mix": "weighted", "max": "best"}[
        args.fusion_mode
    ]
    model = get_model(
        model_config=model_config,
        transform_config=deepcopy(load_yaml(transform_config_path)),
    )
    model.load(str(checkpoint))
    model.set_predict(True)
    model.eval()

    samples = discover_first_per_category(data_root, datalist)
    print(f"checkpoint: {checkpoint}")
    print(f"模型配置: {model_config_path}")
    print(f"类别数: {len(samples)}")
    print(f"输出目录: {output_dir}")

    index_rows: List[Tuple[str, str, int]] = []
    start = time.time()
    for index, (key, model_dir) in enumerate(samples, start=1):
        noisy_path = model_dir / "noisy.npy"
        clean_path = model_dir / "clean.npy"
        noisy = validate_cloud(np.load(noisy_path, allow_pickle=False), noisy_path)
        clean = validate_cloud(
            np.load(clean_path, allow_pickle=False),
            clean_path,
            expected_n=noisy.shape[0],
        )
        with jt.no_grad():
            output = model.predict_step({"pc_noisy": jt.array(noisy[None, ...])})
        pred = output[0]["pc_denoised"]
        if not isinstance(pred, np.ndarray):
            pred = pred.numpy()
        pred = validate_cloud(
            np.asarray(pred),
            Path(f"<prediction:{key}>"),
            expected_n=noisy.shape[0],
        )

        category, model_id = Path(key).parts[-2:]
        filename = f"{category}_{model_id}.html"
        write_visualization(
            output_path=output_dir / filename,
            key=key,
            checkpoint=checkpoint,
            model_config=model_config_path,
            clean=clean,
            pred=pred,
            noisy=noisy,
            max_display_points=args.max_display_points,
            point_size=args.point_size,
            opacity=args.opacity,
            seed=args.seed,
        )
        index_rows.append((key, filename, noisy.shape[0]))

        elapsed = time.time() - start
        remaining = elapsed / index * (len(samples) - index)
        print(
            f"[{index}/{len(samples)}] {key} 完成 "
            f"【已用 {format_duration(elapsed)} / "
            f"预计还需 {format_duration(remaining)}】"
        )
        del output, pred
        jt.gc()

    write_index(output_dir, checkpoint, model_config_path, index_rows)
    print(f"可视化入口: {output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
