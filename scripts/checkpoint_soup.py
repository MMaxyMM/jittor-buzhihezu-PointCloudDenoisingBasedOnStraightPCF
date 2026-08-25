#!/usr/bin/env python3
"""Greedy checkpoint soup：对排名靠前的 checkpoint 做累积权重平均。

用法（--ckpts 必须按 select_best_checkpoint.py 的综合排名从好到差排列）：

    python scripts/checkpoint_soup.py \
        --ckpts rank1.pkl rank2.pkl rank3.pkl rank4.pkl rank5.pkl \
        --out_dir checkpoint_selection_straightpcf_maxagg_endpoint/soup

输出 soup_top2.pkl、soup_top3.pkl、...（top1 就是最佳 checkpoint 本身，
不重复生成）。随后对每个 soup 运行 scripts/evaluate_local_test_models.py
（--fusion-mode max），贪心保留分数不降的最长前缀；soup 后别忘了在
{1.05, 1.10, 1.15} 附近重调 residual_alpha——权重平均会轻微改变位移幅度。

平均规则：
- 浮点参数（含 BN 的 running_mean/running_var）以 float64 累加平均，
  写回时转回原始 dtype；
- 整数型状态（如可能的 num_batches_tracked）不参与平均，直接取 rank-1 的值；
- 任何 key 集合/shape 不一致或结果含 NaN/Inf 都会直接报错退出。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import jittor as jt


def load_state(path: Path) -> dict:
    state = jt.load(str(path))
    if not isinstance(state, dict):
        raise ValueError(f"{path} 不是 state dict（得到 {type(state)}）")
    arrays = {}
    for key, value in state.items():
        arr = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
        arrays[key] = arr
    return arrays


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ckpts",
        nargs="+",
        required=True,
        help="按排名从好到差排列的 checkpoint 路径",
    )
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    ckpt_paths = [Path(p) for p in args.ckpts]
    for path in ckpt_paths:
        if not path.is_file():
            raise SystemExit(f"checkpoint 不存在: {path}")
    if len(ckpt_paths) < 2:
        raise SystemExit("soup 至少需要 2 个 checkpoint")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    states = [load_state(path) for path in ckpt_paths]
    reference = states[0]
    for path, state in zip(ckpt_paths, states):
        if set(state.keys()) != set(reference.keys()):
            missing = set(reference.keys()) - set(state.keys())
            extra = set(state.keys()) - set(reference.keys())
            raise SystemExit(
                f"{path} 的 key 集合与 rank-1 不一致: 缺少 {sorted(missing)}, "
                f"多出 {sorted(extra)}"
            )
        for key in reference:
            if state[key].shape != reference[key].shape:
                raise SystemExit(
                    f"{path} 参数 {key} shape 不一致: "
                    f"{state[key].shape} vs {reference[key].shape}"
                )

    float_keys = [
        key
        for key, value in reference.items()
        if np.issubdtype(value.dtype, np.floating)
    ]
    fixed_keys = [key for key in reference if key not in float_keys]
    if fixed_keys:
        print(f"非浮点状态取 rank-1 原值（共 {len(fixed_keys)} 个）: {fixed_keys}")

    running_sum = {
        key: np.zeros_like(reference[key], dtype=np.float64)
        for key in float_keys
    }
    manifest = {
        "ranked_ckpts": [str(p) for p in ckpt_paths],
        "soups": [],
    }

    for k, (path, state) in enumerate(zip(ckpt_paths, states), start=1):
        for key in float_keys:
            running_sum[key] += state[key]
        if k < 2:
            continue
        soup = {}
        for key in reference:
            if key in float_keys:
                averaged = running_sum[key] / k
                if not np.isfinite(averaged).all():
                    raise SystemExit(
                        f"soup_top{k} 的参数 {key} 含 NaN/Inf，丢弃该候选"
                    )
                soup[key] = jt.array(averaged.astype(reference[key].dtype))
            else:
                soup[key] = jt.array(reference[key])
        out_path = out_dir / f"soup_top{k}.pkl"
        jt.save(soup, str(out_path))
        manifest["soups"].append(
            {"path": str(out_path), "members": [str(p) for p in ckpt_paths[:k]]}
        )
        print(f"[soup] top{k} -> {out_path}")

    manifest_path = out_dir / "soup_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"manifest: {manifest_path}")
    print(
        "下一步：对每个 soup_topK.pkl 运行 evaluate_local_test_models.py，"
        "从 top2 开始贪心保留分数不降的最长前缀，并重调 residual_alpha。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
