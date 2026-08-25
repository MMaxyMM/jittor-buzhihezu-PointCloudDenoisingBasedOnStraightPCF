#!/usr/bin/env python
"""估计含噪点云的噪声强度（用于对齐训练噪声范围）。

原理：对每个点取 k 近邻做局部 PCA，最小特征值对应局部切平面法向的方差。
噪声在各坐标轴各向同性时，法向分量的标准差即单轴噪声 std 的估计。
表面采样自身的不规则也会贡献该特征值，因此结果是偏大的上界估计，
建议用中位数而非均值，并与 visualize 抽查结合判断。

用法：
    python scripts/estimate_noise_level.py --input_dir dataset_test_noisy
    python scripts/estimate_noise_level.py --input_dir dataset_test_noisy --limit 20 --k 24
"""

import argparse
import glob
import os

import numpy as np
from scipy.spatial import cKDTree


def estimate_file(path: str, k: int) -> float:
    pc = np.load(path).astype(np.float64)
    tree = cKDTree(pc)
    _, idx = tree.query(pc, k=k + 1)
    neighbors = pc[idx[:, 1:]]                            # (N, k, 3)
    centered = neighbors - neighbors.mean(axis=1, keepdims=True)
    cov = np.einsum('nki,nkj->nij', centered, centered) / k
    lam_min = np.linalg.eigvalsh(cov)[:, 0]               # 法向方差
    return float(np.sqrt(np.median(lam_min)))


def main():
    parser = argparse.ArgumentParser(description='估计 noisy.npy 的单轴噪声 std')
    parser.add_argument('--input_dir', default='dataset_test_noisy')
    parser.add_argument('--pattern', default=os.path.join('shapenet', '*', '*', 'noisy.npy'))
    parser.add_argument('--k', type=int, default=16, help='局部 PCA 的近邻数')
    parser.add_argument('--limit', type=int, default=None, help='最多处理多少个文件')
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f'未找到文件: {args.input_dir}/{args.pattern}')

    estimates = []
    for i, path in enumerate(files, start=1):
        sigma = estimate_file(path, args.k)
        estimates.append(sigma)
        print(f'[{i}/{len(files)}] {os.path.relpath(path, args.input_dir)}: sigma≈{sigma:.5f}')

    estimates = np.array(estimates)
    print('\n===== 汇总 =====')
    print(f'文件数:            {len(estimates)}')
    print(f'单轴 std 中位数:   {np.median(estimates):.5f}')
    print(f'单轴 std 均值:     {estimates.mean():.5f}')
    print(f'范围:              [{estimates.min():.5f}, {estimates.max():.5f}]')
    print('\n建议: 将 configs/transform/*.yaml 的 noise_std_min/max 对准中位数附近')
    print('注意: 该估计包含表面采样误差的贡献，实际噪声可能略小于估计值')


if __name__ == '__main__':
    main()
