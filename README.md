# Jittor StraightPCF 点云降噪（A/B 榜复现代码）

本仓库是第六届计图挑战赛赛道二“不知何组”的点云降噪方案。输入为
`(N, 3) float32` 的含噪点云 `noisy.npy`，模型预测逐点三维位移，输出与输入
点数、顺序和数据类型一致的 `denoised.npy`。

## 比赛结果

| 项目 | 结果 |
| --- | ---: |
| 赛道 | 赛道二 |
| 队伍 | 不知何组 |
| B 榜名次 | 23 |
| Score | 76.74 |
| CD score | 64.44 |
| P2S score | 89.03 |
| mean CD pred / noisy | 0.000095 / 0.000266 |
| mean P2S pred / noisy | 0.000016 / 0.000161 |

B 榜最优结果由以下完整推理方案产生：

```text
Exp6 StraightPCF Soup Top-4
+ 原始点云分支
+ X 轴旋转 90° 分支
+ 逆旋转后逐点平均（TTA2）
+ residual_alpha = 1.04
+ predict_rounds = 1
+ fusion_mode = best
```

仓库包含 B 榜最优 Soup 权重以及模型构造所需的 CVM 权重：

```text
checkpoint_selection_b_exp6_b32_alpha105_cvm/best_checkpoint.pkl
checkpoint_selection_b_exp6_b32_alpha105_straightpcf_soup_top4/best_checkpoint.pkl
```

对应 SHA-256：

```text
CVM:  2081b2b06bc8e6499a6fa7f0c6d11bd11876f7c3d564f6223ce21947f270c143
Soup: 043f60b854587e474600e9fd65a82e0a88811b0df1582beed7b2906a85651b04
```

## 方法概述

网络基于 StraightPCF 三阶段架构，并使用 Jittor 实现：

1. **VelocityModule（VM）**：Dynamic EdgeConv 提取局部特征并回归逐点速度；
2. **Coupled VelocityModule（CVM）**：串联四个 VM，约束中间轨迹一致性；
3. **StraightPCF**：DistanceModule 估计移动距离，并对最终 endpoint 联合微调。

最终核心实现包括：

- patch 内所有点共享插值时间 `t`，减少局部几何不一致；
- EdgeConv 使用 max aggregation；
- StraightPCF 阶段的 velocity nets 保持 eval 模式，以固定 BN/Dropout 状态；
- 不对 velocity 参数执行 `stop_grad`，endpoint loss 仍能更新 velocity nets；
- 三阶段使用 Charbonnier 向量损失；
- 大点云采用 patch 推理，并保留覆盖点的最高权重 patch 预测；
- 正式预测的 transform 为空，不会给官方 `noisy.npy` 二次加噪。

本项目参考 StraightPCF：

> Dasith de Silva Edirimuni et al. StraightPCF: Straight Point Cloud
> Filtering. CVPR 2024.

官方实现：https://github.com/ddsediri/StraightPCF

第三方来源和许可证说明见 [NOTICE](NOTICE)。

## 相对 A 榜算法的改动

B 榜保持 A 榜的 VM/CVM/StraightPCF 核心网络、损失、MaxAgg、shared-patch-t
和 endpoint-gradient 训练逻辑不变，只做了以下适配与后处理改动：

1. B 榜训练直接读取官方 OBJ，不使用 A 榜的预采样点云缓存；
2. 针对数据规模重新设置 batch size、训练轮数和学习率；
3. 对同一次 Exp6 StraightPCF 训练的后期 checkpoint 做参数平均；
4. 正式推理加入固定 X90 旋转 TTA2；
5. 推理 residual alpha 从 A 榜的 1.10 调整为 1.04。

没有使用 B 榜测试标签、测试网格、外部训练数据或人工逐样本参数。

## 目录结构

```text
.
├── run.py                         # 通用训练/普通预测入口
├── select_best_checkpoint.py      # loss/CD/P2S 两阶段 checkpoint 筛选
├── evaluate.py                    # 已生成结果的离线指标入口
├── requirements.txt
├── configs/
│   ├── data/
│   ├── model/
│   ├── system/
│   ├── task/
│   └── transform/
├── datalist/                      # A/B 榜样本相对路径列表，不含数据
├── src/
│   ├── data/                      # OBJ/NPY 读取、采样、增强和 patch 构造
│   ├── model/                     # VM、CVM、StraightPCF
│   └── system/                    # 训练、验证、预测输出
├── scripts/
│   ├── checkpoint_soup.py
│   ├── predict_b_exp6_soup_top4_tta2_x90_alpha104.py
│   ├── evaluate_b_local200_candidates.py
│   ├── evaluate_b_local200_tta.py
│   ├── evaluate_local_test_models.py
│   ├── visualize_local_test_predictions.py
│   └── ...
├── checkpoint_selection_b_exp6_b32_alpha105_cvm/
│   └── best_checkpoint.pkl
└── checkpoint_selection_b_exp6_b32_alpha105_straightpcf_soup_top4/
    ├── best_checkpoint.pkl
    └── soup_manifest.json
```

## 环境安装

建议复现环境：

- Ubuntu 22.04
- Python 3.10
- CUDA 12.4
- NVIDIA RTX 4090 24 GB
- Jittor 1.3.10

```bash
conda create -n jittor-pcd python=3.10 -y
conda activate jittor-pcd
conda install -c conda-forge gcc=10 gxx=10 libgomp -y
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

首次运行 Jittor 会编译算子，因此第一轮或第一个样本耗时会明显更长。

建议限制每个 dataloader worker 内部的 BLAS/OpenMP 线程，避免多进程线程过量：

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

## 数据准备

仓库不包含任何比赛数据。B 榜训练集和测试集应放在仓库根目录：

```text
dataset_train/
└── shapenet/<synset_id>/<model_id>/models/model_normalized.obj

dataset_test_noisy/
└── shapenet/<synset_id>/<model_id>/noisy.npy
```

代码包提供的 B 榜列表包含：

- 训练：19699 个模型；
- 验证：100 个模型；
- 测试：200 个模型。

快速检查路径：

```bash
train_item=$(sed -n '1p' datalist/train_b.txt)
val_item=$(sed -n '1p' datalist/validate_b.txt)
test_item=$(sed -n '1p' datalist/test_b.txt)

test -f "dataset_train/${train_item}/models/model_normalized.obj"
test -f "dataset_train/${val_item}/models/model_normalized.obj"
test -f "dataset_test_noisy/${test_item}/noisy.npy"
```

## B 榜 Exp6 三阶段训练

### 1. VelocityModule

配置：batch 48、150 epochs、Adam、学习率 `2.5e-4`。

```bash
python run.py \
  --task configs/task/train_b_exp6_b32_alpha105_vm.yaml \
  --seed 123
```

筛选最后 70 个 checkpoint，快速初筛前 10 后完整计算 loss/CD/P2S：

```bash
python select_best_checkpoint.py \
  --metric composite \
  --ckpt_dir experiments_b/exp6_b32_alpha105/vm \
  --task_template configs/task/train_b_exp6_b32_alpha105_vm.yaml \
  --mesh_dir dataset_train \
  --output_dir checkpoint_selection_b_exp6_b32_alpha105_vm \
  --geometry_from_validate_transform \
  --loss_weight 1 --cd_weight 2 --p2s_weight 2 \
  --cd_points 32768 --cd_limit 20 \
  --prefilter_top_k 10 \
  --prefilter_cd_points 4096 --prefilter_cd_limit 10 \
  --prefilter_last_n 70 \
  --validation_workers 0 \
  --seed 123 \
  --copy_best
```

### 2. Coupled VelocityModule

配置：batch 32、100 epochs、Adam、学习率 `1.5e-4`。

```bash
python run.py \
  --task configs/task/train_b_exp6_b32_alpha105_cvm.yaml \
  --seed 123

python select_best_checkpoint.py \
  --metric composite \
  --ckpt_dir experiments_b/exp6_b32_alpha105/cvm \
  --task_template configs/task/train_b_exp6_b32_alpha105_cvm.yaml \
  --mesh_dir dataset_train \
  --output_dir checkpoint_selection_b_exp6_b32_alpha105_cvm \
  --geometry_from_validate_transform \
  --loss_weight 1 --cd_weight 2 --p2s_weight 2 \
  --cd_points 32768 --cd_limit 20 \
  --prefilter_top_k 10 \
  --prefilter_cd_points 4096 --prefilter_cd_limit 10 \
  --prefilter_last_n 70 \
  --validation_workers 0 \
  --seed 123 \
  --copy_best
```

### 3. StraightPCF endpoint 联合微调

配置：batch 32、100 epochs、Adam、学习率 `1.5e-4`。

```bash
python run.py \
  --task configs/task/train_b_exp6_b32_alpha105_straightpcf.yaml \
  --seed 123

python select_best_checkpoint.py \
  --metric composite \
  --ckpt_dir experiments_b/exp6_b32_alpha105/straightpcf \
  --task_template configs/task/train_b_exp6_b32_alpha105_straightpcf.yaml \
  --mesh_dir dataset_train \
  --output_dir checkpoint_selection_b_exp6_b32_alpha105_straightpcf \
  --geometry_from_validate_transform \
  --loss_weight 1 --cd_weight 2 --p2s_weight 2 \
  --cd_points 32768 --cd_limit 20 \
  --prefilter_top_k 10 \
  --prefilter_cd_points 4096 --prefilter_cd_limit 10 \
  --prefilter_last_n 70 \
  --validation_workers 0 \
  --seed 123 \
  --copy_best
```

## 生成 Soup Top-4

最优 Soup 使用同一次 Exp6 StraightPCF 训练的 epoch 93、90、88、94；顺序来自
综合筛选排名，而不是按 epoch 大小排序：

```bash
python scripts/checkpoint_soup.py \
  --ckpts \
    experiments_b/exp6_b32_alpha105/straightpcf/checkpoint_93.pkl \
    experiments_b/exp6_b32_alpha105/straightpcf/checkpoint_90.pkl \
    experiments_b/exp6_b32_alpha105/straightpcf/checkpoint_88.pkl \
    experiments_b/exp6_b32_alpha105/straightpcf/checkpoint_94.pkl \
  --out_dir checkpoint_selection_b_exp6_b32_alpha105_straightpcf/soup
```

生成的 `soup_top4.pkl` 即仓库随附最优权重的来源。Soup 工具会检查所有参数的
key、shape 和有限值，并以 float64 累加浮点参数后写回原 dtype。

## 复现 B 榜最优推理

正式入口：

```bash
python scripts/predict_b_exp6_soup_top4_tta2_x90_alpha104.py \
  --data-root dataset_test_noisy \
  --datalist datalist/test_b.txt \
  --model-config configs/model/straightpcf_b_exp6_b32_alpha100.yaml \
  --transform-config configs/transform/predict.yaml \
  --checkpoint checkpoint_selection_b_exp6_b32_alpha105_straightpcf_soup_top4/best_checkpoint.pkl \
  --output-root results_b_exp6_soup_top4_tta2_x90_alpha104/dataset_test_noisy \
  --alpha 1.04 \
  --seed 123 \
  --use-cuda 1
```

该入口固定执行：

1. 原始点云推理；
2. 点云绕 X 轴旋转 90° 后推理；
3. 第二个结果逆旋转回原坐标；
4. 两个逐点位移取平均；
5. 相对原始 noisy 应用 `residual_alpha=1.04`。

预测支持基于 manifest 的断点复用。已有输出只有在 checkpoint、配置、datalist、
样本数和推理参数完全一致且 shape/dtype 合法时才会跳过。

只检查路径、配置和 200 个输入，不加载模型：

```bash
python scripts/predict_b_exp6_soup_top4_tta2_x90_alpha104.py \
  --check-only
```

## 输出验证与打包

预测输出必须为：

```text
results_b_exp6_soup_top4_tta2_x90_alpha104/
└── dataset_test_noisy/
    └── shapenet/<synset_id>/<model_id>/denoised.npy
```

检查文件数、shape、dtype 和有限值：

```bash
python - <<'PY'
from pathlib import Path
import numpy as np

data_root = Path("dataset_test_noisy")
output_root = Path(
    "results_b_exp6_soup_top4_tta2_x90_alpha104/dataset_test_noisy"
)
items = [
    line.strip()
    for line in Path("datalist/test_b.txt").read_text().splitlines()
    if line.strip()
]
assert len(items) == 200, len(items)
for item in items:
    noisy = np.load(data_root / item / "noisy.npy", mmap_mode="r")
    denoised = np.load(output_root / item / "denoised.npy", mmap_mode="r")
    assert denoised.shape == noisy.shape, item
    assert denoised.dtype == np.float32, item
    assert np.isfinite(denoised).all(), item
print("200 predictions: shape/dtype/finite OK")
PY
```

提交 zip 第一层必须直接是 `shapenet/`：

```bash
cd results_b_exp6_soup_top4_tta2_x90_alpha104/dataset_test_noisy
zip -r ../../result_b_exp6_soup_top4_tta2_x90_alpha104.zip shapenet/
cd ../..
unzip -l result_b_exp6_soup_top4_tta2_x90_alpha104.zip | sed -n '1,30p'
```

## A 榜核心流程

仓库保留 A 榜最终方法所需代码和配置：

```bash
python run.py --task configs/task/train_vm_shared_patch_t.yaml --seed 123
python run.py --task configs/task/train_cvm_maxagg_shared_patch_t.yaml --seed 123
python run.py --task configs/task/train_straightpcf_maxagg_endpoint.yaml --seed 123
```

A 榜使用预采样缓存时，可先运行：

```bash
python scripts/precompute_clean_points.py \
  --input_dir dataset_train \
  --output_dir dataset_train_pcd_disk \
  --train_num_points 200000 \
  --test_num_points 50000 \
  --num_vertex_samples 1024 \
  --workers 16 \
  --seed 123
```

数据集和 A 榜权重不随 B 榜代码包重复发布。

## 本地评测与可视化

- `scripts/create_local_holdout.py`：按类别建立本地留出划分；
- `scripts/generate_local_test_benchmark.py`：从 OBJ 生成固定 clean/noisy benchmark；
- `scripts/evaluate_local_test_models.py`：计算 CD、P2S 和百分制分数；
- `scripts/evaluate_b_local200_candidates.py`：比较 alpha、Soup 和 prediction ensemble；
- `scripts/evaluate_b_local200_tta.py`：比较固定旋转 TTA；
- `scripts/visualize_local_test_predictions.py`：生成 clean/noisy/pred 交互 HTML。

这些工具只用于训练数据构造的本地代理评测，不读取官方测试集 clean 或 mesh。

## 指标说明

- **CD**：预测点云与干净点云之间的双向平均最近邻平方距离；
- **P2S**：预测点到原始干净网格表面的最近距离平方均值；
- **Score**：逐样本以 noisy 为零分基线，CD/P2S 得分各占 50%。

本地留出集只能比较候选的相对趋势，表格顶部成绩为官方 B 榜结果。

## 代码检查压缩包

比赛要求压缩包名称为：

```text
contest2_不知何组_023.zip
```

压缩包结构：

```text
contest2_不知何组_023.zip
├── code/                  # 本仓库内容，包含 B 榜最优 ckpt
├── requirements.txt
└── 提交说明文档.pdf
```

公开仓库不包含联系人手机号和微信；这些信息应只写入提交给组委会的
`提交说明文档.pdf`。

## 复现注意事项

- 正式预测必须使用空的 `predict_transform.augments`，禁止二次加噪；
- StraightPCF 构造阶段仍会读取 CVM checkpoint，因此两个随附权重都要保留；
- 输出必须与输入逐文件保持相同 `(N,3)` shape，dtype 必须为 `float32`；
- 不同 CUDA/Jittor 编译环境可能产生微小浮点差异；
- OBJ 材质、颜色、贴图和 `.mtl` 不参与训练；
- 数据、日志、中间 checkpoint、预测结果和 HTML 不进入 Git。

## License

本仓库代码按 [LICENSE](LICENSE) 发布。StraightPCF 原实现及第三方依赖仍适用
各自许可证。
