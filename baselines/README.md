# PRODEN / PiCO for POI assignment

这两个 baseline 从随机初始化训练，复用原项目的特征与编码器，不调用原项目的 PLL loss、实例 commitment、easy mining、可靠性门控、温度锐化、diffusion 或任何已有 Stage-I checkpoint。

## 与原模型相同的输入信息

| 信息 | 具体实现 |
|---|---|
| 用户、时间、观测坐标、轨迹上下文 | 原 `UserTrajectoryModel`，24 个 weekday/weekend × 2h slots，Transformer |
| POI 身份 | 原 `CandidatePOIEncoder.poi_emb`，全局 POI index + 1；0 为 padding |
| POI category / 层级语义 | 原三层 `qwen_poi_embeddings.pt` (`l1/l2/l3`) 与相同 text projection；不是使用真实访问 category |
| 不同时间的细 category 被 check-in 先验 | 相同 `category_time_distribution_P_Category_given_Time.csv`，`cand_probs` |
| 不同时间的 main category 先验 | 原 category→main-category 映射和聚合方式，`cand_main_cat_probs` |
| 观测–候选距离 | 原实现的 `log1p(norm(rad_coordinate_difference)*6371000)` |
| 用户历史偏好 | 原训练观测候选频率和 log recurrence 特征；不用真实训练 POI 计数 |

直接复用原特征模块，不另加 category embedding、不重新生成文本表示、不使用测试标签生成统计先验。显式 category ID 用于计算先验与评估，分类网络的类别语义入口与原模型一致，为 Qwen 三层语义表示。

## 原方法机制

**PRODEN**：每个实例在固定的候选支持上保存软标签；首次为均匀目标。以已保存的旧目标计算 CE，然后按本次模型预测在候选支持内归一化，更新下次访问该实例时使用的目标。不是 `CE(p.detach(), p)`。

**PiCO**：保留分类 CE、动量 key encoder、投影头、全局 **POI ID** 原型、伪标签正样本、多正样本对比损失、环形队列，以及基于最近原型的 one-hot 目标 EMA 更新（默认 phi 从 0.95 到 0.8）。原型预测在本批原型更新之前计算，与官方实现顺序一致。没有质量门控或样本过滤。预测阶段仅使用 online 分类分支。

非图像输入不能使用 crop/color jitter。本实现使用同一轨迹与候选特征上的两次独立 dropout 前向作为 query/key 视图，不扰动经纬度和候选支持。没有 BatchNorm，因此不需要原图像实现的 DDP batch shuffle。原型按官方逐实例 EMA 更新；同一 POI 在不同候选集合始终对应一个原型。队列只使用已写入的实际表示，不将随机未初始化队列作为有效数据；首批 MoCo 可能没有对比负样本。

`--prototype_start 1` 是 **PiCO 自己**的 MoCo→原型消歧切换（官方 CIFAR 启动脚本也有 `prot_start=1`），不是本项目 Stage-I warm-up。若希望从第一个 epoch 就使用原型，可设 `--prototype_start 0`；未更新的零原型会产生并列，需要验证该设置。

本实现参考官方算法并适配原项目接口，未直接复制官方源码：

- PRODEN / ICML 2020: https://github.com/Lvcrezia77/PRODEN ，参考 commit `637b531a627a149ca494f966f4f41a20ad235853`
- PiCO / ICLR 2022: https://github.com/hbzju/PiCO ，参考 commit `ad687bb4eea533ce584e5a84f0955914c673fb6e`
- 原项目基础 commit: `c9c627367a39c1f2efcfb25b34d727beb4f98c1b`

## 全局归一化与可扩展近似

默认 `--normalizer full`。正候选和全部非候选 POI 均参与 softmax 分母，非候选也按相同的时间先验、文本/category、距离和 recurrence 进行编码。分块使用 activation checkpoint，避免保留全量 POI 的隐向量。**这是完整的全局 POI 分类目标，但计算依然昂贵**：每个观测需要对所有 POI 打分，checkpoint 还会增加反向重计算。不使用类别 0。

`--normalizer sampled --num_negatives 256` 使用“全部候选 + 均匀无放回采样的非候选 POI”作为分类池。要求非候选数量足够，不重复负类。这是 **采样分类近似**，没有 importance correction，不能把它称为与原全类别 softmax 完全等价的实现。论文应标注 adapted/sampled，并让两种 baseline 使用相同 negative budget，必要时验证 128/256/512 个负类的敏感性。

不管使用哪种模式，测试均只在相同的原始空间候选池上计算 Acc@1/Acc@5 和 category/main-category accuracy。

## 数据协议与实验公平性

`data.py` 继承原 `CheckinSequenceDataset` 的过滤、80/10/10 用户时间划分、chunk、噪声坐标和候选生成，只固定每个实例的随机候选支持，增加 stable step ID。model seed 可变，`data_seed` 应固定。验证/测试候选也固定。两种 baseline 产生相同的候选、坐标和输入先验。

**当前仓库代码与 PDF 的描述存在差异**：代码使用半径 `noisy_value` 内的均匀圆盘噪声，且通过随机截断保留真实 POI；PDF 写的是 Gaussian/nearest truncation。本实现忠实沿用仓库代码，没有静默修改成论文协议，也没有移除原 proxy 数据生成时保留 GT 的机制。GT 只用于 inherited proxy 候选构造及评估；训练入口显式移除 `label_pos` / `true_poi_id`，损失不读取它们。

原 `__getitem__` 会反复随机生成候选，baseline 需要稳定支持。因此不能假设旧论文单次测试结果与这里的固定候选逐项一致。最终比较应让主方法也使用同一个 `StableDataset`（训练需要在每个 epoch 设置 `dataset.epoch`，可以忽略新增 `step_id` 字段）；或将共同候选导出给所有方法再运行。不要改变 baseline 的 feature encoder 以补入主方法训练目标。

时间先验文件原样读取，代码无法从 CSV 判断其统计来源。正式实验必须确认 CSV 来自训练集，或是独立外部先验，并让所有方法使用同一个文件。如果现有 CSV 包含 validation/test check-ins，需要统一重建后重跑所有方法。每次运行记录 `protocol.json`：包括全部输入文件 SHA256、候选参数、data seed 和分类归一化模式。

## 运行

在原仓库根目录执行。需要安装原 `requirements_pll.txt` 加适合服务器的 PyTorch（建议已有的 PyTorch 2.x；不强制重新安装 CUDA）。模型依赖原项目 `model.py` / `dataset.py` / `utils.py`，请保留这些文件。

每个数据目录需要：

```text
dataset/<dataset>/filtered_checkin_data.csv
dataset/<dataset>/poi.csv
dataset/<dataset>/main_category.csv
dataset/<dataset>/category_time_distribution_P_Category_given_Time.csv
dataset/<dataset>/qwen_poi_embeddings.pt
```

`category_list.txt` 应使用原实验文件。若缺少，原 processor 会从 POI 类别生成。**公开仓库没有 Qwen `.pt` 缓存，必须使用服务器原实验缓存；不要用测试脚本的随机语义 embedding 替代。** 可通过 `--embedding_path` 指定相同缓存路径。

先在一个数据集上执行：

```bash
CUDA_VISIBLE_DEVICES=0 python -m baselines.train \
  --method proden --dataset tokyo --seed 42 --data_seed 42 \
  --normalizer full --batch_size 4 --class_chunk_size 512

CUDA_VISIBLE_DEVICES=0 python -m baselines.train \
  --method pico --dataset tokyo --seed 42 --data_seed 42 \
  --normalizer full --batch_size 4 --class_chunk_size 512
```

如果全局分类计算太慢，明确切换为采样近似：

```bash
CUDA_VISIBLE_DEVICES=0 python -m baselines.train \
  --method proden --dataset tokyo --seed 42 --data_seed 42 \
  --normalizer sampled --num_negatives 256 --batch_size 16

CUDA_VISIBLE_DEVICES=0 python -m baselines.train \
  --method pico --dataset tokyo --seed 42 --data_seed 42 \
  --normalizer sampled --num_negatives 256 --batch_size 16
```

三个数据集、两个方法、五个 model seeds：

```bash
# 数据目录使用这个仓库中实际存在的名称；如服务器命名不同，可覆盖 DATASETS。
GPU=0 SEEDS="42 43 44 45 46" \
DATASETS="tokyo ny_filterPOI gowalla_filtered" \
bash baselines/run_baselines.sh

# 采样版本，不要与 full 版本覆盖同一个输出目录。
GPU=0 NORMALIZER=sampled BATCH_SIZE=16 \
DATASETS="tokyo ny_filterPOI gowalla_filtered" \
bash baselines/run_baselines.sh
```

默认输出路径中包含归一化模式，full/sampled 分开保存；重新执行同一个 method/dataset/seed/mode 会覆盖 checkpoint。调参实验使用新的 `--output_dir`；固定验证选参规则，不能根据测试结果选超参数。

保存训练日志、每轮验证结果、`last.pt`（model/optimizer/confidence/queue/RNG）和 `best.pt`（验证 Acc@1 最优），最终加载 best 再评估测试集。`metrics.json` 中包括四项指标、模型参数量与协议。

```bash
python -m baselines.train --method proden --dataset tokyo \
  --normalizer full --resume result/tokyo/pll_baselines/proden_full/seed_42/last.pt

python -m baselines.train --method pico --dataset tokyo --normalizer full \
  --eval_only --resume result/tokyo/pll_baselines/pico_full/seed_42/best.pt \
  --benchmark_batches 20

python -m baselines.summarize
```

断点继续训练要求使用同一个输出目录和数据/模型参数。`best.pt` 仅支持 eval；恢复训练使用 `last.pt`。推理成本仅测分类分支的 `predict`，排除 DataLoader 和 CPU→GPU 搬运，CUDA 计时前后同步；计时开始前的三次前向只用于设备预热，不是训练 warm-up。

## 验证

```bash
python -m unittest baselines.test_baselines -v
# 支持本地进程通信的机器可额外验证 DataLoader workers：
PLL_TEST_WORKERS=1 python -m unittest baselines.test_baselines -v
```

测试检查：候选稳定与原特征逐项一致；全局分块 softmax 的数值和梯度与独立构造的 dense 分类一致；PRODEN 旧目标与非零梯度；PiCO 动量 encoder、原型、队列溢出、投影梯度和 checkpoint；训练输入不含真实标签；评估指标分母。合成数据仅验证执行，不构成 Tokyo/NYC/Gowalla 实验结果。

完整 CLI 的额外验证：PRODEN 和 PiCO 各训练两个 epoch，保存 best/last；从 last 恢复，再从 best 单独评估；full 模式反向及 sampled 模式执行。正式训练不要传 `--max_train_steps` 或 `--max_eval_batches`；使用这些选项的结果会标记 `smoke_only`，汇总脚本自动排除。
