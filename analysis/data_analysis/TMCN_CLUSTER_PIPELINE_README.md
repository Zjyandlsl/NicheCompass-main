# TMCN Cluster Pipeline 使用说明

这个文档对应脚本：

- `analysis/data_analysis/tmcn_cluster_pipeline.py`

这个脚本的目标是把你现在“spot 级代谢标签较碎”的结果，改成：

1. 先基于 NicheCompass latent cluster 得到候选空间块；
2. 再在 cluster 层面命名 TMCN；
3. 再按空间连通性拆成真正成片的 niche 区域；
4. 最后再用 pathology label 做后验评估。

---

## 一、你最需要修改的地方

### 位置 1：通路基因集

打开：

- `analysis/data_analysis/tmcn_cluster_pipeline.py`

找到最上面的：

```python
DEFAULT_PATHWAY_GENE_SETS = {...}
```

如果你导师或者文献里定义的通路 marker 不同，就改这里。

---

### 位置 2：你的四元组 CSV 路径

同一个脚本里，找到参数定义：

```python
parser.add_argument(
    "--quadruplet-csv",
    type=str,
    default="data/pre_data/siyuanzu/my_metabolite_network_simplify.csv",
```

如果你要直接使用你本机路径，就把 `default=` 改成：

```python
default="/home/zhangjunyi/xiangmu/nichecompass-main/data/pre_data/siyuanzu/my_metabolite_network_simplify.csv"
```

但更推荐你保持仓库内相对路径不变。

---

### 位置 3：输入数据路径

你有两种方式：

#### 方式 A：读取训练好的 NicheCompass 模型目录

运行时传：

```bash
--model-dir 你的模型目录
```

#### 方式 B：直接读取 h5ad

运行时传：

```bash
--adata-path 你的h5ad路径
```

如果你是小白，我建议你**先导出一个包含 latent embedding 和空间坐标的 h5ad**，然后用 `--adata-path` 跑这个脚本。

---

### 位置 4：病理标签列名

如果你的 `adata.obs` 里病理标签列叫：

- `annot_type`

那么运行时加：

```bash
--pathology-key annot_type
```

注意：

- 这里只做后验评估；
- 不参与模型训练。

---

## 二、脚本运行示例

### 示例 1：直接读取 h5ad

```bash
python analysis/data_analysis/tmcn_cluster_pipeline.py \
  --adata-path /你的/adata.h5ad \
  --quadruplet-csv data/pre_data/siyuanzu/my_metabolite_network_simplify.csv \
  --latent-key nichecompass_latent \
  --spatial-key spatial \
  --pathology-key annot_type \
  --output-dir artifacts/tmcn_cluster_pipeline_run1
```

---

### 示例 2：读取训练好的模型目录

```bash
python analysis/data_analysis/tmcn_cluster_pipeline.py \
  --model-dir /你的/nichecompass/model_dir \
  --adata-file-name adata.h5ad \
  --quadruplet-csv data/pre_data/siyuanzu/my_metabolite_network_simplify.csv \
  --latent-key nichecompass_latent \
  --spatial-key spatial \
  --pathology-key annot_type \
  --output-dir artifacts/tmcn_cluster_pipeline_run1
```

---

## 三、输出结果会在哪里

脚本会在你指定的 `--output-dir` 下生成：

- `tmcn_pipeline_result.h5ad`
- `tmcn_cluster_summary.csv`
- `tmcn_spot_level_results.csv`
- `tmcn_vs_pathology_normalized_crosstab.csv`（如果提供病理标签）
- `tmcn_cluster_labels_spatial.png`
- `tmcn_region_labels_spatial.png`
- `pathology_labels_spatial.png`（如果提供病理标签）

---

## 四、结果怎么看

### 1. `tmcn_cluster_label`

这是 cluster 级别的生态位命名，比如：

- `Healthy_niche`
- `Quiescent_niche`
- `TMCN_Lactate_Axis`
- `TMCN_Adenosine_Axis`
- `TMCN_Mixed_Lactate_Adenosine_Axis`
- `Transitional_niche`

---

### 1.1 `tmcn_cluster_summary.csv` 里新增的重要列

现在 cluster 汇总不再只看“哪条轴赢了”，还会额外输出：

- `Top1_Axis` / `Top2_Axis`
- `Top1_Fraction` / `Top2_Fraction`
- `Dominance_Margin`
- `mixed_spot_fraction`
- `quiescent_fraction`
- `mean_active_axis_count`
- `tmcn_is_quiescent_pattern`（该 cluster 是否满足 Quiescent 判定模式）

你可以把它们理解成：

- `Top1_Fraction`：这个 cluster 的总耦合里，第一主轴占多少；
- `Top2_Fraction`：第二主轴占多少；
- `Dominance_Margin`：Top1 和 Top2 差多少；
- `mixed_spot_fraction`：有多少 spot 同时呈现两条以上 axis 的连续高分；
- `quiescent_fraction`：有多少 spot 属于低耦合 / 背景状态。

这几个量就是为了避免“前面 cluster 已经切得很纯，后面永远只会得到单主轴 niche”的结构性问题。

---

### 2. `tmcn_region_base_label`

这是**最终推荐你用于画图和写论文**的区域级标签。  
因为它已经做了：

- cluster 级命名；
- 连通区域拆分；
- 小碎片合并。

---

### 3. `tmcn_region_label`

这是带区域编号的标签，例如：

- `TMCN_Lactate_Axis__R3`
- `Healthy_niche__R8`

适合做区域级统计。

---

## 五、你后面最可能继续要改的参数

还是在脚本参数区找这些：

- `--spot-high-quantile`
- `--healthy-max-high-risk-fraction`
- `--dominant-axis-min-fraction`
- `--dominant-axis-min-margin`
- `--mixed-axis-min-fraction`
- `--mixed-spot-min-fraction`
- `--mixed-spot-min-share`
- `--quiescent-min-fraction`
- `--merge-quiescent-into-healthy`
- `--min-region-size`

如果你发现结果太碎：

1. 提高 `--min-region-size`
2. 提高 `--dominant-axis-min-fraction`
3. 提高 `--spot-high-quantile`

如果你发现结果太保守，很多区域都被分到 Healthy：

1. 降低 `--spot-high-quantile`
2. 降低 `--dominant-axis-min-fraction`

如果你发现 Mixed niche 还是太少：

1. 降低 `--mixed-spot-min-fraction`
2. 降低 `--mixed-spot-min-share`
3. 降低 `--dominant-axis-min-margin`

如果你发现 Quiescent niche 太少：

1. 降低 `--quiescent-min-fraction`
2. 结合 `quiescent_fraction` 和 `mean_total_coupling` 一起看 cluster summary

如果你希望“识别 Quiescent 模式，但最终标签直接归到 Healthy”：

1. 保持 `--merge-quiescent-into-healthy`（默认开启）
2. 用 `tmcn_is_quiescent_pattern` 这个诊断列追踪哪些 cluster 原本是 Quiescent 模式

---

## 六、你现在应该怎么用

最推荐你的顺序是：

1. 先用默认参数跑一遍；
2. 看 `tmcn_region_labels_spatial.png`；
3. 再看 `tmcn_vs_pathology_normalized_crosstab.csv`；
4. 如果 TMCN 大量落在 Healthy 区，就提高阈值；
5. 如果结果太碎，就增大 `--min-region-size`。

---

## 七、这个脚本和你原来思路的对应关系

这个脚本已经按你的 10 个 step 落地了：

1. 读取四元组  
2. 计算 spot 连续分数  
3. 使用 latent + 空间信息  
4. 先拿小 cluster 当骨架  
5. cluster 内部统计  
6. cluster 级命名生态位  
7. cluster 内二次拆分  
8. 连通区域清理  
9. pathology 后验评估  
10. 结果输出，便于后续做网络拓扑分析  

如果你后面还想做“HubSpot / 源汇比例 / 拟恶性化概率模型”，建议在这个脚本输出的 `tmcn_pipeline_result.h5ad` 基础上继续加分析。
