# 基因程序 (GP) 掩码 - 快速参考表

## 📚 核心概念对照

| 概念 | 英文 | 解释 | 示例 |
|------|------|------|------|
| **基因程序** | Gene Program (GP) | 一组功能相关的基因，共同完成某个生物学过程 | 血管生成 GP: VEGFA → VEGFR1/2 |
| **来源基因** | Source Genes | "发送"信号的基因 | 配体基因、酶基因 |
| **目标基因** | Target Genes | "接收"信号的基因 | 受体基因、传感器基因、靶基因 |
| **GP 掩码** | GP Mask | GP 的二进制矩阵表示，用于模型训练 | 每行=基因，每列=GP，值为0/1 |

---

## 🗄️ 三大数据库对照

| 数据库 | 作用 | 来源基因 | 目标基因 | GP 数量 |
|--------|------|----------|----------|---------|
| **OmniPath** | 配体-受体相互作用 | Ligand (配体) | Receptor (受体) | ~2000-3000 |
| **NicheNet** | 完整信号通路 | Ligand (配体) | Receptor + Target (受体+靶基因) | ~1000-2000 |
| **MEBOCOST** | 代谢相关通信 | Enzyme (酶) | Sensor (传感器) | ~100-200 |

---

## 🔄 处理流程

```
原始 GP 数据
    ├── OmniPath:  2500 个 GP
    ├── NicheNet:  1500 个 GP
    └── MEBOCOST:   150 个 GP
         ↓
    过滤 (移除低质量 GP)
         ↓
    合并 (合并重叠度 > 90% 的 GP)
         ↓
最终 GP: ~297 个
```

---

## 💡 GP 示例详解

### 示例 1: OmniPath GP (配体-受体)
```python
{
    "VEGFA_VEGFR_GP": {
        "sources": ["VEGFA"],                      # 血管内皮生长因子 A
        "targets": ["FLT1", "KDR", "FLT4"],        # VEGF 受体 1/2/3
        "sources_categories": ["ligand"],
        "targets_categories": ["receptor", "receptor", "receptor"]
    }
}
```
**生物学意义**: 血管生成相关的细胞间通信

---

### 示例 2: NicheNet GP (配体-受体-靶基因)
```python
{
    "WNT_combined_GP": {
        "sources": ["WNT3A", "WNT5A"],                          # WNT 配体
        "targets": ["FZD1", "FZD2", "MYC", "CCND1", ...],      # 受体 + 靶基因
        "sources_categories": ["ligand", "ligand"],
        "targets_categories": ["receptor", "receptor", "target", "target", ...]
    }
}
```
**生物学意义**: WNT 信号通路 (发育、干细胞调控)

---

### 示例 3: MEBOCOST GP (酶-传感器)
```python
{
    "Glucose_metabolism_GP": {
        "sources": ["HK1", "GCK", "PFKL"],        # 糖酵解酶
        "targets": ["SLC2A1", "SLC2A4"],          # 葡萄糖转运体 (传感器)
        "sources_categories": ["enzyme", "enzyme", "enzyme"],
        "targets_categories": ["sensor", "sensor"]
    }
}
```
**生物学意义**: 葡萄糖代谢相关的细胞间通信

---

## 🎯 为什么要用 GP 掩码？

### 1. 使模型可解释
```
没有 GP 掩码: 潜在维度 1, 2, 3... → 无法解释
使用 GP 掩码: 潜在维度 1="血管生成", 2="WNT信号", 3="葡萄糖代谢" → 可解释！
```

### 2. 融入先验知识
- 利用已知的生物学发现
- 减少模型搜索空间
- 提高训练稳定性

### 3. 识别细胞生态位
- 不同的 GP 活性模式 → 不同的细胞生态位
- 例如: 高血管生成 GP 活性 = 血管周围生态位

---

## 🔧 函数参数说明

### `extract_gp_dict_from_omnipath_lr_interactions()`

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `species` | 物种 | "mouse" 或 "human" |
| `load_from_disk` | 是否加载缓存 | False (首次处理) |
| `save_to_disk` | 是否保存结果 | True (推荐) |
| `lr_network_file_path` | LR 网络文件路径 | ".../omnipath_lr_network.csv" |
| `gene_orthologs_mapping_file_path` | 基因同源映射文件 | ".../human_mouse_gene_orthologs.csv" |

### `filter_and_combine_gp_dict_gps_v2()`

| 参数 | 说明 |
|------|------|
| `gp_dicts` | GP 字典列表 |
| `verbose` | 是否打印处理详情 |
| `overlap_thresh` | 合并阈值 (默认 0.9 = 90% 重叠) |

---

## 📊 输出结果

### GP 数量变化
```
原始:    4150 个 GP
  ↓ 过滤
剩余:    3800 个 GP
  ↓ 合并
最终:    297 个 GP
```

### 最终 GP 字典结构
```python
combined_gp_dict = {
    "GP_NAME_1": {
        "sources": [...],              # 来源基因列表
        "targets": [...],              # 目标基因列表
        "sources_categories": [...],   # 来源基因类型
        "targets_categories": [...]    # 目标基因类型
    },
    "GP_NAME_2": { ... },
    ...
}
```

---

## ❓ 常见问题

### Q1: 为什么要用三个数据库？
**A**: 每个数据库侧重点不同：
- OmniPath: 经典的配体-受体相互作用
- NicheNet: 包含下游靶基因，更完整的通路
- MEBOCOST: 专注代谢相关通信
综合使用可以覆盖更多类型的细胞间通信

### Q2: GP 越多越好吗？
**A**: 不是。需要平衡：
- 太少: 可能遗漏重要的生物学过程
- 太多: 增加计算负担，可能引入噪声
- 经过过滤和合并后的 ~300 个 GP 是合理数量

### Q3: 可以自定义 GP 吗？
**A**: 可以！如果你研究特定的生物学过程，可以：
1. 创建自定义的 GP 字典
2. 添加到 `gp_dicts` 列表中
3. 一起参与过滤和合并

### Q4: 合并后的 GP 名称为什么叫 "combined"？
**A**: 因为它是由多个相似的 GP 合并而成：
- 例如: "VEGFA_VEGFR1_GP" + "VEGFA_VEGFR2_GP" → "VEGFA_combined_GP"
