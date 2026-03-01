# ============================================================================
# 2.1 创建先验知识基因程序 (Gene Program, GP) 掩码
# ============================================================================
#
# 【核心概念】什么是基因程序 (Gene Program)？
# ================================================
# 基因程序是一组功能相关的基因，它们在生物学过程中共同作用。
# 在 NicheCompass 中，GP 用于表示细胞间的通信和相互作用。
#
# 每个 GP 包含两部分：
#   - 来源基因 (Source Genes): 配体或酶基因，负责"发送"信号
#   - 目标基因 (Target Genes): 受体或传感器基因，负责"接收"信号
#
# 【为什么需要 GP 掩码？】
# ========================
# 1. 使潜在空间可解释：模型学习到的特征可以对应到具体的生物过程
# 2. 融入先验知识：利用已知的生物学数据库信息
# 3. 指导模型训练：通过掩码约束，帮助模型识别有意义的细胞生态位
#
# 【本节流程】
# ============
# 1. 从 OmniPath 数据库提取配体-受体 GP
# 2. 从 NicheNet 数据库提取相互作用 GP
# 3. 从 MEBOCOST 数据库提取酶-传感器 GP
# 4. 过滤并合并所有 GP
# ============================================================================



# ============================================================================
# 步骤 1: 从 OmniPath 数据库提取配体-受体基因程序
# ============================================================================
#
# 【OmniPath 数据库介绍】
# =====================
# OmniPath 是一个综合的信号传导通路数据库，主要包含：
#   - 配体-受体 (Ligand-Receptor, LR) 相互作用
#   - 蛋白质-蛋白质相互作用
#   - 酶-底物关系
#
# 【GP 结构】
# ===========
# 来源基因: 配体基因 (Ligand genes) - 分泌信号分子的基因
# 目标基因: 受体基因 (Receptor genes) - 接收信号的基因
#
# 示例: VEGFA gene (配体) -> VEGFR genes (受体)
#       这个 GP 代表血管生成相关的细胞间通信
# ============================================================================

# 调用函数提取 OmniPath 基因程序
omnipath_gp_dict = extract_gp_dict_from_omnipath_lr_interactions(
    # 物种类型 (mouse 或 human)
    species=species,

    # 是否从磁盘加载之前处理好的数据 (False = 重新处理)
    load_from_disk=False,

    # 是否保存处理结果到磁盘 (True = 保存，下次可以快速加载)
    save_to_disk=True,

    # OmniPath 配体-受体网络文件的路径
    # 这个 CSV 文件包含了所有已知的配体-受体相互作用
    lr_network_file_path=omnipath_lr_network_file_path,

    # 基因同源映射文件路径 (用于人-小鼠基因转换)
    # OmniPath 主要是人类数据，需要转换到小鼠
    gene_orthologs_mapping_file_path=gene_orthologs_mapping_file_path,

    # 是否绘制 GP 基因计数分布图 (用于可视化 GP 的基因数量分布)
    plot_gp_gene_count_distributions=True,

    # 保存分布图的路径
    gp_gene_count_distributions_save_path=f"{figure_folder_path}" \
                                           "/omnipath_gp_gene_count_distributions.svg"
)

# 输出示例:
# omnipath_gp_dict 是一个字典，格式如下:
# {
#   "VEGFA_VEGFR1_GP": {
#       "sources": ["VEGFA"],                    # 配体基因
#       "targets": ["FLT1", "KDR", "FLT4"],      # 受体基因
#       "sources_categories": ["ligand"],
#       "targets_categories": ["receptor", "receptor", "receptor"]
#   },
#   ...
# }

print(f"OmniPath 基因程序数量: {len(omnipath_gp_dict)}")
# 典型输出: OmniPath 基因程序数量: ~2000-3000


# ============================================================================
# 步骤 2: 查看一个 OmniPath GP 示例
# ============================================================================

# 随机选择一个 GP 来查看其结构
omnipath_gp_names = list(omnipath_gp_dict.keys())
random.shuffle(omnipath_gp_names)
omnipath_gp_name = omnipath_gp_names[0]

# 打印 GP 的详细信息
print(f"\n示例 OmniPath GP: {omnipath_gp_name}")
print(f"来源基因: {omnipath_gp_dict[omnipet_gp_name]['sources']}")
print(f"目标基因: {omnipath_gp_dict[omnipath_gp_name]['targets']}")

# 输出示例:
# A2M_combined_GP:
# {
#   'sources': ['A2M'],              # Alpha-2-macroglobulin (配体)
#   'targets': ['LRP1', 'CD91', ...], # 受体基因
#   'sources_categories': ['ligand'],
#   'targets_categories': ['receptor', 'receptor', ...]
# }


# ============================================================================
# 步骤 3: 从 NicheNet 数据库提取基因程序
# ============================================================================
#
# 【NicheNet 数据库介绍】
# ======================
# NicheNet 不仅包含配体-受体相互作用，还包含：
#   - 靶基因 (Target genes): 受配体调控影响的下游基因
#
# 【与 OmniPath 的区别】
# =====================
# OmniPath: 配体 -> 受体 (直接相互作用)
# NicheNet:  配体 -> 受体 + 靶基因 (包含下游调控效应)
#
# 示例: WNT 配体 -> FZD 受体 -> MYC, CCND1 等靶基因
#       这代表完整的信号转导通路
# ============================================================================

# 调用函数提取 NicheNet 基因程序
nichenet_gp_dict = extract_gp_dict_from_nichenet_lrt_interactions(
    species=species,

    # NicheNet 版本 (v2 是最新版本)
    version="v2",

    # 保留多少比例的顶级靶基因
    # 1.0 = 保留所有靶基因
    # 0.01 = 只保留前 1% 最相关的靶基因
    keep_target_genes_ratio=1.,

    # 每个 GP 最多保留多少个靶基因 (防止 GP 过大)
    max_n_target_genes_per_gp=250,

    load_from_disk=False,
    save_to_disk=True,

    # NicheNet 配体-受体网络文件
    lr_network_file_path=nichenet_lr_network_file_path,

    # NicheNet 配体-靶基因矩阵文件
    # 这个矩阵包含了每个配体对每个靶基因的调控强度
    ligand_target_matrix_file_path=nichenet_ligand_target_matrix_file_path,

    gene_orthologs_mapping_file_path=gene_orthologs_mapping_file_path,

    # 绘制 GP 基因计数分布图
    plot_gp_gene_count_distributions=True
)

# 输出示例:
# nichenet_gp_dict 格式:
# {
#   "WNT_combined_GP": {
#       "sources": ["WNT3A", "WNT5A", ...],        # WNT 配体家族
#       "targets": ["FZD1", "MYC", "CCND1", ...],   # 受体 + 靶基因
#       "sources_categories": ["ligand", "ligand", ...],
#       "targets_categories": ["receptor", "target", "target", ...]
#   },
#   ...
# }

print(f"\nNicheNet 基因程序数量: {len(nichenet_gp_dict)}")
# 典型输出: NicheNet 基因程序数量: ~1000-2000


# ============================================================================
# 步骤 4: 查看一个 NicheNet GP 示例
# ============================================================================

nichenet_gp_names = list(nichenet_gp_dict.keys())
random.shuffle(nichenet_gp_names)
nichenet_gp_name = nichenet_gp_names[0]

print(f"\n示例 NicheNet GP: {nichenet_gp_name}")
print(f"来源基因数量: {len(nichenet_gp_dict[nichenet_gp_name]['sources'])}")
print(f"目标基因数量: {len(nichenet_gp_dict[nichenet_gp_name]['targets'])}")


# ============================================================================
# 步骤 5: 从 MEBOCOST 数据库提取基因程序
# ============================================================================
#
# 【MEBOCOST 数据库介绍】
# ======================
# MEBOCOST 专注于代谢相关的细胞间通信：
#   - 酶 (Enzyme): 产生代谢物的基因
#   - 传感器 (Sensor): 检测代谢物的基因 (通常是受体)
#
# 【示例】
# ========
# 酶: 基因 A 产生代谢物 X
# 传感器: 基因 B 检测代谢物 X
# 这代表通过代谢物进行的细胞间通信
# ============================================================================

# 调用函数提取 MEBOCOST 基因程序
mebocost_gp_dict = extract_gp_dict_from_mebocost_ms_interactions(
    # MEBOCOST 数据文件夹路径
    # 这个文件夹包含 TSV 文件，每个文件描述一个代谢物相关的相互作用
    dir_path=mebocost_enzyme_sensor_interactions_folder_path,

    species=species,

    # 绘制 GP 基因计数分布图
    plot_gp_gene_count_distributions=True
)

# 输出示例:
# mebocost_gp_dict 格式:
# {
#   "Glucose_metabolism_GP": {
#       "sources": ["HK1", "GCK", ...],        # 糖酵解酶基因
#       "targets": ["GLUT1", "GLUT4", ...],    # 葡萄糖传感器/转运体
#       "sources_categories": ["enzyme", "enzyme", ...],
#       "targets_categories": ["sensor", "sensor", ...]
#   },
#   ...
# }

print(f"\nMEBOCOST 基因程序数量: {len(mebocost_gp_dict)}")
# 典型输出: MEBOCOST 基因程序数量: ~100-200


# ============================================================================
# 步骤 6: 查看一个 MEBOCOST GP 示例
# ============================================================================

mebocost_gp_names = list(mebocost_gp_dict.keys())
random.shuffle(mebocost_gp_names)
mebocost_gp_name = mebocost_gp_names[0]

print(f"\n示例 MEBOCOST GP: {mebocost_gp_name}")
print(f"详细信息: {mebocost_gp_dict[mebocost_gp_name]}")


# ============================================================================
# 步骤 7: 过滤和合并所有基因程序
# ============================================================================
#
# 【为什么要过滤和合并？】
# ======================
# 1. 过滤: 移除低质量的 GP (基因太少或数据不可靠)
# 2. 合并: 合并高度重叠的 GP (避免冗余)
#
# 【过滤规则】
# ===========
# - 最小基因数: 每个 GP 至少需要一定数量的基因
# - 数据质量: 基于数据库的置信度评分
#
# 【合并规则】
# ===========
# - 如果两个 GP 的来源基因和目标基因重叠度 > 90%，则合并为一个 GP
# - 合并后的 GP 名称通常为 "XXX_combined_GP"
# ============================================================================

# 将三个数据库的 GP 字典放入列表
gp_dicts = [
    omnipath_gp_dict,   # 配体-受体相互作用
    nichenet_gp_dict,   # 配体-受体-靶基因相互作用
    mebocost_gp_dict    # 酶-传感器相互作用
]

# 调用函数过滤和合并 GP
combined_gp_dict = filter_and_combine_gp_dict_gps_v2(
    gp_dicts,           # GP 字典列表
    verbose=True        # 打印处理过程的详细信息
)

# 输出示例:
# 处理过程:
#   过滤前: OmniPath 2500 + NicheNet 1500 + MEBOCOST 150 = 4150 个 GP
#   过滤: 移除基因数 < 2 的 GP
#   合并: 合并重叠度 > 90% 的 GP
#   最终: ~200-300 个 GP

print(f"\n过滤和合并后的基因程序数量: {len(combined_gp_dict)}")
# 典型输出: 过滤和合并后的基因程序数量: 297


# ============================================================================
# 【总结】最终得到的 combined_gp_dict
# ============================================================================
#
# combined_gp_dict 是一个字典，包含：
#
# 键 (Key): GP 名称，例如:
#   - "ALCAM_combined_GP" (细胞黏附相关)
#   - "CCK_combined_GP" (胆囊收缩素信号相关)
#   - "VEGFA_combined_GP" (血管生成相关)
#   - "Add-on_0_GP" (模型自动发现的新 GP)
#
# 值 (Value): GP 信息，包含:
#   - sources: 来源基因列表 (配体/酶)
#   - targets: 目标基因列表 (受体/传感器/靶基因)
#   - sources_categories: 来源基因类型
#   - targets_categories: 目标基因类型
#
# 【下一步】
# ==========
# 这个 GP 字典将在后续步骤中被添加到 AnnData 对象中，
# 作为模型训练的先验知识。
# ============================================================================
