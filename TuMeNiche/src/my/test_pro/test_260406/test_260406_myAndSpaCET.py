# 自动提取自 Jupyter Notebook
# 源文件：/home/zhangjunyi/xiangmu/nichecompass-main/TuMeNiche/src/my/test_260406/test_260406_myAndSpaCET.ipynb

使用SpaCET得出spot反卷积结果和肿瘤非肿瘤标签

# ===================== Cell 1 =====================
# ===================== 1. 导入必需库 =====================
import anndata as ad

# ===================== 2. 读取你的 h5ad 文件 =====================
# 你的文件完整路径（直接用，无需修改）
file_path = "/home/zhangjunyi/xiangmu/nichecompass-main/datasets/Human_breast_cancer/Human_breast_cancer_ViHBC/Human_breast_cancer_integrated.h5ad"

print("正在读取 h5ad 文件...")
adata = ad.read_h5ad(file_path)
print("文件读取成功！")
print("="*60)

# ===================== 3. 查看 基因名（核心！） =====================
print("【前 20 个基因名】：")
gene_names = adata.var.index.tolist()
print(gene_names[:20])
print("="*60)

# ===================== 4. 自动判断基因名格式 =====================
first_gene = gene_names[0]

if first_gene.startswith("ENSG"):
    print("❌ 检测结果：Ensembl ID 格式（例如 ENSG00000141510）")
    print("⚠️  警告：SpaCET 无法直接使用，需要转换为 Gene Symbol！")
else:
    print("✅ 检测结果：标准 Gene Symbol 格式（例如 TP53/ACTB）")
    print("🎉 完美！可以直接运行 SpaCET 代码，无需任何修改！")
    
import anndata as ad
import pandas as pd

# ==========================================
# 1. 设置文件路径
# ==========================================
# 你的原始 h5ad 文件路径
path_h5ad_original = "/home/zhangjunyi/xiangmu/nichecompass-main/datasets/Human_breast_cancer/Human_breast_cancer_ViHBC/Human_breast_cancer_integrated.h5ad"

# 你的 SpaCET 结果 CSV 文件路径
path_spacet_csv = "/home/zhangjunyi/xiangmu/nichecompass-main/datasets/Human_breast_cancer/Human_breast_cancer_ViHBC/SpaCET_CellFractions_Result.csv"

# 输出文件路径（建议不要覆盖原文件，生成一个新的）
path_h5ad_output = "/home/zhangjunyi/xiangmu/nichecompass-main/datasets/Human_breast_cancer/Human_breast_cancer_ViHBC/Human_breast_cancer_integrated_with_SpaCET.h5ad"

# ==========================================
# 2. 读取数据
# ==========================================
print("正在读取数据...")

# 读取原始 h5ad
adata = ad.read_h5ad(path_h5ad_original)

# 读取 SpaCET CSV
# 假设 CSV 的第一列是 Spot ID，我们将其设为 index
spacet_df = pd.read_csv(path_spacet_csv, index_col=0)

print(f"✅ 原始 h5ad 包含 {adata.n_obs} 个 Spots")
print(f"✅ SpaCET 结果包含 {spacet_df.shape[0]} 个 Spots")

# ==========================================
# 3. 【关键】检查并对齐 Spot ID
# ==========================================
# 这一步极其重要，防止因为 ID 不匹配导致数据错位
# 找出共同的 Spot ID
common_spots = adata.obs_names.intersection(spacet_df.index)

if len(common_spots) == 0:
    raise ValueError("❌ 错误：原始 h5ad 和 SpaCET 结果没有匹配的 Spot ID！请检查 CSV 的第一列是否为正确的 Spot 名称。")
elif len(common_spots) < adata.n_obs:
    print(f"⚠️  警告：有 {adata.n_obs - len(common_spots)} 个 Spots 在 SpaCET 结果中未找到（可能是在 SpaCET QC 步骤被过滤了）。")
    # 只保留共同的 Spots（可选，或者你选择保留所有，缺失值填 NA）
    # 这里我们选择只保留共同的，以保证数据完整性
    adata = adata[common_spots].copy()

# 重新排序 SpaCET 的 DataFrame，使其顺序与 h5ad 完全一致
spacet_df_aligned = spacet_df.reindex(adata.obs_names)

# ==========================================
# 4. 合并注入到 h5ad
# ==========================================
print("正在合并数据...")

# 将 SpaCET 的所有列合并到 adata.obs 中
# 为了防止列名冲突，可以给 SpaCET 的列加上前缀（可选，这里直接合并）
# spacet_df_aligned = spacet_df_aligned.add_prefix('SpaCET_') 

adata.obs = adata.obs.join(spacet_df_aligned)

print("✅ 合并完成！adata.obs 中新增了以下列：")
print(spacet_df_aligned.columns.tolist())

# ==========================================
# 5. 保存新的 h5ad 文件
# ==========================================
print(f"正在保存新文件至：{path_h5ad_output}")
adata.write_h5ad(path_h5ad_output)

print("🎉 全部完成！你现在可以在 Python 中加载新的 h5ad 文件进行后续分析了。")

# ===================== Cell 1-1 =====================
import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
import os  # 新增：用于创建文件夹/路径管理

# ==========================================
# 🔥 固定输出路径（你指定的文件夹）
# ==========================================
SAVE_DIR = "/home/zhangjunyi/xiangmu/nichecompass-main/outputs/Human_breast_cancer_ViHBC/zhibiao/SpaCET"
os.makedirs(SAVE_DIR, exist_ok=True)  # 自动创建文件夹，不存在则新建

# ==========================================
# 1. 读取数据
# ==========================================
file_path = "/home/zhangjunyi/xiangmu/nichecompass-main/datasets/Human_breast_cancer/Human_breast_cancer_ViHBC/Human_breast_cancer_integrated_with_SpaCET.h5ad"
print("正在加载数据...")
adata = ad.read_h5ad(file_path)

# ==========================================
# 1. 提取 SpaCET 的 Malignant 分数
# ==========================================
# 过滤掉 NaN 值（如果有背景点）
malignant_scores = adata.obs['Malignant'].dropna().values
# GMM 要求输入是 2D 数组 (n_samples, 1)
X = malignant_scores.reshape(-1, 1)

# ==========================================
# 2. 使用 GMM (高斯混合模型) 自动寻找最优阈值
# ==========================================
print("正在使用高斯混合模型 (GMM) 拟合数据分布...")
# 假设数据分为 2 类（肿瘤 vs 非肿瘤）
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm.fit(X)

# 找出两个分布的均值，区分哪个是非肿瘤(低均值)，哪个是肿瘤(高均值)
means = gmm.means_.flatten()
class_non_tumor_idx = np.argmin(means)
class_tumor_idx = np.argmax(means)

# 寻找两个正态分布交叉的边界点 (即最优阈值)
# 我们在两个均值之间生成 1000 个密集点，看哪一个点模型预测的分类发生了反转
x_test = np.linspace(means[class_non_tumor_idx], means[class_tumor_idx], 1000).reshape(-1, 1)
preds = gmm.predict(x_test)

# 找到分类反转的索引
flip_index = np.where(preds[:-1] != preds[1:])[0]

if len(flip_index) > 0:
    optimal_threshold = x_test[flip_index[0]][0]
else:
    # 极端情况下如果找不到交点，取两个均值的中间值
    optimal_threshold = np.mean(means)

print(f"✨ [无监督] 自动计算出的最优划分阈值为: {optimal_threshold:.4f}")

# ==========================================
# 3. 将新的划分应用到 adata 中
# ==========================================
col_name = f'Unsupervised_Pred_{optimal_threshold:.2f}'
adata.obs[col_name] = np.where(
    adata.obs['Malignant'] > optimal_threshold, 
    'Predicted_Tumor', 
    'Predicted_Non_Tumor'
)
adata.obs[col_name] = adata.obs[col_name].astype('category')

# ==========================================
# 4. 可视化：分布图 (解释为什么选这个阈值) + 空间图
# ==========================================
fig = plt.figure(figsize=(18, 5))

# -----------------
# 子图 1: 分数分布与 GMM 拟合曲线
# -----------------
ax1 = plt.subplot(1, 3, 1)
# 画直方图
sns.histplot(malignant_scores, bins=50, stat='density', alpha=0.5, color='gray', ax=ax1, label='Actual Data')

# 画 GMM 拟合出来的两个正态分布曲线
x_plot = np.linspace(0, 1, 1000).reshape(-1, 1)
logprob = gmm.score_samples(x_plot)
pdf = np.exp(logprob)
responsibilities = gmm.predict_proba(x_plot)

pdf_non_tumor = pdf * responsibilities[:, class_non_tumor_idx]
pdf_tumor = pdf * responsibilities[:, class_tumor_idx]

ax1.plot(x_plot, pdf_non_tumor, color='blue', label='Non-Tumor Distribution', lw=2)
ax1.plot(x_plot, pdf_tumor, color='red', label='Tumor Distribution', lw=2)

# 画阈值垂直线
ax1.axvline(optimal_threshold, color='green', linestyle='--', lw=2.5, 
            label=f'Optimal Threshold = {optimal_threshold:.2f}')

ax1.set_title('Malignant Fraction Distribution & GMM Fit')
ax1.set_xlabel('SpaCET Malignant Fraction')
ax1.set_ylabel('Density')
ax1.legend()

# -----------------
# 子图 2: SpaCET 连续分数的空间分布
# -----------------
ax2 = plt.subplot(1, 3, 2)
sc.pl.spatial(adata, color='Malignant', title='SpaCET Malignant Fraction', 
              spot_size=170, cmap='Reds', ax=ax2, show=False)

# -----------------
# 子图 3: 使用自动阈值划分后的离散空间分布
# -----------------
ax3 = plt.subplot(1, 3, 3)
sc.pl.spatial(adata, color=col_name, title=f'Unsupervised Classification\n(Threshold > {optimal_threshold:.2f})', 
              spot_size=170, ax=ax3, show=False)

plt.tight_layout()

# ==========================================
# 🔥 新增1：保存图片到指定路径（高清300dpi）
# ==========================================
img_save_path = os.path.join(SAVE_DIR, "GMM_Malignant_Threshold_Plot.png")
plt.savefig(img_save_path, dpi=300, bbox_inches='tight')
print(f"\n✅ 图片已保存至: {img_save_path}")

plt.show()

# ==========================================
# 🔥 新增2：保存肿瘤/非肿瘤划分标签为CSV
# ==========================================
label_save_path = os.path.join(SAVE_DIR, "GMM_Tumor_NonTumor_Pred_Labels.csv")
# 保存Spot ID + 划分标签 + 原始Malignant分数
label_df = adata.obs[[col_name, 'Malignant']].copy()
label_df.index.name = "Spot_ID"
label_df.to_csv(label_save_path, encoding='utf-8-sig')
print(f"✅ 划分标签已保存至: {label_save_path}")

# ==========================================
# 🔥 新增3：保存最优阈值结果为CSV
# ==========================================
threshold_save_path = os.path.join(SAVE_DIR, "GMM_Optimal_Threshold_Result.csv")
threshold_df = pd.DataFrame({
    "Metric": ["Optimal_GMM_Threshold"],
    "Value": [round(optimal_threshold, 4)]
})
threshold_df.to_csv(threshold_save_path, index=False, encoding='utf-8-sig')
print(f"✅ 阈值结果已保存至: {threshold_save_path}")

print("\n🎉 全部完成！所有文件已保存至指定文件夹")

# ===================== Cell 1-2 =====================
import anndata as ad
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

# ==========================================
# 1. 准备数据与真实标签映射 (Ground Truth)
# ==========================================
# 映射字典：Tumor/Invasive为1(肿瘤)，Healthy/Surrounding为0(非肿瘤)
gt_map_int = {
    'Tumor': 1,
    'Invasive': 1,
    'Surrounding tumor': 0,
    'Healthy': 0
}

# 为了可视化好看，我们新建一个字符串分类列
gt_map_str = {
    'Tumor': 'Tumor_Region',
    'Invasive': 'Tumor_Region',
    'Surrounding tumor': 'Non_Tumor_Region',
    'Healthy': 'Non_Tumor_Region'
}

adata.obs['Ground_Truth_Binary'] = adata.obs['annot_type'].map(gt_map_str).astype('category')
gt_labels_for_math = adata.obs['annot_type'].map(gt_map_int).values

# ==========================================
# 2. 生成预测标签 (基于 0.4 阈值)
# ==========================================
threshold = 0.5282

# 生成预测的字符串标签（用于画图）
adata.obs[f'Pred_Threshold_{threshold}'] = np.where(
    adata.obs['Malignant'] > threshold, 
    'Predicted_Tumor', 
    'Predicted_Non_Tumor'
)
adata.obs[f'Pred_Threshold_{threshold}'] = adata.obs[f'Pred_Threshold_{threshold}'].astype('category')

# 生成预测的整数标签（用于计算 ARI）
pred_labels_for_math = (adata.obs['Malignant'] > threshold).astype(int).values

# ==========================================
# 3. 空间可视化 (核对 0.25 切得合不合理)
# ==========================================
print("\n===== 正在绘制空间分布对比图 =====")
try:
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # 图1：病理学家的金标准（二分类）
    sc.pl.spatial(adata, color='Ground_Truth_Binary', title='Ground Truth (Pathologist)', spot_size=170, ax=axs[0], show=False)
    
    # 图2：SpaCET 推断的恶性细胞连续比例 (热力图)
    sc.pl.spatial(adata, color='Malignant', title='SpaCET Malignant Fraction (Continuous)', spot_size=170, cmap='Reds', ax=axs[1], show=False)
    
    # 图3：0.25 阈值切割后的预测结果
    sc.pl.spatial(adata, color=f'Pred_Threshold_{threshold}', title=f'Predicted (Threshold > {threshold})', spot_size=170, ax=axs[2], show=False)
    
    plt.tight_layout()
    plt.show()
except Exception as e:
    print(f"可视化失败: {e}")

# ==========================================
# 4. 计算 ARI 指标
# ==========================================
# 过滤掉注释为 NaN 的点（如组织切片外围的背景点）
valid_mask = ~np.isnan(gt_labels_for_math)
gt_valid = gt_labels_for_math[valid_mask]
pred_valid = pred_labels_for_math[valid_mask]

# 计算 ARI
ari = adjusted_rand_score(gt_valid, pred_valid)

print("\n===== 评估指标 =====")
print(f"使用的切割阈值: Malignant > {threshold}")
print(f"有效参与计算的 Spot 数量: {len(gt_valid)}")
print(f"✅ 修正后的宏观肿瘤分割 ARI = {ari:.4f}")

# ==========================================
# [进阶选项]：寻找使 ARI 最大的最优阈值
# ==========================================
# 如果你想知道到底切多少能让 ARI 最高，可以跑下面这段
best_ari = -1
best_t = 0
for t in np.arange(0.1, 0.9, 0.05):
    temp_pred = (adata.obs['Malignant'] > t).astype(int).values[valid_mask]
    temp_ari = adjusted_rand_score(gt_valid, temp_pred)
    if temp_ari > best_ari:
        best_ari = temp_ari
        best_t = t
print(f"💡 [探索] 对于该数据集，使 ARI 最高的阈值其实是 {best_t:.2f} (此时最高 ARI = {best_ari:.4f})")

# ===================== Cell 2 =====================
# =========================================================
# Cell 1: 环境配置与模型导入
# =========================================================
import os
import sys
import warnings

# 1. 强制只使用 GPU 0 (必须在 import torch 之前)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
warnings.filterwarnings("ignore")

import torch
import scanpy as sc
import squidpy as sq
import pandas as pd
import numpy as np
import scipy.sparse as sp
import decoupler as dc
from io import StringIO
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# 2. 导入原版 NicheCompass
LOCAL_SRC = "/home/zhangjunyi/xiangmu/nichecompass-main/src"
if LOCAL_SRC not in sys.path:
    sys.path.insert(0, LOCAL_SRC)
import nichecompass as nc

print("✅ CUDA 可用状态:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("✅ 当前使用 GPU:", torch.cuda.get_device_name(0))
print("✅ NicheCompass 路径:", nc.__file__)

# =========================================================
# Cell 3: 知识库提取与数据加载 (Step 1)
# =========================================================
print("=== Step 1: 构建代谢通讯轴四元组知识库 ===")

# 1. 解析你的四元组知识库
tmcn_csv = """TMCN_Name,Source_Pathways,Source_Genes,Target_Genes,Biologic_Meaning
TMCN_Lactate_Axis,"Hypoxia,EGFR,PI3K,MAPK","SLC2A1,HK2,PKM,LDHA,LDHB,SLC16A3","SLC16A1,SLC16A7,HCAR1,HCAR2",乳酸_肿瘤酸化瓦伯格效应与基质反向代谢共生
TMCN_Adenosine_Axis,"Hypoxia,TGFb,NFkB,MAPK","ENTPD1,ENTPD2,NT5E,CD38,ENPP1,NT5C2","ADORA1,ADORA2A,ADORA2B,ADORA3,SLC29A1,SLC29A2",腺苷_ATP水解级联驱动的强效免疫抑制与M2极化
TMCN_PGE2_Axis,"NFkB,JAK-STAT,Hypoxia,MAPK,TNFa","PLA2G4A,PTGS2,PTGES,ABCC4,SLCO2A1","PTGER1,PTGER2,PTGER3,PTGER4",前列腺素E2_成纤维细胞激活与促癌炎症微环境重塑
TMCN_Glutamine_Axis,"PI3K,MAPK,JAK-STAT,TGFb,WNT","GLUL,SLC38A1,SLC38A3,SLC38A5","SLC1A5,SLC7A5,SLC38A2,GLS,GLUD1",谷氨酰胺_基质向肿瘤供能的代谢寄生与大分子合成
TMCN_Succinate_Axis,"Hypoxia,NFkB,TNFa","SLC13A2,SLC13A3,SLC25A10","SUCNR1",琥珀酸_缺血坏死区释放驱动的TAM致瘤极化与血管生成
TMCN_Kynurenine_Axis,"JAK-STAT,NFkB,TGFb","IDO1,TDO2,KYNU,SLC7A5","AHR",犬尿氨酸_色氨酸剥夺与AHR介导的效应T细胞耗竭
TMCN_ATP_Axis,"Hypoxia,p53,TNFa","PANX1,SLC17A9","P2RX7,P2RY2,P2RY11",胞外ATP_坏死边缘释放的促炎性危险信号(DAMP)传导
TMCN_S1P_Axis,"NFkB,PI3K,MAPK","SPHK1,SPHK2,SPNS2","S1PR1,S1PR2,S1PR3",鞘氨醇-1-磷酸_脂质信号驱动的内皮血管生成与免疫趋化
TMCN_LPA_Axis,"PI3K,MAPK,TGFb","ENPP2,PLA2G4A,LPCAT1","LPAR1,LPAR2,LPAR3,LPAR5",溶血磷脂酸_成纤维细胞基质重塑与肿瘤高侵袭性
TMCN_Glutamate_Axis,"PI3K,MAPK,Hypoxia","GLS,SLC1A5,SLC7A11","GRM3,GRM5,GRIN1",谷氨酸_突触样代谢通讯与微环境神经可塑性
"""
axis_table = pd.read_csv(StringIO(tmcn_csv))
AXES =[x.replace("TMCN_", "").replace("_Axis", "") for x in axis_table["TMCN_Name"]]

# 辅助函数：解析基因列表
def split_items(x):
    return[i.strip() for i in str(x).split(",") if i.strip()]

def get_valid_genes(adata, genes):
    return[g for g in genes if g in adata.var_names]

# 2. 加载带有 SpaCET 结果的空间转录组数据（必须读取 _with_SpaCET.h5ad）
file_path = "/home/zhangjunyi/xiangmu/nichecompass-main/datasets/Human_breast_cancer/Human_breast_cancer_ViHBC/Human_breast_cancer_integrated_with_SpaCET.h5ad"
adata = sc.read_h5ad(file_path)

print(f"✅ 数据加载成功: {adata.n_obs} spots, {adata.n_vars} genes")
display(axis_table)

# =========================================================
# Cell 4: 连续功能分数推断与解耦 (Step 2)
# =========================================================
print("=== Step 2: 计算 Spot 级连续分数 (绝对无监督解耦) ===")

# 1. 运行 PROGENy 提取 Pathway_score
print("--> 正在计算 Pathway_score (PROGENy)...")
net_progeny = dc.get_progeny(organism="human", top=500)
dc.run_mlm(mat=adata, net=net_progeny, source="source", target="target", weight="weight", verbose=False, use_raw=False)
df_pathway = dc.get_acts(adata, obsm_key="mlm_estimate").to_df()
df_pathway.columns =[c.strip().replace("-", "_").replace(" ", "_") for c in df_pathway.columns]

# 2. 运行 AUCell 提取 Enzyme_score
print("--> 正在计算 Enzyme_score (AUCell)...")
enzyme_records =[]
for _, row in axis_table.iterrows():
    ax = row["TMCN_Name"].replace("TMCN_", "").replace("_Axis", "")
    for g in split_items(row["Source_Genes"]):
        if g in adata.var_names:
            enzyme_records.append({"source": ax, "target": g})
dc.run_aucell(mat=adata, net=pd.DataFrame(enzyme_records), source="source", target="target", min_n=1, verbose=False, use_raw=False)
df_enzyme = dc.get_acts(adata, obsm_key="aucell_estimate").to_df()

# 3. 对通路和酶进行 MinMax 归一化 (0~1)
scaler = MinMaxScaler()
df_pathway_scaled = pd.DataFrame(scaler.fit_transform(df_pathway), index=df_pathway.index, columns=df_pathway.columns)
df_enzyme_scaled = pd.DataFrame(scaler.fit_transform(df_enzyme), index=df_enzyme.index, columns=df_enzyme.columns)

# 4. 提取 Receptor_score 并计算 Sender & Receiver 最终分数
print("--> 正在计算 Sender_score 与 Receiver_score...")
def calc_receptor_score(genes):
    valid = get_valid_genes(adata, genes)
    if not valid: return np.zeros(adata.n_obs)
    X_sub = adata[:, valid].X
    mean_exp = np.asarray(X_sub.mean(axis=1)).reshape(-1) if sp.issparse(X_sub) else np.asarray(X_sub).mean(axis=1).reshape(-1)
    return scaler.fit_transform(mean_exp.reshape(-1, 1)).flatten()

for _, row in axis_table.iterrows():
    ax = row["TMCN_Name"].replace("TMCN_", "").replace("_Axis", "")
    pathways =[p.strip().replace("-", "_").replace(" ", "_") for p in split_items(row["Source_Pathways"])]
    target_genes = split_items(row["Target_Genes"])
    
    # 提取存在的通路
    valid_paths =[p for p in pathways if p in df_pathway_scaled.columns]
    
    # Sender = Pathway (均值) * Enzyme
    p_score = df_pathway_scaled[valid_paths].mean(axis=1).values
    e_score = df_enzyme_scaled[ax].values
    adata.obs[f"{ax}_Sender_Score"] = p_score * e_score
    
    # Receiver = Receptor 表达量归一化
    adata.obs[f"{ax}_Receiver_Score"] = calc_receptor_score(target_genes)

print("✅ 所有连续分数提取完毕！保存在 adata.obs 中。")
# 预览前5行打分
score_cols =[c for c in adata.obs.columns if "Sender_Score" in c or "Receiver_Score" in c]
display(adata.obs[score_cols].head())

# # ===================== Cell 5 =====================
# # =========================================================
# # Step3: NicheCompass 无监督拓扑学习与过聚类（作者教程参数对齐版，NaN稳健）
# # 可直接在单元格运行
# # =========================================================
# import scipy.sparse as sp
# import numpy as np
# import pandas as pd
# import scanpy as sc
# import squidpy as sq
# import matplotlib.pyplot as plt

# print("=== Step 3: NicheCompass 构建物理联通图与微型生态位骨架（教程参数对齐版）===")

# # 1) 基础过滤 + 矩阵清洗
# sc.pp.filter_genes(adata, min_cells=10)
# adata_model = adata.copy()

# if "counts" in adata_model.layers:
#     mat = adata_model.layers["counts"]
#     if sp.issparse(mat):
#         mat = mat.copy()
#         mat.data = np.nan_to_num(mat.data, nan=0.0, posinf=0.0, neginf=0.0)
#         mat.data = np.clip(mat.data, a_min=0, a_max=None)
#         adata_model.layers["counts"] = mat
#     else:
#         mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
#         adata_model.layers["counts"] = np.clip(mat, a_min=0, a_max=None)

# # 2) 不再筛选2000个高变基因（已移除），使用所有通过 min_cells=10 过滤的基因。

# # 2.1) 关键修复：去掉总计数为0的spot，避免log(0) -> inf -> NaN
# counts_key = "counts" if "counts" in adata_model.layers else None
# x_for_lib = adata_model.layers[counts_key] if counts_key is not None else adata_model.X
# lib_size = np.asarray(x_for_lib.sum(axis=1)).reshape(-1)
# valid_spot_mask = np.isfinite(lib_size) & (lib_size > 0)
# adata_model = adata_model[valid_spot_mask].copy()
# print(f"✅ 保留 {adata_model.n_obs} 个有效spots（已移除总计数为0/非有限spots）")

# # 再清洗一次，确保无NaN/Inf
# if counts_key is not None:
#     mat = adata_model.layers[counts_key]
#     if sp.issparse(mat):
#         mat = mat.copy()
#         mat.data = np.nan_to_num(mat.data, nan=0.0, posinf=0.0, neginf=0.0)
#         mat.data = np.clip(mat.data, a_min=0, a_max=None)
#         adata_model.layers[counts_key] = mat
#     else:
#         mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
#         adata_model.layers[counts_key] = np.clip(mat, a_min=0, a_max=None)

# # 3) 空间图 + LR先验
# sq.gr.spatial_neighbors(adata_model, coord_type="generic", spatial_key="spatial", n_neighs=8)
# cache_dir = "/home/zhangjunyi/xiangmu/nichecompass-main/cache_nichenet"
# gp_dict = nc.utils.extract_gp_dict_from_nichenet_lrt_interactions(
#     species="human", version="v2",
#     keep_target_genes_ratio=0.25, max_n_target_genes_per_gp=50,
#     load_from_disk=True,
#     lr_network_file_path=f"{cache_dir}/nichenet_lr_network.csv",
#     ligand_target_matrix_file_path=f"{cache_dir}/nichenet_ligand_target_matrix.csv"
# )
# nc.utils.add_gps_from_gp_dict_to_adata(gp_dict=gp_dict, adata=adata_model)

# # 4) 参照官方教程的 key/参数
# adj_key = "spatial_connectivities"
# gp_names_key = "nichecompass_gp_names"
# active_gp_names_key = "nichecompass_active_gp_names"
# gp_targets_mask_key = "nichecompass_gp_targets"
# gp_targets_categories_mask_key = "nichecompass_gp_targets_categories"
# gp_sources_mask_key = "nichecompass_gp_sources"
# gp_sources_categories_mask_key = "nichecompass_gp_sources_categories"
# latent_key = "nichecompass_latent"
# conv_layer_encoder = "gcnconv"
# active_gp_thresh_ratio = 0.01

# # 训练参数（稳健版：降低NaN风险）
# n_epochs = 50
# n_epochs_all_gps = 10
# lr = 1e-4
# lambda_edge_recon = 1e5
# lambda_gene_expr_recon = 100.0
# lambda_l1_masked = 0.0
# edge_batch_size = 64
# node_batch_size = 128
# n_sampled_neighbors = 4

# # 5) Initialize + Train
# model = nc.models.NicheCompass(
#     adata_model,
#     counts_key=counts_key,
#     adj_key=adj_key,
#     gp_names_key=gp_names_key,
#     active_gp_names_key=active_gp_names_key,
#     gp_targets_mask_key=gp_targets_mask_key,
#     gp_targets_categories_mask_key=gp_targets_categories_mask_key,
#     gp_sources_mask_key=gp_sources_mask_key,
#     gp_sources_categories_mask_key=gp_sources_categories_mask_key,
#     latent_key=latent_key,
#     conv_layer_encoder=conv_layer_encoder,
#     active_gp_thresh_ratio=active_gp_thresh_ratio,
# )

# model.train(
#     n_epochs=n_epochs,
#     n_epochs_all_gps=n_epochs_all_gps,
#     lr=lr,
#     lambda_edge_recon=lambda_edge_recon,
#     lambda_gene_expr_recon=lambda_gene_expr_recon,
#     lambda_l1_masked=lambda_l1_masked,
#     edge_batch_size=edge_batch_size,
#     node_batch_size=node_batch_size,
#     n_sampled_neighbors=n_sampled_neighbors,
#     edge_val_ratio=0.0,   # 避免验证阶段NaN指标中断训练
#     node_val_ratio=0.0,
#     use_cuda_if_available=True,
#     verbose=False,
# )

# # =========================================================
# # 保存结果 (AnnData + 模型)
# # =========================================================
# import os

# # -------------------------- 1. 设置保存路径 --------------------------
# save_base_dir = "/home/zhangjunyi/xiangmu/nichecompass-main/TuMeNiche/src/my/test_260325"
# file_prefix = "test_260325_02"

# # 确保目录存在，不存在则自动创建
# os.makedirs(save_base_dir, exist_ok=True)

# # -------------------------- 2. 保存 AnnData 对象 --------------------------
# # 包含了原始数据、空间坐标、训练好的 latent embedding (nichecompass_latent) 等
# adata_save_path = os.path.join(save_base_dir, f"{file_prefix}.h5ad")
# adata_model.write_h5ad(adata_save_path)
# print(f"✅ [1/2] AnnData 结果已保存至: {adata_save_path}")
# # -------------------------- 3. 保存 NicheCompass 模型 --------------------------
# # 保存模型权重和参数，方便后续直接 load 使用
# model_save_dir = os.path.join(save_base_dir, f"{file_prefix}_model")
# model.save(model_save_dir)
# print(f"✅ [2/2] NicheCompass 模型已保存至: {model_save_dir}")

# print("\n=== Step 3 & 4 全部完成 ===")

# ===================== Cell 6 =====================
# =========================================================
# 前置步骤：加载 Step3 保存的结果 (AnnData + 模型)
# =========================================================
import os
import scanpy as sc
# 假设 nichecompass 已正确导入为 nc
import nichecompass as nc 

# -------------------------- 1. 定义加载路径 (与保存路径完全一致) --------------------------
save_base_dir = "/home/zhangjunyi/xiangmu/nichecompass-main/TuMeNiche/src/my/test_260325"
file_prefix = "test_260325_02"

# -------------------------- 2. 加载 AnnData 对象 --------------------------
adata_load_path = os.path.join(save_base_dir, f"{file_prefix}.h5ad")
adata_model = sc.read_h5ad(adata_load_path)
print(f"✅ [1/2] AnnData 已从 {adata_load_path} 加载")

# -------------------------- 3. 加载 NicheCompass 模型 --------------------------
model_load_dir = os.path.join(save_base_dir, f"{file_prefix}_model")
# 注意：load 函数需要传入对应的 adata_model 以恢复图结构等信息
model = nc.models.NicheCompass.load(
    dir_path=model_load_dir,
    adata=adata_model
)
print(f"✅ [2/2] NicheCompass 模型已从 {model_load_dir} 加载")

# -------------------------- 4. 恢复/定义关键参数变量 (供后续代码使用) --------------------------
# 这些变量名必须与你训练时的代码保持一致
latent_key = "nichecompass_latent"

# ✅ 修复：动态检查 counts_key，不要硬编码
if "counts" in adata_model.layers:
    counts_key = "counts"
else:
    counts_key = None  # 如果 layers 里没有 counts，就设为 None (使用 .X)
    print("⚠️ 警告: 在 layers 中未找到 'counts'，将默认使用 adata_model.X")

adj_key = "spatial_connectivities"
node_batch_size = 128  # 与训练时保持一致

print("✅ 关键参数变量已恢复，可继续运行后续代码")



# =========================================================
# Step3: 训练后仅查看 NicheCompass 原生聚类结果（无二次划分）
# 直接接在 model.train(...) 后运行
# =========================================================

# 1) 提取 latent（与训练保持同一 counts_key / adj_key）
adata_model.obsm[latent_key] = model.get_latent_representation(
    adata=adata_model,
    counts_key=counts_key,
    adj_key=adj_key,
    node_batch_size=node_batch_size,
)

# 2) 官方教程风格：latent 上建图 + UMAP
sc.pp.neighbors(adata_model, use_rep=latent_key, key_added=latent_key)
sc.tl.umap(adata_model, neighbors_key=latent_key)

# 3) 原生聚类（不过度切碎）
native_cluster_key = "latent_leiden_0.4"
sc.tl.leiden(
    adata_model,
    neighbors_key=latent_key,
    key_added=native_cluster_key,
    resolution=0.4
)

print(f"✅ 原生聚类完成：{adata_model.obs[native_cluster_key].nunique()} 类")

# 4) 仅可视化原生聚类（空间 + UMAP）
sc.pl.spatial(
    adata_model,
    color=native_cluster_key,
    spot_size=170,
    title="Native clusters (Leiden 0.4)",
    frameon=False
)
sc.pl.umap(
    adata_model,
    color=native_cluster_key,
    title="Native clusters on NicheCompass latent"
)



# =========================================================
# Cell 7: 空间通讯加权分数计算与生态位标注 (阶段二)
# =========================================================
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.mixture import GaussianMixture
from IPython.display import display

print("=== 阶段二: 数据驱动的自适应阈值与代谢生态位初筛 ===")

# ---------------------------------------------------------
# Step 4.1: 构建空间感知的连续通讯分数 (Coupling Score)
# ---------------------------------------------------------
print("--> 1. 计算空间加权代谢通讯分数 (Coupling Score)...")
# 获取基于 squidpy 生成的空间物理邻接矩阵
adj_matrix = adata_model.obsp["spatial_connectivities"]

for k in AXES:
    sender_scores = adata_model.obs[f"{k}_Sender_Score"].values
    receiver_scores = adata_model.obs[f"{k}_Receiver_Score"].values
    
    # 核心公式: 自身 Sender * 周围所有物理邻居的 Receiver 之和
    neighbor_receiver_sum = adj_matrix.dot(receiver_scores)
    raw_coupling = sender_scores * neighbor_receiver_sum
    
    # 为了后续更好拟合高斯分布，对存在严重长尾的连乘分数进行 log1p 转换
    coupling_score = np.log1p(raw_coupling)
    adata_model.obs[f"Coupling_{k}"] = coupling_score

# ---------------------------------------------------------
# Step 4.2: 自适应混合阈值算法 (GMM + Quantile 兜底)
# ---------------------------------------------------------
print("--> 2. 学习每条代谢轴的自适应阈值 (GMM / Quantile)...")
thresholds_dict = {}

# 准备画图，将阈值切分可视化 (非常适合放入论文 Supplementary)
fig, axes = plt.subplots(len(AXES), 1, figsize=(8, 3.5 * len(AXES)))
if len(AXES) == 1: axes =[axes]

for idx, k in enumerate(AXES):
    scores = adata_model.obs[f"Coupling_{k}"].values
    
    # 提取非零有效值用于拟合 (去除纯背景底噪点的影响)
    scores_clean = scores[~np.isnan(scores) & (scores > 0)].reshape(-1, 1)
    
    t_high, t_quiet = None, None
    method_used = "Quantile Fallback"
    
    # -- 优先尝试 2-Component GMM --
    if len(scores_clean) > 50:  # 只有当非零点足够多才跑 GMM
        try:
            gmm = GaussianMixture(n_components=2, random_state=42)
            labels = gmm.fit_predict(scores_clean)
            
            m0, m1 = gmm.means_[0][0], gmm.means_[1][0]
            s0, s1 = np.sqrt(gmm.covariances_[0][0][0]), np.sqrt(gmm.covariances_[1][0][0])
            
            # 校验: 两个峰要有显著区分度 (均值差 > 1倍全局标准差)
            if abs(m0 - m1) > np.std(scores_clean):
                if m0 > m1:
                    t_high = np.min(scores_clean[labels == 0]) # 信号峰最小值
                    t_quiet = m1 + s1                          # 背景峰均值+1SD
                else:
                    t_high = np.min(scores_clean[labels == 1])
                    t_quiet = m0 + s0
                method_used = "GMM (2-Components)"
        except Exception as e:
            pass # 如果GMM不收敛或抛错，平滑过渡到底部分位数策略
            
    # -- 兜底策略: 分位数法 (q85 & q35) --
    if t_high is None:
        t_high = np.quantile(scores, 0.85)
        t_quiet = np.quantile(scores, 0.35)
    
    # 硬基线安全锁: 如果阈值算出来太低，说明全组织都没活化
    if t_high < 0.05:
        t_high = np.inf
        
    thresholds_dict[k] = {'T_high': t_high, 'T_quiet': t_quiet, 'Method': method_used}
    
    # 将 Spot 状态记录回 obs
    adata_model.obs[f"Active_{k}"] = (scores >= t_high).astype(int)
    adata_model.obs[f"Quiet_{k}"] = (scores <= t_quiet).astype(int)
    
    # 画密度分布及阈值线
    sns.kdeplot(scores, ax=axes[idx], fill=True, color="#4CB391", alpha=0.5, bw_adjust=0.5)
    axes[idx].axvline(t_high, color='red', linestyle='--', label=f'T_high (Active): {t_high:.3f}')
    axes[idx].axvline(t_quiet, color='blue', linestyle='--', label=f'T_quiet (Quiet): {t_quiet:.3f}')
    axes[idx].set_title(f"Axis: {k} | Score Distribution | Method: {method_used}")
    axes[idx].set_xlabel("Log1p(Coupling Score)")
    axes[idx].legend()

plt.tight_layout()
plt.show()

# 打印阈值结果表
df_thresholds = pd.DataFrame(thresholds_dict).T
print("\n--- 自适应阈值计算结果 ---")
display(df_thresholds)

# ---------------------------------------------------------
# Step 5: 微型生态位 (Micro-clusters) 统计判定与初标注 (基于超几何富集检验)
# ---------------------------------------------------------
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

print("\n--> 3. 统计微生态位特征，执行纯数据驱动的超几何富集检验...")

# ==========================================
# 1. 计算严格的组织级全局背景 (Global Background)
# 注：adata_model 在 Step3 已剔除 counts=0 的无组织区点，
# 因此 n_obs 即为真实的 Tissue-covered spots 总数。
# ==========================================
M_global = adata_model.n_obs 
bg_active_counts = {k: adata_model.obs[f"Active_{k}"].sum() for k in AXES}
bg_quiet_counts = {k: adata_model.obs[f"Quiet_{k}"].sum() for k in AXES}

bg_active_fracs = {k: bg_active_counts[k] / M_global for k in AXES}
bg_quiet_fracs = {k: bg_quiet_counts[k] / M_global for k in AXES}

# ==========================================
# 2. 收集每个 Cluster 的观察值并计算 P-value
# ==========================================
native_clusters = sorted(adata_model.obs[native_cluster_key].unique())

# 初始化字典用于存储每条轴的 p-value（用于后续 FDR 校正）
pvals_active_dict = {k: [] for k in AXES}
pvals_quiet_dict = {k: [] for k in AXES}

# 初始化 adata.obs 中的保留列
for k in AXES:
    adata_model.obs[f"{k}_Cluster_Active_Frac"] = 0.0

for c in native_clusters:
    c_mask = adata_model.obs[native_cluster_key] == c
    N_cluster = c_mask.sum()
    
    for k in AXES:
        # 统计该 cluster 在本条轴上的实际活跃/休眠点数
        k_obs_act = adata_model.obs.loc[c_mask, f"Active_{k}"].sum()
        k_obs_qui = adata_model.obs.loc[c_mask, f"Quiet_{k}"].sum()
        
        # 实时保存比例到 adata.obs
        adata_model.obs.loc[c_mask, f"{k}_Cluster_Active_Frac"] = k_obs_act / N_cluster
        
        # --- 超几何检验 (Hypergeometric Test) ---
        # sf(k-1, M, n, N) 
        # M: 全局总Spot数 | n: 全局靶向Spot总数 | N: Cluster的总Spot数 | k: Cluster内的靶向Spot数
        pval_a = hypergeom.sf(k_obs_act - 1, M_global, bg_active_counts[k], N_cluster) if k_obs_act > 0 else 1.0
        pval_q = hypergeom.sf(k_obs_qui - 1, M_global, bg_quiet_counts[k], N_cluster) if k_obs_qui > 0 else 1.0
        
        pvals_active_dict[k].append(pval_a)
        pvals_quiet_dict[k].append(pval_q)

# ==========================================
# 3. 多重假设检验校正 (FDR - Benjamini-Hochberg)
# ==========================================
fdr_active_dict = {}
fdr_quiet_dict = {}
for k in AXES:
    _, fdr_a, _, _ = multipletests(pvals_active_dict[k], method='fdr_bh')
    _, fdr_q, _, _ = multipletests(pvals_quiet_dict[k], method='fdr_bh')
    fdr_active_dict[k] = fdr_a
    fdr_quiet_dict[k] = fdr_q

# ==========================================
# 4. 执行基于统计显著性的生态位判定逻辑 (修复背景悖论版)
# ==========================================
cluster_annotations = {}
cluster_stats = []

for idx, c in enumerate(native_clusters):
    c_mask = adata_model.obs[native_cluster_key] == c
    N_cluster = c_mask.sum()
    
    sig_active_axes = []
    cold_axes = []  # 🔥 新增：用于记录“冷轴/耗竭轴”
    
    stat_row = {'Cluster_ID': c, 'Spot_Count': N_cluster}
    
    for k in AXES:
        # 获取基础分数
        act_frac = adata_model.obs.loc[c_mask, f"{k}_Cluster_Active_Frac"].iloc[0]
        qui_frac = adata_model.obs.loc[c_mask, f"Quiet_{k}"].sum() / N_cluster
        
        # 计算富集倍数 (Fold Change)
        fc_act = act_frac / (bg_active_fracs[k] + 1e-9)
        
        # 获取 FDR 值
        fdr_a = fdr_active_dict[k][idx]
        
        # 1. 【活跃判定】：严格的超几何检验富集 (FC > 1.5 且显著)
        if fc_act > 2 and fdr_a < 0.05:
            sig_active_axes.append(k)
            
        # 2. 【休眠/耗竭判定】：不需要显著性检验，只要它比全局背景更“冷”
        # 条件：休眠点比例高于全局平均，或者活跃点被严重耗竭(不足全局的一半)
        if qui_frac > bg_quiet_fracs[k] or fc_act < 0.5:
            cold_axes.append(k)
            
        # 记录用于展示
        stat_row[f"{k}_FC_Act"] = round(fc_act, 2)
        stat_row[f"{k}_FDR_Act"] = f"{fdr_a:.2e}"

    # --- 严格的互斥判定逻辑 ---
    if len(sig_active_axes) == 1:
        c_label = f"Single-axis: {sig_active_axes[0]}"
    elif len(sig_active_axes) >= 2:
        c_label = f"Multi-axis: {'_'.join(sig_active_axes)}"
    elif len(sig_active_axes) == 0 and len(cold_axes) >= 6: 
        # 🔥 修复：没有任何显著活跃轴，且大部分轴(≥6条)处于“冷/耗竭”状态 -> 回归休眠区！
        c_label = "Quiescent (Background)"
    else:
        # 既没有建立起显著的活跃通讯，也没有冷到休眠，属于真正的混乱交界区
        c_label = "Transitional"
        
    cluster_annotations[c] = c_label
    stat_row['Initial_Label'] = c_label
    cluster_stats.append(stat_row)

# 将判定映射回 adata_model
adata_model.obs["Niche_Annotation"] = adata_model.obs[native_cluster_key].map(cluster_annotations)

df_cluster_stats = pd.DataFrame(cluster_stats).set_index('Cluster_ID')
print("\n--- 基于统计显著性的生态位富集初筛表 (已修正背景悖论) ---")
display(df_cluster_stats)




# ---------------------------------------------------------
# 可视化: 初判后的空间生态位地图
# ---------------------------------------------------------
print("\n--> 4. 绘制初步判定的微环境空间地图...")

# ==============================
# 1. 扩充为 20 种高对比度离散颜色 (Tab20 经典色板)
# 足够应对 10 轴带来的十几种生态位类别
# ==============================
color_list =[
    "#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", 
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#ff9896", "#aec7e8", "#98df8a", "#ffbb78", "#c5b0d5", 
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"
]

# ==============================
# 2. 获取数据里的实际类别，并按顺序分配颜色
# ==============================
all_categories = adata_model.obs["Niche_Annotation"].unique().tolist()

# 强制将 Quiescent 和 Transitional 固定为不显眼的底色，让活跃轴更突出
custom_palette = {}
color_idx = 0

for cat in all_categories:
    if "Quiescent" in cat:
        custom_palette[cat] = "#e0e0e0"  # 浅灰色代表无代谢底噪
    elif "Transitional" in cat:
        custom_palette[cat] = "#ffe119"  # 亮黄色代表过渡交界区
    else:
        # 分配大色板中的颜色
        custom_palette[cat] = color_list[color_idx % len(color_list)]
        color_idx += 1

# ==============================
# 3. 画图
# ==============================
sc.pl.spatial(
    adata_model,
    color="Niche_Annotation",
    size=1, # 保持你修改后的 size=1 
    title="Phase 2: Data-Driven Niche Annotation (10 Axes)",
    frameon=False,
    palette=custom_palette
)
print("✅ 阶段二执行完毕！已定位出单轴区、多轴区、休眠区及待重塑的 Transitional 灰度区。")

# =========================================================
# Cell 8: 绘制 10 种代谢物的空间浓度梯度图 (连续分布)
# =========================================================
import matplotlib.pyplot as plt
import scanpy as sc

print("=== 附加分析: 生成十种代谢物空间通讯浓度梯度图 ===")
print("注: 采用 Coupling_Score (Sender * 邻居Receiver) 作为有效代谢物浓度与作用场的推断指标")

# 1. 获取需要绘制的列名 (Coupling_Score 已经在上一步取了 log1p，非常适合梯度可视化)
gradient_columns = [f"Coupling_{k}" for k in AXES]

# 2. 为每张图设置好标题
plot_titles =[f"{k} Gradient\n(Coupling Score)" for k in AXES]

# 3. 设置绘图参数
# 使用 'magma' 或 'inferno' 色系，这种暗底亮色的色谱非常适合展示类似“荧光染色”的浓度梯度
cmap_choice = "magma" 

# 4. 利用 scanpy 自带的多图排列功能绘制
sc.pl.spatial(
    adata_model,
    color=gradient_columns,
    cmap=cmap_choice,
    size=1,              # 保持与你前面一致的 spot 大小
    ncols=5,             # 每行放 5 张图，10 个轴正好 2 行
    frameon=False,       # 去掉坐标轴边框，更美观
    title=plot_titles,
    vmin=0,              # 浓度最低为 0
    vmax='p99',          # 截断最高 1% 的极端离群值，让主体梯度颜色更丰富
    show=False           # 先不显示，稍后通过 plt.show() 统一控制
)

# 调整子图间距并展示
plt.subplots_adjust(wspace=0.1, hspace=0.2)
fig = plt.gcf()
fig.set_size_inches(25, 10) # 设置大画布，保证高分辨率
plt.show()

print("✅ 10 种代谢物的浓度梯度图绘制完毕！")

# =========================================================
# Cell 9: 空间拓扑图剪枝与重塑 (阶段三: Split, Merge & Smooth)
# 从这一步开始后续有保存结果
# =========================================================

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scanpy as sc
import matplotlib.pyplot as plt

print("=== 阶段三: 空间拓扑图剪枝与重塑 (Split, Merge & Smooth) ===")

# 初始化工作列
adata_model.obs["Niche_Type"] = adata_model.obs["Niche_Annotation"].astype(str)
adj = adata_model.obsp["spatial_connectivities"]

# ---------------------------------------------------------
# Step 6: Split（破）—— 图连通优先的异质区拆解
# ---------------------------------------------------------
print("--> 1. Split (破): 正在探索 Transitional 过渡区内的高代谢连通子图...")
min_cc_size = 20  
transitional_mask = (adata_model.obs["Niche_Type"] == "Transitional").values

tentative_labels = np.array([""] * adata_model.n_obs, dtype=object)
for i in range(adata_model.n_obs):
    if transitional_mask[i]:
        active_axes = [k for k in AXES if adata_model.obs[f"Active_{k}"].iloc[i] == 1]
        if len(active_axes) == 1:
            tentative_labels[i] = f"Single-axis: {active_axes[0]}"
        elif len(active_axes) > 1:
            # 🔥 修改：在拆分区域同样应用明确的多轴命名
            tentative_labels[i] = f"Multi-axis: {'_'.join(active_axes)}"
        else:
            tentative_labels[i] = "Transitional"

split_count = 0
for label in np.unique(tentative_labels):
    if not label or label == "Transitional":
        continue
    label_indices = np.where((tentative_labels == label) & transitional_mask)[0]
    if len(label_indices) == 0: continue
    sub_adj = adj[label_indices, :][:, label_indices]
    n_components, labels = sp.csgraph.connected_components(sub_adj, directed=False)
    counts = np.bincount(labels)
    valid_components = np.where(counts >= min_cc_size)[0]
    for comp in valid_components:
        comp_indices = label_indices[labels == comp]
        adata_model.obs.iloc[comp_indices, adata_model.obs.columns.get_loc("Niche_Type")] = label
        split_count += len(comp_indices)
print(f"    [Split] 成功从过渡区中抢救裂变出 {split_count} 个具有空间连通性的活跃 Spots。")

# ---------------------------------------------------------
# Step 7: Merge（立）—— 功能同型合并为宏观生态位
# ---------------------------------------------------------
print("--> 2. Merge (立): 正在跨物理空间合并功能同型的生态位...")
def map_niche_name(raw_name):
    if raw_name.startswith("Single-axis:"):
        axis = raw_name.split(": ")[1]
        return f"TMCN_{axis}_Axis"
    elif raw_name.startswith("Multi-axis:"):
        # 🔥 修改：将 "Multi-axis: Glutamine_Lactate" 转换为 "TMCN_Glutamine_Lactate_Multi_Axis"
        axes_str = raw_name.split(": ")[1]
        return f"TMCN_{axes_str}_Multi_Axis"
    elif raw_name.startswith("Quiescent"):
        return "TMCN_Quiescent"
    else:
        return "TMCN_Transitional"

adata_model.obs["Niche_Type"] = adata_model.obs["Niche_Type"].apply(map_niche_name)
print(f"    [Merge] 当前合并后的宏观生态位类型: {adata_model.obs['Niche_Type'].unique().tolist()}")




# ---------------------------------------------------------
# Step 8: 空间连续性硬约束（平滑去噪 Smooth）
# ---------------------------------------------------------
print("--> 3. Smooth (平滑): 强制注销极小碎片，并入邻接主力生态位...")
min_area_smooth = max(20, int(adata_model.n_obs * 0.005)) # 至少20个点，或0.5%总面积
niche_types = adata_model.obs["Niche_Type"].unique()
noise_mask = np.zeros(adata_model.n_obs, dtype=bool)

# 8.1: 全局扫描寻找碎片
for nt in niche_types:
    indices = np.where(adata_model.obs["Niche_Type"] == nt)[0]
    if len(indices) == 0: continue
    
    sub_adj = adj[indices, :][:, indices]
    n_components, labels = sp.csgraph.connected_components(sub_adj, directed=False)
    counts = np.bincount(labels)
    
    # 找到所有太小的孤立岛屿
    small_comps = np.where(counts < min_area_smooth)[0]
    for comp in small_comps:
        comp_indices = indices[labels == comp]
        noise_mask[comp_indices] = True

print(f"    [Smooth] 扫描发现 {noise_mask.sum()} 个孤立噪音碎片 (面积 < {min_area_smooth})。")

# 8.2: 依据“最大接触边数”进行邻域传播平滑 (KNN Label Propagation)
adata_model.obs["Final_Niche_Type"] = adata_model.obs["Niche_Type"].astype(str)
adata_model.obs.loc[noise_mask, "Final_Niche_Type"] = "Noise"

max_iter = 10
for iteration in range(max_iter):
    current_noise = (adata_model.obs["Final_Niche_Type"] == "Noise").values
    if not current_noise.any():
        break
        
    noise_indices = np.where(current_noise)[0]
    new_labels =[]
    
    for idx in noise_indices:
        # 获取该 spot 的所有物理邻居
        neighbors = adj[idx].indices
        neighbor_labels = adata_model.obs["Final_Niche_Type"].iloc[neighbors]
        valid_labels = neighbor_labels[neighbor_labels != "Noise"]
        
        if len(valid_labels) > 0:
            # 多数投票：并入周围最大的生态位
            new_labels.append(valid_labels.mode()[0])
        else:
            new_labels.append("Noise")
            
    adata_model.obs.iloc[noise_indices, adata_model.obs.columns.get_loc("Final_Niche_Type")] = new_labels

# 兜底：如果还有全是 Noise 组成的孤岛没被吞并，强制标记为休眠或过渡态
leftover = (adata_model.obs["Final_Niche_Type"] == "Noise").values
if leftover.sum() > 0:
    adata_model.obs.loc[leftover, "Final_Niche_Type"] = "TMCN_Transitional"

print("    [Smooth] 平滑去噪完成！所有连续性硬约束均满足。")

# ---------------------------------------------------------
# 可视化: 阶段三重塑后的最终生态位 (Final Niche Type)
# ---------------------------------------------------------
print("\n--> 4. 绘制拓扑精修后的【最终代谢通讯生态位 (TMCN)】地图...")

# 固定的基础配色字典
custom_palette = {
    "TMCN_Lactate_Axis": "#d62728",        
    "TMCN_Adenosine_Axis": "#1f77b4",      
    "TMCN_PGE2_Axis": "#ff7f0e",           
    "TMCN_Glutamine_Axis": "#2ca02c",      
    "TMCN_Succinate_Axis": "#9467bd",      
    "TMCN_Kynurenine_Axis": "#e377c2",     
    "TMCN_ATP_Axis": "#17becf",            
    "TMCN_S1P_Axis": "#7f7f7f",            
    "TMCN_LPA_Axis": "#ff9896",            
    "TMCN_Glutamate_Axis": "#fada5e",      
    "TMCN_Quiescent": "#c7c7c7",           
    "TMCN_Transitional": "#bcbd22"         
}

# 动态多轴备用颜色池（用于随机/顺序分配给生成的复合轴）
multi_axis_colors = ["#8c564b", "#8b008b", "#008080", "#ff1493", "#000080", "#ff8c00", "#4682b4", "#556b2f"]

final_categories = adata_model.obs["Final_Niche_Type"].unique().tolist()
active_palette = {}
color_idx = 0

# 🔥 修改：动态识别后缀为 _Multi_Axis 的类别，并为其分配颜色
for cat in final_categories:
    if cat in custom_palette:
        active_palette[cat] = custom_palette[cat]
    elif "Multi_Axis" in cat:
        active_palette[cat] = multi_axis_colors[color_idx % len(multi_axis_colors)]
        color_idx += 1
    else:
        active_palette[cat] = "#000000"

sc.pl.spatial(
    adata_model,
    color="Final_Niche_Type",
    size=1,
    title="Phase 3: Final Smoothed TMCNs",
    frameon=False,
    palette=active_palette
)
print("✅ 阶段三执行完毕！微型聚类已成功升维并重塑为宏观、空间连续的代谢通讯生态位。")




# =========================================================
# 保存包含 Final_Niche_Type 的结果
# =========================================================
save_base_dir = "/home/zhangjunyi/xiangmu/nichecompass-main/datasets/Human_breast_cancer/Human_breast_cancer_ViHBC"
file_prefix = "Human_breast_cancer_ViHBC_NicheCompassAndmy_Final"

# 覆盖保存 h5ad (或者你可以存成一个新名字，比如 _processed.h5ad)
adata_save_path = os.path.join(save_base_dir, f"{file_prefix}.h5ad")
adata_model.write(adata_save_path)
print(f"✅ 已将包含 Final_Niche_Type 的结果保存至: {adata_save_path}")

# =========================================================
# Cell 10: 跨模态标签对齐与最终层级生态位构建 (SpaCET + NicheCompass)
# =========================================================
import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import os

print("=== 阶段四: 跨模态标签联合 (Macro-Micro Hierarchical Niche) ===")

# ==========================================
# 1. 路径配置 (读取数据与设置输出)
# ==========================================
# 你的 NicheCompass 最终产物 (也就是我们要覆盖保存的原始文件)
path_h5ad_niche = "/home/zhangjunyi/xiangmu/nichecompass-main/datasets/Human_breast_cancer/Human_breast_cancer_ViHBC/Human_breast_cancer_ViHBC_NicheCompassAndmy_Final.h5ad"

# 你上一步保存的 SpaCET GMM 预测结果 CSV
path_spacet_csv = "/home/zhangjunyi/xiangmu/nichecompass-main/outputs/zhibiao/Human_breast_cancer_ViHBC/SpaCET/GMM_Tumor_NonTumor_Pred_Labels.csv"

# 指定输出文件夹（自动创建，用于存CSV和图片）
SAVE_DIR = "/home/zhangjunyi/xiangmu/nichecompass-main/outputs/zhibiao/Human_breast_cancer_ViHBC/my_and_NicheCompass"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# 2. 读取并合并数据
# ==========================================
print("--> 1. 正在加载并对齐数据...")
adata = ad.read_h5ad(path_h5ad_niche)
spacet_df = pd.read_csv(path_spacet_csv, index_col=0)

# 动态获取 GMM 预测的那一列的名字 (例如 'Unsupervised_Pred_0.45')
spacet_pred_col = [c for c in spacet_df.columns if c.startswith('Unsupervised_Pred_')][0]

# 合并到 adata.obs (利用索引自动对齐)
adata.obs = adata.obs.join(spacet_df[[spacet_pred_col]])
print(f"    [对齐完成] 成功导入 SpaCET 预测标签: {spacet_pred_col}")

# ==========================================
# 3. 严格执行核心合并规则 (并集逻辑)
# ==========================================
print("--> 2. 正在执行宏观-微观联合推断逻辑...")

# 初始化新列，默认全为健康区域
adata.obs['Level1_Macro_Region'] = 'Healthy_Region'
adata.obs['Level2_Micro_Niche'] = 'Healthy_Region'

# 提取关键判断条件
is_spacet_tumor = adata.obs[spacet_pred_col] == 'Predicted_Tumor'
is_niche_quiescent = adata.obs['Final_Niche_Type'] == 'TMCN_Quiescent'
# 新增：判断是否为过渡态
is_niche_transitional = adata.obs['Final_Niche_Type'] == 'TMCN_Transitional'


# ----------------- 规则 1: Level 1 (Tumor vs Healthy) ✅ 已修改 -----------------
# 【你的新需求】SpaCET预测为肿瘤  OR  (微环境不是休眠态 且 不是过渡态) → 肿瘤区域
tumor_mask = is_spacet_tumor | ( (~is_niche_quiescent) & (~is_niche_transitional) )
adata.obs.loc[tumor_mask, 'Level1_Macro_Region'] = 'Tumor_Region'



# ----------------- 规则 2: Level 2 (微型代谢生态位) -----------------
# 🔥 核心修复：强制转为普通字符串(str)，打破 Categorical 字典的限制
final_niche_str = adata.obs['Final_Niche_Type'].astype(str)

# 重新定义目标掩码：包含 Quiescent (休眠) 或 Transitional (过渡态)
is_quiescent_or_transitional = (final_niche_str == 'TMCN_Quiescent') | (final_niche_str == 'TMCN_Transitional')

# 1. 真正活跃的代谢区 (排除了休眠和过渡态)：直接保留其详细的多轴/单轴名称
active_niche_mask = ~is_quiescent_or_transitional
adata.obs.loc[active_niche_mask, 'Level2_Micro_Niche'] = final_niche_str[active_niche_mask]

# 2. 【你的新逻辑】将 SpaCET 预测为 Tumor 且 代谢状态为 Quiescent 或 Transitional 的区域，统一视作 TMCN_Quiescent (代谢耗竭/坏死核心)
target_tumor_mask = is_spacet_tumor & is_quiescent_or_transitional
adata.obs.loc[target_tumor_mask, 'Level2_Micro_Niche'] = 'TMCN_Quiescent'

# 🔥 写入完成后，为了方便后续 Scanpy 画图，重新将这列转换回 Category 格式
adata.obs['Level2_Micro_Niche'] = adata.obs['Level2_Micro_Niche'].astype('category')

print("    [逻辑应用完毕] 成功生成 Level1_Macro_Region 与 Level2_Micro_Niche (已将肿瘤内过渡区并入休眠区)。")



# 2. 将 SpaCET 预测为 Tumor 且 你的结果为 TMCN_Quiescent 的区域，视作 TMCN_Quiescent
quiescent_tumor_mask = is_spacet_tumor & is_niche_quiescent
adata.obs.loc[quiescent_tumor_mask, 'Level2_Micro_Niche'] = 'TMCN_Quiescent'

print("    [逻辑应用完毕] 成功生成 Level1_Macro_Region 与 Level2_Micro_Niche。")

# ==========================================
# 4. 可视化检查
# ==========================================
print("--> 3. 正在生成最终层级空间生态位地图...")

macro_palette = {
    "Tumor_Region": "#d62728",    
    "Healthy_Region": "#aec7e8"   
}

base_micro_palette = {
    "TMCN_Lactate_Axis": "#d62728",        
    "TMCN_Adenosine_Axis": "#1f77b4",      
    "TMCN_PGE2_Axis": "#ff7f0e",           
    "TMCN_Glutamine_Axis": "#2ca02c",      
    "TMCN_Succinate_Axis": "#9467bd",      
    "TMCN_Kynurenine_Axis": "#e377c2",     
    "TMCN_ATP_Axis": "#17becf",            
    "TMCN_S1P_Axis": "#7f7f7f",            
    "TMCN_LPA_Axis": "#ff9896",            
    "TMCN_Glutamate_Axis": "#fada5e",      
    "TMCN_Transitional": "#bcbd22",
    "TMCN_Quiescent": "#333333",           
    "Healthy_Region": "#f0f0f0"            
}

multi_axis_colors = ["#8c564b", "#8b008b", "#008080", "#ff1493", "#000080", "#ff8c00", "#4682b4", "#556b2f"]

active_macro_colors = {k: macro_palette[k] for k in adata.obs['Level1_Macro_Region'].unique()}

active_micro_colors = {}
color_idx = 0
for cat in adata.obs['Level2_Micro_Niche'].unique():
    if cat in base_micro_palette:
        active_micro_colors[cat] = base_micro_palette[cat]
    elif "Multi_Axis" in cat:
        active_micro_colors[cat] = multi_axis_colors[color_idx % len(multi_axis_colors)]
        color_idx += 1
    else:
        active_micro_colors[cat] = "#000000"

fig, axs = plt.subplots(1, 2, figsize=(16, 6))
sc.pl.spatial(adata, color='Level1_Macro_Region', size=1, title="Level 1: Macro Region", 
              palette=active_macro_colors, frameon=False, ax=axs[0], show=False)
sc.pl.spatial(adata, color='Level2_Micro_Niche', size=1, title="Level 2: Purified Metabolic Niches", 
              palette=active_micro_colors, frameon=False, ax=axs[1], show=False)
plt.tight_layout()
plt.show()

# ==========================================
# 5. 提取并保存：最终详细生态位 (Level 2) 各代谢轴高活性比例
# ==========================================
print("\n--> 4. 正在计算最终详细生态位的代谢轴高活性 Spot 比例...")

axes_active_cols = [c for c in adata.obs.columns if c.startswith("Active_")]
active_frac_df = adata.obs.groupby('Level2_Micro_Niche')[axes_active_cols].mean().reset_index()

rename_map = {c: c.replace("Active_", "active_fraction_") for c in axes_active_cols}
active_frac_df.rename(columns=rename_map, inplace=True)
active_frac_df.rename(columns={'Level2_Micro_Niche': 'Micro_Niche_Cluster'}, inplace=True)

fraction_csv_path = os.path.join(SAVE_DIR, "Level2_Micro_Niche_Active_Fractions.csv")
active_frac_df.to_csv(fraction_csv_path, index=False)
print(f"✅ 各代谢轴突破高活性阈值的 spot 比例已保存至: \n   {fraction_csv_path}")

# ==========================================
# 6. 精简 h5ad 文件，只保留目标核心列
# ==========================================
print("\n--> 5. 正在精简 adata.obs 冗余中间特征，提取纯净版结果...")

drop_keywords = [
    "Sender_Score", "Receiver_Score", "Coupling_", "Active_", "Quiet_", 
    "Cluster_Active_Frac", "nichecompass_", "latent_leiden", 
    "Pred_Threshold_", "Unsupervised_Pred_", "Niche_Annotation", 
    "Niche_Type", "Final_Niche_Type", "Ground_Truth"
]

cols_to_drop = []
for c in adata.obs.columns:
    if any(k in c for k in drop_keywords):
        cols_to_drop.append(c)

adata.obs.drop(columns=cols_to_drop, inplace=True, errors='ignore')
print(f"    [精简完成] 共移除了 {len(cols_to_drop)} 个生化推断中间列。")

# ==========================================
# 7. 【关键修改】覆盖保存回原 h5ad 文件
# ==========================================
# 直接使用开头加载的 path_h5ad_niche，不新建文件
adata.write_h5ad(path_h5ad_niche)

print(f"\n🎉 终极纯净版数据已成功覆盖保存至原始文件:\n   {path_h5ad_niche}")
print("======================================================")
print("📌 该最终 h5ad 文件的 adata.obs 中保留的列严格符合你的要求：")
print("  1. 原始基础列及病理注释")
print("  2. SpaCET 反卷积的所有细胞成分分数 (被永远合并在内了)")
print("  3. 最终简略版生态位: 'Level1_Macro_Region' (Tumor/Healthy)")
print("  4. 最终详细版生态位: 'Level2_Micro_Niche' (单轴/多轴/休眠等)")
print("======================================================")

# =========================================================
# Cell 11: 顶级生信学者专属 - 空间多维全栈指标综合评估
# 涵盖宏观物理边界 (ARI, F1) + 微观特征纯度 (ASW, SS-C, DBI) + 空间物理聚集度 (Moran's I)
# =========================================================
import scanpy as sc
import squidpy as sq
import pandas as pd
import numpy as np
import os
import warnings
from sklearn.metrics import (
    adjusted_rand_score, 
    f1_score, 
    silhouette_score, 
    silhouette_samples, 
    davies_bouldin_score
)

warnings.filterwarnings("ignore")

print("="*60)
print("🚀 开始执行全维度空间与特征指标综合评估...")
print("="*60)

# 1. 加载数据 (假设直接使用你内存中的 adata，或者重新读取)
# 如果你是接在 Cell 11 后面运行，此时内存中已经有 adata。如果需要重新读取，请取消注释下一行：
# path_h5ad = "/home/zhangjunyi/xiangmu/nichecompass-main/outputs/colorectal_12_cancer_CRC/SN048_A121573_Rep1/SpaCET_And_My/SN048_A121573_Rep1_NicheCompassAndmy_Final.h5ad"
# adata = sc.read_h5ad(path_h5ad)

# =========================================================
# 模块一：宏观边界评估 (Supervised: ARI, F1)
# =========================================================
print("\n🟢 [1/3] 正在计算宏观监督指标 (Level 1 vs Ground Truth)...")

# 严格保留你的肿瘤/正常映射规则
gt_map = {
    'tumor': 1, 'tumor&stroma_IC med to high': 1, 'stroma_fibroblastic_IC high': 1,
    'epithelium&submucosa': 0, 'non neo epithelium': 0, 'submucosa': 0, 
    'IC aggregregate_submucosa': 0, 'exclude': 0
}
pred_map = {'Healthy_Region': 0}

# 映射并清理无效点 (NaN)
gt_raw = adata.obs['annot_type'].map(gt_map).astype(float).values
pred_raw = adata.obs['Level1_Macro_Region'].map(pred_map).fillna(1.0).astype(float).values

valid_mask = ~np.isnan(gt_raw)
y_true = gt_raw[valid_mask]
y_pred = pred_raw[valid_mask]

# 计算 ARI 和 F1
ari_val = adjusted_rand_score(y_true, y_pred)
f1_val = f1_score(y_true, y_pred, pos_label=1.0, average='binary')

print(f"  --> Adjusted Rand Index (ARI): {ari_val:.4f}")
print(f"  --> F1-Score (Tumor Region): {f1_val:.4f}")

# =========================================================
# 模块二：微观特征纯度评估 (Unsupervised: ASW, SS-C, DBI)
# =========================================================
print("\n🔵 [2/3] 正在计算微观特征纯度指标 (Level 2 内部逻辑自洽)...")

# 寻找合适的底层特征矩阵 X
if 'nichecompass_latent' in adata.obsm:
    X_features = adata.obsm['nichecompass_latent']
    print("  --> 使用 NicheCompass Latent 空间作为特征矩阵。")
else:
    print("  --> 未发现 Latent 矩阵，自动降维原始基因表达谱 (PCA) 作为公平特征对照...")
    if 'X_pca' not in adata.obsm:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.tl.pca(adata, svd_solver='arpack')
    X_features = adata.obsm['X_pca']

# 提取细粒度微生态位标签
y_micro = adata.obs['Level2_Micro_Niche'].astype(str).values

# 为了计算无监督指标，必须保证至少有两个类别且不能有过小（<2）的散点群
valid_micro_mask = adata.obs['Level2_Micro_Niche'] != 'nan'
X_valid = X_features[valid_micro_mask]
y_valid = y_micro[valid_micro_mask]

# 计算全局平均轮廓系数 (ASW) 和 DBI
asw_val = silhouette_score(X_valid, y_valid)
dbi_val = davies_bouldin_score(X_valid, y_valid)

# 计算专属通讯特征轮廓系数 (SS-C) - 针对肿瘤区域内的特定代谢分类计算平均凝聚度
sample_silhouette_values = silhouette_samples(X_valid, y_valid)
adata.obs.loc[valid_micro_mask, 'Silhouette_Value'] = sample_silhouette_values

# 提取所有肿瘤代谢生态位的平均轮廓系数 (排除纯Healthy的干扰)
tumor_niches_mask = [True if "TMCN" in str(cat) else False for cat in y_valid]
if any(tumor_niches_mask):
    ssc_val = np.mean(sample_silhouette_values[tumor_niches_mask])
else:
    ssc_val = asw_val # 如果没有细分肿瘤区，兜底等同于全局ASW

print(f"  --> Average Silhouette Width (ASW): {asw_val:.4f}")
print(f"  --> Silhouette Score on Coupling (SS-C): {ssc_val:.4f}")
print(f"  --> Davies-Bouldin Index (DBI): {dbi_val:.4f}")

# =========================================================
# 模块三：空间物理拓扑聚集度 (Spatial Auto-correlation: Moran's I) - 【已修复】
# =========================================================
print("\n🟣 [3/3] 正在计算物理空间连贯性指标 (Moran's I)...")

# 1. 确保存在空间邻接图
if 'spatial_connectivities' not in adata.obsp:
    print("  --> 正在重构基础空间邻接图 (KNN)...")
    sq.gr.spatial_neighbors(adata, coord_type="generic", spatial_key="spatial", n_neighs=6)

# 2. 将 Level2_Micro_Niche 进行 One-hot 编码
niche_dummies = pd.get_dummies(adata.obs['Level2_Micro_Niche'])
dummy_cols = [c for c in niche_dummies.columns if "TMCN" in str(c)] # 仅评估代谢微环境

if len(dummy_cols) > 0:
    # 【核心修复逻辑】：构建轻量级“影子 AnnData”
    # 将生态位的 One-hot 矩阵转为 float，并塞入 tmp_adata.X 中骗过 Squidpy
    tmp_adata = sc.AnnData(X=niche_dummies[dummy_cols].values.astype(float))
    tmp_adata.obs_names = adata.obs_names
    tmp_adata.var_names = dummy_cols
    
    # 继承主 adata 的空间物理连通图
    tmp_adata.obsp['spatial_connectivities'] = adata.obsp['spatial_connectivities']
    
    # 核心计算：现在 Squidpy 可以把这些生态位当做“基因”来计算物理聚集度了
    sq.gr.spatial_autocorr(tmp_adata, mode="moran", genes=dummy_cols, n_perms=100, n_jobs=-1)
    
    # 获取结果
    moran_df = tmp_adata.uns["moranI"]
    moran_val = moran_df['I'].mean()
    
    # 顺便为你打印出各个生态位的具体连续性得分
    print("  --> 各子生态位 Moran's I 得分:")
    for idx, row in moran_df.iterrows():
        print(f"      - {idx}: {row['I']:.4f}")
        
else:
    moran_val = 0.0
    print("  ⚠️ 警告：未检测到细粒度的 TMCN 代谢生态位。")

print(f"  --> 总体平均 Moran's Index (Moran's I): {moran_val:.4f}")


# =========================================================
# 模块四：汇总保存 CSV
# =========================================================
SAVE_DIR = "/home/zhangjunyi/xiangmu/nichecompass-main/outputs/zhibiao/Human_breast_cancer_ViHBC/my_and_NicheCompass"
os.makedirs(SAVE_DIR, exist_ok=True)
csv_save_path = os.path.join(SAVE_DIR, "评估结果_SpaCET_And_My.csv")

# 构建结果表
results_df = pd.DataFrame({
    "Metric Name": [
        "Adjusted Rand Index", 
        "F1-Score", 
        "Average Silhouette Width", 
        "Silhouette Score on Coupling Scores", 
        "Davies-Bouldin Index", 
        "Moran's Index"
    ],
    "Abbreviation": ["ARI", "F1", "ASW", "SS-C", "DBI", "Moran's I"],
    "Value": [
        round(ari_val, 4), 
        round(f1_val, 4), 
        round(asw_val, 4), 
        round(ssc_val, 4), 
        round(dbi_val, 4), 
        round(moran_val, 4)
    ],
    "Ideal Trend": [
        "Closer to 1", "Closer to 1", "Closer to 1", 
        "Closer to 1", "Closer to 0", "Closer to 1 (>0)"
    ],
    "Evaluation Dimension": [
        "Macro Boundary (Supervised)", 
        "Macro Boundary (Supervised)", 
        "Micro Pureness (Unsupervised)", 
        "Micro Pureness (Unsupervised)", 
        "Micro Pureness (Unsupervised)", 
        "Spatial Topology (Physical)"
    ]
})

results_df.to_csv(csv_save_path, index=False, encoding="utf-8-sig")

print("\n" + "="*60)
print(f"🎉 巅峰对决指标计算完毕！最终汇总表格已成功保存至：\n   {csv_save_path}")
print("="*60)

