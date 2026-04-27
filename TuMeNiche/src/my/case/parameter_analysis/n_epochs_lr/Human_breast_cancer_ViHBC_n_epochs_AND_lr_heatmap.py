# 自动提取自 Jupyter Notebook
# 源文件：/home/zhangjunyi/xiangmu/nichecompass-main/TuMeNiche/src/my/case/parameter_analysis/n_epochs_lr/Human_breast_cancer_ViHBC_n_epochs_AND_lr_heatmap.ipynb

# =========================================================
# 指定使用物理 GPU 1
# 注意：必须放在 import torch 之前；如果 torch 已经导入，需要重启 kernel 后重新运行
# =========================================================
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# =========================================================
# Cell 1: import、随机种子、全局配置 CONFIG
# 参数组合热图分析: n_epochs_AND_lr
# =========================================================

import os
import sys
import gc
import random
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt

import anndata as ad
import scanpy as sc
import squidpy as sq
import decoupler as dc

from IPython.display import display
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# --------------------------
# 0. 当前数据集与参数分析配置
# --------------------------
CONFIG = {'dataset_name': 'Human_breast_cancer_ViHBC',
 'input_h5ad': '/home/zhangjunyi/xiangmu/nichecompass-main/datasets/Human_breast_cancer/Human_breast_cancer_ViHBC/Human_breast_cancer_integrated.h5ad',
 'gmm_threshold_csv_candidates': ['/home/zhangjunyi/xiangmu/nichecompass-main/outputs/test/Human_breast_cancer_ViHBC/SpaCET/GMM_Optimal_Threshold_Result.csv',
                                  '/home/zhangjunyi/xiangmu/nichecompass-main/outputs/Human_breast_cancer_ViHBC/SpaCET/GMM_Optimal_Threshold_Result.csv'],
 'save_root': '/home/zhangjunyi/xiangmu/nichecompass-main/outputs/case/parameter_analysis/n_epochs_lr/Human_breast_cancer_ViHBC',
 'file_prefix': 'Human_breast_cancer_ViHBC',
 'pathology_col_candidates': ['annot_type', 'pathology_annotation'],
 'gt_map': {'Tumor': 1, 'Invasive': 1, 'Surrounding tumor': 0, 'Healthy': 0},
 'n_epochs': 50,
 'seed': 42,
 'local_src': '/home/zhangjunyi/xiangmu/nichecompass-main/src',
 'nichenet_cache_dir': '/home/zhangjunyi/xiangmu/nichecompass-main/cache_nichenet',
 'n_top_hvg': 3000,
 'spatial_graph_n_neighs': 8,
 'latent_key': 'nichecompass_latent',
 'native_cluster_key': 'latent_leiden_0.6',
 'leiden_resolution': 0.6,
 'latent_node_batch_size': 128,
 'low_malig_offset': 0.2,
 'MIN_FC': 1.3,
 'MIN_ACTIVE_SPOTS': 10,
 'MIN_INTENSITY_RATIO': 1.2,
 'QUIESCENT_COLD_AXIS_NUM': 6,
 'tmcn_file_path': '/home/zhangjunyi/xiangmu/nichecompass-main/data/pre_data/siyuanzu/my_metabolite_network.csv',
 'analysis_name': 'n_epochs_AND_lr',
 'x_param_name': 'lr',
 'y_param_name': 'n_epochs',
 'lr_range': [1e-05, 3e-05, 5e-05, 0.0001, 0.0003, 0.0005, 0.001],
'n_epochs_range': [30, 50, 75, 100, 125, 150, 175, 200],
 'fixed_n_sampled_neighbors': 4,
 'fixed_lr': 0.0001,
 'fixed_lambda_edge_recon': 100000.0,
 'fixed_lambda_gene_expr_recon': 100.0}

os.makedirs(CONFIG["save_root"], exist_ok=True)

# --------------------------
# 1. 固定随机种子
# --------------------------
def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)

    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as e:
        print("deterministic warn_only 模式未完全启用：", e)
    torch.set_num_threads(1)

set_seed(CONFIG["seed"])

# --------------------------
# 2. 导入 NicheCompass
# --------------------------
LOCAL_SRC = CONFIG["local_src"]
if LOCAL_SRC not in sys.path:
    sys.path.insert(0, LOCAL_SRC)

import nichecompass as nc
import torch

print("✅ CUDA 可用状态:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("✅ 当前使用 GPU:", torch.cuda.get_device_name(0))
print("✅ NicheCompass 路径:", nc.__file__)
print("✅ 数据集:", CONFIG["dataset_name"])
print("✅ 参数分析:", CONFIG["analysis_name"])
print("✅ 输出目录:", CONFIG["save_root"])


# =========================================================
# Cell 2: 工具函数：知识库、Sender/Receiver、训练、生态位判定、ARI
# =========================================================

# --------------------------
# 3. TMCN 代谢通讯轴知识库
# 从外部四元组 CSV 文件读取
# --------------------------

tmcn_file_path = CONFIG["tmcn_file_path"]

if not os.path.exists(tmcn_file_path):
    raise FileNotFoundError(f"❌ 未找到四元组文件: {tmcn_file_path}")

axis_table = pd.read_csv(tmcn_file_path, encoding="utf-8-sig")

required_cols = [
    "TMCN_Name",
    "Source_Pathways",
    "Source_Genes",
    "Target_Genes",
    "Biologic_Meaning"
]

missing_cols = [c for c in required_cols if c not in axis_table.columns]
if len(missing_cols) > 0:
    raise ValueError(
        f"❌ 四元组文件缺少必要列: {missing_cols}\n"
        f"当前文件列名为: {axis_table.columns.tolist()}"
    )

axis_table["TMCN_Name"] = axis_table["TMCN_Name"].astype(str).str.strip()

AXES = [
    x.replace("TMCN_", "").replace("_Axis", "")
    for x in axis_table["TMCN_Name"]
]

print("✅ 已从外部 CSV 文件加载 TMCN 四元组知识库")
print(f"📄 文件路径: {tmcn_file_path}")
print(f"✅ 共加载 {len(axis_table)} 条代谢通讯轴")
print(f"✅ AXES = {AXES}")

display(axis_table)

def split_items(x):
    return [i.strip() for i in str(x).split(",") if i.strip()]


def get_valid_genes(adata_obj, genes):
    return [g for g in genes if g in adata_obj.var_names]


def read_gmm_threshold():
    for p in CONFIG["gmm_threshold_csv_candidates"]:
        if os.path.exists(p):
            gmm_df = pd.read_csv(p)
            val = float(gmm_df.loc[gmm_df["Metric"] == "Optimal_GMM_Threshold", "Value"].iloc[0])
            print(f"✅ 读取 GMM 阈值: {val:.4f} | {p}")
            return val
    raise FileNotFoundError("❌ 未找到 GMM_Optimal_Threshold_Result.csv，请检查 CONFIG['gmm_threshold_csv_candidates']。")


def find_malig_col(adata_obj):
    for c in ["Malignant", "malignant", "Tumor", "tumor", "Cancer", "cancer"]:
        if c in adata_obj.obs.columns:
            return c
    raise KeyError("❌ 未找到 SpaCET 恶性分数字段，如 Malignant。")


def find_pathology_col(adata_obj):
    for c in CONFIG["pathology_col_candidates"]:
        if c in adata_obj.obs.columns:
            return c
    raise KeyError(f"❌ 未找到病理注释列。候选列: {CONFIG['pathology_col_candidates']}\n当前列: {adata_obj.obs.columns.tolist()}")


# =========================================================
# A. 只计算一次：Sender / Receiver 分数
# =========================================================
def compute_sender_receiver_scores():
    print("\n=== A. 加载数据并计算 Sender / Receiver 分数 ===")
    adata_raw = sc.read_h5ad(CONFIG["input_h5ad"])
    print(f"✅ 数据加载成功: {adata_raw.n_obs} spots, {adata_raw.n_vars} genes")

    # PROGENy pathway score
    print("--> 计算 Pathway_score (PROGENy)...")
    net_progeny = dc.get_progeny(organism="human", top=500)
    dc.run_mlm(
        mat=adata_raw,
        net=net_progeny,
        source="source",
        target="target",
        weight="weight",
        verbose=False,
        use_raw=False
    )
    df_pathway = dc.get_acts(adata_raw, obsm_key="mlm_estimate").to_df()
    df_pathway.columns = [c.strip().replace("-", "_").replace(" ", "_") for c in df_pathway.columns]

    # AUCell enzyme score
    print("--> 计算 Enzyme_score (AUCell)...")
    enzyme_records = []
    for _, row in axis_table.iterrows():
        ax = row["TMCN_Name"].replace("TMCN_", "").replace("_Axis", "")
        for g in split_items(row["Source_Genes"]):
            if g in adata_raw.var_names:
                enzyme_records.append({"source": ax, "target": g})

    if len(enzyme_records) == 0:
        raise ValueError("❌ 当前数据中没有任何 TMCN Source_Genes，可检查基因名是否为 Gene Symbol。")

    dc.run_aucell(
        mat=adata_raw,
        net=pd.DataFrame(enzyme_records),
        source="source",
        target="target",
        min_n=1,
        verbose=False,
        use_raw=False
    )
    df_enzyme = dc.get_acts(adata_raw, obsm_key="aucell_estimate").to_df()

    scaler = MinMaxScaler()
    df_pathway_scaled = pd.DataFrame(
        scaler.fit_transform(df_pathway),
        index=df_pathway.index,
        columns=df_pathway.columns
    )
    df_enzyme_scaled = pd.DataFrame(
        scaler.fit_transform(df_enzyme),
        index=df_enzyme.index,
        columns=df_enzyme.columns
    )

    def calc_receptor_score(genes):
        valid = get_valid_genes(adata_raw, genes)
        if not valid:
            return np.zeros(adata_raw.n_obs)
        X_sub = adata_raw[:, valid].X
        if sp.issparse(X_sub):
            mean_exp = np.asarray(X_sub.mean(axis=1)).reshape(-1)
        else:
            mean_exp = np.asarray(X_sub).mean(axis=1).reshape(-1)
        return scaler.fit_transform(mean_exp.reshape(-1, 1)).flatten()

    print("--> 计算 Sender_score 与 Receiver_score...")
    for _, row in axis_table.iterrows():
        ax = row["TMCN_Name"].replace("TMCN_", "").replace("_Axis", "")
        pathways = [p.strip().replace("-", "_").replace(" ", "_") for p in split_items(row["Source_Pathways"])]
        target_genes = split_items(row["Target_Genes"])

        valid_paths = [p for p in pathways if p in df_pathway_scaled.columns]
        if len(valid_paths) > 0:
            p_score = df_pathway_scaled[valid_paths].mean(axis=1).values
        else:
            p_score = np.zeros(adata_raw.n_obs)

        if ax in df_enzyme_scaled.columns:
            e_score = df_enzyme_scaled[ax].values
        else:
            e_score = np.zeros(adata_raw.n_obs)

        adata_raw.obs[f"{ax}_Sender_Score"] = p_score * e_score
        adata_raw.obs[f"{ax}_Receiver_Score"] = calc_receptor_score(target_genes)

    print("✅ Sender / Receiver 分数计算完成。")
    return adata_raw


# =========================================================
# B. 每个参数值重新构建 adata_model 并训练
# =========================================================
def prepare_adata_model(adata_scored):
    adata_model = adata_scored.copy()

    sc.pp.filter_genes(adata_model, min_cells=10)
    sc.pp.highly_variable_genes(
        adata_model,
        n_top_genes=CONFIG["n_top_hvg"],
        flavor="seurat_v3",
        subset=False
    )

    tmcn_gene_set = set()
    for _, row in axis_table.iterrows():
        tmcn_gene_set.update(split_items(row["Source_Genes"]))
        tmcn_gene_set.update(split_items(row["Target_Genes"]))
    tmcn_gene_set = {g for g in tmcn_gene_set if g in adata_model.var_names}

    keep_mask = adata_model.var["highly_variable"].values | adata_model.var_names.isin(list(tmcn_gene_set))
    adata_model = adata_model[:, keep_mask].copy()

    counts_key = "counts" if "counts" in adata_model.layers else None
    x_for_lib = adata_model.layers[counts_key] if counts_key is not None else adata_model.X
    lib_size = np.asarray(x_for_lib.sum(axis=1)).reshape(-1)
    valid_spot_mask = np.isfinite(lib_size) & (lib_size > 0)
    adata_model = adata_model[valid_spot_mask].copy()

    mat = adata_model.layers.get(counts_key) if counts_key else adata_model.X
    if sp.issparse(mat):
        mat = mat.copy()
        mat.data = np.nan_to_num(mat.data, nan=0.0, posinf=0.0, neginf=0.0)
        mat.data = np.clip(mat.data, a_min=0, a_max=None)
    else:
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
        mat = np.clip(mat, a_min=0, a_max=None)

    if counts_key:
        adata_model.layers[counts_key] = mat
    else:
        adata_model.X = mat

    sq.gr.spatial_neighbors(
        adata_model,
        coord_type="generic",
        spatial_key="spatial",
        n_neighs=CONFIG["spatial_graph_n_neighs"]
    )

    cache_dir = CONFIG["nichenet_cache_dir"]
    gp_dict = nc.utils.extract_gp_dict_from_nichenet_lrt_interactions(
        species="human",
        version="v2",
        keep_target_genes_ratio=0.25,
        max_n_target_genes_per_gp=50,
        load_from_disk=True,
        lr_network_file_path=f"{cache_dir}/nichenet_lr_network.csv",
        ligand_target_matrix_file_path=f"{cache_dir}/nichenet_ligand_target_matrix.csv"
    )
    nc.utils.add_gps_from_gp_dict_to_adata(gp_dict=gp_dict, adata=adata_model)

    return adata_model, counts_key


def train_nichecompass(adata_model, counts_key, n_epochs, lr):
    """训练 NicheCompass：本组只改变 n_epochs 和 lr。"""
    latent_key = CONFIG["latent_key"]
    adj_key = "spatial_connectivities"

    model = nc.models.NicheCompass(
        adata_model,
        counts_key=counts_key,
        adj_key=adj_key,
        gp_names_key="nichecompass_gp_names",
        active_gp_names_key="nichecompass_active_gp_names",
        gp_targets_mask_key="nichecompass_gp_targets",
        gp_targets_categories_mask_key="nichecompass_gp_targets_categories",
        gp_sources_mask_key="nichecompass_gp_sources",
        gp_sources_categories_mask_key="nichecompass_gp_sources_categories",
        latent_key=latent_key,
        conv_layer_encoder="gcnconv",
        active_gp_thresh_ratio=0.01,
    )

    full_node_batch_size = adata_model.n_obs
    full_edge_batch_size = int(adata_model.obsp["spatial_connectivities"].nnz)

    print(f"  全量 nodes: {full_node_batch_size}")
    print(f"  全量 edges: {full_edge_batch_size}")
    print(f"  n_epochs = {n_epochs}")
    print(f"  lr = {lr}")

    model.train(
        n_epochs=int(n_epochs),
        n_epochs_all_gps=10,
        lr=lr,
        lambda_edge_recon=CONFIG["fixed_lambda_edge_recon"],
        lambda_gene_expr_recon=CONFIG["fixed_lambda_gene_expr_recon"],
        lambda_l1_masked=0.0,
        edge_batch_size=full_edge_batch_size,
        node_batch_size=full_node_batch_size,
        n_sampled_neighbors=CONFIG["fixed_n_sampled_neighbors"],
        edge_val_ratio=0.0,
        node_val_ratio=0.0,
        use_cuda_if_available=True,
        verbose=False
    )

    adata_model.obsm[latent_key] = model.get_latent_representation(
        adata=adata_model,
        counts_key=counts_key,
        adj_key=adj_key,
        node_batch_size=CONFIG["latent_node_batch_size"],
    )

    sc.pp.neighbors(
        adata_model,
        use_rep=latent_key,
        key_added=latent_key,
        random_state=CONFIG["seed"]
    )
    sc.tl.umap(
        adata_model,
        neighbors_key=latent_key,
        random_state=CONFIG["seed"]
    )
    sc.tl.leiden(
        adata_model,
        neighbors_key=latent_key,
        key_added=CONFIG["native_cluster_key"],
        resolution=CONFIG["leiden_resolution"],
        random_state=CONFIG["seed"]
    )

    return adata_model, model

# =========================================================
# C. TMCN 生态位判定：复刻 Cell 7 核心逻辑
# =========================================================
def annotate_tmcn_niches(adata_model, hard_tumor_threshold):
    native_cluster_key = CONFIG["native_cluster_key"]
    adj_matrix = adata_model.obsp["spatial_connectivities"]

    # 1. Coupling Score
    for k in AXES:
        sender_scores = adata_model.obs[f"{k}_Sender_Score"].values
        receiver_scores = adata_model.obs[f"{k}_Receiver_Score"].values
        neighbor_receiver_sum = adj_matrix.dot(receiver_scores)
        raw_coupling = sender_scores * neighbor_receiver_sum
        adata_model.obs[f"Coupling_{k}"] = np.log1p(raw_coupling)

    # 2. 每条轴自适应阈值
    thresholds_dict = {}
    for k in AXES:
        scores = adata_model.obs[f"Coupling_{k}"].values
        scores_clean = scores[~np.isnan(scores) & (scores > 0)].reshape(-1, 1)
        t_high, t_quiet = None, None
        method_used = "Robust Dynamic Fallback"

        if len(scores_clean) > 50:
            try:
                gmm = GaussianMixture(n_components=2, random_state=CONFIG["seed"])
                labels = gmm.fit_predict(scores_clean)
                m0, m1 = gmm.means_[0][0], gmm.means_[1][0]
                s0 = np.sqrt(gmm.covariances_[0][0][0])
                s1 = np.sqrt(gmm.covariances_[1][0][0])
                if abs(m0 - m1) > 0.8 * np.std(scores_clean):
                    if m0 > m1:
                        t_high = np.min(scores_clean[labels == 0])
                        t_quiet = m1 + s1
                    else:
                        t_high = np.min(scores_clean[labels == 1])
                        t_quiet = m0 + s0
                    method_used = "GMM (2-Components)"
            except Exception:
                pass

        if t_high is None:
            mean_val = np.mean(scores)
            std_val = np.std(scores)
            t_high_candidate_1 = mean_val + 1.5 * std_val
            t_high_candidate_2 = np.quantile(scores, 0.95)
            t_high = max(t_high_candidate_1, t_high_candidate_2)
            t_quiet = mean_val

        if t_high < 0.15:
            t_high = np.inf

        thresholds_dict[k] = {"T_high": t_high, "T_quiet": t_quiet, "Method": method_used}
        adata_model.obs[f"Active_{k}"] = (scores >= t_high).astype(int)
        adata_model.obs[f"Quiet_{k}"] = (scores <= t_quiet).astype(int)

    # 3. low_malig_gate
    malig_col = find_malig_col(adata_model)
    low_malig_gate = max(0.0, hard_tumor_threshold - CONFIG["low_malig_offset"])
    adata_model.obs["Keep_By_LowMalig"] = (adata_model.obs[malig_col].astype(float).values >= low_malig_gate).astype(int)

    # 4. cluster 级统计富集
    M_global = adata_model.n_obs
    bg_active_counts = {k: int(adata_model.obs[f"Active_{k}"].sum()) for k in AXES}
    bg_quiet_counts = {k: int(adata_model.obs[f"Quiet_{k}"].sum()) for k in AXES}
    bg_active_fracs = {k: bg_active_counts[k] / M_global for k in AXES}
    bg_quiet_fracs = {k: bg_quiet_counts[k] / M_global for k in AXES}
    global_mean_scores = {k: adata_model.obs[f"Coupling_{k}"].mean() for k in AXES}

    native_clusters = sorted(adata_model.obs[native_cluster_key].astype(str).unique())
    pvals_active_dict = {k: [] for k in AXES}
    pvals_quiet_dict = {k: [] for k in AXES}

    for k in AXES:
        adata_model.obs[f"{k}_Cluster_Active_Frac"] = 0.0

    for c in native_clusters:
        c_mask = adata_model.obs[native_cluster_key].astype(str) == str(c)
        N_cluster = int(c_mask.sum())
        for k in AXES:
            k_obs_act = int(adata_model.obs.loc[c_mask, f"Active_{k}"].sum())
            k_obs_qui = int(adata_model.obs.loc[c_mask, f"Quiet_{k}"].sum())
            act_frac = k_obs_act / N_cluster if N_cluster > 0 else 0.0
            adata_model.obs.loc[c_mask, f"{k}_Cluster_Active_Frac"] = act_frac
            pval_a = hypergeom.sf(k_obs_act - 1, M_global, bg_active_counts[k], N_cluster) if k_obs_act > 0 else 1.0
            pval_q = hypergeom.sf(k_obs_qui - 1, M_global, bg_quiet_counts[k], N_cluster) if k_obs_qui > 0 else 1.0
            pvals_active_dict[k].append(pval_a)
            pvals_quiet_dict[k].append(pval_q)

    fdr_active_dict = {}
    for k in AXES:
        _, fdr_a, _, _ = multipletests(pvals_active_dict[k], method="fdr_bh")
        fdr_active_dict[k] = fdr_a

    cluster_annotations = {}
    cluster_stats = []
    for idx, c in enumerate(native_clusters):
        c_mask = adata_model.obs[native_cluster_key].astype(str) == str(c)
        N_cluster = int(c_mask.sum())
        sig_active_axes = []
        cold_axes = []

        for k in AXES:
            act_frac = float(adata_model.obs.loc[c_mask, f"{k}_Cluster_Active_Frac"].iloc[0]) if N_cluster > 0 else 0.0
            qui_frac = float(adata_model.obs.loc[c_mask, f"Quiet_{k}"].sum() / N_cluster) if N_cluster > 0 else 0.0
            k_obs_act = int(adata_model.obs.loc[c_mask, f"Active_{k}"].sum())
            cluster_mean_score = float(adata_model.obs.loc[c_mask, f"Coupling_{k}"].mean()) if N_cluster > 0 else 0.0
            fc_act = act_frac / (bg_active_fracs[k] + 1e-9)
            fdr_a = float(fdr_active_dict[k][idx])
            intensity_ratio = cluster_mean_score / (global_mean_scores[k] + 1e-9)

            if (fc_act >= CONFIG["MIN_FC"] and
                k_obs_act >= CONFIG["MIN_ACTIVE_SPOTS"] and
                intensity_ratio >= CONFIG["MIN_INTENSITY_RATIO"] and
                fdr_a < 0.05):
                sig_active_axes.append(k)

            if qui_frac > bg_quiet_fracs[k] or fc_act < 0.5:
                cold_axes.append(k)

        if len(sig_active_axes) == 0 and len(cold_axes) >= CONFIG["QUIESCENT_COLD_AXIS_NUM"]:
            c_label = "Quiescent (Background)"
        elif len(sig_active_axes) == 1:
            c_label = f"Single-axis: {sig_active_axes[0]}"
        elif len(sig_active_axes) >= 2:
            c_label = f"Multi-axis: {'_'.join(sig_active_axes)}"
        else:
            c_label = "Transitional"

        cluster_annotations[c] = c_label
        cluster_stats.append({
            "Cluster_ID": c,
            "Spot_Count": N_cluster,
            "Sig_Active_Axes": ",".join(sig_active_axes) if sig_active_axes else "None",
            "Cold_Axis_Count": len(cold_axes),
            "Initial_Label": c_label
        })

    adata_model.obs["Niche_Annotation_PreFilter"] = adata_model.obs[native_cluster_key].astype(str).map(cluster_annotations).astype(str)
    adata_model.obs["Niche_Annotation"] = adata_model.obs["Niche_Annotation_PreFilter"].astype(str)

    # 5. spot 级 low_malig_gate 过滤
    is_active_niche = adata_model.obs["Niche_Annotation"].str.startswith("Single-axis") | adata_model.obs["Niche_Annotation"].str.startswith("Multi-axis")
    low_malig_fail = adata_model.obs["Keep_By_LowMalig"].astype(int).values == 0
    adata_model.obs.loc[is_active_niche & low_malig_fail, "Niche_Annotation"] = "Transitional"

    # 6. 标准化成 Final_Niche_Type
    def standardize_label(x):
        x = str(x)
        if x.startswith("Single-axis:"):
            ax = x.split(":", 1)[1].strip()
            return f"TMCN_{ax}_Axis"
        if x.startswith("Multi-axis:"):
            axes = x.split(":", 1)[1].strip()
            return f"Multi_Axis_{axes}"
        if x.startswith("Quiescent"):
            return "TMCN_Quiescent"
        return "TMCN_Transitional"

    adata_model.obs["Final_Niche_Type"] = adata_model.obs["Niche_Annotation"].map(standardize_label).astype("category")

    cluster_stats_df = pd.DataFrame(cluster_stats)
    return adata_model, pd.DataFrame(thresholds_dict).T, cluster_stats_df


# =========================================================
# D. Cell 10 核心：生成 Level1 / Level2
# =========================================================
def build_level1_level2(adata_model, hard_tumor_threshold):
    malig_col = find_malig_col(adata_model)
    final_niche_str = adata_model.obs["Final_Niche_Type"].astype(str)

    is_spacet_tumor = adata_model.obs[malig_col].astype(float).values > hard_tumor_threshold
    is_quiescent = final_niche_str == "TMCN_Quiescent"
    is_transitional = final_niche_str == "TMCN_Transitional"
    is_quiescent_or_transitional = is_quiescent | is_transitional

    adata_model.obs["Level1_Macro_Region"] = "Healthy_Region"
    adata_model.obs["Level2_Micro_Niche"] = "Healthy_Region"

    tumor_mask = is_spacet_tumor | (~is_quiescent_or_transitional)
    adata_model.obs.loc[tumor_mask, "Level1_Macro_Region"] = "Tumor_Region"

    active_niche_mask = ~is_quiescent_or_transitional
    adata_model.obs.loc[active_niche_mask, "Level2_Micro_Niche"] = final_niche_str[active_niche_mask]

    target_tumor_quiescent_mask = is_spacet_tumor & is_quiescent_or_transitional
    adata_model.obs.loc[target_tumor_quiescent_mask, "Level2_Micro_Niche"] = "TMCN_Quiescent"

    adata_model.obs["Level1_Macro_Region"] = adata_model.obs["Level1_Macro_Region"].astype("category")
    adata_model.obs["Level2_Micro_Niche"] = adata_model.obs["Level2_Micro_Niche"].astype("category")
    return adata_model


# =========================================================
# E. 计算 ARI
# =========================================================
def calculate_level1_ari(adata_model):
    pathology_col = find_pathology_col(adata_model)
    patho = adata_model.obs[pathology_col].astype(str).str.strip()

    if CONFIG["gt_map"] == "prostate_dynamic":
        gt_raw = patho.map(lambda x: 1 if x == "Invasive carcinoma" else 0).astype(float).values
    else:
        gt_raw = patho.map(CONFIG["gt_map"]).astype(float).values

    pred_map = {
        "Healthy_Region": 0,
        "Tumor_Region": 1
    }
    pred_raw = adata_model.obs["Level1_Macro_Region"].astype(str).map(pred_map).astype(float).values

    valid_mask = (~np.isnan(gt_raw)) & (~np.isnan(pred_raw))
    y_true = gt_raw[valid_mask]
    y_pred = pred_raw[valid_mask]

    if len(y_true) == 0:
        raise ValueError(
            f"❌ ARI 有效 spot 数为 0。请检查病理列 {pathology_col} 和 gt_map。\n"
            f"当前病理标签: {patho.unique().tolist()}"
        )

    ari = adjusted_rand_score(y_true, y_pred)
    return float(ari), int(len(y_true)), pathology_col




# =========================================================
# Cell 3: 主循环：n_epochs × lr 参数组合热图实验
# =========================================================

def format_param_value(v):
    """把 1e-4 / 100.0 这类值转成适合文件夹名的字符串。"""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if abs(v) >= 1000 or (abs(v) > 0 and abs(v) < 0.001):
            return f"{v:.0e}".replace("+", "")
        return str(v).replace(".", "p")
    return str(v).replace(".", "p")


def run_parameter_grid_analysis():
    os.makedirs(CONFIG["save_root"], exist_ok=True)

    hard_tumor_threshold = read_gmm_threshold()
    adata_scored = compute_sender_receiver_scores()

    all_results = []

    y_values = CONFIG["n_epochs_range"]
    x_values = CONFIG["lr_range"]

    total_runs = len(y_values) * len(x_values)
    run_idx = 0

    for n_epochs in y_values:
        for lr in x_values:
            run_idx += 1
            print("\n" + "=" * 90)
            print(
                f"🚀 [{run_idx}/{total_runs}] 参数组合: "
                f"n_epochs={n_epochs}, lr={lr}"
            )
            print("=" * 90)

            set_seed(CONFIG["seed"])

            run_dir = os.path.join(
                CONFIG["save_root"],
                f"n_epochs_{format_param_value(n_epochs)}__lr_{format_param_value(lr)}"
            )
            os.makedirs(run_dir, exist_ok=True)

            adata_model, counts_key = prepare_adata_model(adata_scored)

            adata_model, model = train_nichecompass(
                adata_model=adata_model,
                counts_key=counts_key,
                n_epochs=n_epochs,
                lr=lr
            )

            adata_model, thresholds_df, cluster_stats_df = annotate_tmcn_niches(
                adata_model=adata_model,
                hard_tumor_threshold=hard_tumor_threshold
            )

            adata_model = build_level1_level2(
                adata_model=adata_model,
                hard_tumor_threshold=hard_tumor_threshold
            )

            ari, n_valid, pathology_col = calculate_level1_ari(adata_model)

            print(
                f"✅ n_epochs={n_epochs} | lr={lr} | "
                f"ARI={ari:.4f} | valid spots={n_valid}"
            )

            h5ad_path = os.path.join(
                run_dir,
                f"{CONFIG['file_prefix']}_n_epochs_{format_param_value(n_epochs)}_"
                f"lr_{format_param_value(lr)}.h5ad"
            )
            adata_model.write_h5ad(h5ad_path)

            thresholds_df.to_csv(
                os.path.join(run_dir, "TMCN_thresholds.csv"),
                encoding="utf-8-sig"
            )

            cluster_stats_df.to_csv(
                os.path.join(run_dir, "TMCN_cluster_stats.csv"),
                index=False,
                encoding="utf-8-sig"
            )

            all_results.append({
                "n_epochs": int(n_epochs),
                "lr": lr,
                "ARI": round(ari, 6),
                "valid_spots_for_ARI": n_valid,
                "pathology_column": pathology_col,
                "h5ad_path": h5ad_path
            })

            try:
                del model
            except Exception:
                pass
            del adata_model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    results_df = pd.DataFrame(all_results)

    csv_path = os.path.join(
        CONFIG["save_root"],
        "parameter_sensitivity_n_epochs_AND_lr_ARI.csv"
    )
    results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    matrix_df = results_df.pivot(
        index="n_epochs",
        columns="lr",
        values="ARI"
    )
    matrix_csv_path = os.path.join(
        CONFIG["save_root"],
        "ARI_heatmap_matrix_n_epochs_AND_lr.csv"
    )
    matrix_df.to_csv(matrix_csv_path, encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print("🎉 n_epochs × lr 参数组合分析完成！")
    print(f"✅ 长表 CSV: {csv_path}")
    print(f"✅ 热图矩阵 CSV: {matrix_csv_path}")
    print("=" * 90)

    display(results_df)
    display(matrix_df)

    return results_df, matrix_df


results_df, matrix_df = run_parameter_grid_analysis()


# =========================================================
# Cell 4: 读取结果 CSV 并绘制 n_epochs × lr 的 ARI 热图
# =========================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_path = os.path.join(
    CONFIG["save_root"],
    "parameter_sensitivity_n_epochs_AND_lr_ARI.csv"
)

results_df = pd.read_csv(csv_path)

display(results_df)

matrix_df = results_df.pivot(
    index="n_epochs",
    columns="lr",
    values="ARI"
).sort_index(axis=0).sort_index(axis=1)

fig, ax = plt.subplots(figsize=(9, 6))
im = ax.imshow(matrix_df.values, aspect="auto", cmap="YlGnBu")

ax.set_xticks(np.arange(matrix_df.shape[1]))
ax.set_yticks(np.arange(matrix_df.shape[0]))
ax.set_xticklabels([f"{x:g}" for x in matrix_df.columns])
ax.set_yticklabels([f"{int(y)}" for y in matrix_df.index])

ax.set_xlabel("lr")
ax.set_ylabel("n_epochs")
ax.set_title(
    f"ARI under different parameter combinations\n"
    f"{CONFIG['dataset_name']}"
)

for i in range(matrix_df.shape[0]):
    for j in range(matrix_df.shape[1]):
        val = matrix_df.values[i, j]
        if pd.notna(val):
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=9)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("ARI Score")

plt.tight_layout()

fig_path = os.path.join(
    CONFIG["save_root"],
    "ARI_heatmap_n_epochs_AND_lr.png"
)
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.show()

matrix_csv_path = os.path.join(
    CONFIG["save_root"],
    "ARI_heatmap_matrix_n_epochs_AND_lr.csv"
)
matrix_df.to_csv(matrix_csv_path, encoding="utf-8-sig")

print(f"✅ ARI 热图已保存至: {fig_path}")
print(f"✅ ARI 热图矩阵已保存至: {matrix_csv_path}")
