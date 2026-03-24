#!/usr/bin/env python
# coding: utf-8

"""
TMCN cluster-level refinement pipeline.

This script implements a practical post-processing workflow for
Tumor Metabolic Communication Niche (TMCN) discovery on top of a trained
NicheCompass model or an AnnData object that already contains spatial
coordinates and latent embeddings.

Main ideas:
1. Load a trained NicheCompass model or a prepared h5ad file.
2. Read a custom TMCN quadruplet CSV.
3. Compute spot-level continuous scores:
   - pathway score
   - enzyme score
   - receptor score
   - sender score
   - receiver score
   - coupling score
4. Obtain / reuse latent Leiden clusters.
5. Summarize metabolic axes on cluster level and assign provisional niche labels.
6. Split internally heterogeneous clusters using spatial connected components.
7. Clean tiny isolated regions.
8. Optionally evaluate the final niches against pathology labels.

This script is intentionally explicit and beginner-friendly rather than minimal.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors

try:
    from nichecompass.models import NicheCompass
except Exception:  # pragma: no cover - only used when environment lacks package
    NicheCompass = None


# -----------------------------------------------------------------------------
# 你最需要修改的位置 1：
# 如果你的通路定义和这里不一致，请直接修改这个字典。
# key 必须和 CSV 里 Source_Pathways 列写法一致。
# value 是这个通路对应的一组代表基因，用来近似计算通路活性。
# -----------------------------------------------------------------------------
DEFAULT_PATHWAY_GENE_SETS: Dict[str, List[str]] = {
    "EGFR": ["EGFR", "ERBB2", "GRB2", "SOS1", "SHC1", "STAT3"],
    "PI3K": ["PIK3CA", "PIK3CB", "AKT1", "AKT2", "MTOR", "PTEN"],
    "Hypoxia": ["HIF1A", "VEGFA", "SLC2A1", "CA9", "ENO1", "LDHA"],
    "NFkB": ["NFKB1", "RELA", "RELB", "TNFAIP3", "CXCL8", "ICAM1"],
    "JAK-STAT": ["JAK1", "JAK2", "STAT1", "STAT3", "STAT5A", "SOCS3"],
    "TGFb": ["TGFB1", "TGFBR1", "TGFBR2", "SMAD2", "SMAD3", "SMAD4"],
    "MYC": ["MYC", "MAX", "CDK4", "ODC1", "NCL", "EIF4E"],
}


@dataclass
class AxisDefinition:
    axis_name: str
    pathway_names: List[str]
    source_genes: List[str]
    target_genes: List[str]
    meaning: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TMCN cluster refinement pipeline")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help=(
            "可选：训练好的 NicheCompass 模型目录。"
            "如果提供，将优先使用 NicheCompass.load 读取。"
        ),
    )
    parser.add_argument(
        "--adata-path",
        type=str,
        default=None,
        help=(
            "可选：直接输入 h5ad 文件路径。"
            "如果不提供 model-dir，则必须提供 adata-path。"
        ),
    )
    parser.add_argument(
        "--adata-file-name",
        type=str,
        default="adata.h5ad",
        help="当使用 --model-dir 时，模型目录中的 adata 文件名。",
    )
    parser.add_argument(
        "--quadruplet-csv",
        type=str,
        default="data/pre_data/siyuanzu/my_metabolite_network_simplify.csv",
        help="你的四元组 CSV 路径。",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/tmcn_cluster_pipeline",
        help="输出目录。",
    )
    parser.add_argument(
        "--spatial-key",
        type=str,
        default="spatial",
        help="adata.obsm 中存放空间坐标的 key。",
    )
    parser.add_argument(
        "--latent-key",
        type=str,
        default="nichecompass_latent",
        help="adata.obsm 中存放 NicheCompass latent embedding 的 key。",
    )
    parser.add_argument(
        "--cluster-key",
        type=str,
        default=None,
        help=(
            "如果 adata.obs 中已经有现成的 NicheCompass cluster 列，就填这里；"
            "不填则会根据 latent embedding 重新计算 Leiden。"
        ),
    )
    parser.add_argument(
        "--cluster-resolution",
        type=float,
        default=0.25,
        help="当需要重新计算 Leiden clustering 时使用的分辨率。",
    )
    parser.add_argument(
        "--neighbors-key",
        type=str,
        default="tmcn_latent_neighbors",
        help="重新计算邻居图时使用的 key。",
    )
    parser.add_argument(
        "--spatial-connectivities-key",
        type=str,
        default="spatial_connectivities",
        help="adata.obsp 中空间邻接图的 key；若不存在则会自动重建。",
    )
    parser.add_argument(
        "--pathology-key",
        type=str,
        default=None,
        help="病理标签列名，只用于后验评估，不参与训练。",
    )
    parser.add_argument(
        "--n-spatial-neighbors",
        type=int,
        default=6,
        help="若需要根据坐标自动构建空间图时使用的近邻数。",
    )
    parser.add_argument(
        "--spot-high-quantile",
        type=float,
        default=0.80,
        help="spot 级高分阈值的分位数。",
    )
    parser.add_argument(
        "--healthy-max-high-risk-fraction",
        type=float,
        default=0.20,
        help="cluster 被判为 Healthy niche 时允许的最大高风险 spot 比例。",
    )
    parser.add_argument(
        "--dominant-axis-min-fraction",
        type=float,
        default=0.30,
        help="cluster 命名某条主导代谢轴时，该轴高分 spot 的最小比例。",
    )
    parser.add_argument(
        "--mixed-axis-min-fraction",
        type=float,
        default=0.20,
        help="cluster 命名 Mixed niche 时，每条轴最少需要达到的高分比例。",
    )
    parser.add_argument(
        "--dominant-axis-min-margin",
        type=float,
        default=0.15,
        help="cluster 命名单主轴 niche 时，Top1 与 Top2 的最小占比差值。",
    )
    parser.add_argument(
        "--mixed-spot-min-fraction",
        type=float,
        default=0.25,
        help="cluster 命名 Mixed niche 时，至少需要有多少比例的 spots 呈现多轴并存。",
    )
    parser.add_argument(
        "--mixed-spot-min-share",
        type=float,
        default=0.25,
        help="spot 被判为 Mixed-active 时，第 2 条轴最少要占该 spot 总耦合的比例。",
    )
    parser.add_argument(
        "--quiescent-min-fraction",
        type=float,
        default=0.50,
        help="cluster 命名 Quiescent niche 时，低耦合 / 背景 spot 的最小比例。",
    )
    parser.add_argument(
        "--merge-quiescent-into-healthy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否将 Quiescent niche 直接并入 Healthy_niche 输出。",
    )
    parser.add_argument(
        "--min-region-size",
        type=int,
        default=20,
        help="最终生态位连通区域的最小 spot 数，小于该值会被合并。",
    )
    parser.add_argument(
        "--plot-figures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否保存空间可视化图。",
    )
    return parser.parse_args()


def _ensure_output_dir(path: str) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _parse_csv_list(value: object) -> List[str]:
    if pd.isna(value):
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def load_axis_definitions(csv_path: str) -> List[AxisDefinition]:
    df = pd.read_csv(csv_path)
    required_cols = {
        "TMCN_Name",
        "Source_Pathways",
        "Source_Genes",
        "Target_Genes",
        "Biologic_Meaning",
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"四元组 CSV 缺少列: {sorted(missing_cols)}")

    axes = []
    for _, row in df.iterrows():
        axes.append(
            AxisDefinition(
                axis_name=str(row["TMCN_Name"]).strip(),
                pathway_names=_parse_csv_list(row["Source_Pathways"]),
                source_genes=_parse_csv_list(row["Source_Genes"]),
                target_genes=_parse_csv_list(row["Target_Genes"]),
                meaning=str(row["Biologic_Meaning"]).strip(),
            )
        )
    return axes


def load_adata(args: argparse.Namespace) -> ad.AnnData:
    if args.model_dir:
        if NicheCompass is None:
            raise ImportError(
                "当前环境无法导入 nichecompass.models.NicheCompass，"
                "请改用 --adata-path 直接读取 h5ad。"
            )
        model = NicheCompass.load(
            dir_path=args.model_dir,
            adata=None,
            adata_file_name=args.adata_file_name,
            gp_names_key="nichecompass_gp_names",
        )
        return model.adata

    if args.adata_path is None:
        raise ValueError("必须提供 --model-dir 或 --adata-path 其中之一。")

    return ad.read_h5ad(args.adata_path)


def _match_var_names(adata: ad.AnnData, genes: Sequence[str]) -> List[str]:
    if not genes:
        return []

    var_name_map = {str(g).upper(): str(g) for g in adata.var_names}
    matched = []
    for gene in genes:
        if gene.upper() in var_name_map:
            matched.append(var_name_map[gene.upper()])
    return matched


def _mean_expression(adata: ad.AnnData, genes: Sequence[str]) -> np.ndarray:
    matched_genes = _match_var_names(adata, genes)
    if not matched_genes:
        return np.zeros(adata.n_obs, dtype=float)

    matrix = adata[:, matched_genes].X
    if sp.issparse(matrix):
        values = np.asarray(matrix.mean(axis=1)).reshape(-1)
    else:
        values = np.asarray(matrix).mean(axis=1)
    return np.asarray(values, dtype=float)


def _robust_minmax_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if np.allclose(values, values[0]):
        return np.zeros_like(values)

    lower = np.nanquantile(values, 0.01)
    upper = np.nanquantile(values, 0.99)
    if np.isclose(lower, upper):
        lower = np.nanmin(values)
        upper = np.nanmax(values)
    scaled = (values - lower) / (upper - lower + 1e-8)
    return np.clip(scaled, 0.0, 1.0)


def _pathway_score_from_names(
    adata: ad.AnnData,
    pathway_names: Sequence[str],
    pathway_gene_sets: Dict[str, List[str]],
) -> np.ndarray:
    pathway_scores = []
    for pathway_name in pathway_names:
        genes = pathway_gene_sets.get(pathway_name, [])
        if genes:
            pathway_scores.append(_mean_expression(adata, genes))
    if not pathway_scores:
        return np.zeros(adata.n_obs, dtype=float)
    stacked = np.vstack(pathway_scores)
    return stacked.mean(axis=0)


def ensure_spatial_graph(
    adata: ad.AnnData,
    spatial_key: str,
    spatial_connectivities_key: str,
    n_neighbors: int,
) -> None:
    if spatial_connectivities_key in adata.obsp:
        return

    if spatial_key not in adata.obsm:
        raise KeyError(
            f"未找到 adata.obsm['{spatial_key}']，无法构建空间邻接图。"
        )

    coords = np.asarray(adata.obsm[spatial_key])
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean")
    nn.fit(coords)
    indices = nn.kneighbors(coords, return_distance=False)

    rows = []
    cols = []
    data = []
    for i in range(indices.shape[0]):
        for j in indices[i, 1:]:
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([1.0, 1.0])

    graph = sp.csr_matrix((data, (rows, cols)), shape=(adata.n_obs, adata.n_obs))
    graph = graph.maximum(graph.T)
    adata.obsp[spatial_connectivities_key] = graph


def ensure_latent_clusters(
    adata: ad.AnnData,
    latent_key: str,
    cluster_key: Optional[str],
    neighbors_key: str,
    cluster_resolution: float,
) -> str:
    if cluster_key and cluster_key in adata.obs.columns:
        return cluster_key

    if latent_key not in adata.obsm:
        raise KeyError(
            f"未找到 adata.obsm['{latent_key}']，无法根据 latent embedding 聚类。"
        )

    sc.pp.neighbors(adata, use_rep=latent_key, key_added=neighbors_key)
    cluster_key = f"latent_leiden_{cluster_resolution}"
    sc.tl.leiden(
        adata,
        resolution=cluster_resolution,
        key_added=cluster_key,
        neighbors_key=neighbors_key,
    )
    return cluster_key


def compute_axis_scores(
    adata: ad.AnnData,
    axes: Sequence[AxisDefinition],
    pathway_gene_sets: Dict[str, List[str]],
) -> List[str]:
    axis_names = []

    for axis in axes:
        axis_names.append(axis.axis_name)
        pathway_raw = _pathway_score_from_names(
            adata=adata,
            pathway_names=axis.pathway_names,
            pathway_gene_sets=pathway_gene_sets,
        )
        enzyme_raw = _mean_expression(adata, axis.source_genes)
        receptor_raw = _mean_expression(adata, axis.target_genes)

        pathway_score = _robust_minmax_scale(pathway_raw)
        enzyme_score = _robust_minmax_scale(enzyme_raw)
        receptor_score = _robust_minmax_scale(receptor_raw)

        sender_score = pathway_score * enzyme_score
        receiver_score = receptor_score
        coupling_score = sender_score * receiver_score

        prefix = axis.axis_name
        adata.obs[f"{prefix}__pathway_score"] = pathway_score
        adata.obs[f"{prefix}__enzyme_score"] = enzyme_score
        adata.obs[f"{prefix}__receptor_score"] = receptor_score
        adata.obs[f"{prefix}__sender_score"] = sender_score
        adata.obs[f"{prefix}__receiver_score"] = receiver_score
        adata.obs[f"{prefix}__coupling_score"] = coupling_score

    return axis_names


def assign_spot_level_flags(
    adata: ad.AnnData,
    axis_names: Sequence[str],
    high_quantile: float,
    mixed_spot_min_share: float = 0.25,
) -> None:
    active_frame = {}
    dominant_scores = []
    for axis_name in axis_names:
        sender_col = f"{axis_name}__sender_score"
        receiver_col = f"{axis_name}__receiver_score"
        coupling_col = f"{axis_name}__coupling_score"

        sender_thresh = adata.obs[sender_col].quantile(high_quantile)
        receiver_thresh = adata.obs[receiver_col].quantile(high_quantile)
        coupling_thresh = adata.obs[coupling_col].quantile(high_quantile)

        active = (
            (adata.obs[sender_col] >= sender_thresh)
            & (
                (adata.obs[receiver_col] >= receiver_thresh)
                | (adata.obs[coupling_col] >= coupling_thresh)
            )
        )
        adata.obs[f"{axis_name}__active_spot"] = active.astype(int)
        active_frame[axis_name] = active.astype(int).to_numpy()
        coupling_values = adata.obs[coupling_col].to_numpy()
        dominant_scores.append(coupling_values)

    dominant_scores = np.vstack(dominant_scores).T
    max_idx = dominant_scores.argmax(axis=1)
    max_values = dominant_scores.max(axis=1)
    background_thresh = np.quantile(max_values, high_quantile * 0.5)

    dominant_axis = np.array([axis_names[idx] for idx in max_idx], dtype=object)
    dominant_axis[max_values < background_thresh] = "Background"
    adata.obs["tmcn_dominant_axis"] = pd.Categorical(dominant_axis)
    adata.obs["tmcn_high_risk_spot"] = (
        pd.DataFrame(active_frame).sum(axis=1).to_numpy() > 0
    ).astype(int)
    adata.obs["tmcn_total_coupling_score"] = dominant_scores.sum(axis=1)
    adata.obs["tmcn_max_coupling_score"] = max_values
    adata.obs["tmcn_quiescent_spot"] = (max_values < background_thresh).astype(int)

    total_coupling = dominant_scores.sum(axis=1, keepdims=True)
    share_scores = np.divide(
        dominant_scores,
        total_coupling,
        out=np.zeros_like(dominant_scores),
        where=total_coupling > 0,
    )
    share_df = pd.DataFrame(
        share_scores,
        index=adata.obs_names,
        columns=[f"{axis_name}__coupling_share" for axis_name in axis_names],
    )
    for column in share_df.columns:
        adata.obs[column] = share_df[column].to_numpy()

    sorted_idx = np.argsort(-share_scores, axis=1)
    top1_idx = sorted_idx[:, 0]
    top2_idx = sorted_idx[:, 1] if len(axis_names) > 1 else sorted_idx[:, 0]
    top1_share = share_scores[np.arange(adata.n_obs), top1_idx]
    top2_share = share_scores[np.arange(adata.n_obs), top2_idx]
    adata.obs["tmcn_top1_axis"] = pd.Categorical([axis_names[idx] for idx in top1_idx])
    adata.obs["tmcn_top2_axis"] = pd.Categorical([axis_names[idx] for idx in top2_idx])
    adata.obs["tmcn_top1_share"] = top1_share
    adata.obs["tmcn_top2_share"] = top2_share
    adata.obs["tmcn_dominance_margin"] = top1_share - top2_share
    adata.obs["tmcn_active_axis_count"] = (share_scores >= mixed_spot_min_share).sum(axis=1)
    adata.obs["tmcn_mixed_spot"] = (
        (top2_share >= mixed_spot_min_share) & (max_values >= background_thresh)
    ).astype(int)


def summarize_clusters(
    adata: ad.AnnData,
    cluster_key: str,
    axis_names: Sequence[str],
) -> pd.DataFrame:
    summaries = []
    grouped = adata.obs.groupby(cluster_key, observed=True)

    for cluster_id, df_cluster in grouped:
        record = {
            "cluster_id": cluster_id,
            "n_spots": len(df_cluster),
            "high_risk_fraction": float(df_cluster["tmcn_high_risk_spot"].mean()),
            "quiescent_fraction": float(df_cluster["tmcn_quiescent_spot"].mean()),
            "mixed_spot_fraction": float(df_cluster["tmcn_mixed_spot"].mean()),
            "mean_active_axis_count": float(df_cluster["tmcn_active_axis_count"].mean()),
            "mean_total_coupling": float(df_cluster["tmcn_total_coupling_score"].mean()),
            "mean_max_coupling": float(df_cluster["tmcn_max_coupling_score"].mean()),
            "mean_spot_dominance_margin": float(df_cluster["tmcn_dominance_margin"].mean()),
        }

        coupling_means = []

        for axis_name in axis_names:
            record[f"{axis_name}__active_fraction"] = float(
                df_cluster[f"{axis_name}__active_spot"].mean()
            )
            record[f"{axis_name}__mean_sender"] = float(
                df_cluster[f"{axis_name}__sender_score"].mean()
            )
            record[f"{axis_name}__mean_receiver"] = float(
                df_cluster[f"{axis_name}__receiver_score"].mean()
            )
            record[f"{axis_name}__mean_coupling"] = float(
                df_cluster[f"{axis_name}__coupling_score"].mean()
            )
            coupling_means.append(record[f"{axis_name}__mean_coupling"])

        total_cluster_coupling = float(np.sum(coupling_means))
        if total_cluster_coupling > 0:
            axis_shares = {
                axis_name: record[f"{axis_name}__mean_coupling"] / total_cluster_coupling
                for axis_name in axis_names
            }
        else:
            axis_shares = {axis_name: 0.0 for axis_name in axis_names}

        ranked_axes = sorted(axis_shares.items(), key=lambda item: item[1], reverse=True)
        top1_axis, top1_fraction = ranked_axes[0]
        if len(ranked_axes) > 1:
            top2_axis, top2_fraction = ranked_axes[1]
        else:
            top2_axis, top2_fraction = top1_axis, 0.0

        record["Top1_Axis"] = top1_axis
        record["Top2_Axis"] = top2_axis
        record["Top1_Fraction"] = float(top1_fraction)
        record["Top2_Fraction"] = float(top2_fraction)
        record["Dominance_Margin"] = float(top1_fraction - top2_fraction)

        for axis_name, axis_share in axis_shares.items():
            record[f"{axis_name}__cluster_coupling_fraction"] = float(axis_share)

        summaries.append(record)

    return pd.DataFrame(summaries).sort_values("cluster_id")


def classify_cluster_niche(
    cluster_row: pd.Series,
    axis_names: Sequence[str],
    healthy_max_high_risk_fraction: float,
    dominant_axis_min_fraction: float,
    mixed_axis_min_fraction: float,
    dominant_axis_min_margin: float,
    mixed_spot_min_fraction: float,
    quiescent_min_fraction: float,
    merge_quiescent_into_healthy: bool = True,
) -> str:
    high_risk_fraction = cluster_row["high_risk_fraction"]
    quiescent_fraction = cluster_row["quiescent_fraction"]
    mixed_spot_fraction = cluster_row["mixed_spot_fraction"]
    top1_axis = cluster_row["Top1_Axis"]
    top2_axis = cluster_row["Top2_Axis"]
    top1_fraction = cluster_row["Top1_Fraction"]
    top2_fraction = cluster_row["Top2_Fraction"]
    dominance_margin = cluster_row["Dominance_Margin"]

    if quiescent_fraction >= quiescent_min_fraction:
        if merge_quiescent_into_healthy:
            return "Healthy_niche"
        return "Quiescent_niche"

    if high_risk_fraction <= healthy_max_high_risk_fraction:
        return "Healthy_niche"

    eligible_mixed_axes = []
    for axis_name in axis_names:
        active_fraction = cluster_row[f"{axis_name}__active_fraction"]
        coupling_fraction = cluster_row[f"{axis_name}__cluster_coupling_fraction"]

        if (
            active_fraction >= mixed_axis_min_fraction
            or coupling_fraction >= mixed_axis_min_fraction
        ):
            eligible_mixed_axes.append(axis_name)

    if (
        len(eligible_mixed_axes) >= 2
        and mixed_spot_fraction >= mixed_spot_min_fraction
        and top2_fraction >= mixed_axis_min_fraction
    ):
        candidate_axes = []
        for axis_name in (top1_axis, top2_axis):
            if axis_name in axis_names and axis_name not in candidate_axes:
                candidate_axes.append(axis_name)
        if len(candidate_axes) < 2:
            candidate_axes = eligible_mixed_axes[:2]
        readable_axes = [
            axis_name.replace("TMCN_", "").replace("_Axis", "")
            for axis_name in candidate_axes
        ]
        readable_axes = sorted(readable_axes)
        return "TMCN_Mixed_" + "_".join(readable_axes) + "_Axis"

    if (
        top1_axis in axis_names
        and top1_fraction >= dominant_axis_min_fraction
        and dominance_margin >= dominant_axis_min_margin
    ):
        return top1_axis

    return "Transitional_niche"


def assign_cluster_labels(
    adata: ad.AnnData,
    cluster_key: str,
    axis_names: Sequence[str],
    healthy_max_high_risk_fraction: float,
    dominant_axis_min_fraction: float,
    mixed_axis_min_fraction: float,
    dominant_axis_min_margin: float = 0.15,
    mixed_spot_min_fraction: float = 0.25,
    quiescent_min_fraction: float = 0.50,
    merge_quiescent_into_healthy: bool = True,
) -> pd.DataFrame:
    cluster_summary = summarize_clusters(adata, cluster_key, axis_names)
    cluster_summary["tmcn_is_quiescent_pattern"] = (
        cluster_summary["quiescent_fraction"] >= quiescent_min_fraction
    ).astype(int)
    cluster_summary["tmcn_cluster_label"] = cluster_summary.apply(
        classify_cluster_niche,
        axis=1,
        axis_names=axis_names,
        healthy_max_high_risk_fraction=healthy_max_high_risk_fraction,
        dominant_axis_min_fraction=dominant_axis_min_fraction,
        mixed_axis_min_fraction=mixed_axis_min_fraction,
        dominant_axis_min_margin=dominant_axis_min_margin,
        mixed_spot_min_fraction=mixed_spot_min_fraction,
        quiescent_min_fraction=quiescent_min_fraction,
        merge_quiescent_into_healthy=merge_quiescent_into_healthy,
    )
    label_map = cluster_summary.set_index("cluster_id")["tmcn_cluster_label"].to_dict()
    adata.obs["tmcn_cluster_label"] = adata.obs[cluster_key].map(label_map).astype("category")
    return cluster_summary


def _neighbor_majority_label(
    graph: sp.csr_matrix,
    idx: int,
    labels: np.ndarray,
    current_label: str,
) -> Optional[str]:
    neighbor_indices = graph[idx].indices
    if len(neighbor_indices) == 0:
        return None
    neighbor_labels = [labels[n] for n in neighbor_indices if labels[n] != current_label]
    if not neighbor_labels:
        return None
    return pd.Series(neighbor_labels).value_counts().index[0]


def refine_to_connected_regions(
    adata: ad.AnnData,
    cluster_key: str,
    spatial_connectivities_key: str,
    min_region_size: int,
) -> None:
    graph = adata.obsp[spatial_connectivities_key].tocsr()
    cluster_labels = adata.obs[cluster_key].astype(str).to_numpy()
    niche_labels = adata.obs["tmcn_cluster_label"].astype(str).to_numpy()

    region_labels = np.array([""] * adata.n_obs, dtype=object)
    region_counter = 0

    for cluster_id in pd.unique(cluster_labels):
        cluster_mask = cluster_labels == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_niche_labels = pd.unique(niche_labels[cluster_mask])

        for niche_label in cluster_niche_labels:
            sub_mask = cluster_mask & (niche_labels == niche_label)
            sub_indices = np.where(sub_mask)[0]
            if len(sub_indices) == 0:
                continue

            subgraph = graph[sub_indices][:, sub_indices]
            n_components, component_ids = connected_components(
                subgraph, directed=False, connection="weak"
            )

            for comp_id in range(n_components):
                comp_indices = sub_indices[component_ids == comp_id]
                region_counter += 1
                region_name = f"{niche_label}__R{region_counter}"
                region_labels[comp_indices] = region_name

    # merge tiny regions
    for region_name, region_size in pd.Series(region_labels).value_counts().items():
        if region_size >= min_region_size:
            continue
        idxs = np.where(region_labels == region_name)[0]
        base_label = region_name.split("__R")[0]
        for idx in idxs:
            replacement = _neighbor_majority_label(
                graph=graph,
                idx=idx,
                labels=region_labels,
                current_label=region_name,
            )
            if replacement is None:
                replacement = base_label
            region_labels[idx] = replacement

    adata.obs["tmcn_region_label"] = pd.Categorical(region_labels)
    adata.obs["tmcn_region_base_label"] = pd.Categorical(
        [label.split("__R")[0] for label in region_labels]
    )


def compute_pathology_evaluation(
    adata: ad.AnnData,
    pathology_key: Optional[str],
    output_dir: Path,
) -> None:
    if pathology_key is None or pathology_key not in adata.obs.columns:
        return

    crosstab = pd.crosstab(
        adata.obs["tmcn_region_base_label"],
        adata.obs[pathology_key],
        normalize="index",
    )
    crosstab.to_csv(output_dir / "tmcn_vs_pathology_normalized_crosstab.csv")

    raw_counts = pd.crosstab(
        adata.obs["tmcn_region_base_label"],
        adata.obs[pathology_key],
    )
    raw_counts.to_csv(output_dir / "tmcn_vs_pathology_raw_counts.csv")


def save_axis_metadata(
    axes: Sequence[AxisDefinition],
    output_dir: Path,
) -> None:
    rows = []
    for axis in axes:
        rows.append(
            {
                "axis_name": axis.axis_name,
                "source_pathways": ", ".join(axis.pathway_names),
                "source_genes": ", ".join(axis.source_genes),
                "target_genes": ", ".join(axis.target_genes),
                "meaning": axis.meaning,
            }
        )
    pd.DataFrame(rows).to_csv(output_dir / "tmcn_axis_definitions_used.csv", index=False)


def save_parameters(args: argparse.Namespace, output_dir: Path) -> None:
    with open(output_dir / "tmcn_pipeline_parameters.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)


def make_spatial_plots(
    adata: ad.AnnData,
    spatial_key: str,
    output_dir: Path,
    pathology_key: Optional[str],
) -> None:
    if spatial_key not in adata.obsm:
        return

    sc.pl.spatial(
        adata,
        color=["tmcn_cluster_label"],
        spot_size=1.0,
        show=False,
    )
    import matplotlib.pyplot as plt

    plt.savefig(output_dir / "tmcn_cluster_labels_spatial.png", dpi=200, bbox_inches="tight")
    plt.close()

    sc.pl.spatial(
        adata,
        color=["tmcn_region_base_label"],
        spot_size=1.0,
        show=False,
    )
    plt.savefig(output_dir / "tmcn_region_labels_spatial.png", dpi=200, bbox_inches="tight")
    plt.close()

    if pathology_key and pathology_key in adata.obs.columns:
        sc.pl.spatial(
            adata,
            color=[pathology_key],
            spot_size=1.0,
            show=False,
        )
        plt.savefig(output_dir / "pathology_labels_spatial.png", dpi=200, bbox_inches="tight")
        plt.close()


def main() -> None:
    args = parse_args()
    output_dir = _ensure_output_dir(args.output_dir)

    print("Step 1/10: 读取四元组定义...")
    axes = load_axis_definitions(args.quadruplet_csv)
    save_axis_metadata(axes, output_dir)

    print("Step 2/10: 读取模型/AnnData...")
    adata = load_adata(args)

    print("Step 3/10: 确保空间图存在...")
    ensure_spatial_graph(
        adata=adata,
        spatial_key=args.spatial_key,
        spatial_connectivities_key=args.spatial_connectivities_key,
        n_neighbors=args.n_spatial_neighbors,
    )

    print("Step 4/10: 计算 spot 级连续功能分数...")
    axis_names = compute_axis_scores(
        adata=adata,
        axes=axes,
        pathway_gene_sets=DEFAULT_PATHWAY_GENE_SETS,
    )
    assign_spot_level_flags(
        adata=adata,
        axis_names=axis_names,
        high_quantile=args.spot_high_quantile,
        mixed_spot_min_share=args.mixed_spot_min_share,
    )

    print("Step 5/10: 获取 / 计算 NicheCompass 小 cluster...")
    cluster_key = ensure_latent_clusters(
        adata=adata,
        latent_key=args.latent_key,
        cluster_key=args.cluster_key,
        neighbors_key=args.neighbors_key,
        cluster_resolution=args.cluster_resolution,
    )
    print(f"使用 cluster 列: {cluster_key}")

    print("Step 6/10: cluster 级别命名生态位...")
    cluster_summary = assign_cluster_labels(
        adata=adata,
        cluster_key=cluster_key,
        axis_names=axis_names,
        healthy_max_high_risk_fraction=args.healthy_max_high_risk_fraction,
        dominant_axis_min_fraction=args.dominant_axis_min_fraction,
        mixed_axis_min_fraction=args.mixed_axis_min_fraction,
        dominant_axis_min_margin=args.dominant_axis_min_margin,
        mixed_spot_min_fraction=args.mixed_spot_min_fraction,
        quiescent_min_fraction=args.quiescent_min_fraction,
        merge_quiescent_into_healthy=args.merge_quiescent_into_healthy,
    )
    cluster_summary.to_csv(output_dir / "tmcn_cluster_summary.csv", index=False)

    print("Step 7/10: 对内部异质 cluster 做空间连通拆分...")
    refine_to_connected_regions(
        adata=adata,
        cluster_key=cluster_key,
        spatial_connectivities_key=args.spatial_connectivities_key,
        min_region_size=args.min_region_size,
    )

    print("Step 8/10: 保存 pathology 后验评估结果...")
    compute_pathology_evaluation(
        adata=adata,
        pathology_key=args.pathology_key,
        output_dir=output_dir,
    )

    print("Step 9/10: 保存结果对象与 spot 级表格...")
    spot_columns = [
        cluster_key,
        "tmcn_dominant_axis",
        "tmcn_high_risk_spot",
        "tmcn_quiescent_spot",
        "tmcn_mixed_spot",
        "tmcn_total_coupling_score",
        "tmcn_max_coupling_score",
        "tmcn_top1_axis",
        "tmcn_top2_axis",
        "tmcn_top1_share",
        "tmcn_top2_share",
        "tmcn_dominance_margin",
        "tmcn_active_axis_count",
        "tmcn_cluster_label",
        "tmcn_region_label",
        "tmcn_region_base_label",
    ]
    for axis_name in axis_names:
        spot_columns.extend(
            [
                f"{axis_name}__pathway_score",
                f"{axis_name}__enzyme_score",
                f"{axis_name}__receptor_score",
                f"{axis_name}__sender_score",
                f"{axis_name}__receiver_score",
                f"{axis_name}__coupling_score",
                f"{axis_name}__coupling_share",
                f"{axis_name}__active_spot",
            ]
        )
    if args.pathology_key and args.pathology_key in adata.obs.columns:
        spot_columns.append(args.pathology_key)

    adata.obs[spot_columns].to_csv(output_dir / "tmcn_spot_level_results.csv")
    adata.write_h5ad(output_dir / "tmcn_pipeline_result.h5ad")

    print("Step 10/10: 可视化与参数落盘...")
    save_parameters(args, output_dir)
    if args.plot_figures:
        make_spatial_plots(
            adata=adata,
            spatial_key=args.spatial_key,
            output_dir=output_dir,
            pathology_key=args.pathology_key,
        )

    print("TMCN pipeline 已完成。")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
