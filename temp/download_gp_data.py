#!/usr/bin/env python
"""
Download gene program network data for NicheCompass.
This script will download required network files from their original sources.
"""

import os
import sys
import requests
import pandas as pd
from pathlib import Path
import time

# Define paths
PROJECT_ROOT = Path("/home/zhangjunyi/xiangmu/nichecompass-main")
GP_DATA_FOLDER = PROJECT_ROOT / "datasets/gp_data"
GA_DATA_FOLDER = PROJECT_ROOT / "data/gene_annotations"

# Create directories
GP_DATA_FOLDER.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("NicheCompass 基因程序网络数据下载")
print("=" * 60)

def download_file(url, dest_path, description=""):
    """Download file with progress bar."""
    try:
        print(f"\n正在下载: {description}")
        print(f"URL: {url}")
        print(f"保存到: {dest_path}")

        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0

        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r进度: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')

        print(f"\n✓ 下载完成: {dest_path}")
        return True

    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        return False

# ============================================================================
# Method 1: Download from original sources
# ============================================================================

print("\n方法1: 从原始源下载")
print("-" * 60)

success_count = 0

# 1. OmniPath LR network (from OmniPath)
# Using omnipath R package data export
omnipath_url = "https://omnipathdb.org/interactions?datasets=omnipath&format=tsv"
if download_file(
    omnipath_url,
    GP_DATA_FOLDER / "omnipath_lr_network.csv",
    "OmniPath Ligand-Receptor 网络"
):
    # Convert TSV to CSV and process
    try:
        df = pd.read_csv(GP_DATA_FOLDER / "omnipath_lr_network.csv", sep='\t')
        # Filter for ligand-receptor interactions
        if 'source' in df.columns and 'target' in df.columns:
            df.to_csv(GP_DATA_FOLDER / "omnipath_lr_network.csv", index=False)
            print("✓ OmniPath数据处理完成")
            success_count += 1
    except Exception as e:
        print(f"⚠ OmniPath数据处理跳过: {e}")

# 2. NicheNet networks (from GitHub releases)
# NicheNet v2 data
nichenet_base_url = "https://github.com/aertslab/NicheNet/releases/download/v2.0.0"

# Download mouse LR network
nichenet_lr_mouse_url = f"{nichenet_base_url}/mouse_nicheNet_lr_network.rds"
# Note: RDS files need R or specific Python libraries to read
# We'll use preprocessed CSV if available from alternative sources

# ============================================================================
# Method 2: Use NicheCompass built-in functions (RECOMMENDED)
# ============================================================================

print("\n" + "=" * 60)
print("方法2: 使用NicheCompass内置函数下载 (推荐)")
print("-" * 60)

print("\n正在使用NicheCompass API下载基因程序数据...")
print("这将从OmniPath和其他数据库自动获取最新数据\n")

try:
    from nichecompass.utils import (
        extract_gp_dict_from_omnipath_lr_interactions,
        extract_gp_dict_from_nichenet_lrt_interactions
    )

    # Download OmniPath data for mouse
    print("1. 下载OmniPath配体-受体网络 (小鼠)...")
    extract_gp_dict_from_omnipath_lr_interactions(
        species="mouse",
        load_from_disk=False,
        save_to_disk=True,
        lr_network_file_path=str(GP_DATA_FOLDER / "omnipath_lr_network.csv"),
        gene_orthologs_mapping_file_path=str(GA_DATA_FOLDER / "human_mouse_gene_orthologs.csv"),
        plot_gp_gene_count_distributions=False,
    )
    print("✓ OmniPath网络下载完成")
    success_count += 1

    # Download NicheNet data for mouse
    print("\n2. 下载NicheNet网络 (小鼠v2)...")
    extract_gp_dict_from_nichenet_lrt_interactions(
        species="mouse",
        version="v2",
        keep_target_genes_ratio=1.0,
        max_n_target_genes_per_gp=250,
        load_from_disk=False,
        save_to_disk=True,
        lr_network_file_path=str(GP_DATA_FOLDER / "nichenet_lr_network_v2_mouse.csv"),
        ligand_target_matrix_file_path=str(GP_DATA_FOLDER / "nichenet_ligand_target_matrix_v2_mouse.csv"),
        gene_orthologs_mapping_file_path=str(GA_DATA_FOLDER / "human_mouse_gene_orthologs.csv"),
        plot_gp_gene_count_distributions=False,
    )
    print("✓ NicheNet网络下载完成")
    success_count += 1

    print("\n3. 下载OmniPath配体-受体网络 (人)...")
    extract_gp_dict_from_omnipath_lr_interactions(
        species="human",
        load_from_disk=False,
        save_to_disk=True,
        lr_network_file_path=str(GP_DATA_FOLDER / "omnipath_lr_network_human.csv"),
        gene_orthologs_mapping_file_path=str(GA_DATA_FOLDER / "human_mouse_gene_orthologs.csv"),
        plot_gp_gene_count_distributions=False,
    )
    print("✓ OmniPath网络(人)下载完成")
    success_count += 1

    print("\n4. 下载NicheNet网络 (人v2)...")
    extract_gp_dict_from_nichenet_lrt_interactions(
        species="human",
        version="v2",
        keep_target_genes_ratio=1.0,
        max_n_target_genes_per_gp=250,
        load_from_disk=False,
        save_to_disk=True,
        lr_network_file_path=str(GP_DATA_FOLDER / "nichenet_lr_network_v2_human.csv"),
        ligand_target_matrix_file_path=str(GP_DATA_FOLDER / "nichenet_ligand_target_matrix_v2_human.csv"),
        gene_orthologs_mapping_file_path=str(GA_DATA_FOLDER / "human_mouse_gene_orthologs.csv"),
        plot_gp_gene_count_distributions=False,
    )
    print("✓ NicheNet网络(人)下载完成")
    success_count += 1

except ImportError as e:
    print(f"⚠ NicheCompass未安装或导入失败: {e}")
    print("请先安装: pip install nichecompass")
except Exception as e:
    print(f"⚠ 使用NicheCompass API下载失败: {e}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 60)
print("下载摘要")
print("=" * 60)

files_to_check = [
    (GP_DATA_FOLDER / "omnipath_lr_network.csv", "OmniPath LR网络 (小鼠)"),
    (GP_DATA_FOLDER / "nichenet_lr_network_v2_mouse.csv", "NicheNet LR网络 (小鼠v2)"),
    (GP_DATA_FOLDER / "nichenet_ligand_target_matrix_v2_mouse.csv", "NicheNet配体-靶基因矩阵 (小鼠v2)"),
]

all_exist = True
for file_path, description in files_to_check:
    exists = "✓ 存在" if file_path.exists() else "✗ 缺失"
    print(f"{exists}: {description}")
    if not file_path.exists():
        all_exist = False

print("\n" + "=" * 60)
if all_exist:
    print("✓ 所有基因程序网络数据下载完成！")
    print("现在可以运行NicheCompass教程了。")
else:
    print("⚠ 部分文件下载失败，请检查网络连接或手动下载。")
print("=" * 60)
