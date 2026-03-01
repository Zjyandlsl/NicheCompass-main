#!/usr/bin/env python
"""Check if required data files are present."""

import os
from pathlib import Path

# Define project root
PROJECT_ROOT = Path("/home/zhangjunyi/xiangmu/nichecompass-main")

# Required files
required_files = {
    "Gene Annotations (已有)": [
        PROJECT_ROOT / "data/gene_annotations/human_mouse_gene_orthologs.csv",
        PROJECT_ROOT / "data/gene_annotations/gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz",
    ],
    "Gene Programs (已有)": [
        PROJECT_ROOT / "data/gene_programs/metabolite_enzyme_sensor_gps/human_metabolite_enzymes.tsv",
        PROJECT_ROOT / "data/gene_programs/metabolite_enzyme_sensor_gps/human_metabolite_sensors.tsv",
        PROJECT_ROOT / "data/gene_programs/metabolite_enzyme_sensor_gps/mouse_metabolite_enzymes.tsv",
        PROJECT_ROOT / "data/gene_programs/metabolite_enzyme_sensor_gps/mouse_metabolite_sensors.tsv",
    ],
    "Spatial Omics Data (需要下载)": [
        PROJECT_ROOT / "data/spatial_omics/starmap_plus_mouse_cns_batch1.h5ad",
    ],
    "Gene Programs Network (需要下载)": [
        PROJECT_ROOT / "datasets/gp_data/omnipath_lr_network.csv",
        PROJECT_ROOT / "datasets/gp_data/nichenet_lr_network_v2_mouse.csv",
        PROJECT_ROOT / "datasets/gp_data/nichenet_ligand_target_matrix_v2_mouse.csv",
    ],
}

print("=" * 60)
print("NicheCompass 数据文件检查")
print("=" * 60)

all_present = True
for category, files in required_files.items():
    print(f"\n{category}:")
    for file_path in files:
        exists = file_path.exists()
        status = "✓ 存在" if exists else "✗ 缺失"
        print(f"  {status}: {file_path.relative_to(PROJECT_ROOT)}")
        if not exists:
            all_present = False

print("\n" + "=" * 60)
if all_present:
    print("所有必需文件已就绪！可以开始运行NicheCompass。")
else:
    print("部分文件缺失，请下载缺失的数据。")
print("=" * 60)
