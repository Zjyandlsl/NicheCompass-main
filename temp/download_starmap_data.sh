#!/bin/bash
# Multiple methods to download STARmap PLUS Mouse CNS data
# File: starmap_plus_mouse_cns_batch1.h5ad

echo "============================================================"
echo "STARmap PLUS Mouse CNS 数据下载 - 多种方法"
echo "============================================================"

# Create directory
mkdir -p data/spatial_omics

# Method 1: Using gdown with resume capability
echo ""
echo "方法1: 使用 gdown (支持断点续传)"
echo "------------------------------------------------------------"
echo "如果超时，脚本会自动重试"
echo ""

for i in {1..3}; do
    echo "尝试 $i/3..."
    if gdown --continue "https://drive.google.com/uc?id=1MOjIyue7a-JDAcnAseqIljDyoO7KtH99" \
        -O data/spatial_omics/starmap_plus_mouse_cns_batch1.h5ad; then
        echo "✓ 下载成功！"
        exit 0
    else
        echo "✗ 下载失败，重试中..."
        sleep 2
    fi
done

# Method 2: Using wget with alternative mirror
echo ""
echo "方法2: 使用 wget 从备用源"
echo "------------------------------------------------------------"
echo "如果Google Drive无法访问，尝试备用源"
echo ""

# Alternative: Zenodo or FigShare links (if available)
# You may need to search for alternative hosting

# Method 3: Generate from raw data (last resort)
echo ""
echo "方法3: 从原始数据生成 (需要额外步骤)"
echo "------------------------------------------------------------"
echo "如果自动下载失败，可以从原始STARmap数据生成"
echo ""
echo "原始数据源: https://singlecell.broadinstitute.org/single_cell/study/SCP1830"
echo ""
echo "或使用Squidpy API加载预处理的版本:"
echo "  import squidpy as sq"
echo "  adata = sq.datasets.starmap_plus_mouse_cns()"
echo ""

# Method 4: Request data directly
echo ""
echo "方法4: 从论文作者获取"
echo "------------------------------------------------------------"
echo "联系论文作者获取数据访问权限"
echo ""

# Final check
echo "============================================================"
if [ -f "data/spatial_omics/starmap_plus_mouse_cns_batch1.h5ad" ]; then
    echo "✓ 文件已存在"
    ls -lh data/spatial_omics/starmap_plus_mouse_cns_batch1.h5ad
else
    echo "✗ 文件不存在，请尝试以下方法："
    echo ""
    echo "1. 使用代理/VPN后重试 gdown"
    echo "2. 浏览器下载: https://drive.google.com/file/d/1MOjIyue7a-JDAcnAseqIljDyoO7KtH99/view"
    echo "3. 使用Squidpy API加载 (见下方的Python脚本)"
fi
echo "============================================================"
