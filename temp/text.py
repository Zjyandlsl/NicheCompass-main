import torch
import jax
import pandas as pd
import numpy as np
import nichecompass_2
import torch_geometric
import scanpy

print("\n====== 最终环境体检报告 ======")
print(f"1. NicheCompass Version: {nichecompass_2.__version__}")
print(f"2. PyTorch Version: {torch.__version__} (CUDA Available: {torch.cuda.is_available()})")
if torch.cuda.is_available():
    print(f"   - Device: {torch.cuda.get_device_name(0)}")

print(f"3. JAX Version: {jax.__version__}")
try:
    print(f"   - JAX Devices: {jax.devices()}")
except:
    print("   - [警告] JAX 无法找到 GPU，请检查 jaxlib")

print(f"4. PyG Version: {torch_geometric.__version__}")
print(f"5. Numpy Version: {np.__version__} (Should be < 2.0)")
print(f"6. Pandas Version: {pd.__version__} (Should be < 2.0)")

print("\n====== 尝试导入辅助库 ======")
try:
    import squidpy
    import decoupler
    print("Squidpy & Decoupler 导入成功！")
except ImportError as e:
    print(f"[警告] 辅助库导入失败: {e} (如果不使用相关分析功能可忽略)")

print("\n====== 检查结束 ======")