#!/bin/bash
# 启动Jupyter Lab并允许远程访问

cd /home/zhangjunyi/xiangmu/nichecompass-main

# 生成配置（如果不存在）
if [ ! -f ~/.jupyter/jupyter_lab_config.py ]; then
    jupyter lab --generate-config
fi

# 启动Jupyter Lab
jupyter lab \
  --ip=0.0.0.0 \
  --port=8990 \
  --no-browser \
  --NotebookApp.token='' \
  --NotebookApp.password='' \
  --allow-root
