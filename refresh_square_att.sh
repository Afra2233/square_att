#!/bin/bash

TARGET=/scratch/hpc/07/zhang303/square_att

# 更新根目录
touch -am "$TARGET"

# 递归更新所有文件和目录
find "$TARGET" -exec touch -am {} +

echo "✅ square_att 时间戳已刷新：$(date)"
