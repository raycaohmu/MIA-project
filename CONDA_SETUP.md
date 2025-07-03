# Conda Environment Setup for MIA

本项目使用conda环境管理Python依赖。推荐使用py38环境。

## 🐍 环境要求

- Anaconda 或 Miniconda
- Python 3.8+
- CUDA 11.8+ (可选，用于GPU加速)

## 🚀 快速开始

### 1. 创建conda环境 (如果还没有py38环境)

```bash
# 创建py38环境
conda create -n py38 python=3.8 -y

# 激活环境
conda activate py38
```

### 2. 安装项目依赖

```bash
# 确保在正确的环境中
conda activate py38

# 运行自动安装脚本
bash setup.sh
```

### 3. 手动安装 (可选)

如果自动安装脚本有问题，可以手动安装：

```bash
# 激活环境
conda activate py38

# 安装基础科学计算包
conda install numpy scipy pandas matplotlib seaborn scikit-learn -y

# 安装图像处理包
conda install opencv pillow -y
pip install openslide-python

# 安装PyTorch (已安装版本: torch==2.3.0)
# GPU版本 (当前环境配置)
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 安装PyTorch Geometric (已安装版本)
pip install torch-geometric==1.0.3 
pip install torch-scatter==2.1.2+pt23cu121 torch-sparse==0.6.18+pt23cu121 torch-cluster==1.6.3+pt23cu121

# 安装其他依赖 (基于实际环境版本)
pip install tqdm==4.65.0 PyYAML==6.0.2 shapely==1.8.5.post1 networkx==3.1 
pip install openslide-python==1.3.1 opencv-python==4.5.5.64 rasterio==1.3.10
```

## 🔧 环境验证

验证安装是否成功：

```bash
# 激活环境
conda activate py38

# 测试关键包
python -c "
import torch
import torch_geometric
import openslide
import numpy as np
import pandas as pd
import cv2
import rasterio
print('✅ All packages imported successfully!')
print(f'Python version: 3.8.10')
print(f'PyTorch version: {torch.__version__}')  # 应该显示 2.3.0
print(f'PyTorch Geometric version: {torch_geometric.__version__}')  # 应该显示 1.0.3
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
print(f'NumPy version: {np.__version__}')  # 应该显示 1.23.5
print(f'Pandas version: {pd.__version__}')  # 应该显示 1.5.3
"
```

**预期输出示例**：
```
✅ All packages imported successfully!
Python version: 3.8.10
PyTorch version: 2.3.0
PyTorch Geometric version: 1.0.3
CUDA available: True
CUDA version: 12.1
GPU count: 1
NumPy version: 1.23.5
Pandas version: 1.5.3
```

## 📦 环境管理

### 快速重现环境

本项目提供了基于实际py38环境的配置文件：

```bash
# 方法1: 使用environment.yml (完整环境复制)
conda env create -f environment.yml

# 方法2: 使用requirements.txt (仅Python包)
conda create -n py38_new python=3.8 -y
conda activate py38_new
pip install -r requirements.txt

# 方法3: 从现有py38环境克隆
conda create --name py38_backup --clone py38
```

### 导出当前环境配置

```bash
# 导出完整conda环境 (包含所有依赖)
conda activate py38
conda env export > environment.yml

# 导出pip requirements (仅pip安装的包)
pip freeze > requirements_frozen.txt

# 导出项目核心依赖 (手动筛选的核心包)
pip freeze | grep -E "(torch|numpy|pandas|scipy|matplotlib|seaborn|scikit-learn|opencv|openslide|pillow|tqdm|yaml|shapely|networkx)" > requirements_core.txt
```

### 环境清理

```bash
# 清理conda缓存
conda clean --all -y

# 移除环境 (谨慎使用)
# conda env remove -n py38
```

## 🐛 常见问题

### 1. conda命令未找到

**解决方案**：
```bash
# 初始化conda
~/anaconda3/bin/conda init bash
# 或
~/miniconda3/bin/conda init bash

# 重启终端或
source ~/.bashrc
```

### 2. PyTorch安装失败

**解决方案**：
```bash
# 清理pip缓存
pip cache purge

# 手动安装PyTorch
conda install pytorch torchvision torchaudio -c pytorch -y
```

### 3. OpenSlide安装问题

**解决方案**：
```bash
# Ubuntu/Debian
sudo apt-get install openslide-tools python3-openslide

# CentOS/RHEL
sudo yum install openslide openslide-devel

# 然后重新安装Python包
pip install openslide-python
```

### 4. CUDA版本不匹配

**解决方案**：
```bash
# 检查CUDA版本
nvidia-smi

# 安装对应版本的PyTorch
# 查看：https://pytorch.org/get-started/locally/
```

## 💡 性能优化

### conda配置优化

```bash
# 设置conda配置
conda config --set auto_activate_base false
conda config --set channel_priority strict
conda config --add channels conda-forge
conda config --add channels pytorch
```

### 内存和速度优化

```bash
# 设置环境变量
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0  # 指定GPU
```

## 📋 依赖包清单

### 核心依赖
- torch >= 1.12.0
- torch-geometric >= 2.1.0
- openslide-python >= 1.1.2
- numpy >= 1.21.0
- pandas >= 1.3.0

### 可选依赖
- pathml (高级图特征)
- cucim (GPU加速图像处理)

详细依赖列表请参考 `requirements.txt`。
