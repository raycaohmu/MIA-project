# 基于实际py38环境的安装指南

## 🎯 快速复现环境

### 方法1: 使用完整的environment.yml

```bash
# 完整复制环境 (推荐)
conda env create -f environment.yml
conda activate py38
```

### 方法2: 手动安装核心依赖

```bash
# 创建新环境
conda create -n py38_new python=3.8.10 -y
conda activate py38_new

# 安装核心scientific computing包
conda install numpy=1.23.5 scipy=1.8.1 pandas=1.5.3 matplotlib=3.7.1 seaborn=0.13.2 scikit-learn=1.2.1 -y

# 安装PyTorch (使用实际版本)
pip install torch==2.3.0 torchvision==0.18.0

# 安装PyTorch Geometric相关包 (需要从wheel文件安装)
# 注意: 这些包可能需要从PyTorch Geometric官网下载对应的wheel文件
pip install torch-geometric==1.0.3

# 或者尝试在线安装 (可能版本不完全一致)
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

# 安装图像处理相关包
pip install openslide-python==1.3.1 opencv-python==4.5.5.64 Pillow==10.0.1 rasterio==1.3.10

# 安装其他工具包
pip install tqdm==4.65.0 PyYAML==6.0.2 shapely==1.8.5.post1 networkx==3.1 joblib==1.4.2
```

### 方法3: 从requirements.txt安装

```bash
conda create -n py38_new python=3.8.10 -y
conda activate py38_new

# 安装基础conda包
conda install numpy scipy pandas matplotlib seaborn scikit-learn -y

# 安装其余pip包
pip install -r requirements.txt
```

## 🔍 环境验证

安装完成后运行验证脚本：

```bash
conda activate py38
python -c "
import torch, torch_geometric, openslide, numpy as np, pandas as pd, cv2, rasterio
print('✅ 所有核心包导入成功!')
print(f'Python: 3.8.10 (目标) vs {__import__('sys').version.split()[0]} (当前)')
print(f'PyTorch: 2.3.0 (目标) vs {torch.__version__} (当前)')
print(f'PyTorch Geometric: 1.0.3 (目标) vs {torch_geometric.__version__} (当前)')
print(f'NumPy: 1.23.5 (目标) vs {np.__version__} (当前)')
print(f'Pandas: 1.5.3 (目标) vs {pd.__version__} (当前)')
print(f'CUDA可用: {torch.cuda.is_available()}')
"
```

## 📝 环境文件说明

- `environment.yml`: 完整的conda环境导出 (包含所有依赖)
- `requirements.txt`: 核心Python包列表 (手动整理)
- `requirements_actual.txt`: 实际环境的pip包列表 (自动生成)

## ⚠️ 重要注意事项

1. **PyTorch Geometric包**: 实际环境中的torch-scatter, torch-sparse, torch-cluster是从本地wheel文件安装的，在线安装时版本可能略有不同。

2. **CUDA版本**: 确保您的CUDA版本与PyTorch兼容 (实际环境使用CUDA 12.1)。

3. **OpenSlide**: 在某些系统上可能需要先安装系统级依赖:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install openslide-tools
   
   # CentOS/RHEL  
   sudo yum install openslide openslide-devel
   ```

## 🚀 推荐工作流程

1. 首先尝试使用 `environment.yml` 完整复制环境
2. 如果有兼容性问题，使用方法2手动安装
3. 安装后务必运行验证脚本确认环境正确
4. 开始运行MIA项目pipeline
