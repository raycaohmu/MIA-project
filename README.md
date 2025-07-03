# MIA-Project - Medical Image Analysis for WSI

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

一个基于图神经网络的全切片图像(WSI)分析系统，用于病理学图像的细胞核检测、特征提取和疾病分类。

## 🔬 项目概述

本项目实现了一个完整的病理图像分析流水线，主要功能包括：

- **全切片图像处理**：处理大型病理切片图像(.svs格式)
- **细胞核检测与分类**：使用深度学习模型自动识别和分类不同类型的细胞核
- **图构建与分析**：基于细胞空间位置关系构建图结构
- **疾病分类**：使用图神经网络进行疾病诊断和治疗响应预测

## 🏗️ 项目结构

```
MIA/
├── data/                           # 数据目录
│   ├── slide_data.csv             # 切片标签数据
│   ├── slide_ov_response.csv      # 卵巢癌响应数据
│   ├── wsi/                       # 原始WSI文件
│   └── wsi_output/                # 处理结果输出
├── models/                         # 模型相关
│   ├── pretrained/                # 预训练模型
│   │   └── NuLite/               # NuLite细胞检测模型
│   └── *.py                      # 模型定义文件
├── nuclei_detection/              # 细胞检测模块
│   ├── datamodel/                # 数据模型
│   ├── inference/                # 推理脚本
│   └── training/                 # 训练脚本
├── preprocessing/                 # 预处理模块
│   ├── patch_extraction/         # 图像块提取
│   └── config_examples/          # 配置示例
├── utils/                         # 工具函数
├── output/                        # 输出目录
│   ├── features/                 # 特征文件
│   └── graph_output/             # 图数据输出
└── *.py                          # 主要脚本文件
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- **推荐使用conda环境管理** (py38环境)
- CUDA 11.0+ (GPU推荐)
- 内存：16GB+ 推荐

### 安装依赖

#### 方法1: 使用conda环境 (推荐)

```bash
# 克隆项目
git clone https://github.com/yourusername/MIA.git
cd MIA

# 激活py38环境
conda activate py38

# 运行自动安装脚本
bash setup.sh
```

#### 方法2: 手动安装

```bash
# 确保在py38环境中
conda activate py38

# 安装Python依赖
pip install -r requirements.txt

# 安装PyTorch Geometric (根据您的CUDA版本)
pip install torch-geometric torch-scatter torch-sparse torch-cluster
```

**详细的conda环境配置说明请参考**: [CONDA_SETUP.md](CONDA_SETUP.md)

### 数据准备

1. 将WSI文件(.svs格式)放置在 `data/wsi/` 目录下
2. 准备标签文件，参考 `data/slide_data_template.csv` 格式：

```csv
filename,label
sample1,0
sample2,1
```

**注意**: 由于数据文件过大，本项目不包含实际的WSI数据文件。您需要：
- 自行准备WSI文件并放入 `data/wsi/` 目录
- 创建标签CSV文件（参考模板）
- 确保有足够的存储空间（建议20GB+）

### 下载预训练模型

下载NuLite预训练模型并放置在：
```
models/pretrained/NuLite/NuLite-H-Weights.pth
```

## 🔄 完整工作流程

### 步骤 1: 随机采样细胞核检测

```bash
bash random_sample_nuclei_detection.sh
```

**功能**：
- 从每个WSI随机采样600个图像块(1024×1024像素)
- 使用NuLite模型进行细胞核检测和分类
- 生成GeoJSON格式的细胞位置和类型数据

**输出**：`data/wsi_output/sampled_[filename]/` 目录下的GeoJSON文件

### 步骤 2: 细胞特征提取

```bash
bash 3_cell_feature_extraction.sh
```

**功能**：
- 处理GeoJSON文件，提取细胞形态学特征
- 计算空间分布特征
- 生成特征矩阵

**输出**：细胞特征文件

### 步骤 3: 图构建

```bash
bash 4_graph.sh
```

**功能**：
- 基于细胞空间位置构建k-NN图或最小生成树(MST)
- 设置图的节点(细胞)和边(空间关系)
- 过滤低质量数据

**输出**：`output/graph_output/` 目录下的图数据文件

### 步骤 4: 模型训练

```bash
python train.py
```

**功能**：
- 使用CellNet图神经网络模型
- 进行疾病分类或治疗响应预测
- 输出训练日志和模型权重

## ⚙️ 配置说明

### 主要配置文件

- `random_sample_nuclei_detection.sh`: 细胞检测参数配置
- `preprocessing/config_examples/preprocessing_example.yaml`: 预处理配置
- `data/slide_ov_response.csv`: 数据标签配置

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_samples` | 600 | 每个WSI采样的图像块数量 |
| `patch_size` | 1024 | 图像块大小(像素) |
| `patch_overlap` | 64 | 图像块重叠(像素) |
| `min_tumor_cells_per_patch` | 20 | 每个图像块最小肿瘤细胞数 |
| `batch_size` | 20 | 批处理大小 |

## 📊 数据格式

### 输入数据

1. **WSI文件**: .svs格式的全切片图像
2. **标签文件**: CSV格式，包含文件名和对应标签

### 输出数据

1. **GeoJSON文件**: 包含细胞位置、类型和形态信息
2. **特征文件**: 细胞特征矩阵
3. **图数据**: PyTorch Geometric格式的图数据
4. **模型文件**: 训练后的模型权重

## 💾 数据管理

### 为什么不包含数据文件？

- **文件大小**: WSI文件通常为500MB-5GB，超出GitHub限制
- **隐私保护**: 医学数据需要特殊的隐私保护措施
- **存储成本**: 避免不必要的存储空间占用
- **灵活性**: 用户可以使用自己的数据集

### 目录结构说明

```
data/               # 数据目录 (需要用户自行填充)
├── README.md      # 数据说明文档
├── .gitkeep       # 确保目录被Git跟踪
├── slide_data_template.csv  # 标签文件模板
└── wsi/           # WSI文件目录 (用户添加)

output/             # 输出目录 (自动生成，不提交)
├── README.md      # 输出说明文档
├── .gitkeep       # 确保目录被Git跟踪
├── features/      # 特征文件
└── graph_output/  # 图数据
```

### 数据获取建议

1. **公开数据集**:
   - TCGA (The Cancer Genome Atlas)
   - CAMELYON系列数据集
   - 各大学研究机构公开数据

2. **机构数据**:
   - 医院病理科数据
   - 研究合作数据
   - 确保获得适当的伦理批准

3. **数据要求**:
   - .svs, .tif, .tiff, .ndpi, .mrxs格式
   - 20倍放大倍数推荐
   - 清晰的病理标注

## 🔧 自定义配置

### 修改采样参数

编辑 `random_sample_nuclei_detection.sh`:

```bash
--num_samples 800           # 增加采样数量
--patch_size 512           # 减小图像块大小
--random_seed 123          # 更改随机种子
```

### 修改图构建参数

编辑 `4_graph.sh`:

```bash
--min_tumor_cells_per_patch 30    # 提高细胞数量阈值
--num_workers 4                   # 增加并行工作进程
```

## 🧪 模型架构

### CellNet 图神经网络

- **节点特征**: 细胞形态学特征(大小、形状、纹理等)
- **边特征**: 细胞间距离和空间关系
- **图卷积层**: 基于邻域信息更新节点表示
- **池化层**: 图级别的特征聚合
- **分类层**: 最终疾病分类输出

## 📈 性能优化

### 内存优化

- 使用随机采样减少内存占用
- 批处理处理大型数据集
- 及时释放不需要的变量

### 速度优化

- GPU加速细胞检测和训练
- 多进程并行处理
- 缓存中间结果

## 🐛 常见问题

### 1. 内存不足错误

**解决方案**：
- 减少 `num_samples` 参数
- 减小 `batch_size`
- 使用更少的并行进程

### 2. CUDA错误

**解决方案**：
- 检查CUDA版本兼容性
- 确认GPU内存充足
- 尝试使用CPU模式

### 3. 文件路径错误

**解决方案**：
- 检查WSI文件路径
- 确认模型文件存在
- 验证输出目录权限

## 📚 参考文献

- NuLite: [论文链接]
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
- OpenSlide: https://openslide.org/

## 🤝 贡献指南

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- NuLite团队提供的预训练细胞检测模型
- PyTorch Geometric社区
- OpenSlide项目

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 提交Issue: [GitHub Issues](https://github.com/yourusername/MIA/issues)
- Email: your.email@example.com

---

**注意**: 在使用本项目处理医学数据时，请确保遵守相关的数据隐私和伦理规范。
