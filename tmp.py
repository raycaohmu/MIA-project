import os
import re
import pandas as pd
from sklearn.neighbors import kneighbors_graph as skgraph
from scipy import sparse as sp
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
import logging
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import concurrent.futures
from functools import partial
import gc
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import kneighbors_graph as skgraph
from scipy import sparse as sp
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, NNConv, global_max_pool
from torch_scatter import scatter_mean


class NucleiData(Data):
    """Add some attributes to Data object.
    
    Args:
        * All args mush be torch.tensor. So string is not supported.
        x: Matrix for nodes
        cell_type: cell type
        edge_index: 2*N matrix
        edge_attr: edge type
        y: Label
        pid: Patient ID
        region_id: Region ID
    """
    def __init__(self, x=None, cell_type=None, edge_index=None, edge_attr=None, y=None, pos=None, pid=None, region_id=None):
        super().__init__(x, edge_index, edge_attr, y, pos)
        self.cell_type = cell_type
        self.pid = pid
        self.region_id = region_id
        
    def __repr__(self):
        info = ['{}={}'.format(key, self.size_repr(item)) for key, item in self]
        return '{}({})'.format(self.__class__.__name__, ', '.join(info))
    
    @staticmethod
    def size_repr(value):
        if torch.is_tensor(value):
            return list(value.size())
        elif isinstance(value, int) or isinstance(value, float) or isinstance(value, str):
            return [1]
        else:
            raise ValueError('Unsupported attribute type.')


class DatasetWorker:

    def __init__(self, slide_csv_path: str, output_dir: str,
                 min_tumor_cells_per_patch: int,
                 features_root_path: str = "./output/features/",
                 num_workers: int = None):
        """
        Initialize the DatasetWorker with the path to the slide CSV file,
        output directory, and minimum tumor cells per patch.

        Args:
            slide_csv_path (str): Path to the CSV file containing slide information.
            output_dir (str): Directory to save the output files.
            min_tumor_cells_per_patch (int): Minimum number of tumor cells required per patch.
            features_root_path (str): Root path for features files, default is "./output/features/".
            num_workers (int, optional): Number of worker processes for parallel processing.
                                        If None, uses CPU count.
        """
        self.slide_csv_path = slide_csv_path
        self.output_dir = output_dir
        self.min_tumor_cells_per_patch = min_tumor_cells_per_patch
        self.features_root_path = features_root_path
        self.num_workers = num_workers if num_workers is not None else os.cpu_count()
        
        # 预计算边类型映射，避免字符串操作
        self.edge_type_map = {}
        for src in range(1, 6):
            for dst in range(1, 6):
                self.edge_type_map[(src, dst)] = (src-1)*5 + (dst-1)
        
        self.nuclei_types = {
            "Neoplastic": 1,
            "Inflammatory": 2,
            "Connective": 3,
            "Dead": 4,
            "Epithelial": 5,
        }
        
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def get_edge_type_fast(self, src_type: int, dst_type: int) -> int:
        """快速获取边类型，避免字符串格式化操作"""
        return self.edge_type_map.get((src_type, dst_type), 0)
    
    def get_nuclei_orientation_diff(self, src_idx: int, dst_idx: int, orientations: np.ndarray) -> float:
        """计算细胞方向差的余弦值"""
        return np.cos(orientations[src_idx] - orientations[dst_idx])

    def process_single_file(self, file_info: Tuple[int, str, int]) -> List[NucleiData]:
        """
        处理单个文件并构建图
        
        Args:
            file_info (Tuple[int, str, int]): 包含 (file_index, filename, label) 的元组
            
        Returns:
            List[NucleiData]: 该文件中所有patch的图列表
        """
        file_idx, filename, label = file_info
        file_graphs = []
        file_feature_path = os.path.join(self.features_root_path, filename)
        
        try:
            # 查找特征文件
            cell_features_files = [
                f for f in os.listdir(file_feature_path)
                if re.search("features", f) is not None
            ]
            
            if not cell_features_files:
                return file_graphs
                
            cell_features_file = os.path.join(file_feature_path, cell_features_files[0])
            
            if not os.path.exists(cell_features_file):
                return file_graphs
            
            # 读取CSV并预处理
            cell_feature_summary = pd.read_csv(cell_features_file)
            
            # 映射细胞类型到整数
            cell_feature_summary["cell_type_int"] = cell_feature_summary["cell_type"].map(
                self.nuclei_types
            )
            
            # 获取所有patch IDs
            patch_ids = cell_feature_summary["patch_id"].unique()
            
            # 处理每个patch
            for region_id in patch_ids:
                try:
                    # 过滤得到当前patch的数据
                    patch_summary = cell_feature_summary[
                        cell_feature_summary["patch_id"] == region_id
                    ]
                    
                    # 检查肿瘤细胞数量是否满足要求
                    if sum(patch_summary["cell_type"] == "Neoplastic") <= self.min_tumor_cells_per_patch:
                        continue
                    
                    # 提取坐标并创建近邻图
                    coordinates = np.array(patch_summary[["centroid_x_patch", "centroid_y_patch"]])
                    
                    # 避免因为重复坐标导致的错误
                    if len(coordinates) <= 1:
                        continue
                        
                    # 构建K近邻图
                    graph = skgraph(
                        coordinates,
                        n_neighbors=min(8, len(coordinates)-1),  # 确保不超过可用节点数
                        mode="distance"
                    )
                    
                    # 提取边信息
                    I, J, V = sp.find(graph)
                    
                    # 跳过没有边的情况
                    if len(I) == 0:
                        continue
                        
                    # 构建边列表和权重
                    edge_weights = 1.0 / (V + 1e-6)  # 避免除零错误
                    edge_index = np.vstack((I, J))
                    
                    # 提取节点特征
                    feature_columns = [
                        "area", "perimeter", "convex_area", "equivalent_diameter",
                        "major_axis_length", "minor_axis_length",
                        "eccentricity", "solidity", "extent", "major_minor_ratio",
                        "aspect_ratio", "roundness", "convexity", "pa_ratio",
                        "volume_ratio", "contour_std", "contour_mean",
                        "contour_irregularity", "probability"
                    ]
                    
                    x = np.array(patch_summary.loc[:, feature_columns])
                    
                    # 标准化节点特征
                    if len(x) > 0:
                        # 避免零方差特征导致的错误
                        std = np.std(x, axis=0)
                        std[std == 0] = 1.0
                        x = (x - np.mean(x, axis=0)) / std
                    
                    # 提取细胞类型和方向
                    cell_type = np.array(patch_summary["cell_type_int"])
                    orientation = np.array(patch_summary["orientation"])
                    
                    # 计算边特征 - 向量化操作
                    edge_types = np.array([
                        self.get_edge_type_fast(cell_type[src], cell_type[dst])
                        for src, dst in zip(I, J)
                    ])
                    
                    orientation_diffs = np.array([
                        self.get_nuclei_orientation_diff(src, dst, orientation)
                        for src, dst in zip(I, J)
                    ])
                    
                    # 组合边特征
                    edge_attr = np.vstack([edge_types, orientation_diffs, edge_weights]).T
                    
                    # 创建图数据对象
                    data = NucleiData(
                        x=torch.tensor(x, dtype=torch.float),
                        cell_type=torch.tensor(cell_type, dtype=torch.long),
                        edge_index=torch.tensor(edge_index, dtype=torch.long),
                        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
                        y=torch.tensor([[label]]),
                        pid=torch.tensor([[file_idx]]),
                        region_id=torch.tensor([[region_id]])
                    )
                    
                    file_graphs.append(data)
                
                except Exception as e:
                    self.logger.error(f"Error processing patch {region_id} in file {file_feature_path}: {str(e)}")
                    continue
            
            # 清理内存
            del cell_feature_summary
            gc.collect()
            
            return file_graphs
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_feature_path}: {str(e)}")
            return []

    def _graph_builder(self, filenames: List[str], label: int) -> List[NucleiData]:
        """
        并行构建多个文件的图
        
        Args:
            filenames (List[str]): 要处理的文件名列表
            label (int): 这些文件的标签
            
        Returns:
            List[NucleiData]: 所有构建的图列表
        """
        # 准备并行处理的输入
        file_infos = [(i, filename, label) for i, filename in enumerate(filenames)]
        
        dataset = []
        
        # 使用并行处理加速
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            # 并行处理所有文件
            results = list(tqdm(
                executor.map(self.process_single_file, file_infos),
                total=len(file_infos),
                desc=f"Processing label {label} files"
            ))
            
            # 合并所有结果
            for file_graphs in results:
                dataset.extend(file_graphs)
        
        return dataset

    def create_dataset(self):
        """
        创建用于训练的数据集
        """
        self.logger.info(f"Reading slide CSV file from {self.slide_csv_path}")
        
        # 读取CSV文件
        try:
            slide_df = pd.read_csv(self.slide_csv_path)
        except Exception as e:
            self.logger.error(f"Error reading slide CSV file: {str(e)}")
            return
            
        # 按标签分组文件名
        label_0_filenames = slide_df[slide_df['label'] == 0]['filename'].tolist()
        label_1_filenames = slide_df[slide_df['label'] == 1]['filename'].tolist()
        
        self.logger.info(f"Found {len(label_0_filenames)} slides with label 0 and {len(label_1_filenames)} slides with label 1.")
        
        # 处理标签0的文件
        if label_0_filenames:
            self.logger.info("Building graphs for label 0 slides...")
            dataset_0 = self._graph_builder(label_0_filenames, 0)
            self.logger.info(f"Created {len(dataset_0)} graphs for label 0.")
            
            # 保存标签0的数据集
            output_path = os.path.join(self.output_dir, "dataset_label_0.pt")
            torch.save(dataset_0, output_path)
            self.logger.info(f"Saved label 0 dataset to {output_path}")
            
            # 清理内存
            del dataset_0
            gc.collect()
        
        # 处理标签1的文件
        if label_1_filenames:
            self.logger.info("Building graphs for label 1 slides...")
            dataset_1 = self._graph_builder(label_1_filenames, 1)
            self.logger.info(f"Created {len(dataset_1)} graphs for label 1.")
            
            # 保存标签1的数据集
            output_path = os.path.join(self.output_dir, "dataset_label_1.pt")
            torch.save(dataset_1, output_path)
            self.logger.info(f"Saved label 1 dataset to {output_path}")
            
            # 清理内存
            del dataset_1
            gc.collect()
        
        self.logger.info("Dataset creation completed successfully.")
        self.logger.info(f"Datasets saved to {self.output_dir}.")


class EnhancedNucleiData(Data):
    """Enhanced data object for multi-patch-type analysis"""
    
    def __init__(self, x=None, cell_type=None, edge_index=None, edge_attr=None, 
                 y=None, pos=None, pid=None, region_id=None, 
                 patch_type=None, cell_composition=None, microenv_features=None):
        super().__init__(x, edge_index, edge_attr, y, pos)
        self.cell_type = cell_type
        self.pid = pid
        self.region_id = region_id
        self.patch_type = patch_type  # 新增：patch类型
        self.cell_composition = cell_composition  # 新增：细胞组成比例
        self.microenv_features = microenv_features  # 新增：微环境特征
        
    def __repr__(self):
        info = ['{}={}'.format(key, self.size_repr(item)) for key, item in self]
        return '{}({})'.format(self.__class__.__name__, ', '.join(info))
    
    @staticmethod
    def size_repr(value):
        if torch.is_tensor(value):
            return list(value.size())
        elif isinstance(value, int) or isinstance(value, float) or isinstance(value, str):
            return [1]
        else:
            raise ValueError('Unsupported attribute type.')
        

class EnhancedDatasetWorker(DatasetWorker):
    
    def __init__(self, slide_csv_path: str, output_dir: str,
                 min_tumor_cells_per_patch: int,
                 features_root_path: str = "./output/features/",
                 num_workers: int = None,
                 use_adaptive_filtering: bool = True,
                 min_cells_per_patch: int = 50):
        super().__init__(slide_csv_path, output_dir, min_tumor_cells_per_patch, 
                        features_root_path, num_workers)
        
        self.use_adaptive_filtering = use_adaptive_filtering
        self.min_cells_per_patch = min_cells_per_patch
        
        # 定义patch类型
        self.patch_types = {
            "tumor_rich": 0,
            "immune_rich": 1, 
            "stroma_rich": 2,
            "vascular_rich": 3,
            "mixed": 4,
            "sparse": 5
        }
    
    def analyze_patch_composition(self, patch_summary: pd.DataFrame) -> Dict[str, Any]:
        """分析patch的细胞组成和微环境特征"""
        
        # 计算各种细胞类型的数量和比例
        cell_counts = {}
        total_cells = len(patch_summary)
        
        for cell_type_name, cell_type_id in self.nuclei_types.items():
            count = sum(patch_summary["cell_type"] == cell_type_name)
            cell_counts[cell_type_name] = {
                'count': count,
                'ratio': count / total_cells if total_cells > 0 else 0
            }
        
        # 确定主导细胞类型
        dominant_type = max(cell_counts.keys(), 
                          key=lambda x: cell_counts[x]['count'])
        dominant_ratio = cell_counts[dominant_type]['ratio']
        
        # 计算微环境特征
        microenv_features = self.calculate_microenvironment_features(patch_summary)
        
        return {
            'cell_counts': cell_counts,
            'dominant_type': dominant_type,
            'dominant_ratio': dominant_ratio,
            'total_cells': total_cells,
            'microenv_features': microenv_features
        }
    
    def classify_patch_type(self, composition: Dict[str, Any]) -> str:
        """根据细胞组成分类patch类型"""
        
        cell_counts = composition['cell_counts']
        total_cells = composition['total_cells']
        
        # 获取各种细胞的数量
        tumor_count = cell_counts['Neoplastic']['count']
        immune_count = cell_counts['Inflammatory']['count']
        stroma_count = cell_counts['Connective']['count']
        epithelial_count = cell_counts['Epithelial']['count']
        
        # 多层判断逻辑
        if total_cells < self.min_cells_per_patch:
            return "sparse"
        
        # 肿瘤主导 (原始标准)
        if tumor_count >= self.min_tumor_cells_per_patch and tumor_count / total_cells >= 0.4:
            return "tumor_rich"
        
        # 免疫主导 (对化疗反应重要)
        elif immune_count >= 30 and immune_count / total_cells >= 0.3:
            return "immune_rich"
        
        # 基质主导 (影响药物渗透)
        elif stroma_count >= 40 and stroma_count / total_cells >= 0.4:
            return "stroma_rich"
        
        # 血管/上皮主导 (影响药物递送)
        elif epithelial_count >= 25 and epithelial_count / total_cells >= 0.25:
            return "vascular_rich"
        
        # 混合类型 (多种细胞混合，但总数充足)
        elif total_cells >= self.min_cells_per_patch:
            # 进一步判断是否有价值
            valuable_cells = tumor_count + immune_count + stroma_count
            if valuable_cells >= 30:  # 至少30个有价值的细胞
                return "mixed"
        
        return "sparse"  # 默认为稀疏类型，会被过滤
    
    def calculate_microenvironment_features(self, patch_summary: pd.DataFrame) -> torch.Tensor:
        """计算微环境特征"""
        features = []
        
        try:
            # 1. 细胞密度特征
            total_cells = len(patch_summary)
            tumor_density = sum(patch_summary["cell_type"] == "Neoplastic") / total_cells
            immune_density = sum(patch_summary["cell_type"] == "Inflammatory") / total_cells
            stroma_density = sum(patch_summary["cell_type"] == "Connective") / total_cells
            
            features.extend([tumor_density, immune_density, stroma_density])
            
            # 2. 细胞间距离特征
            coordinates = np.array(patch_summary[["centroid_x_patch", "centroid_y_patch"]])
            if len(coordinates) > 1:
                # 计算平均细胞间距离
                from scipy.spatial.distance import pdist
                distances = pdist(coordinates)
                avg_distance = np.mean(distances)
                std_distance = np.std(distances)
                features.extend([avg_distance, std_distance])
            else:
                features.extend([0.0, 0.0])
            
            # 3. 肿瘤-免疫相互作用特征
            tumor_coords = coordinates[patch_summary["cell_type"] == "Neoplastic"]
            immune_coords = coordinates[patch_summary["cell_type"] == "Inflammatory"]
            
            if len(tumor_coords) > 0 and len(immune_coords) > 0:
                from scipy.spatial.distance import cdist
                tumor_immune_distances = cdist(tumor_coords, immune_coords)
                min_tumor_immune_dist = np.min(tumor_immune_distances)
                avg_tumor_immune_dist = np.mean(tumor_immune_distances)
                features.extend([min_tumor_immune_dist, avg_tumor_immune_dist])
            else:
                features.extend([float('inf'), float('inf')])
            
            # 4. 细胞形态异质性
            if total_cells > 1:
                area_cv = np.std(patch_summary["area"]) / (np.mean(patch_summary["area"]) + 1e-6)
                shape_cv = np.std(patch_summary["eccentricity"]) / (np.mean(patch_summary["eccentricity"]) + 1e-6)
                features.extend([area_cv, shape_cv])
            else:
                features.extend([0.0, 0.0])
            
        except Exception as e:
            # 如果计算失败，返回零向量
            features = [0.0] * 9
            
        return torch.tensor(features, dtype=torch.float)
    
    def should_include_patch(self, patch_type: str, composition: Dict[str, Any]) -> bool:
        """判断是否应该包含这个patch"""
        
        if not self.use_adaptive_filtering:
            # 使用原始策略：只要肿瘤细胞≥阈值
            return composition['cell_counts']['Neoplastic']['count'] >= self.min_tumor_cells_per_patch
        
        # 自适应策略：根据patch类型使用不同标准
        if patch_type == "sparse":
            return False
        
        # 所有非稀疏patch都有潜在价值
        return True
    
    def process_single_file(self, file_info: Tuple[int, str, int]) -> List[EnhancedNucleiData]:
        """修改后的文件处理函数"""
        file_idx, filename, label = file_info
        file_graphs = []
        file_feature_path = os.path.join(self.features_root_path, filename)
        
        try:
            # [保持原有的文件读取逻辑]
            cell_features_files = [
                f for f in os.listdir(file_feature_path)
                if re.search("features", f) is not None
            ]
            
            if not cell_features_files:
                return file_graphs
                
            cell_features_file = os.path.join(file_feature_path, cell_features_files[0])
            
            if not os.path.exists(cell_features_file):
                return file_graphs
            
            cell_feature_summary = pd.read_csv(cell_features_file)
            cell_feature_summary["cell_type_int"] = cell_feature_summary["cell_type"].map(
                self.nuclei_types
            )
            
            patch_ids = cell_feature_summary["patch_id"].unique()
            
            # 统计不同patch类型的数量
            patch_type_counts = {ptype: 0 for ptype in self.patch_types.keys()}
            
            for region_id in patch_ids:
                try:
                    patch_summary = cell_feature_summary[
                        cell_feature_summary["patch_id"] == region_id
                    ]
                    
                    # 分析patch组成
                    composition = self.analyze_patch_composition(patch_summary)
                    patch_type = self.classify_patch_type(composition)
                    
                    # 判断是否包含此patch
                    if not self.should_include_patch(patch_type, composition):
                        continue
                    
                    patch_type_counts[patch_type] += 1
                    
                    # [保持原有的图构建逻辑，但添加新特征]
                    coordinates = np.array(patch_summary[["centroid_x_patch", "centroid_y_patch"]])
                    
                    if len(coordinates) <= 1:
                        continue
                        
                    graph = skgraph(
                        coordinates,
                        n_neighbors=min(8, len(coordinates)-1),
                        mode="distance"
                    )
                    
                    I, J, V = sp.find(graph)
                    
                    if len(I) == 0:
                        continue
                        
                    edge_weights = 1.0 / (V + 1e-6)
                    edge_index = np.vstack((I, J))
                    
                    # 节点特征提取
                    feature_columns = [
                        "area", "perimeter", "convex_area", "equivalent_diameter",
                        "major_axis_length", "minor_axis_length",
                        "eccentricity", "solidity", "extent", "major_minor_ratio",
                        "aspect_ratio", "roundness", "convexity", "pa_ratio",
                        "volume_ratio", "contour_std", "contour_mean",
                        "contour_irregularity", "probability"
                    ]
                    
                    x = np.array(patch_summary.loc[:, feature_columns])
                    
                    if len(x) > 0:
                        std = np.std(x, axis=0)
                        std[std == 0] = 1.0
                        x = (x - np.mean(x, axis=0)) / std
                    
                    cell_type = np.array(patch_summary["cell_type_int"])
                    orientation = np.array(patch_summary["orientation"])
                    
                    # 边特征计算
                    edge_types = np.array([
                        self.get_edge_type_fast(cell_type[src], cell_type[dst])
                        for src, dst in zip(I, J)
                    ])
                    
                    orientation_diffs = np.array([
                        self.get_nuclei_orientation_diff(src, dst, orientation)
                        for src, dst in zip(I, J)
                    ])
                    
                    edge_attr = np.vstack([edge_types, orientation_diffs, edge_weights]).T
                    
                    # 计算细胞组成向量
                    cell_composition = torch.tensor([
                        composition['cell_counts'][name]['ratio'] 
                        for name in self.nuclei_types.keys()
                    ], dtype=torch.float)
                    
                    # 计算微环境特征
                    microenv_features = composition['microenv_features']
                    
                    # 创建增强的图数据对象
                    data = EnhancedNucleiData(
                        x=torch.tensor(x, dtype=torch.float),
                        cell_type=torch.tensor(cell_type, dtype=torch.long),
                        edge_index=torch.tensor(edge_index, dtype=torch.long),
                        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
                        y=torch.tensor([[label]]),
                        pid=torch.tensor([[file_idx]]),
                        region_id=torch.tensor([[region_id]]),
                        patch_type=torch.tensor([[self.patch_types[patch_type]]]),
                        cell_composition=cell_composition,
                        microenv_features=microenv_features
                    )
                    
                    file_graphs.append(data)
                
                except Exception as e:
                    self.logger.error(f"Error processing patch {region_id}: {str(e)}")
                    continue
            
            # 记录patch类型分布
            self.logger.info(f"File {filename} patch distribution: {patch_type_counts}")
            
            del cell_feature_summary
            gc.collect()
            
            return file_graphs
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_feature_path}: {str(e)}")
            return []
        

class EdgeNN(nn.Module):
    """
    Embedding according to edge type, and then modulated by edge features.
    """

    def __init__(self, in_channels, out_channels, device, n_edge_types=25):
        super(EdgeNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device

        self.edge_type_embedding = nn.Embedding(n_edge_types, out_channels)
        self.fc_h = nn.Linear(in_channels, out_channels)
        self.fc_g = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        """
        Args:
            x: [batch_size, n_edges, 1(edge_type) + in_channels]
        Return: [batch_size, out_channels]
        """
        y = self.edge_type_embedding(x[..., 0].clone().detach().type(torch.long).to(self.device))
        h = self.fc_h(x[..., 1:(1 + self.in_channels)].clone().detach().type(torch.float).to(self.device))
        g = self.fc_g(x[..., 1:(1 + self.in_channels)].clone().detach().type(torch.float).to(self.device))
        y = y * h + g
        return F.relu(y, inplace=True)
    

class CellNet(nn.Module):

    def __init__(self, in_channels, out_channels, device,
                 batch=True, edge_features=2, n_edge_types=25):
        """
        Args:
            in_channels: number of node features
            out_channels: number of output node features
            batch: True if from DataLoader; False if single Data object
            edge_features: number of edge features (excluding edge type)
            n_edge_types: number of edge types
        """
        super(CellNet, self).__init__()
        self.device = device
        self.batch = batch

        self.conv1 = NNConv(
            in_channels, 10,
            EdgeNN(edge_features, in_channels*10, device, n_edge_types=n_edge_types),
            aggr='mean', root_weight=True, bias=True
        )
        self.conv2 = NNConv(
            10, 10,
            EdgeNN(edge_features, 10*10, device, n_edge_types=n_edge_types),
            aggr='mean', root_weight=True, bias=True
        )
        self.conv3 = NNConv(
            10, out_channels,
            EdgeNN(edge_features, 10*out_channels, device, n_edge_types=n_edge_types),
            aggr='mean', root_weight=True, bias=True
        )

    def forward(self, data):
        """
        Args:
            data: Data in torch_geometric.data
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)

        gate = torch.eq(
            data.cell_type, 1
        ).clone().detach().requires_grad_(False).type(torch.long).to(self.device)
        if self.batch:
            _batch_size = data.batch[-1] + 1
            x = scatter_mean(x, gate * (data.batch+1), dim=0).to(self.device)[1:_batch_size+1, :]
        else:
            x = scatter_mean(x, gate, dim=0).to(self.device)[1, :]

        return x
    

class MultiPatchTypeCellNet(nn.Module):
    """支持多种patch类型的增强CellNet"""
    
    def __init__(self, in_channels, out_channels, device,
                 batch=True, edge_features=2, n_edge_types=25,
                 n_patch_types=6, microenv_features=9):
        super(MultiPatchTypeCellNet, self).__init__()
        self.device = device
        self.batch = batch
        self.n_patch_types = n_patch_types
        
        # 原有的图卷积层
        self.conv1 = NNConv(
            in_channels, 10,
            EdgeNN(edge_features, in_channels*10, device, n_edge_types=n_edge_types),
            aggr='mean', root_weight=True, bias=True
        )
        self.conv2 = NNConv(
            10, 10,
            EdgeNN(edge_features, 10*10, device, n_edge_types=n_edge_types),
            aggr='mean', root_weight=True, bias=True
        )
        self.conv3 = NNConv(
            10, out_channels,
            EdgeNN(edge_features, 10*out_channels, device, n_edge_types=n_edge_types),
            aggr='mean', root_weight=True, bias=True
        )
        
        # patch类型嵌入
        self.patch_type_embedding = nn.Embedding(n_patch_types, 16)
        
        # 细胞组成特征处理
        self.composition_fc = nn.Linear(5, 16)  # 5种细胞类型
        
        # 微环境特征处理
        self.microenv_fc = nn.Linear(microenv_features, 16)
        
        # 多模态注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=out_channels + 48,  # graph_features + patch_embedding + composition + microenv
            num_heads=4,
            batch_first=True
        )
        
        # 最终融合层
        self.fusion_fc = nn.Sequential(
            nn.Linear(out_channels + 48, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, out_channels)
        )
    
    def forward(self, data):
        """前向传播"""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 图卷积特征提取
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)
        
        # 聚合肿瘤细胞特征 (保持原有逻辑)
        gate = torch.eq(data.cell_type, 1).clone().detach().requires_grad_(False).type(torch.long).to(self.device)
        
        if self.batch:
            _batch_size = data.batch[-1] + 1
            graph_features = scatter_mean(x, gate * (data.batch+1), dim=0).to(self.device)[1:_batch_size+1, :]
            
            # 提取patch级别的特征
            patch_type_emb = self.patch_type_embedding(data.patch_type.view(-1))
            composition_features = self.composition_fc(data.cell_composition)
            microenv_features = self.microenv_fc(data.microenv_features)
            
        else:
            graph_features = scatter_mean(x, gate, dim=0).to(self.device)[1, :].unsqueeze(0)
            
            # 单个数据的特征提取
            patch_type_emb = self.patch_type_embedding(data.patch_type.view(-1))
            composition_features = self.composition_fc(data.cell_composition.unsqueeze(0))
            microenv_features = self.microenv_fc(data.microenv_features.unsqueeze(0))
        
        # 特征融合
        combined_features = torch.cat([
            graph_features, 
            patch_type_emb, 
            composition_features, 
            microenv_features
        ], dim=1)
        
        # 自注意力机制
        attended_features, _ = self.attention(
            combined_features.unsqueeze(1), 
            combined_features.unsqueeze(1), 
            combined_features.unsqueeze(1)
        )
        attended_features = attended_features.squeeze(1)
        
        # 最终特征融合
        output = self.fusion_fc(attended_features)
        
        return output
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create enhanced dataset for ovarian cancer chemotherapy response prediction.")
    parser.add_argument("--slide_csv_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--min_tumor_cells_per_patch", type=int, default=20)
    parser.add_argument("--features_root_path", type=str, default="./output/features/")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--use_adaptive_filtering", action="store_true", 
                       help="Use adaptive filtering instead of tumor-only filtering")
    parser.add_argument("--min_cells_per_patch", type=int, default=50,
                       help="Minimum total cells per patch for adaptive filtering")

    args = parser.parse_args()

    worker = EnhancedDatasetWorker(
        slide_csv_path=args.slide_csv_path,
        output_dir=args.output_dir,
        min_tumor_cells_per_patch=args.min_tumor_cells_per_patch,
        features_root_path=args.features_root_path,
        num_workers=args.num_workers,
        use_adaptive_filtering=args.use_adaptive_filtering,
        min_cells_per_patch=args.min_cells_per_patch
    )
    worker.create_dataset()