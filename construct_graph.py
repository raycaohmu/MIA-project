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
import networkx as nx

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, NNConv, global_max_pool
from torch_scatter import scatter_mean

# PathML imports
try:
    from pathml.graph.preprocessing import KNNGraphBuilder, MSTGraphBuilder, GraphFeatureExtractor
    from torch_geometric.utils.convert import from_networkx, to_networkx
    PATHML_AVAILABLE = True
    print("PathML successfully imported")
except ImportError as e:
    print(f"PathML not available: {e}")
    PATHML_AVAILABLE = False


class PathMLGraphEnhancer:
    """PathML图构建算法的包装器"""
    
    def __init__(self, method='mst', k=8, use_mst=True, enable_pathml=True):
        self.method = method
        self.k = k
        self.use_mst = use_mst
        self.enable_pathml = enable_pathml and PATHML_AVAILABLE
        
        if self.enable_pathml:
            # 初始化图特征提取器
            self.feature_extractor = GraphFeatureExtractor(use_weight=True, alpha=0.85)
        
    def build_enhanced_graph(self, coordinates, cell_types, patch_summary):
        """构建增强的图结构"""
        
        if not self.enable_pathml:
            return self._fallback_to_original(coordinates, cell_types)
        
        try:
            # 将您的数据转换为PathML格式
            centroids_tensor = torch.tensor(coordinates, dtype=torch.float)
            
            if self.method == 'knn':
                builder = KNNGraphBuilder(
                    k=self.k, 
                    thresh=None,  # 可以设置距离阈值
                    return_networkx=True,
                    add_loc_feats=True
                )
            elif self.method == 'mst':
                builder = MSTGraphBuilder(
                    k=self.k,
                    thresh=None,
                    return_networkx=True,
                    add_loc_feats=True
                )
            else:
                return self._fallback_to_original(coordinates, cell_types)
            
            # 准备特征
            feature_columns = [
                "area", "perimeter", "convex_area", "equivalent_diameter",
                "major_axis_length", "minor_axis_length", "eccentricity", 
                "solidity", "extent", "major_minor_ratio", "aspect_ratio",
                "roundness", "convexity", "pa_ratio", "volume_ratio",
                "contour_std", "contour_mean", "contour_irregularity", "probability"
            ]
            
            if all(col in patch_summary.columns for col in feature_columns):
                features = np.array(patch_summary[feature_columns])
                cell_type_annotations = np.array(cell_types)
                
                # 使用PathML构建图
                nx_graph = builder.process_with_centroids(
                    centroids=centroids_tensor,
                    features=torch.tensor(features, dtype=torch.float),
                    annotation=torch.tensor(cell_type_annotations, dtype=torch.long)
                )
                
                # 提取高级图特征
                graph_features = self.feature_extractor.process(nx_graph)
                
                # 转换回PyTorch Geometric格式
                edge_index, edge_weights = self._networkx_to_pyg(nx_graph)
                
                return edge_index, edge_weights, graph_features, nx_graph
            else:
                return self._fallback_to_original(coordinates, cell_types)
                
        except Exception as e:
            print(f"PathML graph building failed: {e}, falling back to original method")
            return self._fallback_to_original(coordinates, cell_types)
    
    def _networkx_to_pyg(self, nx_graph):
        """将NetworkX图转换为PyTorch Geometric格式"""
        edge_list = []
        edge_weights = []
        
        for u, v, data in nx_graph.edges(data=True):
            edge_list.append([u, v])
            edge_list.append([v, u])  # 无向图
            weight = data.get('weight', 1.0)
            edge_weights.extend([weight, weight])
        
        if len(edge_list) == 0:
            return torch.tensor([[0], [0]], dtype=torch.long), torch.tensor([1.0])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        
        return edge_index, edge_weights
    
    def _fallback_to_original(self, coordinates, cell_types):
        """回退到原始方法"""
        try:
            graph = skgraph(coordinates, n_neighbors=min(self.k, len(coordinates)-1), mode="distance")
            I, J, V = sp.find(graph)
            edge_weights = 1.0 / (V + 1e-6)
            edge_index = np.vstack((I, J))
            
            return torch.tensor(edge_index, dtype=torch.long), torch.tensor(edge_weights, dtype=torch.float), {}, None
        except Exception as e:
            print(f"Fallback graph building also failed: {e}")
            return torch.tensor([[0], [0]], dtype=torch.long), torch.tensor([1.0]), {}, None


class EnhancedSpatialFeatureExtractor:
    """集成PathML的空间特征提取器"""
    
    def __init__(self, enable_pathml=True, graph_method='mst'):
        self.enable_pathml = enable_pathml and PATHML_AVAILABLE
        self.nuclei_types = {
            "Neoplastic": 1,
            "Inflammatory": 2,
            "Connective": 3,
            "Dead": 4,
            "Epithelial": 5,
        }
        
        if self.enable_pathml:
            self.pathml_enhancer = PathMLGraphEnhancer(method=graph_method, k=8)
    
    def calculate_enhanced_microenvironment_features(self, patch_summary: pd.DataFrame) -> torch.Tensor:
        """计算增强的微环境特征"""
        
        features = []
        coordinates = np.array(patch_summary[["centroid_x_patch", "centroid_y_patch"]])
        
        # 1. 原始空间特征
        basic_features = self._calculate_basic_spatial_features(patch_summary, coordinates)
        features.extend(basic_features)
        
        # 2. PathML图特征（如果可用）
        if self.enable_pathml:
            pathml_features = self._calculate_pathml_graph_features(patch_summary, coordinates)
            features.extend(pathml_features)
        else:
            # 如果PathML不可用，添加零特征
            features.extend([0.0] * 12)
        
        # 3. 细胞间相互作用特征
        interaction_features = self._calculate_interaction_features(patch_summary, coordinates)
        features.extend(interaction_features)
        
        # 4. 空间异质性特征
        heterogeneity_features = self._calculate_spatial_heterogeneity(patch_summary, coordinates)
        features.extend(heterogeneity_features)
        
        return torch.tensor(features, dtype=torch.float)
    
    def _calculate_basic_spatial_features(self, patch_summary, coordinates):
        """计算基础空间特征"""
        features = []
        
        try:
            # 1. 细胞密度特征
            total_cells = len(patch_summary)
            if total_cells > 0:
                tumor_density = sum(patch_summary["cell_type"] == "Neoplastic") / total_cells
                immune_density = sum(patch_summary["cell_type"] == "Inflammatory") / total_cells
                stroma_density = sum(patch_summary["cell_type"] == "Connective") / total_cells
            else:
                tumor_density = immune_density = stroma_density = 0.0
            
            features.extend([tumor_density, immune_density, stroma_density])
            
            # 2. 细胞间距离特征
            if len(coordinates) > 1:
                try:
                    from scipy.spatial.distance import pdist
                    distances = pdist(coordinates)
                    avg_distance = np.mean(distances) if len(distances) > 0 else 0.0
                    std_distance = np.std(distances) if len(distances) > 0 else 0.0
                    features.extend([avg_distance, std_distance])
                except:
                    features.extend([0.0, 0.0])
            else:
                features.extend([0.0, 0.0])
            
            # 3. 肿瘤-免疫相互作用特征
            tumor_mask = patch_summary["cell_type"] == "Neoplastic"
            immune_mask = patch_summary["cell_type"] == "Inflammatory"
            
            tumor_coords = coordinates[tumor_mask]
            immune_coords = coordinates[immune_mask]
            
            if len(tumor_coords) > 0 and len(immune_coords) > 0:
                try:
                    from scipy.spatial.distance import cdist
                    tumor_immune_distances = cdist(tumor_coords, immune_coords)
                    min_tumor_immune_dist = np.min(tumor_immune_distances)
                    avg_tumor_immune_dist = np.mean(tumor_immune_distances)
                    features.extend([min_tumor_immune_dist, avg_tumor_immune_dist])
                except:
                    features.extend([0.0, 0.0])
            else:
                features.extend([float('inf'), float('inf')])
            
            # 4. 细胞形态异质性
            if total_cells > 1:
                try:
                    area_mean = np.mean(patch_summary["area"])
                    area_cv = np.std(patch_summary["area"]) / (area_mean + 1e-6)
                    
                    ecc_mean = np.mean(patch_summary["eccentricity"])
                    shape_cv = np.std(patch_summary["eccentricity"]) / (ecc_mean + 1e-6)
                    
                    features.extend([area_cv, shape_cv])
                except:
                    features.extend([0.0, 0.0])
            else:
                features.extend([0.0, 0.0])
                
        except Exception as e:
            print(f"Error in basic spatial features: {e}")
            features = [0.0] * 9
            
        return features
    
    def _calculate_pathml_graph_features(self, patch_summary, coordinates):
        """使用PathML提取图特征"""
        try:
            cell_types = [self.nuclei_types.get(ct, 0) for ct in patch_summary["cell_type"]]
            
            # 构建PathML图
            edge_index, edge_weights, graph_features, nx_graph = self.pathml_enhancer.build_enhanced_graph(
                coordinates, cell_types, patch_summary
            )
            
            # 提取关键图特征
            key_features = [
                graph_features.get('diameter', 0),
                graph_features.get('radius', 0), 
                graph_features.get('density', 0),
                graph_features.get('transitivity_undir', 0),
                graph_features.get('assortativity_degree', 0),
                graph_features.get('degree_mean', 0),
                graph_features.get('degree_std', 0),
                graph_features.get('egvec_centr_mean', 0),
                graph_features.get('egvec_centr_std', 0),
                graph_features.get('constraint_mean', 0),
                graph_features.get('coreness_mean', 0),
                graph_features.get('personalized_pgrank_mean', 0)
            ]
            
            return key_features
            
        except Exception as e:
            print(f"PathML graph features failed: {e}")
            # 失败时返回零特征
            return [0.0] * 12
    
    def _calculate_interaction_features(self, patch_summary, coordinates):
        """计算细胞间相互作用特征"""
        features = []
        
        try:
            # 按细胞类型分组
            cell_groups = {}
            for i, cell_type in enumerate(patch_summary["cell_type"]):
                if cell_type not in cell_groups:
                    cell_groups[cell_type] = []
                cell_groups[cell_type].append(coordinates[i])
            
            # 计算类型间距离特征
            type_pairs = [
                ('Neoplastic', 'Inflammatory'),  # 肿瘤-免疫
                ('Neoplastic', 'Connective'),    # 肿瘤-基质
                ('Inflammatory', 'Connective'),  # 免疫-基质
                ('Neoplastic', 'Epithelial'),    # 肿瘤-上皮
            ]
            
            for type1, type2 in type_pairs:
                if type1 in cell_groups and type2 in cell_groups:
                    coords1 = np.array(cell_groups[type1])
                    coords2 = np.array(cell_groups[type2])
                    
                    if len(coords1) > 0 and len(coords2) > 0:
                        try:
                            from scipy.spatial.distance import cdist
                            distances = cdist(coords1, coords2)
                            
                            # 距离统计特征
                            features.extend([
                                np.min(distances),      # 最小距离
                                np.mean(distances),     # 平均距离
                                np.std(distances),      # 距离标准差
                                np.percentile(distances, 10)  # 10%分位数
                            ])
                        except:
                            features.extend([0.0] * 4)
                    else:
                        features.extend([float('inf')] * 4)
                else:
                    features.extend([float('inf')] * 4)
            
            # 计算局部密度特征
            for cell_type in ['Neoplastic', 'Inflammatory', 'Connective']:
                if cell_type in cell_groups and len(cell_groups[cell_type]) > 1:
                    try:
                        coords = np.array(cell_groups[cell_type])
                        # 计算每个细胞周围的局部密度
                        from scipy.spatial.distance import pdist, squareform
                        dist_matrix = squareform(pdist(coords))
                        
                        # 在50像素半径内的邻居数量
                        neighbor_counts = np.sum(dist_matrix < 50, axis=1) - 1  # 减去自己
                        features.extend([
                            np.mean(neighbor_counts),
                            np.std(neighbor_counts)
                        ])
                    except:
                        features.extend([0.0, 0.0])
                else:
                    features.extend([0.0, 0.0])
            
        except Exception as e:
            print(f"Error in interaction features: {e}")
            # 失败时返回默认值
            features = [0.0] * (len(type_pairs) * 4 + 3 * 2)
        
        return features
    
    def _calculate_spatial_heterogeneity(self, patch_summary, coordinates):
        """计算空间异质性特征"""
        features = []
        
        try:
            # 1. 基于网格的异质性
            grid_size = 4  # 4x4网格
            patch_size = 256  # 假设patch大小为256x256
            cell_grid = np.zeros((grid_size, grid_size, 5))  # 5种细胞类型
            
            for i, (x, y) in enumerate(coordinates):
                grid_x = min(int(x / patch_size * grid_size), grid_size - 1)
                grid_y = min(int(y / patch_size * grid_size), grid_size - 1)
                cell_type_idx = self.nuclei_types.get(patch_summary.iloc[i]["cell_type"], 0)
                if cell_type_idx > 0:  # 只有有效的细胞类型
                    cell_grid[grid_y, grid_x, cell_type_idx-1] += 1
            
            # 计算每个网格的香农多样性指数
            diversities = []
            for i in range(grid_size):
                for j in range(grid_size):
                    cell_counts = cell_grid[i, j, :]
                    total = np.sum(cell_counts)
                    if total > 0:
                        proportions = cell_counts / total
                        # 香农指数
                        shannon = -np.sum(proportions * np.log(proportions + 1e-10))
                        diversities.append(shannon)
                    else:
                        diversities.append(0)
            
            features.extend([
                np.mean(diversities),    # 平均多样性
                np.std(diversities),     # 多样性变异
                np.max(diversities),     # 最大多样性
            ])
            
            # 2. 空间聚集性特征
            if len(coordinates) > 1:
                try:
                    from scipy.spatial.distance import pdist, squareform
                    
                    # 计算细胞类型的空间自相关
                    cell_type_numeric = [self.nuclei_types.get(ct, 0) for ct in patch_summary["cell_type"]]
                    
                    if len(set(cell_type_numeric)) > 1:  # 有多种细胞类型
                        dist_matrix = squareform(pdist(coordinates))
                        # 创建邻接矩阵（距离小于50的为邻居）
                        adjacency = (dist_matrix < 50) & (dist_matrix > 0)
                        
                        # 计算简化的空间自相关指标
                        spatial_autocorr = 0
                        count = 0
                        for i in range(len(cell_type_numeric)):
                            neighbors = np.where(adjacency[i])[0]
                            if len(neighbors) > 0:
                                # 邻居中相同类型的比例
                                same_type_ratio = np.mean([
                                    cell_type_numeric[j] == cell_type_numeric[i] 
                                    for j in neighbors
                                ])
                                spatial_autocorr += same_type_ratio
                                count += 1
                        
                        if count > 0:
                            spatial_autocorr /= count
                            features.append(spatial_autocorr)
                        else:
                            features.append(0.0)
                    else:
                        features.append(1.0)  # 所有细胞同类型
                except:
                    features.append(0.0)
            else:
                features.append(0.0)
                
        except Exception as e:
            print(f"Error in heterogeneity features: {e}")
            features = [0.0] * 4
        
        return features


# 修改原有的数据类
class EnhancedNucleiData(Data):
    """Enhanced data object for multi-patch-type analysis"""
    
    def __init__(self, x=None, cell_type=None, edge_index=None, edge_attr=None, 
                 y=None, pos=None, pid=None, region_id=None, 
                 patch_type=None, cell_composition=None, microenv_features=None,
                 pathml_features=None):  # 新增PathML特征
        super().__init__(x, edge_index, edge_attr, y, pos)
        self.cell_type = cell_type
        self.pid = pid
        self.region_id = region_id
        self.patch_type = patch_type
        self.cell_composition = cell_composition
        self.microenv_features = microenv_features
        self.pathml_features = pathml_features  # PathML图级别特征
        
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


# 保持原有的基础类不变
class NucleiData(Data):
    """Add some attributes to Data object."""
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
    """保持原有的DatasetWorker类不变"""
    
    def __init__(self, slide_csv_path: str, output_dir: str,
                 min_tumor_cells_per_patch: int,
                 features_root_path: str = "./output/features/",
                 num_workers: int = None):
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
        """原有的process_single_file方法保持不变"""
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
        """原有方法保持不变"""
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
        """原有方法保持不变"""
        self.logger.info(f"Reading slide CSV file from {self.slide_csv_path}")
        
        # 读取CSV文件
        try:
            slide_df = pd.read_csv(self.slide_csv_path)
        except Exception as e:
            self.logger.error(f"Error reading slide CSV file: {str(e)}")
            return
            
        # 按标签分组文件名
        label_0_filenames = slide_df[slide_df['label'] == 0]['wsi_name'].tolist()
        label_1_filenames = slide_df[slide_df['label'] == 1]['wsi_name'].tolist()
        
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


# 修改增强版的Worker类以集成PathML
class PathMLEnhancedDatasetWorker(DatasetWorker):
    """集成PathML图分析的增强版本"""
    
    def __init__(self, slide_csv_path: str, output_dir: str,
                 min_tumor_cells_per_patch: int,
                 features_root_path: str = "./output/features/",
                 num_workers: int = None,
                 use_adaptive_filtering: bool = True,
                 min_cells_per_patch: int = 50,
                 use_pathml_graph: bool = True,
                 graph_method: str = 'mst'):
        
        super().__init__(slide_csv_path, output_dir, min_tumor_cells_per_patch, 
                        features_root_path, num_workers)
        
        self.use_adaptive_filtering = use_adaptive_filtering
        self.min_cells_per_patch = min_cells_per_patch
        self.use_pathml_graph = use_pathml_graph
        self.graph_method = graph_method
        
        # 定义patch类型
        self.patch_types = {
            "tumor_rich": 0,
            "immune_rich": 1, 
            "stroma_rich": 2,
            "vascular_rich": 3,
            "mixed": 4,
            "sparse": 5
        }
        
        # 初始化空间特征提取器
        self.spatial_extractor = EnhancedSpatialFeatureExtractor(
            enable_pathml=use_pathml_graph,
            graph_method=graph_method
        )
        self.spatial_extractor.nuclei_types = self.nuclei_types
    
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
        
        # 计算增强的微环境特征
        microenv_features = self.spatial_extractor.calculate_enhanced_microenvironment_features(patch_summary)
        
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
    
    def build_enhanced_graph(self, patch_summary: pd.DataFrame):
        """使用PathML构建增强的图结构"""
        
        coordinates = np.array(patch_summary[["centroid_x_patch", "centroid_y_patch"]])
        
        if not self.use_pathml_graph or len(coordinates) <= 1:
            # 回退到原始方法
            return self._build_original_graph(coordinates, patch_summary)
        
        try:
            cell_types = [self.nuclei_types.get(ct, 0) for ct in patch_summary["cell_type"]]
            enhancer = PathMLGraphEnhancer(method=self.graph_method, k=8, enable_pathml=True)
            
            edge_index, edge_weights, graph_features, nx_graph = enhancer.build_enhanced_graph(
                coordinates, cell_types, patch_summary
            )
            
            # 计算改进的边特征
            edge_attr = self._calculate_enhanced_edge_features(
                edge_index, patch_summary, edge_weights, graph_features
            )
            
            # 提取PathML图级别特征
            pathml_features = self._extract_pathml_features(graph_features)
            
            return edge_index, edge_attr, pathml_features
            
        except Exception as e:
            self.logger.warning(f"PathML graph building failed: {e}")
            return self._build_original_graph(coordinates, patch_summary)
    
    def _build_original_graph(self, coordinates, patch_summary):
        """原始图构建方法"""
        try:
            graph = skgraph(coordinates, n_neighbors=min(8, len(coordinates)-1), mode="distance")
            I, J, V = sp.find(graph)
            edge_weights = 1.0 / (V + 1e-6)
            edge_index = np.vstack((I, J))
            
            # 计算原始边特征
            cell_type = np.array([self.nuclei_types[ct] for ct in patch_summary["cell_type"]])
            orientation = np.array(patch_summary["orientation"])
            
            edge_types = np.array([
                self.get_edge_type_fast(cell_type[src], cell_type[dst])
                for src, dst in zip(I, J)
            ])
            
            orientation_diffs = np.array([
                self.get_nuclei_orientation_diff(src, dst, orientation)
                for src, dst in zip(I, J)
            ])
            
            edge_attr = np.vstack([edge_types, orientation_diffs, edge_weights]).T
            
            return torch.tensor(edge_index, dtype=torch.long), torch.tensor(edge_attr, dtype=torch.float), None
        except Exception as e:
            self.logger.error(f"Original graph building failed: {e}")
            return torch.tensor([[0], [0]], dtype=torch.long), torch.tensor([[0, 0, 1]]), None
    
    def _calculate_enhanced_edge_features(self, edge_index, patch_summary, edge_weights, graph_features):
        """计算增强的边特征"""
        
        # 原有的边特征计算逻辑
        cell_types = np.array([self.nuclei_types[ct] for ct in patch_summary["cell_type"]])
        orientations = np.array(patch_summary["orientation"])
        
        I, J = edge_index
        
        edge_types = np.array([
            self.get_edge_type_fast(cell_types[src], cell_types[dst])
            for src, dst in zip(I, J)
        ])
        
        orientation_diffs = np.array([
            self.get_nuclei_orientation_diff(src, dst, orientations)
            for src, dst in zip(I, J)
        ])
        
        # 添加图结构特征到边特征
        edge_centrality = self._calculate_edge_centrality(edge_index, graph_features)
        
        edge_attr = np.vstack([edge_types, orientation_diffs, edge_weights, edge_centrality]).T
        
        return torch.tensor(edge_attr, dtype=torch.float)
    
    def _calculate_edge_centrality(self, edge_index, graph_features):
        """计算边的中心性特征"""
        try:
            # 简化的边中心性计算
            I, J = edge_index
            
            # 基于度中心性的简单边权重
            degree_mean = graph_features.get('degree_mean', 1.0)
            edge_centrality = np.ones(len(I)) * degree_mean / 10.0  # 归一化
            
            return edge_centrality
        except:
            return np.ones(len(edge_index[0]))
    
    def _extract_pathml_features(self, graph_features):
        """提取PathML图级别特征"""
        if not graph_features:
            return torch.zeros(12)
        
        key_features = [
            graph_features.get('diameter', 0),
            graph_features.get('radius', 0), 
            graph_features.get('density', 0),
            graph_features.get('transitivity_undir', 0),
            graph_features.get('assortativity_degree', 0),
            graph_features.get('degree_mean', 0),
            graph_features.get('degree_std', 0),
            graph_features.get('egvec_centr_mean', 0),
            graph_features.get('egvec_centr_std', 0),
            graph_features.get('constraint_mean', 0),
            graph_features.get('coreness_mean', 0),
            graph_features.get('personalized_pgrank_mean', 0)
        ]
        
        return torch.tensor(key_features, dtype=torch.float)
    
    def process_single_file(self, file_info: Tuple[int, str, int]) -> List[EnhancedNucleiData]:
        """修改后的文件处理函数，集成PathML功能"""
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
                    
                    # 使用增强的图构建方法
                    coordinates = np.array(patch_summary[["centroid_x_patch", "centroid_y_patch"]])
                    
                    if len(coordinates) <= 1:
                        continue
                    
                    # 构建增强的图
                    edge_index, edge_attr, pathml_features = self.build_enhanced_graph(patch_summary)
                    
                    if edge_index is None or len(edge_index[0]) == 0:
                        continue
                    
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
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        y=torch.tensor([[label]]),
                        pid=torch.tensor([[file_idx]]),
                        region_id=torch.tensor([[region_id]]),
                        patch_type=torch.tensor([[self.patch_types[patch_type]]]),
                        cell_composition=cell_composition,
                        microenv_features=microenv_features,
                        pathml_features=pathml_features
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


# 保持原有的网络组件
class EdgeNN(nn.Module):
    """Embedding according to edge type, and then modulated by edge features."""

    def __init__(self, in_channels, out_channels, device, n_edge_types=25):
        super(EdgeNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device

        self.edge_type_embedding = nn.Embedding(n_edge_types, out_channels)
        self.fc_h = nn.Linear(in_channels, out_channels)
        self.fc_g = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        y = self.edge_type_embedding(x[..., 0].clone().detach().type(torch.long).to(self.device))
        h = self.fc_h(x[..., 1:(1 + self.in_channels)].clone().detach().type(torch.float).to(self.device))
        g = self.fc_g(x[..., 1:(1 + self.in_channels)].clone().detach().type(torch.float).to(self.device))
        y = y * h + g
        return F.relu(y, inplace=True)


class CellNet(nn.Module):
    """原有的CellNet保持不变"""

    def __init__(self, in_channels, out_channels, device,
                 batch=True, edge_features=2, n_edge_types=25):
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


# 增强的网络架构
class PathMLEnhancedCellNet(nn.Module):
    """集成PathML特征的增强网络"""
    
    def __init__(self, in_channels, out_channels, device,
                 batch=True, edge_features=4,  # 增加边特征维度
                 n_edge_types=25, n_patch_types=6, 
                 microenv_features=33,  # 增加微环境特征维度 (9+12+16+4)
                 pathml_features=12,    # PathML图特征维度
                 **kwargs):
        
        super(PathMLEnhancedCellNet, self).__init__()
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
        
        # PathML图特征处理
        self.pathml_features_fc = nn.Linear(pathml_features, 16)
        
        # 多模态注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=out_channels + 64,  # graph_features + patch_embedding + composition + microenv + pathml
            num_heads=4,
            batch_first=True
        )
        
        # 最终融合层
        self.fusion_fc = nn.Sequential(
            nn.Linear(out_channels + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_channels)
        )
    
    def forward(self, data):
        """增强的前向传播"""
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
        
        # 聚合肿瘤细胞特征
        gate = torch.eq(data.cell_type, 1).clone().detach().requires_grad_(False).type(torch.long).to(self.device)
        
        if self.batch:
            _batch_size = data.batch[-1] + 1
            graph_features = scatter_mean(x, gate * (data.batch+1), dim=0).to(self.device)[1:_batch_size+1, :]
            
            # 提取各种特征
            patch_type_emb = self.patch_type_embedding(data.patch_type.view(-1))
            composition_features = self.composition_fc(data.cell_composition)
            microenv_features = self.microenv_fc(data.microenv_features)
            
            # PathML图特征
            if hasattr(data, 'pathml_features') and data.pathml_features is not None:
                pathml_graph_features = self.pathml_features_fc(data.pathml_features)
            else:
                pathml_graph_features = torch.zeros(_batch_size, 16, device=self.device)
            
        else:
            graph_features = scatter_mean(x, gate, dim=0).to(self.device)[1, :].unsqueeze(0)
            
            # 单个数据的特征提取
            patch_type_emb = self.patch_type_embedding(data.patch_type.view(-1))
            composition_features = self.composition_fc(data.cell_composition.unsqueeze(0))
            microenv_features = self.microenv_fc(data.microenv_features.unsqueeze(0))
            
            if hasattr(data, 'pathml_features') and data.pathml_features is not None:
                pathml_graph_features = self.pathml_features_fc(data.pathml_features.unsqueeze(0))
            else:
                pathml_graph_features = torch.zeros(1, 16, device=self.device)
        
        # 特征融合
        combined_features = torch.cat([
            graph_features, 
            patch_type_emb, 
            composition_features, 
            microenv_features,
            pathml_graph_features
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


# 为了向后兼容，保持MultiPatchTypeCellNet
class MultiPatchTypeCellNet(PathMLEnhancedCellNet):
    """向后兼容的别名"""
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create enhanced dataset with PathML integration for ovarian cancer chemotherapy response prediction.")
    parser.add_argument("--slide_csv_path", type=str, required=True,
                       help="Path to CSV file containing slide information")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save output files")
    parser.add_argument("--min_tumor_cells_per_patch", type=int, default=20,
                       help="Minimum number of tumor cells per patch")
    parser.add_argument("--features_root_path", type=str, default="./output/features/",
                       help="Root path for features files")
    parser.add_argument("--num_workers", type=int, default=None,
                       help="Number of worker processes")
    parser.add_argument("--use_adaptive_filtering", action="store_true", 
                       help="Use adaptive filtering instead of tumor-only filtering")
    parser.add_argument("--min_cells_per_patch", type=int, default=50,
                       help="Minimum total cells per patch for adaptive filtering")
    
    # PathML相关参数
    parser.add_argument("--use_pathml_graph", action="store_true",
                       help="Enable PathML graph building algorithms")
    parser.add_argument("--graph_method", type=str, choices=['knn', 'mst'], default='mst',
                       help="Graph building method: knn or mst (minimum spanning tree)")
    parser.add_argument("--worker_type", type=str, choices=['basic', 'enhanced', 'pathml'], 
                       default='enhanced',
                       help="Type of worker to use: basic, enhanced, or pathml")

    args = parser.parse_args()

    # 根据参数选择合适的Worker类
    if args.worker_type == 'pathml':
        print("Using PathML Enhanced Dataset Worker")
        if not PATHML_AVAILABLE:
            print("Warning: PathML not available, falling back to enhanced worker")
            worker_class = EnhancedDatasetWorker
        else:
            worker_class = PathMLEnhancedDatasetWorker
            
        worker = worker_class(
            slide_csv_path=args.slide_csv_path,
            output_dir=args.output_dir,
            min_tumor_cells_per_patch=args.min_tumor_cells_per_patch,
            features_root_path=args.features_root_path,
            num_workers=args.num_workers,
            use_adaptive_filtering=args.use_adaptive_filtering,
            min_cells_per_patch=args.min_cells_per_patch,
            use_pathml_graph=args.use_pathml_graph,
            graph_method=args.graph_method
        )
        
    elif args.worker_type == 'enhanced':
        print("Using Enhanced Dataset Worker")
        worker = EnhancedDatasetWorker(
            slide_csv_path=args.slide_csv_path,
            output_dir=args.output_dir,
            min_tumor_cells_per_patch=args.min_tumor_cells_per_patch,
            features_root_path=args.features_root_path,
            num_workers=args.num_workers,
            use_adaptive_filtering=args.use_adaptive_filtering,
            min_cells_per_patch=args.min_cells_per_patch
        )
        
    else:  # basic
        print("Using Basic Dataset Worker")
        worker = DatasetWorker(
            slide_csv_path=args.slide_csv_path,
            output_dir=args.output_dir,
            min_tumor_cells_per_patch=args.min_tumor_cells_per_patch,
            features_root_path=args.features_root_path,
            num_workers=args.num_workers
        )
    
    # 创建数据集
    worker.create_dataset()
    
    print(f"\nDataset creation completed!")
    print(f"Results saved to: {args.output_dir}")
    
    if args.worker_type == 'pathml' and PATHML_AVAILABLE:
        print(f"PathML graph method used: {args.graph_method}")
        print("Enhanced features include:")
        print("  - PathML graph topology features")
        print("  - Advanced spatial interaction features") 
        print("  - Multi-scale heterogeneity analysis")
        print("  - Cell composition embeddings")


# 基础原有功能
#python your_script.py \
#    --slide_csv_path /path/to/slides.csv \
#    --output_dir /path/to/output \
#    --min_tumor_cells_per_patch 20 \
#    --worker_type basic

# 增强版本（您原有的EnhancedDatasetWorker）
#python your_script.py \
#    --slide_csv_path /path/to/slides.csv \
#    --output_dir /path/to/output \
#    --min_tumor_cells_per_patch 20 \
#    --worker_type enhanced \
#    --use_adaptive_filtering \
#    --min_cells_per_patch 50

# PathML集成版本（新功能）
#python your_script.py \
#    --slide_csv_path /path/to/slides.csv \
#    --output_dir /path/to/output \
#    --min_tumor_cells_per_patch 20 \
#    --worker_type pathml \
#    --use_pathml_graph \
#    --graph_method mst \
#    --use_adaptive_filtering \
#    --min_cells_per_patch 50