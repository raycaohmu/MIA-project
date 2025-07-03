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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create a dataset for WSI graph classification.")
    parser.add_argument("--slide_csv_path", type=str, required=True, help="Path to the slide CSV file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output files.")
    parser.add_argument("--min_tumor_cells_per_patch", type=int, default=20, help="Minimum tumor cells per patch.")
    parser.add_argument("--features_root_path", type=str, default="./output/features/", help="Root path for features files.")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of worker processes. Default is CPU count.")

    args = parser.parse_args()

    worker = DatasetWorker(
        slide_csv_path=args.slide_csv_path,
        output_dir=args.output_dir,
        min_tumor_cells_per_patch=args.min_tumor_cells_per_patch,
        features_root_path=args.features_root_path,
        num_workers=args.num_workers
    )
    worker.create_dataset()