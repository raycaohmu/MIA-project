import os
import sys
import random
import numpy as np
import logging
import torch
import pandas as pd
from pathlib import Path
import argparse
import shutil
from typing import Callable, List, Tuple, Union, Dict

# Add parent directories to path
currentdir = os.path.dirname(os.path.abspath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# Disable duplicate loggers
logging.getLogger("pandarallel").propagate = False
os.environ["PYTHONWARNINGS"] = "ignore"

from preprocessing.patch_extraction.src.cli import PreProcessingConfig
from preprocessing.patch_extraction.src.patch_extraction import PreProcessor
from preprocessing.patch_extraction.src.utils.patch_util import get_files_from_dir, generate_thumbnails
from nuclei_detection.inference.nuclei_detection import CellSegmentationInference
from nuclei_detection.datamodel.wsi_datamodel import WSI
from utils.logger import Logger
from configs.python.config import LOGGING_EXT, WSI_EXT


class RandomPatchSampler:
    def __init__(
    self, 
    wsi_path: str = None,  # 可以为None, 如果使用CSV读取
    csv_path: str = None,  # 如果提供了CSV路径，则使用它
    filename_column: str = "filename",  # CSV中WSI文件名的列名
    label_column: str = "label",  # CSV中标签列名
    output_path: str = None,
    model_path: str = None,
    num_samples: int = 600, 
    patch_size: int = 1024, 
    patch_overlap: int = 64,
    gpu: int = 0,
    random_seed: int = 42,
    hardware_selection: str = "openslide",  # 添加默认值为openslide的参数
    mpp: float = 0.5,
    save_all_files: bool = False,  # 是否保存所有文件
    save_visualization: bool = True,  # 是否保存可视化随机选择的patches以及heatmap
    filter_by_label: int = None, # 如果提供了标签，则只处理该标签的WSI文件
):
        """
        Randomly sample patches from WSI and perform nuclei detection
        
        Args:
            wsi_path: Path to WSI file or directory of WSI files (用于处理单个文件或目录)
            csv_path: Path to CSV file containing WSI file paths
            filename_column: Column name containing file paths in CSV
            label_column: Column name containing labels in CSV
            output_path: Path to save results
            model_path: Path to nuclei detection model
            num_samples: Number of patches to sample
            patch_size: Size of patches
            patch_overlap: Overlap between patches
            gpu: GPU ID to use
            random_seed: Random seed for reproducibility
            hardware_selection: Hardware selection for slide reading ('cucim', 'openslide')
            filter_by_label: Filter files by specific label value (optional)
        """
        # Validate inputs
        if not wsi_path and not csv_path:
            raise ValueError("Either wsi_path or csv_path must be provided.")
        if wsi_path and csv_path:
            raise ValueError("Only one of wsi_path or csv_path should be provided.")
        
        self.wsi_path = Path(wsi_path) if wsi_path else None
        self.csv_path = Path(csv_path) if csv_path else None
        self.filename_column = filename_column
        self.label_column = label_column
        self.filter_by_label = filter_by_label  # 添加标签过滤参数
        self.output_path = Path(output_path)
        self.model_path = Path(model_path)
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.gpu = gpu
        self.hardware_selection = hardware_selection  # 保存hardware_selection参数
        self.mpp = mpp  # 添加mpp参数
        self.save_all_files = save_all_files  # 是否保存所有文件
        self.save_visualization = save_visualization  # 是否保存可视化结果
        
        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Create logger - ensure we're not duplicating
        logger_handlers = [h for h in logging.root.handlers]
        for handler in logger_handlers:
            logging.root.removeHandler(handler)
            
        self.logger = Logger(level="INFO").create_logger()
        
        # Initialize nuclei detection model
        self.nuclei_detector = CellSegmentationInference(
            model_path=self.model_path,
            gpu=self.gpu,
            save_all_files=self.save_all_files,  # 是否保存所有文件
        )
        # if use CSV, read WSI paths from CSV
        if self.csv_path:
            self.wsi_files_info = self._load_wsi_files_from_csv()
        else:
            # if use path, read WSI files from directory
            self.wsi_files_info = self._load_wsi_files_from_path()
        
    def _load_wsi_files_from_csv(self) -> List[Dict]:
        """
        load wsi from csv
        """
        try:
            # read csv file
            df = pd.read_csv(self.csv_path)
            self.logger.info(f"Loaded CSV file: {self.csv_path} with {len(df)} entries")
            # check if filename_column exists
            if self.filename_column not in df.columns:
                raise ValueError(f"Column '{self.filename_column}' not found in CSV file.")
            # filter by label if specified
            if self.filter_by_label is not None:
                if self.label_column not in df.columns:
                    raise ValueError(f"Column '{self.label_column}' not found in CSV file.")
                else:
                    original_count = len(df)
                    df = df[df[self.label_column] == self.filter_by_label]
                    self.logger.info(f"Filtered CSV to {len(df)} entries with label '{self.filter_by_label}' (original: {original_count})")

            # check if file exists
            wsi_files_info = []
            for idx, row in df.iterrows():
                file_path = Path(row[self.filename_column])
                if not file_path.exists():
                    self.logger.warning(f"File {file_path} does not exist, skipping.")
                    continue
                file_info = {
                    "path": file_path,
                    "index": idx}
                if self.label_column in df.columns:
                    file_info["label"] = row[self.label_column]

                wsi_files_info.append(file_info)

            self.logger.info(f"Found {len(wsi_files_info)} valid WSI files in CSV")
            return wsi_files_info
        except Exception as e:
            self.logger.error(f"Error loading WSI files from CSV: {e}")
            raise

    def _load_wsi_files_from_path(self) -> List[Dict]:
        """
        Load WSI files from dir
        """
        wsi_files_info = []
        if self.wsi_path.is_file():
            # single WSI file
            wsi_files_info.append({
                "path": self.wsi_path,
                "index": 0})
        else:
            # directory of WSI files
            wsi_files = get_files_from_dir(
                self.wsi_path, file_path="svs")
            for idx, wsi_file in enumerate(wsi_files):
                wsi_files_info.append({
                    "path": wsi_file,
                    "index": idx})
        return wsi_files_info

    def _setup_temporary_extraction(self, wsi_file):
        """Setup temporary directory for patch extraction"""
        # Create temporary output directory
        temp_dir = self.output_path / f"temp_{wsi_file.stem}"
        temp_dir.mkdir(exist_ok=True, parents=True)
        
        # Get file extension without the dot and verify it's supported
        file_ext = wsi_file.suffix[1:]
        if file_ext not in WSI_EXT:
            self.logger.warning(f"File extension {file_ext} may not be supported. Supported extensions: {WSI_EXT}")
        
        # Make sure log_level is one of the allowed values
        if "info" not in LOGGING_EXT:
            log_level = LOGGING_EXT[0]  # Use the first allowed value as fallback
        else:
            log_level = "info"
        
        # Ensure patch_overlap is a percentage value (0-100)
        # The NuLite code expects overlap as a percentage, not pixels
        if self.patch_overlap >= 1:
            # Convert from pixels to percentage
            overlap_percentage = (self.patch_overlap / self.patch_size) * 100
        else:
            # Already a percentage
            overlap_percentage = self.patch_overlap * 100
            
        # 创建wsi_properties字典传递MPP值
        wsi_properties = {
            "slide_mpp": self.mpp,       # 这里是关键，使用"slide_mpp"键
            "magnification": 20.0,       # 也可以提供默认放大倍数
        }
        
        # Create configuration directly instead of using parser
        config_dict = {
            "wsi_paths": str(wsi_file),
            "output_path": str(temp_dir),
            "wsi_extension": file_ext,
            "patch_size": self.patch_size,
            "patch_overlap": overlap_percentage,  # Use percentage value
            "processes": 8,
            "min_intersection_ratio": 0.4,  # Only keep patches with at least 40% tissue
            "filter_patches": False,  # No additional filtering
            "hardware_selection": self.hardware_selection,
            "log_level": log_level,
            "mpp": self.mpp,
            "wsi_properties": wsi_properties,  # 添加wsi_properties参数
        }
        
        # Create the PreProcessingConfig directly
        try:
            config = PreProcessingConfig(**config_dict)
            return temp_dir, config
        except Exception as e:
            self.logger.error(f"Error creating configuration: {e}")
            raise

    def _randomly_select_patches(self, patched_wsi_dir, num_samples):
        """Randomly select a subset of patches"""
        # 首先尝试直接在temp_dir下查找patches目录
        patches_dir = patched_wsi_dir / "patches"
        
        # 如果没找到，则检查是否有WSI名称子目录（NuLite创建的嵌套结构）
        if not patches_dir.exists():
            # 假设子目录名称与WSI文件名相同（不带扩展名）
            wsi_subdir = list(patched_wsi_dir.glob("*"))
            
            # 查找第一个是目录且包含patches子目录的项
            for item in wsi_subdir:
                if item.is_dir() and (item / "patches").exists():
                    patches_dir = item / "patches"
                    self.logger.info(f"Found patches directory at: {patches_dir}")
                    break
        
        # 仍然找不到patches目录
        if not patches_dir.exists():
            self.logger.error(f"No patches directory found at or under {patched_wsi_dir}")
            # 列出临时目录结构以便调试
            self.logger.info(f"Directory structure of {patched_wsi_dir}: {list(patched_wsi_dir.glob('**/*'))[:20]}")
            return []
            
        all_patches = list(patches_dir.glob("*.png"))
        self.logger.info(f"Found {len(all_patches)} patches in {patches_dir}")
        
        # If we have fewer patches than requested, use all of them
        if len(all_patches) <= num_samples:
            self.logger.warning(f"Only {len(all_patches)} patches available, using all {len(all_patches)} patches")
            return all_patches
            
        # Randomly select patches
        selected_patches = random.sample(all_patches, num_samples)
        self.logger.info(f"Randomly selected {len(selected_patches)} patches from {len(all_patches)} total patches")
        
        return selected_patches
        
    def _create_sampled_dataset(self, temp_dir, selected_patches, wsi_name):
        """Create a new dataset with only the selected patches"""
        # Create output directory for sampled patches
        sampled_dir = self.output_path / f"sampled_{wsi_name}"
        sampled_dir.mkdir(exist_ok=True, parents=True)
        patches_dir = sampled_dir / "patches"
        patches_dir.mkdir(exist_ok=True)
        metadata_dir = sampled_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        
        # 尝试在temp_dir或其子目录下找metadata文件
        metadata_file = None
        if (temp_dir / "metadata.json").exists():
            metadata_file = temp_dir / "metadata.json"
        elif (temp_dir / "metadata.yaml").exists():
            metadata_file = temp_dir / "metadata.yaml"
        else:
            # 查找可能在wsi_name子目录下的metadata文件
            wsi_subdir = temp_dir / wsi_name
            if wsi_subdir.exists():
                if (wsi_subdir / "metadata.json").exists():
                    metadata_file = wsi_subdir / "metadata.json"
                elif (wsi_subdir / "metadata.yaml").exists():
                    metadata_file = wsi_subdir / "metadata.yaml"
        
        if metadata_file:
            self.logger.info(f"Found metadata file: {metadata_file}")
            shutil.copy(metadata_file, sampled_dir / metadata_file.name)
        else:
            self.logger.warning(f"No metadata file found in {temp_dir} or its subdirectories")
        
        # Copy selected patches and their metadata
        copied_count = 0
        patch_metadata_list = []
        
        for patch_path in selected_patches:
            try:
                # Copy patch
                patch_filename = patch_path.name
                shutil.copy(patch_path, patches_dir / patch_filename)
                copied_count += 1
                
                # 尝试找到对应的metadata文件
                # 首先尝试直接在metadata目录下查找
                metadata_path = patch_path.parent.parent / "metadata" / f"{patch_path.stem}.yaml"
                if not metadata_path.exists():
                    # 然后尝试在嵌套目录中查找
                    metadata_path = patch_path.parent.parent.parent / "metadata" / f"{patch_path.stem}.yaml"
                
                if metadata_path.exists():
                    shutil.copy(metadata_path, metadata_dir / f"{patch_path.stem}.yaml")
                    # 添加到patch_metadata列表中
                    relative_metadata_path = f"metadata/{patch_path.stem}.yaml"
                    patch_metadata_entry = {
                        patch_filename: {
                            "metadata_path": relative_metadata_path
                        }
                    }
                    patch_metadata_list.append(patch_metadata_entry)
                else:
                    self.logger.warning(f"No metadata found for patch: {patch_path.name}")
            except Exception as e:
                self.logger.error(f"Error copying patch {patch_path}: {e}")
        
        # 创建patch_metadata.json文件
        if patch_metadata_list:
            self.logger.info(f"Creating patch_metadata.json with {len(patch_metadata_list)} entries")
            with open(sampled_dir / "patch_metadata.json", "w") as f:
                import json
                json.dump(patch_metadata_list, f, indent=2)
        else:
            self.logger.error("No patch metadata collected, can't create patch_metadata.json")
            
        # 如果没有从原始目录复制metadata.yaml，创建一个基本的版本
        if not (sampled_dir / "metadata.yaml").exists() and not (sampled_dir / "metadata.json").exists():
            self.logger.info("Creating basic metadata.yaml file")
            # 可能需要从原始temp_dir/WSI_NAME中复制元数据或手动创建
            try:
                # 尝试查找原始的patch提取配置，从中复制关键信息
                source_metadata = None
                patches_info = {}
                
                # 尝试从第一个patch的元数据中获取基本信息
                first_patch_metadata = None
                if patch_metadata_list and len(patch_metadata_list) > 0:
                    first_patch_name = list(patch_metadata_list[0].keys())[0]
                    first_patch_meta_path = patch_metadata_list[0][first_patch_name]["metadata_path"]
                    first_patch_meta_full_path = sampled_dir / first_patch_meta_path
                    
                    if first_patch_meta_full_path.exists():
                        try:
                            import yaml
                            with open(first_patch_meta_full_path, "r") as f:
                                first_patch_metadata = yaml.safe_load(f)
                        except Exception as e:
                            self.logger.error(f"Error loading first patch metadata: {e}")
                
                # 创建一个基本的metadata.yaml
                basic_metadata = {
                    "patch_size": self.patch_size,
                    "downsampling": 1,  # 默认值，可能需要从原始metadata中获取
                    "wsi_name": wsi_name,
                    "patch_overlap": self.patch_overlap,
                    "magnification": 40,  # 默认值，可能需要从原始metadata中获取
                    "base_magnification": 40,  # 默认值，可能需要从原始metadata中获取
                    "label_map": {"Background": 0, "Tumor": 1},  # 默认值
                    "number_patches": len(patch_metadata_list)
                }
                
                # 如果有第一个patch的元数据，尝试从中获取更准确的信息
                if first_patch_metadata:
                    if "magnification" in first_patch_metadata:
                        basic_metadata["magnification"] = first_patch_metadata["magnification"]
                    if "base_magnification" in first_patch_metadata:
                        basic_metadata["base_magnification"] = first_patch_metadata["base_magnification"]
                    if "downsampling" in first_patch_metadata:
                        basic_metadata["downsampling"] = first_patch_metadata["downsampling"]
                
                # 写入metadata.yaml
                import yaml
                with open(sampled_dir / "metadata.yaml", "w") as f:
                    yaml.dump(basic_metadata, f)
                    
                self.logger.info(f"Created basic metadata.yaml in {sampled_dir}")
            except Exception as e:
                self.logger.error(f"Failed to create metadata.yaml: {e}")
                        
        self.logger.info(f"Created sampled dataset at {sampled_dir} with {copied_count} patches")
        return sampled_dir
        
    def _run_nuclei_detection(self, sampled_dir, wsi_name, original_slide_path):
        """Run nuclei detection on sampled patches"""
        # 检查必要文件是否存在
        required_files = [
            "patch_metadata.json",
            "metadata.yaml"
        ]
        
        missing_files = []
        for file in required_files:
            if not (sampled_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            self.logger.error(f"Missing required files for nuclei detection: {missing_files}")
            self.logger.info(f"Directory contents: {list(sampled_dir.glob('*'))}")
            return
        
        # Create WSI object with required parameters
        self.logger.info(f"Creating WSI object for {wsi_name} at {sampled_dir}")
        try:
            # 在这里添加所需的参数
            wsi = WSI(
                name=wsi_name,
                patient=wsi_name,  # 使用WSI名称作为患者ID
                slide_path=str(original_slide_path),  # 使用原始WSI路径
                patched_slide_path=sampled_dir  # 提供处理后的幻灯片路径
            )
            
            # Run nuclei detection
            self.logger.info(f"Running nuclei detection on {wsi_name}")
            self.nuclei_detector.process_wsi(
                wsi=wsi,
                subdir_name=None,
                patch_size=self.patch_size,
                overlap=self.patch_overlap,
                batch_size=16,
                geojson=True,
                save_all_files=self.save_all_files,
            )
            
            self.logger.info(f"Nuclei detection completed for {wsi_name}")
            
            # 检查结果
            cell_detection_dir = sampled_dir / "cell_detection"
            if cell_detection_dir.exists():
                self.logger.info(f"Cell detection results available at: {cell_detection_dir}")
                try:
                    files = list(cell_detection_dir.glob("*"))
                    self.logger.info(f"Cell detection files: {[f.name for f in files]}")
                except Exception as e:
                    self.logger.error(f"Error listing cell detection files: {e}")
            else:
                self.logger.error(f"Cell detection directory not created: {cell_detection_dir}")
        except Exception as e:
            self.logger.error(f"Error in nuclei detection: {e}")
            import traceback
            self.logger.error(traceback.format_exc())

    def visualize_sampled_patches(
        self,
        wsi_file: Path,
        selected_patches: List[Path],
        sampled_dir: Path
    ):
        """在WSI缩略图上可视化随机抽样的patch位置"""
        if not self.save_visualization:
            self.logger.info("Visualization is disabled, skipping...")
            return None
        
        import yaml
        import re
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        import time
        import openslide
        
        self.logger.info(f"为WSI {wsi_file.stem}创建随机抽样patch可视化图像...")
        start_time = time.time()
        
        try:
            # 1. 直接从原始WSI文件获取无网格的缩略图
            slide = openslide.OpenSlide(str(wsi_file))
            
            # 获取MPP值和放大倍数
            if "openslide.mpp-x" in slide.properties:
                slide_mpp = float(slide.properties.get("openslide.mpp-x"))
            else:
                slide_mpp = self.mpp  # 使用默认值
                
            slide_properties = {"mpp": slide_mpp}
            
            # 使用自己定义的生成缩略图函数，避免使用可能包含网格的缩略图
            sample_factor = 128  # 使用较小的下采样因子获得更清晰的图像
            thumbnails = generate_thumbnails(
                slide, 
                slide_properties["mpp"], 
                sample_factors=[sample_factor]
            )
            
            # 获取缩略图 - 从您的函数中直接获取干净的缩略图
            thumbnail = thumbnails[f"downsample_{sample_factor}"] 
            if not thumbnail:
                thumbnail = thumbnails.get("thumbnail")
            
            # 确保是RGB格式
            if thumbnail.mode != 'RGB':
                thumbnail = thumbnail.convert('RGB')
                
            self.logger.info(f"成功获取WSI缩略图，尺寸: {thumbnail.width}x{thumbnail.height}")
            
            # 2. 加载元数据信息
            metadata_dir = sampled_dir / "metadata"
            if not metadata_dir.exists():
                self.logger.error(f"元数据目录不存在: {metadata_dir}")
                return None
                
            # 获取全局元数据
            global_metadata = {}
            global_metadata_path = sampled_dir / "metadata.yaml"
            if global_metadata_path.exists():
                try:
                    with open(global_metadata_path, 'r') as f:
                        global_metadata = yaml.safe_load(f)
                except Exception as e:
                    self.logger.warning(f"读取全局元数据时出错: {e}")
            
            # 获取关键参数
            downsample = global_metadata.get('downsampling', 1)
            patch_size = global_metadata.get('patch_size', self.patch_size)
            patch_overlap = global_metadata.get('patch_overlap', self.patch_overlap)
            
            self.logger.info(f"全局参数: downsample={downsample}, patch_size={patch_size}, overlap={patch_overlap}")
            
            # 3. 解析所有选中patch的位置
            patch_locations = []
            patch_info = []
            
            for idx, patch_path in enumerate(selected_patches):
                try:
                    # 从文件名中提取信息
                    patch_name = patch_path.name
                    patch_stem = patch_path.stem
                    
                    # 寻找对应的元数据文件
                    metadata_file = None
                    
                    # 首先尝试直接匹配
                    direct_match = metadata_dir / f"{patch_stem}.yaml"
                    if direct_match.exists():
                        metadata_file = direct_match
                    else:
                        # 尝试根据截图中的命名模式构建路径
                        # 格式: {wsi_name}_Y_{row}_{col}.yaml
                        metadata_files = list(metadata_dir.glob(f"*_{patch_stem.split('_')[-2]}_{patch_stem.split('_')[-1]}.yaml"))
                        if metadata_files:
                            metadata_file = metadata_files[0]
                        else:
                            # 如果以上方法都失败，尝试遍历目录查找包含相似模式的文件
                            similar_files = []
                            for mf in metadata_dir.glob("*.yaml"):
                                if patch_stem.split('_')[-2] in mf.name and patch_stem.split('_')[-1] in mf.name:
                                    similar_files.append(mf)
                            if similar_files:
                                metadata_file = similar_files[0]
                    
                    # 如果找到了元数据文件，解析它
                    if metadata_file:
                        with open(metadata_file, 'r') as f:
                            metadata = yaml.safe_load(f)
                        
                        # 获取行列信息
                        if 'row' in metadata and 'col' in metadata:
                            row = int(metadata['row'])
                            col = int(metadata['col'])
                            
                            # 计算原始坐标
                            x_coord = col * (patch_size - patch_overlap) * downsample
                            y_coord = row * (patch_size - patch_overlap) * downsample
                            
                            # 计算缩略图坐标
                            thumb_x = int(x_coord / sample_factor)
                            thumb_y = int(y_coord / sample_factor)
                            thumb_size = max(5, int(patch_size / sample_factor))
                            
                            # 添加到位置列表
                            patch_locations.append((thumb_x, thumb_y, thumb_size))
                            patch_info.append({
                                'index': idx,
                                'file': patch_name,
                                'row': row,
                                'col': col,
                                'original_coords': (x_coord, y_coord),
                                'thumb_coords': (thumb_x, thumb_y),
                                'background_ratio': metadata.get('background_ratio', 0)
                            })
                    else:
                        # 如果找不到元数据，尝试从文件名解析
                        # 例如: patches/wsi_name_Y_1_26.png 表示 row=1, col=26
                        match = re.search(r'_(\d+)_(\d+)\.', patch_name)
                        if match:
                            row = int(match.group(1))
                            col = int(match.group(2))
                            
                            # 计算坐标
                            x_coord = col * (patch_size - patch_overlap) * downsample
                            y_coord = row * (patch_size - patch_overlap) * downsample
                            
                            # 计算缩略图坐标
                            thumb_x = int(x_coord / sample_factor)
                            thumb_y = int(y_coord / sample_factor)
                            thumb_size = max(5, int(patch_size / sample_factor))
                            
                            patch_locations.append((thumb_x, thumb_y, thumb_size))
                            patch_info.append({
                                'index': idx,
                                'file': patch_name,
                                'row': row,
                                'col': col,
                                'original_coords': (x_coord, y_coord),
                                'thumb_coords': (thumb_x, thumb_y),
                                'background_ratio': 0  # 无法获取，使用默认值
                            })
                        else:
                            self.logger.warning(f"无法解析文件名中的行列信息: {patch_name}")
                except Exception as e:
                    self.logger.warning(f"处理patch {patch_path.name}时出错: {e}")
            
            self.logger.info(f"成功解析{len(patch_locations)}/{len(selected_patches)}个patch位置")
            
            # 4. 在缩略图上绘制patch (确保只使用原始缩略图，不使用任何带网格的图像)
            vis_img = thumbnail.copy()
            draw = ImageDraw.Draw(vis_img)
            
            # 尝试加载字体
            try:
                font = ImageFont.truetype("Arial", 12)
            except IOError:
                try:
                    font = ImageFont.truetype("DejaVuSans.ttf", 12)
                except IOError:
                    font = ImageFont.load_default()
            
            # 使用不同颜色标记patch，提高可视化效果
            #colors = ["red", "green", "blue", "yellow", "cyan", "magenta"]
            
            # 在缩略图上绘制所有patch
            for i, (x, y, size) in enumerate(patch_locations):
                #color_idx = (i // 100) % len(colors)  # 每100个patch一种颜色
                
                # 绘制矩形标记patch位置
                draw.rectangle(
                    [x, y, x + size, y + size],
                    outline="red",  # 使用红色边框
                    width=1
                )
            
            # 添加标题信息
            title_height = 40
            title_img = Image.new('RGB', (vis_img.width, title_height), color='black')
            title_draw = ImageDraw.Draw(title_img)

            title_text = f"WSI: {wsi_file.stem} - {len(selected_patches)} sampled patches visualization"
            title_draw.text((10, 10), title_text, fill='white', font=font)
            
            # 合并标题和图像
            combined_img = Image.new('RGB', (vis_img.width, vis_img.height + title_height))
            combined_img.paste(title_img, (0, 0))
            combined_img.paste(vis_img, (0, title_height))
            
            # 5. 保存可视化图像
            vis_path = sampled_dir / f"{wsi_file.stem}_random_sampled_patches.png"
            combined_img.save(vis_path)
            self.logger.info(f"保存可视化图像至: {vis_path}")
            
            # 6. 保存matplotlib版本以获得更好的质量
            plt.figure(figsize=(12, 10))
            plt.imshow(np.array(vis_img))
            plt.title(title_text, fontsize=14)
            plt.axis('off')
            plt.tight_layout()
            
            mpl_vis_path = sampled_dir / f"{wsi_file.stem}_random_sampled_patches_mpl.png"
            plt.savefig(mpl_vis_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 7. 保存详细的patch位置信息
            info_path = sampled_dir / f"{wsi_file.stem}_random_patch_locations.txt"
            with open(info_path, 'w') as f:
                f.write(f"WSI: {wsi_file.stem}\n")
                f.write(f"处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总patch数: {len(selected_patches)}\n")
                f.write(f"可视化patch数: {len(patch_locations)}\n")
                f.write(f"缩略图下采样因子: {sample_factor}\n")
                f.write(f"缩略图尺寸: {thumbnail.width}x{thumbnail.height}\n\n")
                
                f.write("Patch位置信息:\n")
                sorted_info = sorted(patch_info, key=lambda x: x['index'])
                for info in sorted_info:
                    f.write(f"Patch {info['index']+1}: {info['file']}\n")
                    f.write(f"  行列: (row={info['row']}, col={info['col']})\n")
                    f.write(f"  原始坐标: ({info['original_coords'][0]}, {info['original_coords'][1]})\n")
                    f.write(f"  缩略图坐标: ({info['thumb_coords'][0]}, {info['thumb_coords'][1]})\n")
                    f.write(f"  背景比例: {info['background_ratio']}\n\n")
            
            self.logger.info(f"可视化完成，用时: {time.time() - start_time:.2f}秒")
            # 9. 创建热力图版本 - 这可以让您看到patch分布情况
            self.create_heatmap_visualization(patch_locations, thumbnail, wsi_file.stem, sampled_dir)
            return vis_path
        
        except Exception as e:
            self.logger.error(f"创建可视化图像时出错: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
        
    def create_heatmap_visualization(self, patch_locations, thumbnail, wsi_name, output_dir):
        """创建patch分布热力图
        
        Args:
            patch_locations: patch位置列表 [(x, y, size), ...]
            thumbnail: WSI缩略图
            wsi_name: WSI名称
            output_dir: 输出目录
        """
        if not self.save_visualization:
            return
        
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.ndimage import gaussian_filter
        
        try:
            # 创建热力图
            heatmap = np.zeros((thumbnail.height, thumbnail.width))
            
            # 将所有patch位置添加到热力图
            for x, y, size in patch_locations:
                # 确保坐标在图像范围内
                if 0 <= x < thumbnail.width and 0 <= y < thumbnail.height:
                    for i in range(size):
                        for j in range(size):
                            if 0 <= x+i < thumbnail.width and 0 <= y+j < thumbnail.height:
                                heatmap[y+j, x+i] += 1
            
            # 应用高斯平滑
            sigma = 5
            heatmap = gaussian_filter(heatmap, sigma)
            
            # 创建图形
            plt.figure(figsize=(16, 14))
            
            # 绘制缩略图
            plt.imshow(np.array(thumbnail))
            
            # 叠加热力图
            plt.imshow(heatmap, alpha=0.5, cmap='hot')
            
            # 设置标题
            plt.title(f"WSI: {wsi_name} - Patch Distribution Heatmap", fontsize=16)
            
            # 添加颜色条
            cbar = plt.colorbar()
            cbar.set_label('Patch Density', rotation=270, labelpad=15)
            
            # 关闭坐标轴
            plt.axis('off')
            plt.tight_layout()
            
            # 保存图像
            heatmap_path = output_dir / f"{wsi_name}_patch_distribution_heatmap.png"
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"保存热力图至: {heatmap_path}")
        except Exception as e:
            self.logger.warning(f"创建热力图时出错: {e}")
        
    def process_single_wsi(self, wsi_info):
        """Process a single WSI file"""
        wsi_file = wsi_info['path']
        self.logger.info(f"Processing WSI: {wsi_file}")
        
        # 如果有标签信息，在日志中显示
        if 'label' in wsi_info:
            self.logger.info(f"WSI Label: {wsi_info['label']}")
        
        try:
            # Setup temporary extraction
            temp_dir, config = self._setup_temporary_extraction(wsi_file)
            self.logger.info(f"Temporary directory created: {temp_dir}")
                    
            # Extract all patches
            self.logger.info(f"Extracting patches from {wsi_file} using {self.hardware_selection}")
            try:
                slide_processor = PreProcessor(slide_processor_config=config)
                slide_processor.sample_patches_dataset()
                self.logger.info("Patch extraction completed successfully")
            except Exception as e:
                # 如果使用当前库失败，尝试切换到另一个库
                if "more than one image with Subfile Type 0" in str(e) or "is not supported" in str(e):
                    self.logger.error(f"Error with {self.hardware_selection}: {e}")
                    
                    # 切换到另一个库
                    new_hardware = "openslide" if self.hardware_selection == "cucim" else "cucim"
                    self.logger.info(f"Trying with {new_hardware} instead")
                    
                    # 保存原始选择以便之后恢复
                    original_hardware = self.hardware_selection
                    self.hardware_selection = new_hardware
                    
                    # 重新创建配置并尝试提取
                    try:
                        # 确保干净的环境
                        if temp_dir.exists():
                            shutil.rmtree(temp_dir)
                            
                        temp_dir, config = self._setup_temporary_extraction(wsi_file)
                        slide_processor = PreProcessor(slide_processor_config=config)
                        slide_processor.sample_patches_dataset()
                        self.logger.info(f"Patch extraction completed successfully using {new_hardware}")
                    except Exception as retry_error:
                        self.logger.error(f"Failed with {new_hardware} as well: {retry_error}")
                        # 恢复原始选择
                        self.hardware_selection = original_hardware
                        return
                    
                    # 恢复原始选择
                    self.hardware_selection = original_hardware
                else:
                    self.logger.error(f"Error during patch extraction: {e}")
                    raise
            
            # Randomly select patches
            self.logger.info("Starting random patch selection")
            selected_patches = self._randomly_select_patches(temp_dir, self.num_samples)
            self.logger.info(f"Selected {len(selected_patches)} patches")
            
            if not selected_patches:
                self.logger.error(f"No patches were selected for {wsi_file}")
                return
                    
            # Create sampled dataset
            self.logger.info(f"Creating sampled dataset for {wsi_file.stem}")
            try:
                sampled_dir = self._create_sampled_dataset(temp_dir, selected_patches, wsi_file.stem)
                self.logger.info(f"Sampled dataset created at: {sampled_dir}")
                
                if self.save_visualization:
                    # 调用可视化函数，确保传入的是selected_patches参数
                    vis_path = self.visualize_sampled_patches(
                        wsi_file=wsi_file,
                        selected_patches=selected_patches,  # 这里使用randomly_selected_patches
                        sampled_dir=sampled_dir
                    )
                    
                    if vis_path:
                        self.logger.info(f"Visualization created successfully: {vis_path}")
                    else:
                        self.logger.warning("Failed to create visualization")
                
                # Check if directory was actually created
                if not sampled_dir.exists():
                    self.logger.error(f"Failed to create sampled directory: {sampled_dir}")
                    return
                    
                # Check contents
                self.logger.info(f"Sampled directory contents: {list(sampled_dir.glob('*'))}")
                
                # Run nuclei detection
                self.logger.info(f"Starting nuclei detection for {wsi_file.stem}")
                self._run_nuclei_detection(sampled_dir, wsi_file.stem, wsi_file)
                
                # Verify cell_detection directory
                cell_detection_dir = sampled_dir / "cell_detection"
                if cell_detection_dir.exists():
                    self.logger.info(f"Cell detection results created at: {cell_detection_dir}")
                    self.logger.info(f"Cell detection directory contents: {list(cell_detection_dir.glob('*'))}")
                else:
                    self.logger.error(f"Cell detection directory not created: {cell_detection_dir}")
                
                # Clean up temporary directory
                self.logger.info(f"Cleaning up temporary directory {temp_dir}")
                shutil.rmtree(temp_dir)
                
            except Exception as e:
                self.logger.error(f"Error creating sampled dataset or running nuclei detection: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
            
        except Exception as e:
            self.logger.error(f"Error processing WSI {wsi_file}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
        
    def process(self):
        """Process all WSI files"""
        self.logger.info(f"开始处理 {len(self.wsi_files_info)} 个WSI文件")
        
        for i, wsi_info in enumerate(self.wsi_files_info):
            self.logger.info(f"处理进度: {i+1}/{len(self.wsi_files_info)}")
            self.process_single_wsi(wsi_info)
        
        self.logger.info("所有WSI文件处理完成")


if __name__ == "__main__":
    # Suppress duplicate logging
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    # Set logging level for all loggers
    logging.basicConfig(level=logging.ERROR)
    for logger_name in logging.root.manager.loggerDict:
        logging.getLogger(logger_name).setLevel(logging.ERROR)
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Randomly sample patches from WSI and perform nuclei detection")
    
    # 输入方式选择（互斥组）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--wsi_path", type=str, help="Path to WSI file or directory of WSI files")
    input_group.add_argument("--csv_path", type=str, help="Path to CSV file containing WSI file paths")
    
    # CSV相关参数
    parser.add_argument("--filename_column", type=str, default="filename", help="Column name containing file paths in CSV (default: filename)")
    parser.add_argument("--label_column", type=str, default="label", help="Column name containing labels in CSV (default: label)")
    parser.add_argument("--filter_by_label", type=int, help="Filter files by specific label value (e.g., 0 or 1)")
    
    # 其他参数
    parser.add_argument("--output_path", type=str, required=True, help="Path to save results")
    parser.add_argument("--model_path", type=str, required=True, help="Path to nuclei detection model")
    parser.add_argument("--num_samples", type=int, default=600, help="Number of patches to sample (if fewer are available, all will be used)")
    parser.add_argument("--patch_size", type=int, default=1024, help="Size of patches")
    parser.add_argument("--patch_overlap", type=int, default=64, help="Overlap between patches (in pixels)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--hardware_selection", type=str, default="openslide", choices=["openslide", "cucim"], 
                        help="Hardware selection for slide reading (openslide or cucim)")
    parser.add_argument("--mpp", type=float, default=0.5, 
                    help="Microns per pixel (default: 0.5)")
    parser.add_argument("--save_all_files", action="store_true", 
                help="Save all output files including json and pt files. If not set, only geojson files will be saved.")
    parser.add_argument("--save_visualization", action="store_true", default=True,
                    help="Save visualization images of sampled patches (default: True)")
    parser.add_argument("--no_visualization", dest="save_visualization", action="store_false",
                        help="Skip generating visualization images")
    
    args = parser.parse_args()
    
    sampler = RandomPatchSampler(
        wsi_path=args.wsi_path,
        csv_path=args.csv_path,  # 新增
        filename_column=args.filename_column,  # 新增
        label_column=args.label_column,  # 新增
        filter_by_label=args.filter_by_label,  # 新增
        output_path=args.output_path,
        model_path=args.model_path,
        num_samples=args.num_samples,
        patch_size=args.patch_size,
        patch_overlap=args.patch_overlap,
        gpu=args.gpu,
        random_seed=args.random_seed,
        hardware_selection=args.hardware_selection,
        mpp=args.mpp,
        save_all_files=args.save_all_files,
        save_visualization=args.save_visualization
    )
    
    sampler.process()