import numpy as np
import pandas as pd
from skimage.measure import regionprops
from skimage.feature import graycomatrix, graycoprops
import openslide
import os
import logging
from datetime import datetime
import json


def rgb2gray(rgb):
    # matlab's (NTSC/PAL) implementation:
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    # Replace NaN with 0
    gray = np.nan_to_num(gray, nan=0.0)
    return gray.astype(np.uint8)


class SlideNucStatObject:
    """细胞核统计对象
    
    Author: raycaohmu
    Date: 2025-06-02 12:33:30
    """
    def __init__(self, region, intensity_image=None, logger=None):
        self.region = region
        self._intensity_image = intensity_image
        self.label = region.label
        self.logger = logger or logging.getLogger(__name__)
        
        # 获取区域对应的强度图像
        if intensity_image is not None:
            self.intensity_image = intensity_image[
                region.slice[0].start:region.slice[0].stop,
                region.slice[1].start:region.slice[1].stop
            ]
        else:
            self.intensity_image = None
            
        # 获取掩码
        self.mask = region.image
        
        # 计算特征时使用的mask区域
        self.mask_indices = np.where(self.mask)

    def get_boundary_features(self):
        """获取边界特征"""
        try:
            area = self.region.area
            perimeter = self.region.perimeter
            convex_area = self.region.convex_area
            equivalent_diameter = self.region.equivalent_diameter
            major_axis_length = self.region.major_axis_length
            minor_axis_length = self.region.minor_axis_length
            orientation = self.region.orientation
            features = {
                # 基础特征
                'area': area,
                'perimeter': perimeter,
                #'perimeter_crofton': self.region.perimeter_crofton,
                #'bbox_area': self.region.bbox_area,
                'convex_area': convex_area,
                'equivalent_diameter': equivalent_diameter,
                'major_axis_length': major_axis_length,
                'minor_axis_length': minor_axis_length,
                'orientation': orientation,
                "orientation_degree": orientation * (180 / np.pi) + 90,
                'eccentricity': self.region.eccentricity,
                'solidity': self.region.solidity,
                'extent': self.region.extent,  # 新增
                "major_minor_ratio": major_axis_length / minor_axis_length,
                
                # 计算的特征
                'aspect_ratio': major_axis_length / 
                            (minor_axis_length + 1e-6),
                #'compactness': (4 * np.pi * area) / 
                #            (perimeter ** 2 + 1e-6),
                'roundness': (4 * area) / 
                            (np.pi * major_axis_length ** 2 + 1e-6),  # 新增
                'convexity': convex_area / (area + 1e-6),  # 新增
                'pa_ratio': (perimeter ** 2) / (area + 1e-6)
            }
            # 添加球形体积特征
            sfer_volume = (4/3) * np.pi * (np.sqrt(area / (4 * np.pi)) ** 3)
            #features['sfer_volume'] = sfer_volume
            # 添加椭球体积特征
            ellis_volume = (4/3) * np.pi * (
                (minor_axis_length / 2) ** 2) * (major_axis_length / 2)
            features.update({
                "volume_ratio": ellis_volume / (sfer_volume + 1e-6)
            })
            
            # 添加轮廓不规则度指标
            if hasattr(self.region, 'coords'):
                centroid = self.region.centroid
                coords = self.region.coords
                distances = np.sqrt(np.sum((coords - centroid)**2, axis=1))
                features.update({
                    'contour_std': np.std(distances),
                    'contour_mean': np.mean(distances),
                    'contour_irregularity': np.std(distances) / (np.mean(distances) + 1e-6)
                })
                
            return features
            
        except Exception as e:
            self.logger.error(f"Error computing boundary features: {str(e)}")
            return {}

    def get_intensity_features(self):
        """获取强度特征"""
        try:
            if self.intensity_image is None or self.mask is None:
                return {}
                
            # 获取mask区域内的像素值
            pixel_values = self.intensity_image[self.mask]
            
            # 基础统计特征
            features = {
                'mean_intensity': np.mean(pixel_values),
                'max_intensity': np.max(pixel_values),
                'min_intensity': np.min(pixel_values),
                'std_intensity': np.std(pixel_values),
                'variance_intensity': np.var(pixel_values),  # 新增
                'intensity_range': np.ptp(pixel_values),
                
                # 百分位数特征
                'intensity_quartile_25': np.percentile(pixel_values, 25),
                'intensity_median': np.median(pixel_values),
                'intensity_quartile_75': np.percentile(pixel_values, 75),
                'intensity_iqr': np.percentile(pixel_values, 75) - 
                            np.percentile(pixel_values, 25),
                'intensity_percentile_10': np.percentile(pixel_values, 10),  # 新增
                'intensity_percentile_90': np.percentile(pixel_values, 90),  # 新增
                
                # 形状特征
                'intensity_skewness': self._compute_skewness(pixel_values),
                'intensity_kurtosis': self._compute_kurtosis(pixel_values),
                
                # 模式相关特征
                'intensity_mode': self._compute_mode(pixel_values),  # 新增
                'intensity_energy': np.sum(pixel_values ** 2),  # 新增
                'intensity_entropy': self._compute_entropy(pixel_values),  # 新增
            }
            
            return features
            
        except Exception as e:
            logging.error(f"Error computing intensity features: {str(e)}")
            return {}

    def _compute_mode(self, data):
        """计算众数"""
        if len(data) == 0:
            return 0
        values, counts = np.unique(data, return_counts=True)
        return values[np.argmax(counts)]

    def _compute_entropy(self, data):
        """计算熵"""
        if len(data) == 0:
            return 0
        hist, _ = np.histogram(data, bins=256)
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))

    def get_texture_features(self):
        """获取纹理特征"""
        try:
            if self.intensity_image is None or not self.mask.any():
                return {}
                
            # 准备GLCM计算
            if self.intensity_image.max() <= 1.0:
                img = (self.intensity_image * 255).astype(np.uint8)
            else:
                img = self.intensity_image.astype(np.uint8)
                
            # 只使用mask区域
            masked_img = img.copy()
            masked_img[~self.mask] = 0
            
            # GLCM参数
            distances = [1, 2, 3]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            
            glcm = graycomatrix(
                masked_img, 
                distances=distances,
                angles=angles,
                symmetric=True,
                normed=True
            )
            
            # 计算GLCM属性
            texture_features = {}
            properties = [
                'contrast',       # 对比度
                'correlation',    # 相关性
                'energy',        # 能量
                'homogeneity',   # 同质性
                'dissimilarity', # 不相似性
                'ASM'           # 角二阶矩
            ]
            
            for prop in properties:
                values = graycoprops(glcm, prop)
                for d_idx, d in enumerate(distances):
                    for a_idx, a in enumerate(angles):
                        feature_name = f'texture_{prop}_d{d}_a{int(a*180/np.pi)}'
                        texture_features[feature_name] = values[d_idx, a_idx]
                        
            return texture_features
            
        except Exception as e:
            logging.error(f"Error computing texture features: {str(e)}")
            return {}

    def get_gradient_features(self):
        """获取梯度特征"""
        try:
            if self.intensity_image is None or not self.mask.any():
                return {}
                
            from skimage.filters import sobel, scharr, prewitt
            
            features = {}
            
            # 使用不同的梯度算子
            gradient_ops = {
                'sobel': sobel,
                'scharr': scharr,
                'prewitt': prewitt
            }
            
            for op_name, gradient_op in gradient_ops.items():
                # 计算梯度
                gradient = gradient_op(self.intensity_image)
                gradient_values = gradient[self.mask]
                
                prefix = f"gradient_{op_name}_"
                features.update({
                    f'{prefix}mean': np.mean(gradient_values),
                    f'{prefix}std': np.std(gradient_values),
                    f'{prefix}max': np.max(gradient_values),
                    f'{prefix}min': np.min(gradient_values),
                    f'{prefix}quartile_25': np.percentile(gradient_values, 25),
                    f'{prefix}median': np.median(gradient_values),
                    f'{prefix}quartile_75': np.percentile(gradient_values, 75),
                    f'{prefix}iqr': np.percentile(gradient_values, 75) - 
                                np.percentile(gradient_values, 25)
                })
                
            return features
            
        except Exception as e:
            logging.error(f"Error computing gradient features: {str(e)}")
            return {}
            
    def _compute_skewness(self, data):
        """计算偏度"""
        n = len(data)
        if n < 3:
            return 0
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            return 0
        return (np.sum((data - mean) ** 3) / ((n-1) * std ** 3))
        
    def _compute_kurtosis(self, data):
        """计算峰度"""
        n = len(data)
        if n < 4:
            return 0
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            return 0
        return (np.sum((data - mean) ** 4) / ((n-1) * std ** 4)) - 3
        
    def get_all_features(self):
        """获取所有特征"""
        features = {}
        
        # 添加各类特征
        features.update(self.get_boundary_features())
        #features.update(self.get_intensity_features())
        #features.update(self.get_texture_features())
        #features.update(self.get_gradient_features())
        
        return features
    

class WSIPatchFeatureExtractor:
    def __init__(self, wsi_path, patch_info, output_dir, save_log=False):
        """初始化WSI patch特征提取器
        
        Args:
            wsi_path (str): WSI文件路径
            patch_info (dict): patch相关信息
            output_dir (str): 输出目录
            save_log (bool, optional): 是否保存日志文件
        """
        self.wsi_path = wsi_path
        self.output_dir = output_dir
        self.patch_info = patch_info
        self.save_log = save_log
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置日志
        self.logger = self._setup_logger()
        
        # 初始化参数
        self._init_patch_params()
        
        # 记录初始化信息
        self.logger.info(f"Initialized WSIPatchFeatureExtractor")
        self.logger.info(f"WSI path: {wsi_path}")
        self.logger.info(f"Output directory: {output_dir}")
        self.logger.info(f"Patch info: {patch_info}")
        
    def _setup_logger(self):
        """设置日志系统"""
        logger = logging.getLogger('WSIPatchFeatureExtractor')
        logger.setLevel(logging.WARNING)  # 只输出警告和错误
        
        if not logger.handlers:
            # 创建控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 创建格式器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            
            # 添加控制台处理器
            logger.addHandler(console_handler)
            
            # 如果需要保存日志，添加文件处理器
            if self.save_log:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                log_file = os.path.join(
                    self.output_dir, 
                    f'feature_extraction_{timestamp}.log'
                )
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(logging.INFO)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                
        return logger
        
    def _init_patch_params(self):
        """初始化patch参数"""
        # patch基本参数
        self.patch_size = self.patch_info['patch_size']
        self.overlap = self.patch_size * self.patch_info['patch_overlap'] / 100.
        self.row = self.patch_info['row']
        self.col = self.patch_info['col']
        
        # 计算WSI中的位置
        self.wsi_scaling_factor = 1
        self.x_start = int(
            self.row * self.patch_size * self.wsi_scaling_factor
            - (self.row + 0.5) * self.overlap
        )
        self.y_start = int(
            self.col * self.patch_size * self.wsi_scaling_factor
            - (self.col + 0.5) * self.overlap
        )
        
    def create_cell_mask(self, cell_doc):
        """从细胞标注创建mask"""
        from skimage.draw import polygon
        
        # 创建label mask
        mask = np.zeros((self.patch_size, self.patch_size), dtype=np.uint16)
        cell_dict = []
        
        self.logger.info(f"Processing {len(cell_doc)} cells")
        
        for idx, row in cell_doc.iterrows():
            try:
                # 获取WSI坐标并转换为patch坐标
                wsi_coordinates = row.geometry['coordinates'][0][0]
                polygon_points = np.array([
                    self._convert_to_patch_coordinates(coord)
                    for coord in wsi_coordinates
                ])
                
                # 检查是否有点在patch范围内
                valid_points = (
                    (polygon_points[:, 0] >= 0) & 
                    (polygon_points[:, 0] < self.patch_size) & 
                    (polygon_points[:, 1] >= 0) & 
                    (polygon_points[:, 1] < self.patch_size)
                )
                
                if np.any(valid_points):
                    # 使用scikit-image的polygon函数直接在numpy数组上绘制多边形
                    rr, cc = polygon(
                        polygon_points[:, 1], 
                        polygon_points[:, 0], 
                        shape=mask.shape
                    )
                    mask[rr, cc] = idx + 1
                    
                    # 获取细胞信息
                    wsi_centroid = row.properties['measurements']['centroid']
                    probability = row.properties['measurements']['probability']
                    # 转换centroid为patch坐标
                    patch_centroid = self._convert_to_patch_coordinates(wsi_centroid)
                    
                    cell_info = {
                        'type': row.properties['classification']['name'],
                        'centroid_wsi': wsi_centroid,
                        'centroid_patch': patch_centroid,
                        'probability': probability,
                    }
                    cell_dict.append(cell_info)
                    
            except Exception as e:
                self.logger.error(f"Error processing cell {idx}: {str(e)}")
                continue
        
        return mask, cell_dict
    
    def _convert_to_patch_coordinates(self, wsi_coordinates):
        """将WSI坐标转换为patch局部坐标"""
        x = int(wsi_coordinates[0] - self.y_start)
        y = int(wsi_coordinates[1] - self.x_start)
        return [x, y]
    
    def compute_spatial_features(self, cell_dict):
        """计算空间特征
        
        Args:
            cell_dict (list): 细胞信息字典列表
            
        Returns:
            list: 空间特征列表
        """
        try:
            centroids = np.array([cell['centroid'] for cell in cell_dict])
            if len(centroids) < 3:
                return [self._empty_spatial_features()] * len(centroids)
                
            from scipy.spatial import Delaunay
            tri = Delaunay(centroids)
            features_list = []
            
            for i in range(len(centroids)):
                neighbors = []
                for simplex in tri.simplices:
                    if i in simplex:
                        neighbors.extend([j for j in simplex if j != i])
                neighbors = list(set(neighbors))
                
                if neighbors:
                    distances = np.linalg.norm(centroids[neighbors] - centroids[i], axis=1)
                    angles = np.arctan2(
                        centroids[neighbors][:,1] - centroids[i][1],
                        centroids[neighbors][:,0] - centroids[i][0]
                    )
                    
                    features = {
                        'spatial_mean_distance': np.mean(distances),
                        'spatial_std_distance': np.std(distances),
                        'spatial_min_distance': np.min(distances),
                        'spatial_max_distance': np.max(distances),
                        'spatial_num_neighbors': len(neighbors),
                        'spatial_mean_angle': np.mean(angles),
                        'spatial_std_angle': np.std(angles),
                        'spatial_angle_dispersion': np.std(angles) / (np.mean(angles) + 1e-6)
                    }
                else:
                    features = self._empty_spatial_features()
                    
                features_list.append(features)
                
            return features_list
            
        except Exception as e:
            self.logger.error(f"Error computing spatial features: {str(e)}")
            return [self._empty_spatial_features()] * len(cell_dict)
        
    def _empty_spatial_features(self):
        """返回空的空间特征"""
        return {
            'spatial_mean_distance': 0,
            'spatial_std_distance': 0,
            'spatial_min_distance': 0,
            'spatial_max_distance': 0,
            'spatial_num_neighbors': 0,
            'spatial_mean_angle': 0,
            'spatial_std_angle': 0,
            'spatial_angle_dispersion': 0
        }

    def extract_features(self, patch_image, cell_doc):
        """提取所有特征
        
        Args:
            patch_image: PIL.Image对象，patch图像
            cell_doc: pd.DataFrame，细胞标注信息
        """
        self.logger.info("Starting feature extraction")
        
        # 转换图像为numpy数组
        image_np = np.array(patch_image)
        if len(image_np.shape) == 3:
            # 使用标准的RGB转灰度公式
            r, g, b = image_np[:,:,0], image_np[:,:,1], image_np[:,:,2]
            intensity_image = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.uint8)
        else:
            intensity_image = image_np
            
        # 为了更好的特征提取，可以考虑保存原始RGB值
        self.rgb_image = image_np if len(image_np.shape) == 3 else None
                
        # 创建cell mask
        mask, cell_dict = self.create_cell_mask(cell_doc)
        
        if len(cell_dict) == 0:
            self.logger.warning("No valid cells found in this patch")
            return pd.DataFrame()
            
        # 获取区域属性
        regions = regionprops(mask, intensity_image=intensity_image)
        
        # 计算空间特征
        #spatial_features = self.compute_spatial_features(cell_dict)
        
        # 提取所有特征
        all_features = []
        for idx, (region, cell_info) in enumerate(
            zip(regions, cell_dict)
        ):
            try:
                # 创建SlideNucStatObject
                nuc_stat = SlideNucStatObject(region, intensity_image, logger=self.logger)
                
                # 获取基本信息
                bbox = region.bbox  # regionprops的bbox格式是(min_row, min_col, max_row, max_col)
                features = {
                    #'cell_id': cell_info['id'],
                    #'original_id': cell_info['original_id'],
                    'cell_type': cell_info['type'],
                    'probability': cell_info['probability'],
                    'centroid_x_wsi': cell_info['centroid_wsi'][0],
                    'centroid_y_wsi': cell_info['centroid_wsi'][1],
                    'centroid_x_patch': cell_info['centroid_patch'][0],
                    'centroid_y_patch': cell_info['centroid_patch'][1],
                    'coord_x_patch': self.y_start,
                    'coord_y_patch': self.x_start,
                    'bbox_x1': bbox[1],
                    'bbox_y1': bbox[0],
                    'bbox_x2': bbox[3],
                    'bbox_y2': bbox[2]
                }
                
                # 添加各类特征
                features.update(nuc_stat.get_all_features())
                #features.update(spatial_feat)
                
                all_features.append(features)
                
                if (idx + 1) % 100 == 0:
                    self.logger.info(f"Processed features for {idx + 1} cells")
                    
            except Exception as e:
                self.logger.error(f"Error processing features for cell {idx}: {str(e)}")
                continue
        
        features_df = pd.DataFrame(all_features)
        
        # 记录处理结果
        self.logger.info(f"Feature extraction complete:")
        self.logger.info(f"Input cells: {len(cell_doc)}")
        self.logger.info(f"Processed cells: {len(features_df)}")
        self.logger.info(f"Total features: {len(features_df.columns)}")
        
        return features_df
        
    def save_features(self, features_df, patch_name):
        """保存特征到文件
        
        Args:
            features_df (pd.DataFrame): 特征DataFrame
            patch_name (str): patch名称
            
        Returns:
            tuple: (特征文件路径, 元数据文件路径)
        """
        try:
            # 创建时间戳
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 保存特征
            output_path = os.path.join(
                self.output_dir,
                f"features_{patch_name}_{timestamp}.csv"
            )
            features_df.to_csv(output_path, index=False)
            
            # 保存元数据
            metadata = {
                'wsi_path': self.wsi_path,
                'patch_info': self.patch_info,
                'timestamp': timestamp,
                'user': os.getlogin(),
                'num_cells': len(features_df),
                'num_features': len(features_df.columns),
                'feature_names': list(features_df.columns),
                'cell_types': features_df['cell_type'].unique().tolist(),
                'cell_type_counts': features_df['cell_type'].value_counts().to_dict()
            }
            
            metadata_path = os.path.join(
                self.output_dir,
                f"metadata_{patch_name}_{timestamp}.json"
            )
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
                
            self.logger.info(f"Features saved to: {output_path}")
            self.logger.info(f"Metadata saved to: {metadata_path}")
            
            return output_path, metadata_path
            
        except Exception as e:
            self.logger.error(f"Error saving features: {str(e)}")
            return None, None

def main():
    """主函数示例"""
    # 配置参数
    wsi_path = "./data/wsi/CMU-1.svs"
    patch_info = {
        'patch_size': 1024,
        'patch_overlap': 6.25,
        'row': 1,
        'col': 31
    }
    output_dir = "./output/features"
    
    try:
        # 创建特征提取器
        extractor = WSIPatchFeatureExtractor(
            wsi_path=wsi_path,
            patch_info=patch_info,
            output_dir=output_dir
        )
        
        # 读取WSI和patch
        slide = openslide.OpenSlide(wsi_path)
        patch = slide.read_region(
            (extractor.y_start, extractor.x_start),
            level=0,
            size=(patch_info['patch_size'], patch_info['patch_size'])
        ).convert("RGB")
        
        # 读取geojson文件
        patch_name = f"patch_{patch_info['row']}_{patch_info['col']}"
        geojson_path = "/home/stat-caolei/code/MIA/data/wsi_output/CMU-1/cell_detection/patch_geojson/cells_patch_1_31.geojson"
        
        cell_doc = pd.read_json(geojson_path)
        
        # 提取特征
        features_df = extractor.extract_features(patch, cell_doc)
        import ipdb;ipdb.set_trace()
        
        # 保存特征
        if not features_df.empty:
            output_path, metadata_path = extractor.save_features(features_df, patch_name)
            
            if output_path and metadata_path:
                print(f"\nFeature extraction successful!")
                print(f"Features saved to: {output_path}")
                print(f"Metadata saved to: {metadata_path}")
                print(f"\nFeature summary:")
                print(f"Total cells processed: {len(features_df)}")
                print(f"Total features extracted: {len(features_df.columns)}")
                print("\nFeature preview:")
                print(features_df.head())
            else:
                print("Error saving features and metadata")
        else:
            print("No features were extracted")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        
if __name__ == "__main__":
    main()