import os
import pandas as pd
import openslide
import logging
from datetime import datetime
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from tqdm import tqdm
import json
from wsi_feature_extractor import WSIPatchFeatureExtractor
import cProfile
import pstats


class WSIProcessor:
    def __init__(self, wsi_path, geojson_dir, output_dir, patch_info=None, save_log=False):
        """初始化WSI处理器
        
        Args:
            wsi_path (str): WSI文件路径
            geojson_dir (str): 包含patch geojson文件的目录
            output_dir (str): 输出目录
            patch_info (dict, optional): patch相关信息
            save_log (bool, optional): 是否保存日志文件
        """
        self.wsi_path = wsi_path
        self.geojson_dir = geojson_dir
        self.output_dir = output_dir
        self.patch_info = patch_info or {
            'patch_size': 1024,
            #'patch_overlap': 0
            'patch_overlap': 6.25
        }
        self.save_log = save_log
        
        # 首先创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 然后设置日志
        self.logger = self._setup_logger()
        
        # 记录初始化信息
        self.logger.info(f"Initialized WSIProcessor for {self.wsi_path}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Geojson directory: {self.geojson_dir}")

        self._slide = None
        
    def _setup_logger(self):
        """设置日志系统"""
        logger = logging.getLogger(f'WSIProcessor_{Path(self.wsi_path).stem}')
        logger.setLevel(logging.ERROR)  # 只记录错误
        
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
                    f'wsi_processing_{Path(self.wsi_path).stem}_{timestamp}.log'
                )
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(logging.INFO)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                
        return logger
        
    def get_patch_files(self):
        """获取所有patch的geojson文件"""
        geojson_pattern = re.compile(r'cells_patch_(\d+)_(\d+)\.geojson$')
        patch_files = []
        patch_id = 0
        for file in os.listdir(self.geojson_dir):
            if file.endswith('.geojson'):
                match = geojson_pattern.match(file)
                if match:
                    patch_id += 1
                    row, col = map(int, match.groups())
                    patch_files.append({
                        'file': os.path.join(self.geojson_dir, file),
                        'row': row,
                        'col': col,
                        'patch_id': patch_id
                    })
                    
        return sorted(patch_files, key=lambda x: (x['row'], x['col']))
    
    def __del__(self):
        """析构函数确保slide对象被正确关闭"""
        if self._slide is not None:
            self._slide.close()
            self._slide = None

    @property
    def slide(self):
        """懒加载WSI文件"""
        if self._slide is None:
            self._slide = openslide.OpenSlide(self.wsi_path)
        return self._slide
        
    def process_single_patch(self, patch_file):
        try:
            patch_info = self.patch_info.copy()
            patch_info.update({
                'row': patch_file['row'],
                'col': patch_file['col']
            })
            
            extractor = WSIPatchFeatureExtractor(
                wsi_path=self.wsi_path,
                patch_info=patch_info,
                output_dir=self.output_dir,
                save_log=self.save_log
            )
            
            # 使用缓存的slide对象
            patch = self.slide.read_region(
                (extractor.y_start, extractor.x_start),
                level=0,
                size=(patch_info['patch_size'], patch_info['patch_size'])
            ).convert("RGB")
            
            # 读取geojson文件
            cell_doc = pd.read_json(patch_file['file'])
            
            # 提取特征
            features_df = extractor.extract_features(patch, cell_doc)
            
            if not features_df.empty:
                # 添加patch信息
                features_df['patch_row'] = patch_file['row']
                features_df['patch_col'] = patch_file['col']
                
            return features_df
            
        except Exception as e:
            self.logger.error(
                f"Error processing patch {patch_file['row']}_{patch_file['col']}: {str(e)}"
            )
            return pd.DataFrame()
            
    def process_wsi(self, max_workers=4):
        """处理整个WSI的所有patch"""
        try:
            patch_files = self.get_patch_files()
            # 预先读取slide对象
            slide = openslide.OpenSlide(self.wsi_path)
            
            # 使用ProcessPoolExecutor替代ThreadPoolExecutor
            # 因为Python的GIL限制，CPU密集型任务用多进程更好
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for patch_file in patch_files:
                    # 为每个patch创建独立的上下文
                    context = {
                        'patch_info': self.patch_info.copy(),
                        'wsi_path': self.wsi_path,
                        'output_dir': self.output_dir,
                        'patch_file': patch_file,
                        'save_log': self.save_log
                    }
                    futures.append(
                        executor.submit(self._process_patch_in_process, context)
                    )
                
                all_features = []
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Processing patches"
                ):
                    result = future.result()
                    if result is not None:
                        all_features.append(result)
            
            # 最后合并结果
            if all_features:
                final_df = pd.concat(all_features, ignore_index=True)
                return self._save_results(final_df, patch_files)
            return None
            
        finally:
            if slide:
                slide.close()

    @staticmethod
    def _process_patch_in_process(context):
        """在独立进程中处理patch"""
        try:
            patch_info = context['patch_info']
            patch_info.update({
                'row': context['patch_file']['row'],
                'col': context['patch_file']['col'],
                'patch_id': context['patch_file']['patch_id']
            })
            
            # 在进程中创建新的slide对象
            with openslide.OpenSlide(context['wsi_path']) as slide:
                extractor = WSIPatchFeatureExtractor(
                    wsi_path=context['wsi_path'],
                    patch_info=patch_info,
                    output_dir=context['output_dir'],
                    save_log=context['save_log']
                )
                
                # 读取patch
                patch = slide.read_region(
                    (extractor.y_start, extractor.x_start),
                    level=0,
                    size=(patch_info['patch_size'], patch_info['patch_size'])
                ).convert("RGB")
                
                # 读取geojson文件
                cell_doc = pd.read_json(context['patch_file']['file'])
                
                # 提取特征
                features_df = extractor.extract_features(patch, cell_doc)
                
                if not features_df.empty:
                    features_df['patch_row'] = context['patch_file']['row']
                    features_df['patch_col'] = context['patch_file']['col']
                    features_df['patch_id'] = context['patch_file']['patch_id']
                    
                return features_df
                
        except Exception as e:
            logging.error(f"Error processing patch: {str(e)}")
            return None

    def _save_results(self, all_features, patch_files):
        if all_features is None or all_features.empty:
            return None
            
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(
                self.output_dir,
                f"features_{Path(self.wsi_path).stem}_{timestamp}.csv"
            )
            
            # 直接保存，不使用分块
            all_features.to_csv(output_path, index=False)
            
            # 使用已有的DataFrame计算元数据
            metadata = {
                'wsi_path': self.wsi_path,
                'timestamp': timestamp,
                'user': os.getlogin(),
                'num_patches': len(patch_files),
                'num_cells': len(all_features),
                'num_features': len(all_features.columns),
                'feature_names': list(all_features.columns),
                'cell_types': all_features['cell_type'].unique().tolist(),
                'cell_type_counts': all_features['cell_type'].value_counts().to_dict()
            }
            
            metadata_path = os.path.join(
                self.output_dir,
                f"metadata_{Path(self.wsi_path).stem}_{timestamp}.json"
            )
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            return None
        

class WSIBatchProcessor:
    def __init__(self, wsi_dir, output_base_dir, wsi_output_dir, save_log=False):
        """初始化WSI批处理器
        
        Args:
            wsi_dir (str): 存放WSI文件的目录
            output_base_dir (str): 特征输出的基础目录
            wsi_output_dir (str): 存放WSI处理结果的目录
            save_log (bool, optional): 是否保存日志文件
        """
        self.wsi_dir = wsi_dir
        self.output_base_dir = output_base_dir
        self.wsi_output_dir = wsi_output_dir
        self.save_log = save_log
        
        # 创建输出目录
        os.makedirs(output_base_dir, exist_ok=True)
        
        # 设置日志
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """设置日志系统"""
        logger = logging.getLogger('WSIBatchProcessor')
        logger.setLevel(logging.INFO)
        
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
                    self.output_base_dir,
                    f'batch_processing_{timestamp}.log'
                )
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(logging.INFO)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                
        return logger
        
    def get_wsi_files(self):
        """获取所有WSI文件及其对应的geojson目录"""
        wsi_files = []
        
        wsi_dir_df = pd.read_csv(self.wsi_dir)
        # 遍历WSI目录
        for wsi_file_row in wsi_dir_df.itertuples():
            wsi_path = wsi_file_row.filename
            wsi_name = wsi_path.split("/")[-1].strip(".svs")
                
            # 构建对应的geojson目录路径
            geojson_dir = os.path.join(
                self.wsi_output_dir,
                f"sampled_{wsi_name}",
                'cell_detection',
                'patch_geojson'
            )
            
            if os.path.exists(geojson_dir):
                wsi_files.append({
                    'wsi_path': wsi_path,
                    'wsi_name': f"sampled_{wsi_name}",
                    'geojson_dir': geojson_dir
                })
            else:
                self.logger.warning(
                    f"Geojson directory not found for {wsi_file}: {geojson_dir}"
                )
                    
        return wsi_files
        
    def process_single_wsi(self, wsi_file):
        try:
            wsi_output_dir = os.path.join(
                self.output_base_dir,
                Path(wsi_file['wsi_path']).stem
            )
            os.makedirs(wsi_output_dir, exist_ok=True)
            
            processor = WSIProcessor(
                wsi_path=wsi_file['wsi_path'],
                geojson_dir=wsi_file['geojson_dir'],
                output_dir=wsi_output_dir,
                save_log=self.save_log
            )
            
            # 使用更多的线程处理patches
            return processor.process_wsi(max_workers=8)  # 增加线程数
            
        except Exception as e:
            self.logger.error(f"Error processing WSI {wsi_file['wsi_path']}: {str(e)}")
            return None
            
    def process_batch(self, max_workers=2):
        """处理所有WSI文件
        
        Args:
            max_workers (int): 最大进程数
        """
        self.logger.info("Starting batch processing")
        
        # 获取所有WSI文件
        wsi_files = self.get_wsi_files()
        self.logger.info(f"Found {len(wsi_files)} WSI files")
        
        # 使用进程池处理WSI
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_wsi = {
                executor.submit(self.process_single_wsi, wsi_file): wsi_file 
                for wsi_file in wsi_files
            }
            
            # 处理完成的任务
            results = []
            for future in tqdm(
                as_completed(future_to_wsi),
                total=len(wsi_files),
                desc="Processing WSIs"
            ):
                wsi_file = future_to_wsi[future]
                try:
                    output_path = future.result()
                    if output_path:
                        results.append({
                            'wsi': Path(wsi_file['wsi_path']).stem,
                            'output': output_path
                        })
                except Exception as e:
                    self.logger.error(
                        f"Error processing WSI {wsi_file['wsi_path']}: {str(e)}"
                    )
        
        # 保存处理结果摘要
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        summary_path = os.path.join(
            self.output_base_dir,  # 这里改用 output_base_dir
            f"processing_summary_{timestamp}.json"
        )
        
        with open(summary_path, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'total_wsi': len(wsi_files),
                'successful_wsi': len(results),
                'results': results
            }, f, indent=4)
            
        self.logger.info(f"Batch processing complete")
        self.logger.info(f"Total WSI: {len(wsi_files)}")
        self.logger.info(f"Successful: {len(results)}")
        self.logger.info(f"Summary saved to: {summary_path}")


def main():
    # 配置参数
    base_dir = os.path.dirname(os.path.abspath(__file__))
    #wsi_dir = os.path.join(base_dir, "data/wsi")
    wsi_dir = os.path.join(base_dir, "data/slide_ov_response.csv")
    wsi_output_dir = os.path.join(base_dir, "data/wsi_output")
    output_base_dir = os.path.join(base_dir, "output/features")
    save_log = False  # 设置是否保存日志
    
    try:
        # 确保目录存在
        os.makedirs(output_base_dir, exist_ok=True)
        
        if not os.path.exists(wsi_dir):
            raise FileNotFoundError(f"WSI directory not found: {wsi_dir}")
        if not os.path.exists(wsi_output_dir):
            raise FileNotFoundError(f"WSI output directory not found: {wsi_output_dir}")
        
        # 创建批处理器
        processor = WSIBatchProcessor(
            wsi_dir=wsi_dir,
            output_base_dir=output_base_dir,
            wsi_output_dir=wsi_output_dir,
            save_log=save_log  # 传递save_log参数
        )
        
        # 处理所有WSI
        processor.process_batch(max_workers=12)
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        

def profile_processing():
    pr = cProfile.Profile()
    pr.enable()
    
    # 运行你的处理代码
    main()
    
    pr.disable()
    stats = pstats.Stats(pr)
    stats.sort_stats('cumtime').print_stats(30)

if __name__ == '__main__':
    #profile_processing()
    main()
