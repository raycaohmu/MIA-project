import pandas as pd
import os
import logging
from openslide import OpenSlide, OpenSlideError
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
import time

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('svs_check.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def check_svs_file(file_path: str) -> Dict[str, any]:
    """
    检查单个SVS文件的完整性
    
    Args:
        file_path: SVS文件路径
        
    Returns:
        包含检查结果的字典
    """
    result = {
        'filename': file_path,
        'exists': False,
        'readable': False,
        'valid_format': False,
        'dimensions': None,
        'level_count': None,
        'error_message': None,
        'file_size_mb': None
    }
    
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            result['error_message'] = 'File does not exist'
            return result
        
        result['exists'] = True
        
        # 获取文件大小
        file_size = os.path.getsize(file_path)
        result['file_size_mb'] = round(file_size / (1024 * 1024), 2)
        
        # 检查文件是否为空
        if file_size == 0:
            result['error_message'] = 'File is empty (0 bytes)'
            return result
        
        # 尝试打开SVS文件
        try:
            slide = OpenSlide(file_path)
            result['readable'] = True
            result['valid_format'] = True
            
            # 获取基本信息
            result['dimensions'] = slide.dimensions
            result['level_count'] = slide.level_count
            
            # 尝试读取一个小的区域来验证数据完整性
            try:
                # 读取左上角的小块区域 (100x100)
                tile = slide.read_region((0, 0), 0, (100, 100))
                if tile.size == (0, 0):
                    result['error_message'] = 'Cannot read image data'
                    result['valid_format'] = False
            except Exception as e:
                result['error_message'] = f'Cannot read image data: {str(e)}'
                result['valid_format'] = False
            
            slide.close()
            
        except OpenSlideError as e:
            result['error_message'] = f'OpenSlide error: {str(e)}'
        except Exception as e:
            result['error_message'] = f'Unexpected error: {str(e)}'
            
    except Exception as e:
        result['error_message'] = f'General error: {str(e)}'
    
    return result

def check_svs_files_from_csv(csv_path: str, filename_column: str = 'filename') -> pd.DataFrame:
    """
    从CSV文件读取SVS文件路径并检查每个文件
    
    Args:
        csv_path: CSV文件路径
        filename_column: 包含文件路径的列名
        
    Returns:
        包含检查结果的DataFrame
    """
    try:
        # 读取CSV文件
        df = pd.read_csv(csv_path)
        logger.info(f"读取CSV文件: {csv_path}, 共 {len(df)} 行数据")
        
        if filename_column not in df.columns:
            raise ValueError(f"列 '{filename_column}' 不存在于CSV文件中")
        
        # 检查每个SVS文件
        results = []
        total_files = len(df)
        
        for index, row in df.iterrows():
            file_path = row[filename_column]
            logger.info(f"检查文件 {index + 1}/{total_files}: {file_path}")
            
            result = check_svs_file(file_path)
            results.append(result)
            
            # 输出检查结果
            if result['valid_format']:
                logger.info(f"  ✓ 文件正常 - 尺寸: {result['dimensions']}, 层级: {result['level_count']}, 大小: {result['file_size_mb']}MB")
            else:
                logger.warning(f"  ✗ 文件有问题 - {result['error_message']}")
        
        # 创建结果DataFrame
        results_df = pd.DataFrame(results)
        
        # 添加原始数据的其他列
        for col in df.columns:
            if col != filename_column:
                results_df[col] = df[col].values
        
        return results_df
        
    except Exception as e:
        logger.error(f"处理CSV文件时出错: {str(e)}")
        raise

def generate_summary(results_df: pd.DataFrame) -> None:
    """
    生成检查结果摘要
    
    Args:
        results_df: 包含检查结果的DataFrame
    """
    total_files = len(results_df)
    existing_files = results_df['exists'].sum()
    readable_files = results_df['readable'].sum()
    valid_files = results_df['valid_format'].sum()
    
    logger.info("\n" + "="*50)
    logger.info("检查结果摘要:")
    logger.info(f"总文件数: {total_files}")
    logger.info(f"存在的文件: {existing_files}")
    logger.info(f"可读取的文件: {readable_files}")
    logger.info(f"格式正确的文件: {valid_files}")
    logger.info(f"有问题的文件: {total_files - valid_files}")
    
    if total_files > 0:
        logger.info(f"成功率: {valid_files/total_files*100:.1f}%")
    
    # 显示有问题的文件
    problem_files = results_df[~results_df['valid_format']]
    if len(problem_files) > 0:
        logger.info("\n有问题的文件:")
        for _, row in problem_files.iterrows():
            logger.info(f"  - {row['filename']}: {row['error_message']}")
    
    logger.info("="*50)

def main():
    parser = argparse.ArgumentParser(description='检查SVS文件完整性')
    parser.add_argument('csv_file', help='包含SVS文件路径的CSV文件')
    parser.add_argument('--column', default='filename', help='包含文件路径的列名 (默认: filename)')
    parser.add_argument('--output', default='svs_check_results.csv', help='输出结果文件名 (默认: svs_check_results.csv)')
    
    args = parser.parse_args()
    
    try:
        # 检查CSV文件是否存在
        if not os.path.exists(args.csv_file):
            logger.error(f"CSV文件不存在: {args.csv_file}")
            return
        
        # 执行检查
        start_time = time.time()
        results_df = check_svs_files_from_csv(args.csv_file, args.column)
        end_time = time.time()
        
        # 保存结果
        results_df.to_csv(args.output, index=False)
        logger.info(f"\n结果已保存到: {args.output}")
        
        # 生成摘要
        generate_summary(results_df)
        
        logger.info(f"\n检查完成，耗时: {end_time - start_time:.2f} 秒")
        
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")

if __name__ == "__main__":
    main()