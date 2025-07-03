#!/usr/bin/env python3
"""
WSI内存监控演示脚本
模拟WSI处理过程，用于测试监控功能
"""

import time
import numpy as np
import psutil
import os
from pathlib import Path

def simulate_wsi_processing():
    """模拟WSI处理过程"""
    print("🧪 开始模拟WSI处理...")
    
    # 模拟随机采样和细胞核检测
    data_arrays = []
    
    for i in range(50):
        print(f"处理批次 {i+1}/50", end='\r')
        
        # 模拟读取WSI patches
        patch_data = np.random.rand(512, 512, 3) * 255
        data_arrays.append(patch_data)
        
        # 模拟处理延迟
        time.sleep(0.2)
        
        # 模拟内存积累（每10个批次释放一些）
        if i % 10 == 9:
            # 释放部分数据
            data_arrays = data_arrays[-5:]
            print(f"\n🧹 批次 {i+1}: 清理内存")
        
        # 模拟一些内存泄漏
        if i % 15 == 14:
            # 故意不释放的数据
            leaked_data = np.random.rand(256, 256, 3) * 255
            print(f"\n⚠️  批次 {i+1}: 模拟内存泄漏")
    
    print(f"\n✅ WSI处理完成!")
    
    # 最终清理
    del data_arrays
    print("🧹 最终内存清理完成")

def main():
    """主函数"""
    print("WSI处理模拟器")
    print("此脚本会模拟WSI数据处理过程，包括:")
    print("- 随机patch采样")
    print("- 内存使用变化")
    print("- 模拟内存泄漏")
    print("- 批次处理")
    print()
    print("🔍 请在另一个终端运行: python wsi_memory_tracker.py")
    print("然后回到这里按回车开始模拟...")
    
    input()
    
    start_time = time.time()
    simulate_wsi_processing()
    end_time = time.time()
    
    print(f"\n📊 处理统计:")
    print(f"总耗时: {end_time - start_time:.1f}秒")
    
    process = psutil.Process()
    memory_mb = process.memory_info().rss / (1024**2)
    print(f"当前内存使用: {memory_mb:.1f}MB")
    
    print("\n💡 现在可以检查监控报告和可视化图表!")

if __name__ == "__main__":
    main()
