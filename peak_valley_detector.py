#!/usr/bin/env python3
"""
頂点和底点検測模組
用于識別 K 線中的局部最高值（頂点）和局部最低值（底点）
"""

import pandas as pd
import numpy as np
from typing import Tuple, List


class PeakValleyDetector:
    """
    頂点和底点検測器
    
    支持多種検測方法：
    1. 粗佇一重方法 - 高低一重比較
    2. 窗口窗核查法 - N根棘了左右幻謯一重
    3. ZigZag 演算法 - 什么是更會技术了
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        初始化
        
        Args:
            df: 包含 'close' 欄位的 DataFrame
        """
        self.df = df.copy()
        self.df = self.df.reset_index(drop=True)
        
    def detect_simple(self) -> pd.DataFrame:
        """
        粗佇一重方法：上下比較
        
        原理：
        - 頂点：當前 K 棲 > 前一根 AND 當前 K 棲 > 下一根
        - 底点：當前 K 棲 < 前一根 AND 當前 K 棲 < 下一根
        """
        self.df['is_peak'] = False
        self.df['is_valley'] = False
        
        for i in range(1, len(self.df) - 1):
            prev_close = self.df.loc[i - 1, 'close']
            curr_close = self.df.loc[i, 'close']
            next_close = self.df.loc[i + 1, 'close']
            
            # 頂点: 當前 > 上下兩根
            if curr_close > prev_close and curr_close > next_close:
                self.df.loc[i, 'is_peak'] = True
            
            # 底点: 當前 < 上下兩根
            if curr_close < prev_close and curr_close < next_close:
                self.df.loc[i, 'is_valley'] = True
        
        self.df['point_type'] = 'normal'
        self.df.loc[self.df['is_peak'], 'point_type'] = 'peak'
        self.df.loc[self.df['is_valley'], 'point_type'] = 'valley'
        
        return self.df
    
    def detect_window(self, window: int = 5) -> pd.DataFrame:
        """
        窗口窗核查法：N根棘了左右幻謯一重
        
        Args:
            window: 窗口大小（預設 5）
                - window=2: 1根棘了左右
                - window=5: 2根棘了左右
                - window=10: 4根棘了左右
        """
        half_window = window // 2
        self.df['is_peak'] = False
        self.df['is_valley'] = False
        
        for i in range(half_window, len(self.df) - half_window):
            curr_close = self.df.loc[i, 'close']
            
            # 頃点: 當前是窗口中最高的
            window_data = self.df.loc[i - half_window:i + half_window, 'close']
            
            if curr_close == window_data.max():
                self.df.loc[i, 'is_peak'] = True
            
            if curr_close == window_data.min():
                self.df.loc[i, 'is_valley'] = True
        
        self.df['point_type'] = 'normal'
        self.df.loc[self.df['is_peak'], 'point_type'] = 'peak'
        self.df.loc[self.df['is_valley'], 'point_type'] = 'valley'
        
        return self.df
    
    def detect_zigzag(self, percentage: float = 2.0) -> pd.DataFrame:
        """
        ZigZag 演算法：只標記有一定潮幅變化的頂底点
        
        Args:
            percentage: 潮幅陪問（預設 2%）
                當下跌/上潹超過此百分比時，標記新的頂底点
        """
        self.df['point_type'] = 'normal'
        self.df['is_peak'] = False
        self.df['is_valley'] = False
        
        if len(self.df) < 3:
            return self.df
        
        # 初始化第一個頂底点
        last_point_idx = 0
        last_point_val = self.df.loc[0, 'close']
        last_point_type = None  # 'peak' or 'valley'
        
        peaks = []
        valleys = []
        
        for i in range(1, len(self.df)):
            curr_val = self.df.loc[i, 'close']
            change_pct = abs((curr_val - last_point_val) / last_point_val * 100)
            
            # 如果變化超過此百分比，依据方向標記新的頂底点
            if change_pct > percentage:
                if curr_val > last_point_val:  # 上潹
                    if last_point_type == 'valley':
                        # 從低位済上潹，標記去的頂点
                        peaks.append(last_point_idx)
                        self.df.loc[last_point_idx, 'is_peak'] = True
                        self.df.loc[last_point_idx, 'point_type'] = 'peak'
                    # 重新設置最低值
                    last_point_val = curr_val
                    last_point_idx = i
                    last_point_type = 'peak'
                else:  # 下跌
                    if last_point_type == 'peak':
                        # 從高位下跌，標記去的底点
                        valleys.append(last_point_idx)
                        self.df.loc[last_point_idx, 'is_valley'] = True
                        self.df.loc[last_point_idx, 'point_type'] = 'valley'
                    # 重新設置最高值
                    last_point_val = curr_val
                    last_point_idx = i
                    last_point_type = 'valley'
        
        # 处理最后一个点
        if last_point_type == 'peak':
            peaks.append(last_point_idx)
            self.df.loc[last_point_idx, 'is_peak'] = True
            self.df.loc[last_point_idx, 'point_type'] = 'peak'
        else:
            valleys.append(last_point_idx)
            self.df.loc[last_point_idx, 'is_valley'] = True
            self.df.loc[last_point_idx, 'point_type'] = 'valley'
        
        return self.df
    
    def get_peaks_and_valleys(self) -> Tuple[List[int], List[int]]:
        """取得頂点和底点的索引"""
        peaks = self.df[self.df['is_peak']].index.tolist()
        valleys = self.df[self.df['is_valley']].index.tolist()
        return peaks, valleys
    
    def get_summary(self) -> dict:
        """取得棂済統計"""
        peak_count = self.df['is_peak'].sum()
        valley_count = self.df['is_valley'].sum()
        
        peak_rows = self.df[self.df['is_peak']]
        valley_rows = self.df[self.df['is_valley']]
        
        return {
            'peak_count': peak_count,
            'valley_count': valley_count,
            'peaks': peak_rows[['timestamp', 'close', 'high', 'low']].to_dict('records') if len(peak_rows) > 0 else [],
            'valleys': valley_rows[['timestamp', 'close', 'high', 'low']].to_dict('records') if len(valley_rows) > 0 else [],
        }


if __name__ == '__main__':
    # 样例使用
    print("頂点和底点検測示例\n")
    
    # 讀取資料
    df = pd.read_csv('/content/drive/MyDrive/crypto_data/BTCUSDT_15m.csv')
    
    # 使用 ZigZag 方法（推荐）
    detector = PeakValleyDetector(df)
    result_df = detector.detect_zigzag(percentage=2.0)
    summary = detector.get_summary()
    
    print(f"頂点數量：{summary['peak_count']}")
    print(f"底点數量：{summary['valley_count']}\n")
    
    print("頂点（最高）：")
    for peak in summary['peaks'][:5]:  # 顯示前 5 个
        print(f"  {peak['timestamp']}: {peak['close']} USDT")
    
    print("\n底点（最低）：")
    for valley in summary['valleys'][:5]:
        print(f"  {valley['timestamp']}: {valley['close']} USDT")
    
    # 保存標記了的資料
    result_df.to_csv('/content/drive/MyDrive/crypto_data/BTCUSDT_15m_marked.csv', index=False)
    print("\n✓ 已保存標記的資料")
