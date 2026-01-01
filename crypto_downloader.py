"""
Crypto Data Downloader from HuggingFace Hub
下載 HuggingFace 上的加密貨幣 OHLCV 資料並轉換為 CSV 格式
"""

import os
import pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download
from typing import Optional, List


class CryptoDataDownloader:
    """
    從 HuggingFace Hub 下載加密貨幣 OHLCV 資料並轉換為 CSV 格式
    """
    
    def __init__(self, repo_id: str = "zongowo111/v2-crypto-ohlcv-data"):
        """
        初始化下載器
        
        Args:
            repo_id: HuggingFace 倉庫 ID，預設為 zongowo111/v2-crypto-ohlcv-data
        """
        self.repo_id = repo_id
        self.repo_type = "dataset"
        self.output_dir = Path("crypto_data")
        self.output_dir.mkdir(exist_ok=True)
    
    def download_single_file(
        self,
        symbol: str,
        timeframe: str = "15m",
        output_format: str = "csv"
    ) -> Optional[str]:
        """
        下載單一幣種的資料
        
        Args:
            symbol: 幣種符號，例如 'BTCUSDT', 'ETHUSDT'
            timeframe: 時間框架，'15m' 或 '1h'
            output_format: 輸出格式，'csv' 或 'parquet'
        
        Returns:
            輸出檔案路徑，如果失敗則返回 None
        
        Example:
            >>> downloader = CryptoDataDownloader()
            >>> downloader.download_single_file('BTCUSDT', '15m')
            'crypto_data/BTCUSDT_15m.csv'
        """
        try:
            # 構造檔案路徑
            filename = f"klines/{symbol}/{symbol.split('USDT')[0]}_{timeframe}.parquet"
            
            print(f"正在下載 {symbol} {timeframe} 資料...")
            
            # 從 HuggingFace 下載 Parquet 檔案
            local_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=filename,
                repo_type=self.repo_type
            )
            
            # 讀取 Parquet 檔案
            df = pd.read_parquet(local_path)
            
            # 重新命名欄位以確保一致性
            column_mapping = {
                'open_time': 'timestamp',
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume'
            }
            
            # 只選擇存在的欄位
            available_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
            df = df.rename(columns=available_cols)
            
            # 確保時間戳是 datetime 格式
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', errors='coerce')
            
            # 排序資料
            df = df.sort_values('timestamp', ascending=True)
            
            # 保存為 CSV
            if output_format.lower() == 'csv':
                output_path = self.output_dir / f"{symbol}_{timeframe}.csv"
                df.to_csv(output_path, index=False)
                print(f"✓ 成功儲存: {output_path}")
                return str(output_path)
            
            # 保存為 Parquet
            elif output_format.lower() == 'parquet':
                output_path = self.output_dir / f"{symbol}_{timeframe}.parquet"
                df.to_parquet(output_path, index=False)
                print(f"✓ 成功儲存: {output_path}")
                return str(output_path)
            
        except Exception as e:
            print(f"✗ 下載失敗 {symbol}: {str(e)}")
            return None
    
    def download_multiple_files(
        self,
        symbols: List[str],
        timeframes: Optional[List[str]] = None,
        output_format: str = "csv"
    ) -> dict:
        """
        下載多個幣種的資料
        
        Args:
            symbols: 幣種列表，例如 ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
            timeframes: 時間框架列表，預設為 ['15m', '1h']
            output_format: 輸出格式，'csv' 或 'parquet'
        
        Returns:
            成功下載的檔案字典
        
        Example:
            >>> downloader = CryptoDataDownloader()
            >>> results = downloader.download_multiple_files(
            ...     ['BTCUSDT', 'ETHUSDT'],
            ...     ['15m']
            ... )
        """
        if timeframes is None:
            timeframes = ['15m', '1h']
        
        results = {}
        total = len(symbols) * len(timeframes)
        current = 0
        
        for symbol in symbols:
            for timeframe in timeframes:
                current += 1
                print(f"\n[{current}/{total}] 處理中...")
                
                result = self.download_single_file(
                    symbol,
                    timeframe,
                    output_format
                )
                
                if result:
                    key = f"{symbol}_{timeframe}"
                    results[key] = result
        
        print(f"\n完成！共成功下載 {len(results)} 個檔案")
        return results
    
    def combine_csv_files(
        self,
        symbol: str,
        timeframes: List[str],
        output_filename: Optional[str] = None
    ) -> Optional[str]:
        """
        合併同一幣種不同時間框架的 CSV 檔案
        
        Args:
            symbol: 幣種符號
            timeframes: 時間框架列表
            output_filename: 輸出檔案名稱，預設為 '{symbol}_combined.csv'
        
        Returns:
            合併後檔案的路徑
        """
        try:
            dataframes = []
            
            for timeframe in timeframes:
                filepath = self.output_dir / f"{symbol}_{timeframe}.csv"
                if filepath.exists():
                    df = pd.read_csv(filepath)
                    df['timeframe'] = timeframe
                    dataframes.append(df)
                else:
                    print(f"警告：找不到 {filepath}")
            
            if not dataframes:
                print(f"錯誤：沒有找到任何 {symbol} 的資料")
                return None
            
            # 合併所有資料
            combined_df = pd.concat(dataframes, ignore_index=True)
            combined_df = combined_df.sort_values('timestamp', ascending=True)
            
            # 保存合併的檔案
            if output_filename is None:
                output_filename = f"{symbol}_combined.csv"
            
            output_path = self.output_dir / output_filename
            combined_df.to_csv(output_path, index=False)
            
            print(f"✓ 合併成功: {output_path}")
            print(f"  總行數: {len(combined_df)}")
            
            return str(output_path)
            
        except Exception as e:
            print(f"✗ 合併失敗: {str(e)}")
            return None
    
    def get_data_info(self, symbol: str, timeframe: str = "15m") -> Optional[dict]:
        """
        獲取資料的基本資訊
        
        Args:
            symbol: 幣種符號
            timeframe: 時間框架
        
        Returns:
            包含資料資訊的字典
        """
        try:
            filepath = self.output_dir / f"{symbol}_{timeframe}.csv"
            
            if not filepath.exists():
                print(f"檔案不存在: {filepath}")
                return None
            
            df = pd.read_csv(filepath)
            
            info = {
                'symbol': symbol,
                'timeframe': timeframe,
                'rows': len(df),
                'columns': list(df.columns),
                'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
                'date_range': f"{df['timestamp'].min()} 到 {df['timestamp'].max()}" if 'timestamp' in df.columns else "N/A",
            }
            
            if 'close' in df.columns:
                info['price_range'] = f"{df['close'].min():.2f} - {df['close'].max():.2f}"
            
            return info
            
        except Exception as e:
            print(f"✗ 無法獲取資訊: {str(e)}")
            return None


if __name__ == "__main__":
    # 使用示例
    downloader = CryptoDataDownloader()
    result = downloader.download_single_file('BTCUSDT', '15m')
    if result:
        info = downloader.get_data_info('BTCUSDT', '15m')
        print("\n資料資訊:")
        for key, value in info.items():
            print(f"  {key}: {value}")
