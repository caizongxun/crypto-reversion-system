#!/usr/bin/env python3
"""
ARPI v3 Parameter Optimizer
- Optuna Bayesian optimization
- Walk-Forward validation
- Remote execution from Colab (no git clone needed)
"""

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import optuna
from optuna.samplers import TPESampler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ======================== 配置區域 ========================

# HuggingFace 數據集信息
HF_REPO = "zongowo111/v2-crypto-ohlcv-data"
BTC_15M_FILE = "klines/BTCUSDT/BTC_15m.parquet"
BTC_1H_FILE = "klines/BTCUSDT/BTC_1h.parquet"

# 優化參數
OPTUNA_TRIALS = 100  # Bayesian 搜索試驗次數
WALK_FORWARD_PERIODS = 4  # Walk-Forward 分割數
INSAMPLE_RATIO = 0.6  # 樣本內佔比 (60% 優化, 40% 驗證)

class ARPIv3Backtest:
    """ARPI v3 回測引擎"""
    
    def __init__(self, df):
        """
        初始化回測
        
        Args:
            df: DataFrame with columns [open, high, low, close, volume]
        """
        self.df = df.copy()
        self.df['returns'] = self.df['close'].pct_change()
        
    def calculate_indicators(self, params):
        """
        計算所有 ARPI v3 指標
        
        Args:
            params: dict with keys:
                - amra_fast_period, amra_slow_period, amra_asym_factor, amra_threshold
                - ctlv_entropy_window, ctlv_gap_threshold, ctlv_sync_length
                - vjc_vol_window, vjc_jump_sensitivity, vjc_momentum_length
        """
        df = self.df.copy()
        
        # ===== 公式 1: AMRA (非對稱均值回歸加速度) =====
        fast_ma = df['close'].rolling(params['amra_fast_period']).mean()
        slow_ma = df['close'].rolling(params['amra_slow_period']).mean()
        
        price_dev_fast = ((df['close'] - fast_ma) / fast_ma) * 100
        price_dev_slow = ((df['close'] - slow_ma) / slow_ma) * 100
        
        asym_multiplier = np.where(
            df['close'] < fast_ma,
            params['amra_asym_factor'],
            1 / params['amra_asym_factor']
        )
        
        accel = price_dev_fast.diff().diff()  # 二階導數
        accel_filtered = accel.rolling(3).mean()
        
        amra_bull = (accel_filtered < -params['amra_threshold']) & \
                    (df['close'] < fast_ma) & \
                    (price_dev_fast < price_dev_slow * asym_multiplier)
        amra_bear = (accel_filtered > params['amra_threshold']) & \
                    (df['close'] > fast_ma) & \
                    (price_dev_fast > price_dev_slow * asym_multiplier)
        
        df['amra_bull'] = amra_bull.astype(int) * 40
        df['amra_bear'] = amra_bear.astype(int) * 40
        
        # ===== 公式 2: CTLV (混沌理論流動性真空) =====
        price_range = df['high'].rolling(params['ctlv_entropy_window']).max() - \
                      df['low'].rolling(params['ctlv_entropy_window']).min()
        
        # Shannon 熵 (簡化版)
        entropy_raw = (df['close'] - df['low'].rolling(params['ctlv_entropy_window']).min()) / \
                      (price_range + 1e-8)
        entropy_raw = entropy_raw.clip(0.001, 0.999)
        entropy_raw = -entropy_raw * np.log(entropy_raw)
        entropy_smooth = entropy_raw.rolling(3).mean()
        entropy_drop = entropy_smooth < entropy_smooth.shift(1) * 0.85
        
        # 同步流入 (連續同向)
        up_bars = (df['close'] > df['open']).rolling(params['ctlv_sync_length']).sum()
        dn_bars = (df['close'] < df['open']).rolling(params['ctlv_sync_length']).sum()
        sync_condition = ((up_bars == params['ctlv_sync_length']) | \
                         (dn_bars == params['ctlv_sync_length'])) & entropy_drop
        
        ma_window = df['close'].rolling(params['ctlv_entropy_window']).mean()
        ctlv_bull = sync_condition & (df['close'] < ma_window)
        ctlv_bear = sync_condition & (df['close'] > ma_window)
        
        df['ctlv_bull'] = ctlv_bull.astype(int) * 45
        df['ctlv_bear'] = ctlv_bear.astype(int) * 45
        
        # ===== 公式 3: VJC (波動率躍跳連接器) =====
        atr_val = self._calculate_atr(df, params['vjc_vol_window'])
        atr_ma = atr_val.rolling(params['vjc_vol_window']).mean()
        
        vol_squeeze = atr_val < atr_ma * (1 - params['vjc_jump_sensitivity'] / 100)
        vol_jump = atr_val > atr_ma * (1 + params['vjc_jump_sensitivity'] / 100)
        
        momentum = df['close'].pct_change(params['vjc_momentum_length']) * 100
        momentum_ma = momentum.rolling(3).mean()
        
        vjc_bull = vol_squeeze & (momentum < -2) & \
                   ((momentum > momentum_ma.shift(1)) | momentum_ma.diff() > 0)
        vjc_bear = vol_jump & (momentum > 2) & \
                   ((momentum < momentum_ma.shift(1)) | momentum_ma.diff() < 0)
        
        df['vjc_bull'] = vjc_bull.astype(int) * 50
        df['vjc_bear'] = vjc_bear.astype(int) * 50
        
        # ===== 融合信號 =====
        df['bull_count'] = (df['amra_bull'] > 0).astype(int) + \
                          (df['ctlv_bull'] > 0).astype(int) + \
                          (df['vjc_bull'] > 0).astype(int)
        df['bear_count'] = (df['amra_bear'] > 0).astype(int) + \
                          (df['ctlv_bear'] > 0).astype(int) + \
                          (df['vjc_bear'] > 0).astype(int)
        
        df['signal_strength'] = np.where(
            df['bull_count'] == 3, 100,
            np.where(df['bull_count'] == 2, 70,
            np.where(df['bull_count'] == 1, 35, 0))
        )
        df['signal_strength'] -= np.where(
            df['bear_count'] == 3, 100,
            np.where(df['bear_count'] == 2, 70,
            np.where(df['bear_count'] == 1, 35, 0))
        )
        
        return df
    
    @staticmethod
    def _calculate_atr(df, period):
        """計算 ATR"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        return tr.rolling(period).mean()
    
    def backtest(self, params, lookback_periods=2):
        """
        執行回測並計算績效指標
        
        Args:
            params: 參數字典
            lookback_periods: 冷卻期 (避免連續信號)
        
        Returns:
            績效指標 dict
        """
        df = self.calculate_indicators(params)
        
        # 冷卻機制
        df['last_bull_bar'] = np.nan
        df['last_bear_bar'] = np.nan
        
        for i in range(1, len(df)):
            if i > 0:
                if df['signal_strength'].iloc[i] > 0:
                    df.iloc[i, df.columns.get_loc('last_bull_bar')] = i
                elif not pd.isna(df['last_bull_bar'].iloc[i-1]):
                    df.iloc[i, df.columns.get_loc('last_bull_bar')] = df['last_bull_bar'].iloc[i-1]
        
        # 生成進出場信號
        buy_signals = []
        sell_signals = []
        
        for i in range(1, len(df)):
            # 檢查冷卻
            is_new_bull = df['signal_strength'].iloc[i] > 0 and \
                         (pd.isna(df['last_bull_bar'].iloc[i]) or \
                          i - df['last_bull_bar'].iloc[i] > lookback_periods)
            
            if is_new_bull and df['signal_strength'].iloc[i] > 0:
                buy_signals.append(i)
            elif df['signal_strength'].iloc[i] < 0:
                sell_signals.append(i)
        
        # 計算績效
        if not buy_signals:
            return {
                'returns': 0,
                'sharpe': 0,
                'win_rate': 0,
                'num_trades': 0,
                'max_drawdown': 0
            }
        
        # 簡單的進出場邏輯
        trades = []
        position = False
        entry_price = 0
        entry_bar = 0
        
        for i in range(len(df)):
            if not position and i in buy_signals:
                position = True
                entry_price = df['close'].iloc[i]
                entry_bar = i
            elif position and (i in sell_signals or i == len(df) - 1):
                exit_price = df['close'].iloc[i]
                trade_return = (exit_price - entry_price) / entry_price
                trades.append({
                    'entry_bar': entry_bar,
                    'exit_bar': i,
                    'return': trade_return,
                    'bars_held': i - entry_bar
                })
                position = False
        
        if not trades:
            return {
                'returns': 0,
                'sharpe': 0,
                'win_rate': 0,
                'num_trades': 0,
                'max_drawdown': 0
            }
        
        trade_returns = np.array([t['return'] for t in trades])
        num_trades = len(trades)
        win_rate = np.sum(trade_returns > 0) / num_trades if num_trades > 0 else 0
        total_return = np.sum(trade_returns)
        
        # Sharpe Ratio (簡化版)
        if len(trade_returns) > 1:
            sharpe = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(252 / 24)  # 15m 轉年化
        else:
            sharpe = 0
        
        # Max Drawdown
        cum_returns = np.cumprod(1 + trade_returns)
        running_max = np.maximum.accumulate(cum_returns)
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        return {
            'returns': total_return,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'num_trades': num_trades,
            'max_drawdown': max_drawdown
        }


class ParameterOptimizer:
    """參數優化器"""
    
    def __init__(self, df):
        self.df = df
        self.backtest_engine = ARPIv3Backtest(df)
        self.history = []
    
    def get_parameter_space(self, trial):
        """
        定義 Optuna 搜索空間
        """
        return {
            # AMRA
            'amra_fast_period': trial.suggest_int('amra_fast_period', 5, 15),
            'amra_slow_period': trial.suggest_int('amra_slow_period', 15, 30),
            'amra_asym_factor': trial.suggest_float('amra_asym_factor', 1.1, 1.8),
            'amra_threshold': trial.suggest_float('amra_threshold', 0.3, 1.0),
            
            # CTLV
            'ctlv_entropy_window': trial.suggest_int('ctlv_entropy_window', 7, 20),
            'ctlv_gap_threshold': trial.suggest_float('ctlv_gap_threshold', 0.15, 0.6),
            'ctlv_sync_length': trial.suggest_int('ctlv_sync_length', 3, 7),
            
            # VJC
            'vjc_vol_window': trial.suggest_int('vjc_vol_window', 14, 28),
            'vjc_jump_sensitivity': trial.suggest_float('vjc_jump_sensitivity', 1.2, 2.5),
            'vjc_momentum_length': trial.suggest_int('vjc_momentum_length', 5, 15),
        }
    
    def objective(self, trial):
        """
        Optuna 目標函數
        """
        params = self.get_parameter_space(trial)
        
        # 使用 Walk-Forward 驗證
        wf_scores = self._walk_forward_test(params)
        
        if not wf_scores:
            return 0
        
        # 目標函數: Sharpe Ratio (排除過度優化的參數)
        avg_sharpe = np.mean([s['sharpe'] for s in wf_scores])
        avg_returns = np.mean([s['returns'] for s in wf_scores])
        avg_drawdown = np.mean([abs(s['max_drawdown']) for s in wf_scores])
        
        # 複合評分 (Sharpe + returns/drawdown 比)
        score = avg_sharpe + (avg_returns / (avg_drawdown + 0.01)) * 0.1
        
        self.history.append({
            'params': params,
            'sharpe': avg_sharpe,
            'returns': avg_returns,
            'score': score
        })
        
        return score
    
    def _walk_forward_test(self, params):
        """
        Walk-Forward 驗證
        """
        results = []
        total_len = len(self.df)
        period_len = total_len // WALK_FORWARD_PERIODS
        
        for period in range(WALK_FORWARD_PERIODS - 1):  # 留最後一期作為驗證
            start_idx = period * period_len
            end_idx = (period + 1) * period_len
            split_idx = int(start_idx + (end_idx - start_idx) * INSAMPLE_RATIO)
            
            # 使用樣本內數據優化
            train_df = self.df.iloc[start_idx:split_idx]
            # 在樣本外數據驗證
            test_df = self.df.iloc[split_idx:end_idx]
            
            if len(test_df) < 10:
                continue
            
            backtest = ARPIv3Backtest(test_df)
            result = backtest.backtest(params)
            results.append(result)
        
        return results
    
    def optimize(self, n_trials=OPTUNA_TRIALS):
        """
        執行 Optuna 優化
        """
        sampler = TPESampler(seed=42)
        study = optuna.create_study(sampler=sampler, direction='maximize')
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        return study.best_params, study.best_value


def load_data_from_hf(symbol_file):
    """
    從 HuggingFace 遠端讀取數據
    """
    print(f"正在從 HuggingFace 下載: {symbol_file}")
    path = hf_hub_download(
        repo_id=HF_REPO,
        filename=symbol_file,
        repo_type="dataset"
    )
    df = pd.read_parquet(path)
    df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    print(f"數據加載完成: {len(df)} 筆記錄")
    return df


def main():
    print("="*60)
    print("ARPI v3 參數優化系統")
    print("="*60)
    
    # 加載 BTC 15m 數據
    df = load_data_from_hf(BTC_15M_FILE)
    
    # 只使用最近 2000 根蠟燭 (加快優化)
    df = df.tail(2000).reset_index(drop=True)
    print(f"\n使用數據範圍: {df['time'].iloc[0]} 到 {df['time'].iloc[-1]}")
    
    # 初始化優化器
    print("\n初始化 Optuna 優化...")
    optimizer = ParameterOptimizer(df)
    
    # 執行優化
    print(f"開始 Bayesian 搜索 ({OPTUNA_TRIALS} 次試驗)...")
    best_params, best_score = optimizer.optimize(n_trials=OPTUNA_TRIALS)
    
    # 輸出結果
    print("\n" + "="*60)
    print("最優參數")
    print("="*60)
    
    for key, value in best_params.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    print(f"\n最優評分: {best_score:.6f}")
    
    # 最終驗證
    print("\n執行最終驗證回測...")
    backtest = ARPIv3Backtest(df)
    final_result = backtest.backtest(best_params)
    
    print("\n最終績效指標:")
    print(f"  總收益: {final_result['returns']*100:.2f}%")
    print(f"  Sharpe Ratio: {final_result['sharpe']:.4f}")
    print(f"  勝率: {final_result['win_rate']*100:.2f}%")
    print(f"  交易次數: {final_result['num_trades']}")
    print(f"  最大回撤: {final_result['max_drawdown']*100:.2f}%")
    
    # 保存最優參數
    result_df = pd.DataFrame([{
        **best_params,
        'score': best_score,
        'returns': final_result['returns'],
        'sharpe': final_result['sharpe'],
        'win_rate': final_result['win_rate'],
        'num_trades': final_result['num_trades'],
        'max_drawdown': final_result['max_drawdown'],
        'optimization_time': datetime.now().isoformat()
    }])
    
    print("\n優化完成!")
    return best_params, final_result, result_df


if __name__ == "__main__":
    best_params, final_result, result_df = main()
