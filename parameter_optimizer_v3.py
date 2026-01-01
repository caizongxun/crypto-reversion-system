#!/usr/bin/env python3
"""
ARPI v3 Parameter Optimizer - FIXED
- Weighted signal fusion instead of strict consensus
- Optuna Bayesian optimization
- Walk-Forward validation
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

HF_REPO = "zongowo111/v2-crypto-ohlcv-data"
BTC_15M_FILE = "klines/BTCUSDT/BTC_15m.parquet"

OPTUNA_TRIALS = 100
WALK_FORWARD_PERIODS = 4
INSAMPLE_RATIO = 0.6

class ARPIv3Backtest:
    """ARPI v3 回測引擎 - 加權信號融合版本"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.df['returns'] = self.df['close'].pct_change()
        
    def calculate_indicators(self, params):
        """
        計算 ARPI v3 三大指標，使用加權融合代替嚴格共識
        """
        df = self.df.copy()
        
        # ===== 指標 1: AMRA (非對稱均值回歸加速度) =====
        fast_ma = df['close'].rolling(params['amra_fast_period']).mean()
        slow_ma = df['close'].rolling(params['amra_slow_period']).mean()
        
        price_dev_fast = ((df['close'] - fast_ma) / fast_ma) * 100
        price_dev_slow = ((df['close'] - slow_ma) / slow_ma) * 100
        
        asym_multiplier = np.where(
            df['close'] < fast_ma,
            params['amra_asym_factor'],
            1 / params['amra_asym_factor']
        )
        
        accel = price_dev_fast.diff().diff()
        accel_filtered = accel.rolling(3).mean()
        
        amra_bull = (accel_filtered < -params['amra_threshold']) & \
                    (df['close'] < fast_ma) & \
                    (price_dev_fast < price_dev_slow * asym_multiplier)
        amra_bear = (accel_filtered > params['amra_threshold']) & \
                    (df['close'] > fast_ma) & \
                    (price_dev_fast > price_dev_slow * asym_multiplier)
        
        # 使用強度值而非二元信號
        df['amra_score'] = np.where(amra_bull, 1.0, np.where(amra_bear, -1.0, 0))
        
        # ===== 指標 2: CTLV (混沌理論流動性真空) =====
        price_range = df['high'].rolling(params['ctlv_entropy_window']).max() - \
                      df['low'].rolling(params['ctlv_entropy_window']).min()
        
        entropy_raw = (df['close'] - df['low'].rolling(params['ctlv_entropy_window']).min()) / \
                      (price_range + 1e-8)
        entropy_raw = entropy_raw.clip(0.001, 0.999)
        entropy_raw = -entropy_raw * np.log(entropy_raw)
        entropy_smooth = entropy_raw.rolling(3).mean()
        entropy_drop = entropy_smooth < entropy_smooth.shift(1) * 0.85
        
        # 改進：不要求連續同向，只要求多數
        up_bars = (df['close'] > df['open']).rolling(params['ctlv_sync_length']).sum()
        dn_bars = (df['close'] < df['open']).rolling(params['ctlv_sync_length']).sum()
        majority_threshold = params['ctlv_sync_length'] * 0.6  # 60% 以上同向
        
        sync_condition = ((up_bars >= majority_threshold) | \
                         (dn_bars >= majority_threshold)) & entropy_drop
        
        ma_window = df['close'].rolling(params['ctlv_entropy_window']).mean()
        ctlv_bull = sync_condition & (df['close'] < ma_window)
        ctlv_bear = sync_condition & (df['close'] > ma_window)
        
        df['ctlv_score'] = np.where(ctlv_bull, 0.8, np.where(ctlv_bear, -0.8, 0))
        
        # ===== 指標 3: VJC (波動率躍跳連接器) =====
        atr_val = self._calculate_atr(df, params['vjc_vol_window'])
        atr_ma = atr_val.rolling(params['vjc_vol_window']).mean()
        
        vol_squeeze = atr_val < atr_ma * (1 - params['vjc_jump_sensitivity'] / 100)
        vol_jump = atr_val > atr_ma * (1 + params['vjc_jump_sensitivity'] / 100)
        
        momentum = df['close'].pct_change(params['vjc_momentum_length']) * 100
        momentum_ma = momentum.rolling(3).mean()
        
        # 改進：不要求動量方向，只要求波動率異常
        vjc_bull = vol_squeeze & (momentum_ma.diff() > 0)
        vjc_bear = vol_jump & (momentum_ma.diff() < 0)
        
        df['vjc_score'] = np.where(vjc_bull, 0.6, np.where(vjc_bear, -0.6, 0))
        
        # ===== 改進的信號融合：加權平均而非嚴格共識 =====
        weights = {'amra': 0.4, 'ctlv': 0.35, 'vjc': 0.25}
        df['signal_score'] = (
            weights['amra'] * df['amra_score'] + \
            weights['ctlv'] * df['ctlv_score'] + \
            weights['vjc'] * df['vjc_score']
        )
        
        # 使用閾值生成二元信號
        signal_threshold = 0.15  # 調整靈敏度
        df['signal'] = np.where(df['signal_score'] > signal_threshold, 1, 
                               np.where(df['signal_score'] < -signal_threshold, -1, 0))
        
        return df
    
    @staticmethod
    def _calculate_atr(df, period):
        """計算 ATR"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        return tr.rolling(period).mean()
    
    def backtest(self, params, lookback_periods=1):
        """
        執行回測
        """
        df = self.calculate_indicators(params)
        
        # 簡化的進出場邏輯
        trades = []
        position = False
        entry_price = 0
        entry_bar = 0
        cooldown = 0
        
        for i in range(1, len(df)):
            if cooldown > 0:
                cooldown -= 1
                continue
            
            current_signal = df['signal'].iloc[i]
            
            # 進場
            if not position and current_signal == 1:
                position = True
                entry_price = df['close'].iloc[i]
                entry_bar = i
                cooldown = lookback_periods
                
            # 出場
            elif position and current_signal == -1:
                exit_price = df['close'].iloc[i]
                trade_return = (exit_price - entry_price) / entry_price
                trades.append({
                    'entry_bar': entry_bar,
                    'exit_bar': i,
                    'return': trade_return,
                    'bars_held': i - entry_bar
                })
                position = False
                cooldown = lookback_periods
        
        # 平倉未結部位
        if position:
            exit_price = df['close'].iloc[-1]
            trade_return = (exit_price - entry_price) / entry_price
            trades.append({
                'entry_bar': entry_bar,
                'exit_bar': len(df) - 1,
                'return': trade_return,
                'bars_held': len(df) - 1 - entry_bar
            })
        
        if not trades:
            return {
                'returns': 0,
                'sharpe': 0,
                'win_rate': 0,
                'num_trades': 0,
                'max_drawdown': 0,
                'avg_bars_held': 0
            }
        
        trade_returns = np.array([t['return'] for t in trades])
        num_trades = len(trades)
        win_rate = np.sum(trade_returns > 0) / num_trades if num_trades > 0 else 0
        total_return = np.sum(trade_returns)
        avg_bars_held = np.mean([t['bars_held'] for t in trades])
        
        # Sharpe Ratio
        if len(trade_returns) > 1 and np.std(trade_returns) > 0:
            sharpe = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(252 * 24 / 15)  # 15m 轉年化
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
            'max_drawdown': max_drawdown,
            'avg_bars_held': avg_bars_held
        }


class ParameterOptimizer:
    """參數優化器"""
    
    def __init__(self, df):
        self.df = df
        self.backtest_engine = ARPIv3Backtest(df)
        self.history = []
    
    def get_parameter_space(self, trial):
        """定義搜索空間"""
        return {
            'amra_fast_period': trial.suggest_int('amra_fast_period', 5, 15),
            'amra_slow_period': trial.suggest_int('amra_slow_period', 15, 30),
            'amra_asym_factor': trial.suggest_float('amra_asym_factor', 1.1, 1.8),
            'amra_threshold': trial.suggest_float('amra_threshold', 0.3, 1.0),
            'ctlv_entropy_window': trial.suggest_int('ctlv_entropy_window', 7, 20),
            'ctlv_gap_threshold': trial.suggest_float('ctlv_gap_threshold', 0.15, 0.6),
            'ctlv_sync_length': trial.suggest_int('ctlv_sync_length', 3, 7),
            'vjc_vol_window': trial.suggest_int('vjc_vol_window', 14, 28),
            'vjc_jump_sensitivity': trial.suggest_float('vjc_jump_sensitivity', 1.2, 2.5),
            'vjc_momentum_length': trial.suggest_int('vjc_momentum_length', 5, 15),
        }
    
    def objective(self, trial):
        """Optuna 目標函數"""
        params = self.get_parameter_space(trial)
        wf_scores = self._walk_forward_test(params)
        
        if not wf_scores:
            return 0
        
        avg_sharpe = np.mean([s['sharpe'] for s in wf_scores])
        avg_returns = np.mean([s['returns'] for s in wf_scores])
        avg_trades = np.mean([s['num_trades'] for s in wf_scores])
        avg_drawdown = np.mean([abs(s['max_drawdown']) for s in wf_scores])
        
        # 目標函數：考慮收益、Sharpe、交易數、回撤
        # 要求至少有交易
        if avg_trades < 2:
            return -1000  # 懲罰信號過少的參數
        
        score = avg_sharpe * 0.5 + (avg_returns / (avg_drawdown + 0.01)) * 0.5
        
        self.history.append({
            'params': params,
            'sharpe': avg_sharpe,
            'returns': avg_returns,
            'trades': avg_trades,
            'score': score
        })
        
        return score
    
    def _walk_forward_test(self, params):
        """Walk-Forward 驗證"""
        results = []
        total_len = len(self.df)
        period_len = total_len // WALK_FORWARD_PERIODS
        
        for period in range(WALK_FORWARD_PERIODS - 1):
            start_idx = period * period_len
            end_idx = (period + 1) * period_len
            split_idx = int(start_idx + (end_idx - start_idx) * INSAMPLE_RATIO)
            
            test_df = self.df.iloc[split_idx:end_idx]
            
            if len(test_df) < 10:
                continue
            
            backtest = ARPIv3Backtest(test_df)
            result = backtest.backtest(params)
            results.append(result)
        
        return results
    
    def optimize(self, n_trials=OPTUNA_TRIALS):
        """執行優化"""
        sampler = TPESampler(seed=42)
        study = optuna.create_study(sampler=sampler, direction='maximize')
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        
        return study.best_params, study.best_value


def load_data_from_hf(symbol_file):
    """
    從 HuggingFace 加載數據
    """
    print(f"正在從 HuggingFace 下載: {symbol_file}")
    path = hf_hub_download(
        repo_id=HF_REPO,
        filename=symbol_file,
        repo_type="dataset"
    )
    df = pd.read_parquet(path)
    
    print(f"原始列: {list(df.columns)}")
    
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    
    if all(col in df.columns for col in required_cols):
        df = df[required_cols].astype(float)
    else:
        if len(df.columns) >= 6:
            df.columns = ['timestamp'] + list(df.columns[1:7])
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna()
        else:
            raise ValueError(f"不源混亂的 parquet 模式: {df.columns}")
    
    df = df.reset_index(drop=True)
    print(f"數據加載完成: {len(df)} 筆記錄")
    return df


def main():
    print("="*70)
    print("ARPI v3 參數優化系統 - 加權信號融合版本")
    print("="*70)
    
    df = load_data_from_hf(BTC_15M_FILE)
    df = df.tail(2000).reset_index(drop=True)
    print(f"\n使用數據: {len(df)} 根 OHLCV K線")
    
    print("\n初始化 Optuna 優化...")
    optimizer = ParameterOptimizer(df)
    
    print(f"開始 Bayesian 搜索 ({OPTUNA_TRIALS} 次試驗)...\n")
    best_params, best_score = optimizer.optimize(n_trials=OPTUNA_TRIALS)
    
    print("\n" + "="*70)
    print("最優參數")
    print("="*70)
    
    for key, value in best_params.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    print(f"\n最優評分: {best_score:.6f}")
    
    print("\n執行最終驗證回測...")
    backtest = ARPIv3Backtest(df)
    final_result = backtest.backtest(best_params)
    
    print("\n最終績效指標:")
    print(f"  總收益: {final_result['returns']*100:.2f}%")
    print(f"  Sharpe Ratio: {final_result['sharpe']:.4f}")
    print(f"  勝率: {final_result['win_rate']*100:.2f}%")
    print(f"  交易次數: {final_result['num_trades']}")
    print(f"  平均持倉根數: {final_result['avg_bars_held']:.1f}")
    print(f"  最大回撤: {final_result['max_drawdown']*100:.2f}%")
    
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
