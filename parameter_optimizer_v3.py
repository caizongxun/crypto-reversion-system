#!/usr/bin/env python3
"""
ARPI v3 Parameter Optimizer - Complete Redesign
Simplified signal generation with actual trade volume
"""

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import optuna
from optuna.samplers import TPESampler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

HF_REPO = "zongowo111/v2-crypto-ohlcv-data"
BTC_15M_FILE = "klines/BTCUSDT/BTC_15m.parquet"

OPTUNA_TRIALS = 100
WALK_FORWARD_PERIODS = 4
INSAMPLE_RATIO = 0.6

class ARPIv3Backtest:
    """ARPI v3 回測引擎 - 簡化設計"""
    
    def __init__(self, df):
        self.df = df.copy()
        self.df['returns'] = self.df['close'].pct_change()
        
    def calculate_indicators(self, params):
        """
        計算簡化的 ARPI 指標
        三個獨立的信號源，各自完整
        """
        df = self.df.copy()
        
        # ===== 信號 1: MA Mean Reversion =====
        # 簡單的均值回歸 - 當價格偏離 MA 時買入
        fast_ma = df['close'].rolling(params['ma_fast_period']).mean()
        slow_ma = df['close'].rolling(params['ma_slow_period']).mean()
        
        # 價格相對 MA 的百分比偏離
        price_ratio = df['close'] / (fast_ma + 1e-8)
        
        # 簡單邏輯: 價格低於 MA 時看多
        signal1_bull = (price_ratio < (1 - params['ma_threshold'])) & (fast_ma < slow_ma)
        signal1_bear = (price_ratio > (1 + params['ma_threshold'])) & (fast_ma > slow_ma)
        
        df['signal1'] = np.where(signal1_bull, 1, np.where(signal1_bear, -1, 0))
        
        # ===== 信號 2: Momentum Reversal =====
        # 動量反轉 - 當動量變化時交易
        momentum = df['close'].pct_change(params['momentum_period']) * 100
        momentum_sma = momentum.rolling(params['momentum_sma']).mean()
        momentum_change = momentum - momentum_sma
        
        # 動量從負變正 (看多), 或從正變負 (看空)
        signal2_bull = (momentum_change > 0) & (momentum_change.shift(1) <= 0) & (momentum < 0)
        signal2_bear = (momentum_change < 0) & (momentum_change.shift(1) >= 0) & (momentum > 0)
        
        df['signal2'] = np.where(signal2_bull, 1, np.where(signal2_bear, -1, 0))
        
        # ===== 信號 3: Volatility Breakout =====
        # 波動率突破 - 當波動率上升且有明確方向時
        atr = self._calculate_atr(df, params['atr_period'])
        atr_sma = atr.rolling(params['atr_sma']).mean()
        
        # 波動率擴張
        vol_expand = atr > atr_sma * (1 + params['vol_expansion'])
        
        # 方向確認: 連續上升或下降
        up_count = (df['close'] > df['close'].shift(1)).rolling(params['direction_period']).sum()
        dn_count = (df['close'] < df['close'].shift(1)).rolling(params['direction_period']).sum()
        
        signal3_bull = vol_expand & (up_count >= params['direction_threshold'])
        signal3_bear = vol_expand & (dn_count >= params['direction_threshold'])
        
        df['signal3'] = np.where(signal3_bull, 1, np.where(signal3_bear, -1, 0))
        
        # ===== 融合信號: 簡單投票制 =====
        # 至少 1-2 個指標同向則交易
        df['bull_votes'] = (df['signal1'] > 0).astype(int) + \
                          (df['signal2'] > 0).astype(int) + \
                          (df['signal3'] > 0).astype(int)
        df['bear_votes'] = (df['signal1'] < 0).astype(int) + \
                          (df['signal2'] < 0).astype(int) + \
                          (df['signal3'] < 0).astype(int)
        
        # 至少 1 票就交易 (寬鬆條件)
        min_votes = 1
        df['final_signal'] = np.where(
            df['bull_votes'] >= min_votes, 1,
            np.where(df['bear_votes'] >= min_votes, -1, 0)
        )
        
        return df
    
    @staticmethod
    def _calculate_atr(df, period):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        return tr.rolling(period).mean()
    
    def backtest(self, params):
        """
        簡單進出場邏輯
        """
        df = self.calculate_indicators(params)
        
        trades = []
        position = False
        entry_price = 0
        entry_idx = 0
        
        for i in range(1, len(df)):
            signal = df['final_signal'].iloc[i]
            
            # 進場
            if not position and signal == 1:
                position = True
                entry_price = df['close'].iloc[i]
                entry_idx = i
            
            # 出場
            elif position and signal == -1:
                exit_price = df['close'].iloc[i]
                ret = (exit_price - entry_price) / entry_price
                trades.append({
                    'entry': entry_idx,
                    'exit': i,
                    'return': ret,
                    'days': i - entry_idx
                })
                position = False
        
        # 平倉未結部位
        if position:
            exit_price = df['close'].iloc[-1]
            ret = (exit_price - entry_price) / entry_price
            trades.append({
                'entry': entry_idx,
                'exit': len(df) - 1,
                'return': ret,
                'days': len(df) - 1 - entry_idx
            })
        
        if not trades:
            return {
                'returns': 0,
                'sharpe': 0,
                'win_rate': 0,
                'num_trades': 0,
                'max_dd': 0
            }
        
        rets = np.array([t['return'] for t in trades])
        n = len(trades)
        
        # 績效指標
        total_ret = np.sum(rets)
        wins = np.sum(rets > 0)
        win_rate = wins / n if n > 0 else 0
        
        sharpe = 0
        if np.std(rets) > 0:
            sharpe = np.mean(rets) / np.std(rets) * np.sqrt(252 * 24 / 15)
        
        # Max Drawdown
        cum_rets = np.cumprod(1 + rets)
        dd = (np.maximum.accumulate(cum_rets) - cum_rets) / np.maximum.accumulate(cum_rets)
        max_dd = np.max(dd) if len(dd) > 0 else 0
        
        return {
            'returns': total_ret,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'num_trades': n,
            'max_dd': max_dd
        }


class ParameterOptimizer:
    def __init__(self, df):
        self.df = df
        self.history = []
    
    def get_parameter_space(self, trial):
        return {
            # MA Mean Reversion
            'ma_fast_period': trial.suggest_int('ma_fast_period', 5, 20),
            'ma_slow_period': trial.suggest_int('ma_slow_period', 20, 50),
            'ma_threshold': trial.suggest_float('ma_threshold', 0.01, 0.1),
            
            # Momentum
            'momentum_period': trial.suggest_int('momentum_period', 5, 20),
            'momentum_sma': trial.suggest_int('momentum_sma', 3, 10),
            
            # Volatility
            'atr_period': trial.suggest_int('atr_period', 10, 20),
            'atr_sma': trial.suggest_int('atr_sma', 10, 20),
            'vol_expansion': trial.suggest_float('vol_expansion', 0.1, 0.5),
            
            # Direction
            'direction_period': trial.suggest_int('direction_period', 3, 8),
            'direction_threshold': trial.suggest_int('direction_threshold', 2, 6),
        }
    
    def objective(self, trial):
        params = self.get_parameter_space(trial)
        results = []
        
        # Walk-Forward
        period_len = len(self.df) // WALK_FORWARD_PERIODS
        
        for p in range(WALK_FORWARD_PERIODS - 1):
            start = p * period_len
            end = (p + 1) * period_len
            split = int(start + (end - start) * INSAMPLE_RATIO)
            
            test_df = self.df.iloc[split:end]
            if len(test_df) < 20:
                continue
            
            bt = ARPIv3Backtest(test_df)
            res = bt.backtest(params)
            results.append(res)
        
        if not results:
            return -1000
        
        # 評分
        avg_ret = np.mean([r['returns'] for r in results])
        avg_sharpe = np.mean([r['sharpe'] for r in results])
        avg_trades = np.mean([r['num_trades'] for r in results])
        
        # 交易太少懲罰
        if avg_trades < 2:
            return -1000
        
        score = avg_sharpe * 0.6 + avg_ret * 10 * 0.4
        
        self.history.append({
            'params': params,
            'sharpe': avg_sharpe,
            'returns': avg_ret,
            'trades': avg_trades,
            'score': score
        })
        
        return score
    
    def optimize(self, n_trials=OPTUNA_TRIALS):
        sampler = TPESampler(seed=42)
        study = optuna.create_study(sampler=sampler, direction='maximize')
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        return study.best_params, study.best_value


def load_data_from_hf(symbol_file):
    print(f"正在下載: {symbol_file}")
    path = hf_hub_download(repo_id=HF_REPO, filename=symbol_file, repo_type="dataset")
    df = pd.read_parquet(path)
    
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
            raise ValueError(f"無效的 parquet 格式: {df.columns}")
    
    df = df.reset_index(drop=True)
    print(f"加載完成: {len(df)} 筆記錄")
    return df


def main():
    print("="*70)
    print("ARPI v3 參數優化 - 重新設計版本")
    print("="*70)
    
    df = load_data_from_hf(BTC_15M_FILE)
    df = df.tail(2000).reset_index(drop=True)
    print(f"使用 {len(df)} 根 K 線")
    
    optimizer = ParameterOptimizer(df)
    print(f"\n開始 {OPTUNA_TRIALS} 次試驗...\n")
    best_params, best_score = optimizer.optimize(n_trials=OPTUNA_TRIALS)
    
    print("\n" + "="*70)
    print("最優參數")
    print("="*70)
    for k, v in best_params.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
    
    print(f"\n最優評分: {best_score:.4f}\n")
    
    # 最終驗證
    bt = ARPIv3Backtest(df)
    final = bt.backtest(best_params)
    
    print("最終績效:")
    print(f"  收益: {final['returns']*100:.2f}%")
    print(f"  Sharpe: {final['sharpe']:.4f}")
    print(f"  勝率: {final['win_rate']*100:.2f}%")
    print(f"  交易數: {final['num_trades']}")
    print(f"  最大回撤: {final['max_dd']*100:.2f}%")
    
    print("\n優化完成!")
    return best_params, final


if __name__ == "__main__":
    best_params, final = main()
