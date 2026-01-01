# ARPI v3 參數優化 - Colab 執行指南

## 快速開始（5 分鐘）

### 步驟 1: 打開 Google Colab

訪問 https://colab.research.google.com

### 步驟 2: 複製以下代碼到第一個 Cell 並執行

```python
# 安裝必要的套件
!pip install optuna pandas huggingface-hub numpy -q

print("✓ 套件安裝完成")
```

### 步驟 3: 複製優化腳本到第二個 Cell

```python
# 直接從 GitHub 讀取優化腳本
import urllib.request

url = "https://raw.githubusercontent.com/caizongxun/crypto-reversion-system/main/parameter_optimizer_v3.py"
with urllib.request.urlopen(url) as response:
    script_content = response.read().decode('utf-8')

# 執行腳本
exec(script_content)
```

### 步驟 4: 執行優化（等待 3-5 分鐘）

```python
# 運行優化
best_params, final_result, result_df = main()
```

### 步驟 5: 查看結果

```python
# 顯示最優參數
print("\n最優參數:")
for key, value in best_params.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

# 顯示性能指標
print("\n性能指標:")
print(f"年化 Sharpe Ratio: {final_result['sharpe']:.4f}")
print(f"勝率: {final_result['win_rate']*100:.2f}%")
print(f"交易次數: {final_result['num_trades']}")

# 導出為 CSV（可選）
result_df.to_csv('arpi_v3_optimization_results.csv', index=False)
print("\n結果已保存到 arpi_v3_optimization_results.csv")
```

---

## 完整腳本模板（複製整個 Cell）

如果想要在一個 Cell 裡完成所有操作，使用這個模板：

```python
# 1. 安裝套件
!pip install optuna pandas huggingface-hub numpy -q

# 2. 導入所有必要模組
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import optuna
from optuna.samplers import TPESampler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 3. 配置
HF_REPO = "zongowo111/v2-crypto-ohlcv-data"
BTC_15M_FILE = "klines/BTCUSDT/BTC_15m.parquet"
OPTUNA_TRIALS = 100
WALK_FORWARD_PERIODS = 4
INSAMPLE_RATIO = 0.6

# 4. 讀取遠端數據
print("正在從 HuggingFace 下載數據...")
path = hf_hub_download(
    repo_id=HF_REPO,
    filename=BTC_15M_FILE,
    repo_type="dataset"
)
df = pd.read_parquet(path)
df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time').reset_index(drop=True)
df = df.tail(2000).reset_index(drop=True)  # 只使用最近 2000 根蠟燭
print(f"✓ 數據加載完成: {len(df)} 筆記錄")
print(f"  時間範圍: {df['time'].iloc[0]} 到 {df['time'].iloc[-1]}")

# 5. 定義指標計算函數
class ARPIv3Backtest:
    def __init__(self, df):
        self.df = df.copy()
        self.df['returns'] = self.df['close'].pct_change()
    
    def calculate_indicators(self, params):
        df = self.df.copy()
        
        # AMRA (非對稱均值回歸加速度)
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
        
        df['amra_bull'] = amra_bull.astype(int) * 40
        df['amra_bear'] = amra_bear.astype(int) * 40
        
        # CTLV (混沌理論流動性真空)
        price_range = df['high'].rolling(params['ctlv_entropy_window']).max() - \
                      df['low'].rolling(params['ctlv_entropy_window']).min()
        
        entropy_raw = (df['close'] - df['low'].rolling(params['ctlv_entropy_window']).min()) / \
                      (price_range + 1e-8)
        entropy_raw = entropy_raw.clip(0.001, 0.999)
        entropy_raw = -entropy_raw * np.log(entropy_raw)
        entropy_smooth = entropy_raw.rolling(3).mean()
        entropy_drop = entropy_smooth < entropy_smooth.shift(1) * 0.85
        
        up_bars = (df['close'] > df['open']).rolling(params['ctlv_sync_length']).sum()
        dn_bars = (df['close'] < df['open']).rolling(params['ctlv_sync_length']).sum()
        sync_condition = ((up_bars == params['ctlv_sync_length']) | \
                         (dn_bars == params['ctlv_sync_length'])) & entropy_drop
        
        ma_window = df['close'].rolling(params['ctlv_entropy_window']).mean()
        ctlv_bull = sync_condition & (df['close'] < ma_window)
        ctlv_bear = sync_condition & (df['close'] > ma_window)
        
        df['ctlv_bull'] = ctlv_bull.astype(int) * 45
        df['ctlv_bear'] = ctlv_bear.astype(int) * 45
        
        # VJC (波動率躍跳連接器)
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
        
        # 融合信號
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
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        return tr.rolling(period).mean()
    
    def backtest(self, params, lookback_periods=2):
        df = self.calculate_indicators(params)
        
        # 生成進出場信號的簡化版本
        trade_returns = []
        position = False
        entry_price = 0
        
        for i in range(len(df)):
            if not position and df['signal_strength'].iloc[i] > 0:
                position = True
                entry_price = df['close'].iloc[i]
            elif position and df['signal_strength'].iloc[i] < 0:
                exit_price = df['close'].iloc[i]
                trade_returns.append((exit_price - entry_price) / entry_price)
                position = False
        
        if not trade_returns:
            return {
                'returns': 0, 'sharpe': 0, 'win_rate': 0,
                'num_trades': 0, 'max_drawdown': 0
            }
        
        trade_returns = np.array(trade_returns)
        total_return = np.sum(trade_returns)
        win_rate = np.sum(trade_returns > 0) / len(trade_returns)
        sharpe = np.mean(trade_returns) / np.std(trade_returns) * np.sqrt(252 / 24) if len(trade_returns) > 1 else 0
        
        cum_returns = np.cumprod(1 + trade_returns)
        running_max = np.maximum.accumulate(cum_returns)
        max_drawdown = np.min((cum_returns - running_max) / running_max)
        
        return {
            'returns': total_return,
            'sharpe': sharpe,
            'win_rate': win_rate,
            'num_trades': len(trade_returns),
            'max_drawdown': max_drawdown
        }

# 6. 定義優化器
class ParameterOptimizer:
    def __init__(self, df):
        self.df = df
        self.backtest_engine = ARPIv3Backtest(df)
        self.history = []
    
    def get_parameter_space(self, trial):
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
        params = self.get_parameter_space(trial)
        wf_scores = self._walk_forward_test(params)
        
        if not wf_scores:
            return 0
        
        avg_sharpe = np.mean([s['sharpe'] for s in wf_scores])
        avg_returns = np.mean([s['returns'] for s in wf_scores])
        avg_drawdown = np.mean([abs(s['max_drawdown']) for s in wf_scores])
        
        score = avg_sharpe + (avg_returns / (avg_drawdown + 0.01)) * 0.1
        self.history.append({'params': params, 'score': score})
        return score
    
    def _walk_forward_test(self, params):
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
        sampler = TPESampler(seed=42)
        study = optuna.create_study(sampler=sampler, direction='maximize')
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)
        return study.best_params, study.best_value

# 7. 執行優化
print("\n開始 Bayesian 優化...\n")
optimizer = ParameterOptimizer(df)
best_params, best_score = optimizer.optimize(n_trials=OPTUNA_TRIALS)

print("\n" + "="*60)
print("最優參數")
print("="*60)
for key, value in best_params.items():
    if isinstance(value, float):
        print(f"{key}: {value:.4f}")
    else:
        print(f"{key}: {value}")

print(f"\n最優評分: {best_score:.6f}")

# 8. 最終驗證回測
print("\n執行最終驗證回測...")
backtest = ARPIv3Backtest(df)
final_result = backtest.backtest(best_params)

print("\n最終性能指標:")
print(f"  總收益: {final_result['returns']*100:.2f}%")
print(f"  Sharpe Ratio: {final_result['sharpe']:.4f}")
print(f"  勝率: {final_result['win_rate']*100:.2f}%")
print(f"  交易次數: {final_result['num_trades']}")
print(f"  最大回撤: {final_result['max_drawdown']*100:.2f}%")

# 9. 導出結果
result_df = pd.DataFrame([{
    **best_params,
    'score': best_score,
    'returns': final_result['returns'],
    'sharpe': final_result['sharpe'],
    'win_rate': final_result['win_rate'],
    'num_trades': final_result['num_trades'],
    'max_drawdown': final_result['max_drawdown'],
}])

print("\n優化完成！")
```

---

## 自定義優化配置

### 修改試驗次數

```python
# 預設 100 次試驗 (~3-5 分鐘)
# 如果想要更多試驗，改這行 (建議不超過 200)
OPTUNA_TRIALS = 50  # 快速模式
OPTUNA_TRIALS = 200  # 深度優化 (10+ 分鐘)
```

### 改變使用的幣種

```python
# 改變 BTC_15M_FILE 為其他幣種
# 可用的幣種: ETHUSDT, BNBUSDT, ADAUSDT, XRPUSDT, 等等

BTC_15M_FILE = "klines/ETHUSDT/ETH_15m.parquet"
# 或
BTC_15M_FILE = "klines/BNBUSDT/BNB_1h.parquet"  # 1h 也支持
```

### 調整 Walk-Forward 窗口

```python
# 更多回測週期 = 更穩健但更慢
WALK_FORWARD_PERIODS = 2   # 快速 (1 分鐘)
WALK_FORWARD_PERIODS = 6   # 深度 (10+ 分鐘)
```

---

## 常見問題

### Q: 需要多久?
A: 
- 快速模式 (50 試驗): 1-2 分鐘
- 預設模式 (100 試驗): 3-5 分鐘  
- 深度模式 (200 試驗): 10-15 分鐘

### Q: 會超時嗎?
A: Colab 有 12 小時限制。即使 200 試驗也只需 15 分鐘，不會超時。

### Q: 可以重新執行嗎?
A: 可以。重新執行 Cell 會使用新的隨機種子運行新的優化。

### Q: 如何保存結果?
A: 
```python
# 保存為 CSV
result_df.to_csv('arpi_v3_best_params.csv', index=False)
print("保存成功!")
```

### Q: 最優參數如何用在 TradingView?
A: 
1. 複製優化出的參數值
2. 打開 TradingView 上的 ARPI v3 指標設定
3. 將對應參數修改為優化值
4. 應用即可

---

## 結果解讀

**Sharpe Ratio** > 1.0 = 優秀  
**勝率** > 50% = 有效  
**最大回撤** < -20% = 可接受  
**交易次數** > 10 = 充足樣本

---

**最後更新**: 2026-01-01  
**版本**: ARPI v3.0
