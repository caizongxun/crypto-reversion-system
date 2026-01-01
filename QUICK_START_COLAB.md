# 30 秒快速開始

## 打開 Google Colab

https://colab.research.google.com

---

## 在新 Notebook 中，複製下面整個代碼到一個 Cell 並執行

```python
# 一鍵執行 - 無需任何設置
!pip install optuna pandas huggingface-hub numpy -q
import urllib.request
url = "https://raw.githubusercontent.com/caizongxun/crypto-reversion-system/main/parameter_optimizer_v3.py"
with urllib.request.urlopen(url) as response:
    exec(response.read().decode('utf-8'))
```

---

## 等待完成 (3-5 分鐘)

✓ 下載數據 (30 秒)  
✓ Bayesian 搜索 100 次試驗 (3 分鐘)  
✓ 顯示最優參數  
✓ 輸出性能指標  

---

## 獲得最優參數

看到類似輸出：

```
最優參數
============================================================
amra_fast_period: 8
amra_slow_period: 23
amra_asym_factor: 1.4521
amra_threshold: 0.5621
ctlv_entropy_window: 12
... 等等

性能指標:
  Sharpe Ratio: 1.2345
  勝率: 54.32%
  交易次數: 23
```

---

## 複製到 TradingView

1. 打開 TradingView → Pine Editor
2. 打開 `arpi_next_gen_v3.pine` 
3. 點擊指標設定 (⚙️ 符號)
4. 修改參數為上面輸出的值
5. 應用即可

---

## 可選：用不同幣種優化

修改這一行：

```python
BTC_15M_FILE = "klines/ETHUSDT/ETH_15m.parquet"  # 改成 ETH
# 或
BTC_15M_FILE = "klines/BNBUSDT/BNB_1h.parquet"   # 改成 BNB (1h)
# 或
BTC_15M_FILE = "klines/ADAUSDT/ADA_15m.parquet"  # 改成 ADA
```

---

## 完整參數列表 (可在 HF 查看)

https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data/tree/main/klines

所有幣種：
- BTCUSDT (BTC_15m.parquet, BTC_1h.parquet)
- ETHUSDT (ETH_15m.parquet, ETH_1h.parquet)
- BNBUSDT (BNB_15m.parquet, BNB_1h.parquet)
- ADAUSDT (ADA_15m.parquet, ADA_1h.parquet)
- ... 等共 23 種

---

## 保存結果

```python
# 在優化完成後，執行這個保存結果
result_df.to_csv('arpi_v3_best_params.csv', index=False)
# 或下載到本地
from google.colab import files
files.download('arpi_v3_best_params.csv')
```

---

## 遇到問題?

### 網絡超時
→ 重新執行 Cell (Colab 會重試)

### 內存不足
→ 修改這行，使用更少數據：
```python
df = df.tail(1000)  # 改成 1000 而不是 2000
```

### 想要更精確的優化
→ 修改這行：
```python
OPTUNA_TRIALS = 200  # 改成 200 (10 分鐘)
```

---

**就這樣！去 Colab 一鍵運行吧。** 🚀
