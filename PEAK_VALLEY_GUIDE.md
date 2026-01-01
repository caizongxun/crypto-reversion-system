# 頂點和底點檢測完整指南

## 概述

本指南教你如何使用 `peak_valley_detector.py` 在 K 線圖中自動識別局部最高點（頂點）和局部最低點（底點）。

## 三種檢測方法對比

| 方法 | 優點 | 缺點 | 適用場景 |
|------|------|------|----------|
| **Simple** | 速度快、邏輯簡單 | 容易誤判小波動 | 快速測試 |
| **Window** | 更穩定、考慮周邊 | 邊界處理不完美 | 一般分析 |
| **ZigZag** | 最準確、反映真實波浪 | 參數調整複雜 | 專業交易（推薦） |

---

## 在 Colab 中使用

### 方法 1：簡單比較法（Quick Start）

```python
import pandas as pd
import matplotlib.pyplot as plt
from peak_valley_detector import PeakValleyDetector

# 讀取資料
df = pd.read_csv('/content/drive/MyDrive/crypto_data/BTCUSDT_15m.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 檢測頂點和底點
detector = PeakValleyDetector(df)
result_df = detector.detect_simple()  # 上下比較

# 查看結果
peaks, valleys = detector.get_peaks_and_valleys()
print(f"頂點數量: {len(peaks)}")
print(f"底點數量: {len(valleys)}")

# 顯示前 10 個頂點
print("\n頂點：")
print(result_df[result_df['is_peak']][['timestamp', 'close']].head(10))

# 顯示前 10 個底點
print("\n底點：")
print(result_df[result_df['is_valley']][['timestamp', 'close']].head(10))
```

### 方法 2：窗口法（推薦用於一般分析）

```python
# 使用窗口法（以 5 根 K 線為窗口）
detector = PeakValleyDetector(df)
result_df = detector.detect_window(window=5)  # 2 根棒左右比較

summary = detector.get_summary()

print(f"頂點: {summary['peak_count']} 個")
print(f"底點: {summary['valley_count']} 個")

# 顯示所有頂點
print("\n所有頂點：")
for peak in summary['peaks']:
    print(f"  {peak['timestamp']}: {peak['close']} USDT")

# 顯示所有底點
print("\n所有底點：")
for valley in summary['valleys']:
    print(f"  {valley['timestamp']}: {valley['close']} USDT")
```

### 方法 3：ZigZag 法（最準確，推薦）

```python
# 使用 ZigZag 法（只標記有 2% 以上波幅的頂底點）
detector = PeakValleyDetector(df)
result_df = detector.detect_zigzag(percentage=2.0)  # 2% 波幅閾值

summary = detector.get_summary()

print(f"頂點: {summary['peak_count']} 個")
print(f"底點: {summary['valley_count']} 個")
print(f"比例: 頂點數 / 底點數 = {summary['peak_count'] / summary['valley_count']:.2f}")
```

---

## 參數調整

### ZigZag 方法的 percentage 參數

`percentage` 控制多大的波幅才算是一個新的頂/底點

```python
# 例 1: 只標記 1% 以上的波動（更敏感，頂底點更多）
result_df = detector.detect_zigzag(percentage=1.0)

# 例 2: 只標記 2% 以上的波動（中等敏感）
result_df = detector.detect_zigzag(percentage=2.0)

# 例 3: 只標記 5% 以上的波動（不敏感，只抓主要趨勢）
result_df = detector.detect_zigzag(percentage=5.0)
```

**建議值**
- 15 分鐘圖：1-2%
- 1 小時圖：2-3%
- 日圖：3-5%

### Window 方法的 window 參數

`window` 控制左右各看多少根 K 線

```python
# window=2 → 左右各看 1 根棒
result_df = detector.detect_window(window=2)

# window=5 → 左右各看 2 根棒（推薦）
result_df = detector.detect_window(window=5)

# window=10 → 左右各看 4 根棒（看得更遠）
result_df = detector.detect_window(window=10)
```

**建議值**
- 快速交易（短線）：window=3-5
- 中期交易：window=5-10
- 長期趨勢：window=10-20

---

## 完整的視覺化（帶圖表）

```python
import pandas as pd
import matplotlib.pyplot as plt
from peak_valley_detector import PeakValleyDetector

# 讀取資料
df = pd.read_csv('/content/drive/MyDrive/crypto_data/BTCUSDT_15m.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 檢測
detector = PeakValleyDetector(df)
result_df = detector.detect_zigzag(percentage=2.0)

# 繪圖
fig, ax = plt.subplots(figsize=(18, 6))

# 繪製收盤價
ax.plot(result_df['timestamp'], result_df['close'], 
        color='black', linewidth=1, label='Close Price', zorder=1)

# 標記頂點（紅色▼）
peaks = result_df[result_df['is_peak']]
ax.scatter(peaks['timestamp'], peaks['close'], 
          color='red', marker='v', s=150, label=f'Peak ({len(peaks)})', zorder=5)

# 標記底點（綠色▲）
valleys = result_df[result_df['is_valley']]
ax.scatter(valleys['timestamp'], valleys['close'], 
          color='green', marker='^', s=150, label=f'Valley ({len(valleys)})', zorder=5)

# 連接頂底點的線（可選）
points = result_df[result_df['point_type'] != 'normal'].sort_values('timestamp')
if len(points) > 0:
    ax.plot(points['timestamp'], points['close'], 
           color='blue', linewidth=0.5, alpha=0.5, linestyle='--', zorder=2)

ax.set_title('BTC 15m - 頂點和底點檢測', fontsize=14, fontweight='bold')
ax.set_xlabel('時間')
ax.set_ylabel('價格 (USDT)')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_facecolor('#f8f9fa')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print(f"\n頂點: {len(peaks)} 個")
print(f"底點: {len(valleys)} 個")
```

---

## 保存標記後的資料

```python
# 方式 1: 保存為 CSV
result_df.to_csv('/content/drive/MyDrive/crypto_data/BTCUSDT_15m_marked.csv', index=False)
print("✓ 已保存到 CSV")

# 方式 2: 只保存頂底點的資訊
points_only = result_df[result_df['point_type'] != 'normal'][['timestamp', 'close', 'point_type']]
points_only.to_csv('/content/drive/MyDrive/crypto_data/BTCUSDT_15m_points.csv', index=False)
print("✓ 已保存頂底點")

# 方式 3: 導出為 JSON（便於後續處理）
import json
points_json = {
    'peaks': peaks[['timestamp', 'close']].to_dict('records'),
    'valleys': valleys[['timestamp', 'close']].to_dict('records')
}
with open('/content/drive/MyDrive/crypto_data/BTCUSDT_15m_points.json', 'w') as f:
    json.dump(points_json, f, indent=2, default=str)
print("✓ 已保存為 JSON")
```

---

## 分析頂底點的統計

```python
import pandas as pd
from peak_valley_detector import PeakValleyDetector

df = pd.read_csv('/content/drive/MyDrive/crypto_data/BTCUSDT_15m.csv')
detector = PeakValleyDetector(df)
result_df = detector.detect_zigzag(percentage=2.0)

# 計算頂底點的價差
peaks_df = result_df[result_df['is_peak']].copy()
valleys_df = result_df[result_df['is_valley']].copy()

print("=== 頂點統計 ===")
print(f"數量: {len(peaks_df)}")
print(f"平均價格: {peaks_df['close'].mean():.2f}")
print(f"最高價格: {peaks_df['close'].max():.2f}")
print(f"最低價格: {peaks_df['close'].min():.2f}")

print("\n=== 底點統計 ===")
print(f"數量: {len(valleys_df)}")
print(f"平均價格: {valleys_df['close'].mean():.2f}")
print(f"最高價格: {valleys_df['close'].max():.2f}")
print(f"最低價格: {valleys_df['close'].min():.2f}")

# 計算平均波幅
if len(peaks_df) > 0 and len(valleys_df) > 0:
    avg_amplitude = (peaks_df['close'].mean() - valleys_df['close'].mean()) / valleys_df['close'].mean() * 100
    print(f"\n平均波幅: {avg_amplitude:.2f}%")

# 計算頂點之間的平均間隔
if len(peaks_df) > 1:
    peak_intervals = (peaks_df.index.to_series().diff().dropna())
    print(f"\n頂點之間平均間隔: {peak_intervals.mean():.0f} 根 K 線")

# 計算底點之間的平均間隔
if len(valleys_df) > 1:
    valley_intervals = (valleys_df.index.to_series().diff().dropna())
    print(f"底點之間平均間隔: {valley_intervals.mean():.0f} 根 K 線")
```

---

## 用於交易策略

```python
from peak_valley_detector import PeakValleyDetector
import pandas as pd

df = pd.read_csv('/content/drive/MyDrive/crypto_data/BTCUSDT_15m.csv')
detector = PeakValleyDetector(df)
result_df = detector.detect_zigzag(percentage=2.0)

# 策略 1: 在底點買入，在頂點賣出
result_df['signal'] = 0
result_df.loc[result_df['is_valley'], 'signal'] = 1   # 買入信號
result_df.loc[result_df['is_peak'], 'signal'] = -1    # 賣出信號

print("買入信號:")
print(result_df[result_df['signal'] == 1][['timestamp', 'close']].head(10))

print("\n賣出信號:")
print(result_df[result_df['signal'] == -1][['timestamp', 'close']].head(10))

# 計算可能的收益
buys = result_df[result_df['signal'] == 1]['close'].values
sells = result_df[result_df['signal'] == -1]['close'].values

if len(buys) > 0 and len(sells) > 0:
    potential_gains = []
    for buy_price in buys:
        future_sells = sells[sells > buy_price]
        if len(future_sells) > 0:
            gain_pct = (future_sells[0] - buy_price) / buy_price * 100
            potential_gains.append(gain_pct)
    
    if potential_gains:
        print(f"\n潛在收益率: {sum(potential_gains) / len(potential_gains):.2f}%")
```

---

## 常見問題

### Q1: 為什麼檢測出的頂底點那麼多/那麼少？

**A:** 調整 `percentage` 參數（ZigZag 方法）或 `window` 參數（Window 方法）

```python
# 頂底點太多 → 增加 percentage
result_df = detector.detect_zigzag(percentage=5.0)

# 頂底點太少 → 減少 percentage
result_df = detector.detect_zigzag(percentage=1.0)
```

### Q2: Simple 和 Window 的區別？

**A:** 
- **Simple**: 只看上一根和下一根棒 (簡單但容易誤判)
- **Window**: 看多根棒 (更穩定)

### Q3: 能否跨越多個時間框架檢測？

**A:** 可以！分別下載不同時間框架的資料，各自檢測

```python
for timeframe in ['15m', '1h', '4h']:
    df = pd.read_csv(f'/content/drive/MyDrive/crypto_data/BTCUSDT_{timeframe}.csv')
    detector = PeakValleyDetector(df)
    result_df = detector.detect_zigzag()
    result_df.to_csv(f'/content/drive/MyDrive/crypto_data/BTCUSDT_{timeframe}_marked.csv')
```

### Q4: 如何與機器學習模型結合？

**A:** 可以將頂底點作為特徵或標籤

```python
# 將頂底點標記作為訓練標籤
result_df['label'] = result_df['point_type'].map({'peak': 1, 'valley': -1, 'normal': 0})

# 然後用其他特徵預測 label
from sklearn.ensemble import RandomForestClassifier

X = result_df[['close', 'volume', 'high', 'low']]
y = result_df['label']

model = RandomForestClassifier()
model.fit(X, y)
```

---

## 推薦工作流

```python
# 1. 下載資料
from crypto_downloader import CryptoDataDownloader
downloader = CryptoDataDownloader()
downloader.download_single_file('BTCUSDT', '15m')

# 2. 檢測頂底點
from peak_valley_detector import PeakValleyDetector
df = pd.read_csv('/content/drive/MyDrive/crypto_data/BTCUSDT_15m.csv')
detector = PeakValleyDetector(df)
result_df = detector.detect_zigzag(percentage=2.0)

# 3. 視覺化
# （執行上面的 plt 代碼）

# 4. 統計分析
# （執行上面的統計代碼）

# 5. 用於交易策略
# （執行上面的交易策略代碼）

# 6. 保存結果
result_df.to_csv('/content/drive/MyDrive/crypto_data/BTCUSDT_15m_marked.csv', index=False)
```

---

**祝你分析愉快！** 🚀
