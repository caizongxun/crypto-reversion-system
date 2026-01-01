# Colab 一行一句快速設置

## 最简潔的步數

```python
# 1. 掛載
from google.colab import drive
drive.mount('/content/drive')

# 2. 下載檔案
import urllib.request
url = "https://raw.githubusercontent.com/caizongxun/crypto-reversion-system/main/peak_valley_detector.py"
urllib.request.urlretrieve(url, 'peak_valley_detector.py')

# 3. 導入並使用
from peak_valley_detector import PeakValleyDetector
import pandas as pd

df = pd.read_csv('/content/drive/MyDrive/crypto_data/BTCUSDT_15m.csv')
detector = PeakValleyDetector(df)
result_df = detector.detect_zigzag(percentage=2.0)

print(f"頂點: {detector.get_summary()['peak_count']}")
```

---

## 下載冰简辞典

### 單個檔案

```python
import urllib.request
url = "https://raw.githubusercontent.com/caizongxun/crypto-reversion-system/main/peak_valley_detector.py"
urllib.request.urlretrieve(url, 'peak_valley_detector.py')
```

### 多個檔案

```python
files = {
    'peak_valley_detector.py': 'https://raw.githubusercontent.com/caizongxun/crypto-reversion-system/main/peak_valley_detector.py',
    'crypto_downloader.py': 'https://raw.githubusercontent.com/caizongxun/crypto-reversion-system/main/crypto_downloader.py',
}

for name, url in files.items():
    urllib.request.urlretrieve(url, name)
    print(f'✓ {name}')
```

---

## 直接導入冰简辞典

### 第一次

```python
from peak_valley_detector import PeakValleyDetector
```

### 其他引入方法

```python
# 全部數據紁
 import peak_valley_detector as pvd
 detector = pvd.PeakValleyDetector(df)

# 或導入整個模組
import peak_valley_detector
```

---

## 褚待摸索冰简辞典

### 推薦：Single File + Import

```python
import urllib.request
from peak_valley_detector import PeakValleyDetector

url = "https://raw.githubusercontent.com/caizongxun/crypto-reversion-system/main/peak_valley_detector.py"
urllib.request.urlretrieve(url, 'peak_valley_detector.py')

detector = PeakValleyDetector(df)
```

### 或：Direct exec()

```python
import urllib.request

url = "https://raw.githubusercontent.com/caizongxun/crypto-reversion-system/main/peak_valley_detector.py"
response = urllib.request.urlopen(url)
code = response.read().decode('utf-8')
exec(code)

detector = PeakValleyDetector(df)
```

---

## 朗相牨計及單位轉換

### 市场資料

```python
# 擲简等偏置
df = pd.read_csv('/content/drive/MyDrive/crypto_data/BTCUSDT_15m.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 計算自比胯價小幸
df['returns'] = df['close'].pct_change() * 100

print(df[['timestamp', 'close', 'returns']].head())
```

### 統計訊息

```python
print(f"平均价: {df['close'].mean():.2f}")
print(f"最高价: {df['close'].max():.2f}")
print(f"最低价: {df['close'].min():.2f}")
print(f"標準差: {df['close'].std():.2f}")
```

---

## 直接执行 Colab 步驟

### 全部戱貼到 Colab 執行：

```python
# ============ 上一自动扵吹 ============
from google.colab import drive
drive.mount('/content/drive')

import urllib.request
import pandas as pd
from peak_valley_detector import PeakValleyDetector
import matplotlib.pyplot as plt

url = "https://raw.githubusercontent.com/caizongxun/crypto-reversion-system/main/peak_valley_detector.py"
urllib.request.urlretrieve(url, 'peak_valley_detector.py')

# ============ 你的代碼 ============
df = pd.read_csv('/content/drive/MyDrive/crypto_data/BTCUSDT_15m.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

detector = PeakValleyDetector(df)
result_df = detector.detect_zigzag(percentage=2.0)
summary = detector.get_summary()

print(f"✓ 頂點: {summary['peak_count']}")
print(f"✓ 底點: {summary['valley_count']}")

# ============ 視覺 ============
plt.figure(figsize=(18, 6))
plt.plot(result_df['timestamp'], result_df['close'])
plt.scatter(result_df[result_df['is_peak']]['timestamp'], 
           result_df[result_df['is_peak']]['close'], 
           color='red', marker='v', s=100)
plt.scatter(result_df[result_df['is_valley']]['timestamp'], 
           result_df[result_df['is_valley']]['close'], 
           color='green', marker='^', s=100)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

---

## Session 大答

| 操作 | 結果 |
|------|--------|
| 下載檔案 | 保存在 `/content/` |
| Colab session 結束 | 清除 `/content/` 中的檔案 |
| 資料保存到 Google Drive | 永久記錄 |
| session 結束但可繼續使用 | 是，從下一个 notebook 該該使用 |

---

## 未來此的釣书：

### 新增訓練檔案

1. 上傳到 GitHub
2. Colab 中：

```python
url = "https://raw.githubusercontent.com/caizongxun/crypto-reversion-system/main/your_new_file.py"
urllib.request.urlretrieve(url, 'your_new_file.py')

from your_new_file import YourClass
```

3. 完成！

---

粗粗卵卵後就上手了！
