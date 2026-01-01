# Google Colab 執行加密貨幣資料下載器完整指南

## 概述

本指南將教你如何在 Google Colab 上執行 `crypto_downloader.py`，從 HuggingFace Hub 下載加密貨幣 OHLCV 資料並轉換為 CSV 格式。

## 為什麼使用 Google Colab？

- 免費的雲端計算資源
- 無需本地環境配置
- 自動儲存於 Google Drive
- GPU/TPU 支援
- 易於分享和協作

---

## 方法一：完整的 Colab 筆記本（推薦）

### 步驟 1：建立新的 Colab 筆記本

1. 開啟 [Google Colab](https://colab.research.google.com/)
2. 點擊「檔案」→「新增筆記本」

### 步驟 2：掛載 Google Drive（可選但推薦）

```python
# 掛載 Google Drive 以儲存下載的資料
from google.colab import drive
drive.mount('/content/drive')
```

執行後會出現授權連結，複製連結到瀏覽器並授權。

### 步驟 3：安裝依賴

```python
# 在 Colab 中安裝必需的套件
!pip install -r https://raw.githubusercontent.com/caizongxun/crypto-reversion-system/main/requirements.txt
```

或逐個安裝：

```python
!pip install pandas pyarrow huggingface-hub requests scikit-learn numpy matplotlib
```

### 步驟 4：下載 crypto_downloader.py

```python
# 從 GitHub 下載主程式
import urllib.request

url = "https://raw.githubusercontent.com/caizongxun/crypto-reversion-system/main/crypto_downloader.py"
urllib.request.urlretrieve(url, 'crypto_downloader.py')

print("✓ crypto_downloader.py 已下載")
```

### 步驟 5：執行下載 (選擇一個方案)

#### 方案 A：下載單個幣種

```python
from crypto_downloader import CryptoDataDownloader

# 初始化下載器
downloader = CryptoDataDownloader()

# 設定輸出目錄（儲存到 Google Drive）
from pathlib import Path
downloader.output_dir = Path('/content/drive/MyDrive/crypto_data')
downloader.output_dir.mkdir(parents=True, exist_ok=True)

# 下載 BTC 15分鐘資料
result = downloader.download_single_file('BTCUSDT', '15m')
print(f"下載完成: {result}")

# 查看資料統計
if result:
    info = downloader.get_data_info('BTCUSDT', '15m')
    print("\n資料統計:")
    for key, value in info.items():
        print(f"  {key}: {value}")
```

#### 方案 B：批量下載多個幣種

```python
from crypto_downloader import CryptoDataDownloader
from pathlib import Path

downloader = CryptoDataDownloader()
downloader.output_dir = Path('/content/drive/MyDrive/crypto_data')
downloader.output_dir.mkdir(parents=True, exist_ok=True)

# 批量下載
symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOGEUSDT']
results = downloader.download_multiple_files(
    symbols,
    ['15m']  # 只下載 15 分鐘
)

print(f"\n成功下載 {len(results)} 個檔案")
for key, path in results.items():
    print(f"  - {key}: {path}")
```

#### 方案 C：下載後合併資料

```python
from crypto_downloader import CryptoDataDownloader
from pathlib import Path

downloader = CryptoDataDownloader()
downloader.output_dir = Path('/content/drive/MyDrive/crypto_data')
downloader.output_dir.mkdir(parents=True, exist_ok=True)

# 下載不同時框
downloader.download_single_file('BTCUSDT', '15m')
downloader.download_single_file('BTCUSDT', '1h')

# 合併資料
combined_path = downloader.combine_csv_files('BTCUSDT', ['15m', '1h'])
print(f"\n合併後的檔案: {combined_path}")
```

### 步驟 6：驗證下載的資料

```python
import pandas as pd

# 讀取下載的 CSV
df = pd.read_csv('/content/drive/MyDrive/crypto_data/BTCUSDT_15m.csv')

print(f"資料形狀: {df.shape}")
print(f"\n前 5 行:")
print(df.head())

print(f"\n資料統計:")
print(df.describe())

print(f"\n欄位名稱: {list(df.columns)}")
```

### 步驟 7：資料分析示例

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/content/drive/MyDrive/crypto_data/BTCUSDT_15m.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 計算每根蠟燭線的收益率
df['return'] = ((df['close'] - df['open']) / df['open'] * 100).round(4)

# 繪製收盤價
plt.figure(figsize=(15, 5))
plt.plot(df['timestamp'], df['close'], linewidth=0.5)
plt.title('BTC 15分鐘收盤價')
plt.xlabel('時間')
plt.ylabel('價格 (USDT)')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 統計收益率分布
print(f"\n收益率統計:")
print(f"  平均: {df['return'].mean():.4f}%")
print(f"  最大: {df['return'].max():.4f}%")
print(f"  最小: {df['return'].min():.4f}%")
print(f"  標準差: {df['return'].std():.4f}%")
```

---

## 方法二：直接從 GitHub 克隆

如果你想完整的專案結構：

```python
# 克隆整個倉庫
!git clone https://github.com/caizongxun/crypto-reversion-system.git

# 進入專案目錄
import os
os.chdir('crypto-reversion-system')

# 安裝依賴
!pip install -r requirements.txt
```

然後執行：

```python
from crypto_downloader import CryptoDataDownloader
from pathlib import Path

downloader = CryptoDataDownloader()
downloader.output_dir = Path('/content/drive/MyDrive/crypto_data')
downloader.download_single_file('BTCUSDT', '15m')
```

---

## 方法三：使用現成的 Colab 筆記本

### 快速開啟（推薦）

如果我們上傳了現成的 Colab 筆記本，你可以直接點擊開啟：

1. 訪問 [Colab 筆記本連結]
2. 點擊「在 Colab 中開啟」
3. 按順序執行所有儲存格

---

## 完整示例：一份完整的 Colab 筆記本

將以下內容複製到 Colab 中，從上到下執行：

```python
# ============================================
# 第 1 格：掛載 Google Drive
# ============================================
from google.colab import drive
drive.mount('/content/drive')

# ============================================
# 第 2 格：安裝依賴
# ============================================
!pip install pandas pyarrow huggingface-hub requests scikit-learn

# ============================================
# 第 3 格：下載 crypto_downloader.py
# ============================================
import urllib.request

url = "https://raw.githubusercontent.com/caizongxun/crypto-reversion-system/main/crypto_downloader.py"
urllib.request.urlretrieve(url, 'crypto_downloader.py')

print("✓ 下載完成")

# ============================================
# 第 4 格：設定參數
# ============================================
from pathlib import Path

# 設定要下載的幣種和時框
SYMBOLS = ['BTCUSDT', 'ETHUSDT']  # 修改你要的幣種
TIMEFRAMES = ['15m']  # 可選 '15m' 或 '1h'
OUTPUT_DIR = Path('/content/drive/MyDrive/crypto_data')  # 儲存路徑

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"設定完成:")
print(f"  幣種: {SYMBOLS}")
print(f"  時框: {TIMEFRAMES}")
print(f"  存放位置: {OUTPUT_DIR}")

# ============================================
# 第 5 格：執行下載
# ============================================
from crypto_downloader import CryptoDataDownloader

downloader = CryptoDataDownloader()
downloader.output_dir = OUTPUT_DIR

print("\n開始下載...\n")
results = downloader.download_multiple_files(SYMBOLS, TIMEFRAMES)

print(f"\n✓ 下載完成! 共 {len(results)} 個檔案")
for key, path in results.items():
    print(f"  - {key}")

# ============================================
# 第 6 格：驗證資料
# ============================================
import pandas as pd

for symbol in SYMBOLS:
    filepath = OUTPUT_DIR / f"{symbol}_{TIMEFRAMES[0]}.csv"
    if filepath.exists():
        df = pd.read_csv(filepath)
        print(f"\n{symbol} 資料統計:")
        print(f"  行數: {len(df)}")
        print(f"  欄位: {list(df.columns)}")
        print(f"\n前 3 行:")
        print(df.head(3))

# ============================================
# 第 7 格：視覺化（可選）
# ============================================
import matplotlib.pyplot as plt
import pandas as pd

fig, axes = plt.subplots(len(SYMBOLS), 1, figsize=(15, 5*len(SYMBOLS)))
if len(SYMBOLS) == 1:
    axes = [axes]

for idx, symbol in enumerate(SYMBOLS):
    filepath = OUTPUT_DIR / f"{symbol}_{TIMEFRAMES[0]}.csv"
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    axes[idx].plot(df['timestamp'], df['close'], linewidth=0.5)
    axes[idx].set_title(f"{symbol} 收盤價")
    axes[idx].set_ylabel('價格 (USDT)')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 常見問題

### Q1: 下載速度很慢？

**A:** 這很正常。HuggingFace Hub 的速度取決於你的網路。Colab 通常有較好的連接。

### Q2: 執行出錯「FileNotFoundError」？

**A:** 檢查：
1. 幣種符號是否正確（需包含 USDT）
2. 網路連接是否正常
3. 該幣種是否在資料集中

### Q3: 儲存空間不夠？

**A:** 
- 檢查 Google Drive 剩餘空間
- 或改為輸出到 Colab 的 `/content/` 目錄（環境變更時會清除）
- 或使用 Parquet 格式（更省空間）

### Q4: 如何在 Colab 中使用下載的資料進行分析？

**A:** 見上面的「第 7 格」示例。也可以使用其他分析工具如 Pandas、NumPy、Scikit-learn 等。

### Q5: 可以下載後自動進行回測嗎？

**A:** 可以！下載後可以直接在 Colab 中執行回測。見其他檔案如 `parameter_optimizer_v3.py` 的用法。

---

## 最佳實踐

1. **儲存到 Google Drive**：使用 `/content/drive/MyDrive/` 以保留資料
2. **批量下載**：使用 `download_multiple_files()` 比逐個下載更快
3. **檢查下載**：總是驗證下載的資料再進行分析
4. **使用資料夾**：建立清晰的檔案結構
5. **記錄日誌**：在每個步驟列印進度訊息

---

## 進階用法

### 自訂輸出目錄

```python
from pathlib import Path
downloader = CryptoDataDownloader()
downloader.output_dir = Path('/content/drive/MyDrive/my_crypto_data')
downloader.output_dir.mkdir(parents=True, exist_ok=True)
downloader.download_single_file('BTCUSDT', '15m')
```

### 輸出為 Parquet（更節省空間）

```python
downloader.download_single_file('BTCUSDT', '15m', output_format='parquet')
```

### 下載後立即分析

```python
import pandas as pd
from crypto_downloader import CryptoDataDownloader

downloader = CryptoDataDownloader()
downloader.download_single_file('BTCUSDT', '15m')

df = pd.read_csv('crypto_data/BTCUSDT_15m.csv')
df['return'] = ((df['close'] - df['open']) / df['open'] * 100)
print(df['return'].describe())
```

---

## 資源連結

- [GitHub 倉庫](https://github.com/caizongxun/crypto-reversion-system)
- [HuggingFace 資料集](https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data)
- [Google Colab 官方文件](https://colab.research.google.com/)
- [Pandas 文件](https://pandas.pydata.org/)

---

## 支援

若有問題，請：
1. 檢查網路連接
2. 確認幣種名稱正確
3. 查看 GitHub 的 Issues
4. 提交 Bug 報告

祝你分析愉快！
