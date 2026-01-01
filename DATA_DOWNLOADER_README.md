# 加密貨幣資料下載器 - 完整文檔

> 從 HuggingFace Hub 下載加密貨幣 OHLCV 資料並轉換為 CSV 格式

## 概述

### 功能特性

✓ 一次性下載單一或多個幣種的 OHLCV 資料
✓ 自動從 Parquet 轉換為 CSV 格式
✓ 支持多個時間框架 (15分鐘、母小時等)
✓ 合併同一幣種不同時間框架的資料
✓ 提供資料统計信息
✓ 完全支援 Google Colab

### 資料來源

[zongowo111/v2-crypto-ohlcv-data](https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data)

- 总计：46 个檔案、481万个数据点、110.57 MB
- 支援 23 種加密貨幣
- 2 种時間框架：15m, 1h

---

## 第一次使用 - Google Colab

### 步驟 1-3: 執行初始化

按順執行以下 3 个包段：

```python
# 売出 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 安裝依賶
!pip install pandas pyarrow huggingface-hub requests scikit-learn

# 下載 crypto_downloader.py
import urllib.request
urllib.request.urlretrieve(
    'https://raw.githubusercontent.com/caizongxun/crypto-reversion-system/main/crypto_downloader.py',
    'crypto_downloader.py'
)
```

### 步驟 4-6: 執行下載

**方案一：下載單一幣種**

```python
from crypto_downloader import CryptoDataDownloader
from pathlib import Path

# 初始化
downloader = CryptoDataDownloader()
downloader.output_dir = Path('/content/drive/MyDrive/crypto_data')
downloader.output_dir.mkdir(parents=True, exist_ok=True)

# 下載 BTC 15 分鐘資料
result = downloader.download_single_file('BTCUSDT', '15m')
print(f"✓ 下載完成: {result}")

# 查看資料資訊
info = downloader.get_data_info('BTCUSDT', '15m')
print("\n資料統計:")
for key, value in info.items():
    print(f"  {key}: {value}")
```

**方案二：批量下載**

```python
# 批量下載多个幣种
symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOGEUSDT']
results = downloader.download_multiple_files(
    symbols,
    ['15m', '1h']  # 下載两个时钱框架
)

print(f"\n成功下載 {len(results)} 个檔案")
for key, path in results.items():
    print(f"  - {key}: {path}")
```

**方案三：合併資料**

```python
# 合併的不呜时薵棧时帧
 downloader.download_single_file('BTCUSDT', '15m')
downloader.download_single_file('BTCUSDT', '1h')

combined_path = downloader.combine_csv_files('BTCUSDT', ['15m', '1h'])
print(f"\n合併後: {combined_path}")
```

---

## 驗證資料

```python
import pandas as pd

df = pd.read_csv('/content/drive/MyDrive/crypto_data/BTCUSDT_15m.csv')

print(f"資料形狀: {df.shape}")
print(f"\n上量 5 行:")
print(df.head())

print(f"\n欄位: {list(df.columns)}")
```

---

## 分析示例

### 基本統計

```python
import pandas as pd

df = pd.read_csv('/content/drive/MyDrive/crypto_data/BTCUSDT_15m.csv')

# 計算江时探抢
df['return'] = ((df['close'] - df['open']) / df['open'] * 100).round(4)

print("\n返回狫統計:")
print(f"  平均: {df['return'].mean():.4f}%")
print(f"  最大: {df['return'].max():.4f}%")
print(f"  最小: {df['return'].min():.4f}%")
print(f"  標準差: {df['return'].std():.4f}%")
```

### 可視化

```python
import matplotlib.pyplot as plt

df['timestamp'] = pd.to_datetime(df['timestamp'])

plt.figure(figsize=(15, 5))
plt.plot(df['timestamp'], df['close'], linewidth=0.5)
plt.title('BTC 15分鐘收盤價')
plt.xlabel('時間')
plt.ylabel('價格 (USDT)')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

---

## API 念一寶注

### 初始化

```python
from crypto_downloader import CryptoDataDownloader

downloader = CryptoDataDownloader(
    repo_id="zongowo111/v2-crypto-ohlcv-data"  # 可省非
)
```

### 方法

#### 1. 下載單一幣種

```python
downloader.download_single_file(
    symbol='BTCUSDT',      # 幣種符號
    timeframe='15m',       # 時時框架: '15m' or '1h'
    output_format='csv'    # 輸出格式: 'csv' or 'parquet'
)
```

**返回**: 輸出檔案路徑 (str) 或 None

#### 2. 批量下載

```python
downloader.download_multiple_files(
    symbols=['BTCUSDT', 'ETHUSDT'],  # 幣種列表
    timeframes=['15m', '1h'],         # 時時框架列表
    output_format='csv'               # 輸出格式
)
```

**返回**: {檔案名: 路徑} 字典

#### 3. 合併 CSV

```python
downloader.combine_csv_files(
    symbol='BTCUSDT',                  # 幣種符號
    timeframes=['15m', '1h'],          # 要合併的時楋
    output_filename='BTCUSDT_all.csv'  # 輸出檔案名
)
```

**返回**: 輸出檔案路徑

#### 4. 查看資料統計

```python
downloader.get_data_info(
    symbol='BTCUSDT',
    timeframe='15m'
)
```

**返回**: 包含資料統計的字典

---

## 支援的幣種

| 幣種 | 符號 | 幣種 | 符號 |
|--------|--------|--------|--------|
| Bitcoin | BTCUSDT | Cardano | ADAUSDT |
| Ethereum | ETHUSDT | Solana | SOLUSDT |
| BNB | BNBUSDT | Ripple | XRPUSDT |
| Polkadot | DOTUSDT | Litecoin | LTCUSDT |
| Dogecoin | DOGEUSDT | Polygon | MATICUSDT |

*詳後显示詳源: [HuggingFace 資料集](https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data)*

---

## 想要對流残

### 違犩 1: 下載速度慢

**原因**: 有時鎕批量連接下載需要時間

**解決**: 第一次下載後会缑存，下次下載會很快

### 違犩 2: "FileNotFoundError"

**原因**: 幣種不存在或幣種符號錯誤

**解決**: 檢查:
1. 幣種支收是驗USDT
2. 幣種是否在 HuggingFace 資料集中
3. 網路連接是否正常

### 違犩 3: 儲存空間不足

**解決**:
- 輸出到 `/content/` (環境變更時清除)
- 或使用 Parquet 格式 (需沙草組敬)

---

## 最佳實踐

1. ✔️ 儲存到 Google Drive 以保留資料
2. ✔️ 批量下載比逐個下載更有效率
3. ✔️ 總是驗證下載的資料再進行分析
4. ✔️ 使用清晰的檔案結構
5. ✔️ 這些日誌記錄每一偏歩

---

## 相關檔案

- [COLAB_DATA_DOWNLOADER_GUIDE.md](COLAB_DATA_DOWNLOADER_GUIDE.md) - 詳細說明指南
- [COLAB_QUICK_REFERENCE.md](COLAB_QUICK_REFERENCE.md) - 快速參考卡
- [crypto_downloader.py](crypto_downloader.py) - 源程式

---

## 貨幣綁幣次數

**這事一自種不就是勲金檔案了？** 也包括其了 ❤️

---

## 支援與反饋

遇到問題？
- 查看 [COLAB_DATA_DOWNLOADER_GUIDE.md](COLAB_DATA_DOWNLOADER_GUIDE.md) 的「常見問題」篇章
- 提交 GitHub Issue 報告

---

**祝你分析愉快！** 🚀
