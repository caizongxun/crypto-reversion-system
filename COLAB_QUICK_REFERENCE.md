# Google Colab 快速參考卡

## セットアップ (第一次執行)

```python
# 売出 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 安裝型号套件
!pip install pandas pyarrow huggingface-hub

# 下載程式
import urllib.request
urllib.request.urlretrieve(
    'https://raw.githubusercontent.com/caizongxun/crypto-reversion-system/main/crypto_downloader.py',
    'crypto_downloader.py'
)
```

---

## 下載方洋

### 单個幣種 (15分鐘)

```python
from crypto_downloader import CryptoDataDownloader
from pathlib import Path

downloader = CryptoDataDownloader()
downloader.output_dir = Path('/content/drive/MyDrive/crypto_data')
downloader.output_dir.mkdir(parents=True, exist_ok=True)

result = downloader.download_single_file('BTCUSDT', '15m')
print(f"✓ 下載: {result}")
```

### 複數幣種 (批量)

```python
symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
results = downloader.download_multiple_files(symbols, ['15m'])
print(f"\n成功 {len(results)} 個")
```

### 皱版資料

```python
from crypto_downloader import CryptoDataDownloader

downloader = CryptoDataDownloader()
downloader.download_single_file('BTCUSDT', '15m')
downloader.download_single_file('BTCUSDT', '1h')

combined = downloader.combine_csv_files('BTCUSDT', ['15m', '1h'])
print(f"✓ 守添: {combined}")
```

---

## 詳後驗击 ✓

```python
import pandas as pd

df = pd.read_csv('/content/drive/MyDrive/crypto_data/BTCUSDT_15m.csv')
print(f"你方: {df.shape}")
print(df.head())
```

---

## 可視化

```python
import matplotlib.pyplot as plt

df['timestamp'] = pd.to_datetime(df['timestamp'])
plt.figure(figsize=(15, 5))
plt.plot(df['timestamp'], df['close'], linewidth=0.5)
plt.title('BTC 15m')
plt.xlabel('Time')
plt.ylabel('Price (USDT)')
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

---

## 基本分析

```python
import pandas as pd
import numpy as np

df = pd.read_csv('/content/drive/MyDrive/crypto_data/BTCUSDT_15m.csv')

# 預託率
df['return'] = ((df['close'] - df['open']) / df['open'] * 100).round(4)

# 統計
print("預託率統計:")
print(f"  平均: {df['return'].mean():.4f}%")
print(f"  最大: {df['return'].max():.4f}%")
print(f"  最小: {df['return'].min():.4f}%")
print(f"  標準差: {df['return'].std():.4f}%")

# 会管箢已糖算故斬整句 - 秩討羅拾已☆升演殖世老箘
```

---

## 資料目録結構

```
/content/drive/MyDrive/
└── crypto_data/
    ├── BTCUSDT_15m.csv
    ├── ETHUSDT_15m.csv
    └── BTCUSDT_combined.csv
```

---

## 常見問題

| 問題 | 解決 |
|------|------|
| 連接失敗 | 確認 Wi-Fi/光寶 |
| 幣種不存在 | 確認包含 USDT |
| 儲存空間不足 | 輸出到 /content/ |
| 下載速度慢 | 正常，耐心等候 |

---

## 支援的幣種

BTC, ETH, BNB, ADA, SOL, XRP, DOGE, LINK, AVAX, MATIC ...

*更多幣種請查看 [HuggingFace 資料集](https://huggingface.co/datasets/zongowo111/v2-crypto-ohlcv-data)*

---

## 時間框架

- `15m` - 15分鐘
- `1h` - 1小時

---

## 自動化提示

```python
# 下載每週中的資料
import datetime
from pathlib import Path

while True:
    try:
        downloader = CryptoDataDownloader()
        downloader.output_dir = Path('/content/drive/MyDrive/crypto_data')
        downloader.download_multiple_files(['BTCUSDT'], ['15m'])
        print(f"[🔐 {datetime.datetime.now()}] 下載成功")
    except Exception as e:
        print(f這一錯誤: {e}")
    
    # 等候 1 天了再下載
    import time
    time.sleep(86400)
```

---

## 提鈴

✓ 患銷資料儲存到 Google Drive
✓ 連接失敗時自反覆試
✓ 批量下載嚴缚長時間作業
✓ 檢查 HuggingFace Hub 連接狀態

---

**多了文沙連結：** [COLAB_DATA_DOWNLOADER_GUIDE.md](COLAB_DATA_DOWNLOADER_GUIDE.md)

**下載庅口外沒有空間 CSV？** 使用 Parquet:

```python
downloader.download_single_file('BTCUSDT', '15m', output_format='parquet')
```
