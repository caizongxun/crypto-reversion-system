# Colab 不克隆倉庫直接執行檔案指南

## 核心概念

不需要 `git clone` 整個倉庫，直接從 GitHub 原始檔案 URL 匯入 Python 模組。

---

## 方法一：直接下載 + 導入（推薦用於單個檔案）

### 步驟 1: 下載 Python 檔案

```python
import urllib.request

# 下載檔案
url = "https://raw.githubusercontent.com/caizongxun/crypto-reversion-system/main/peak_valley_detector.py"
urllib.request.urlretrieve(url, 'peak_valley_detector.py')

print("✓ 已下載 peak_valley_detector.py")
```

### 步驟 2: 直接導入

```python
# 方式 A: 直接導入
from peak_valley_detector import PeakValleyDetector

# 或方式 B: 導入整個模組
import peak_valley_detector
detector = peak_valley_detector.PeakValleyDetector(df)
```

### 完整例子

```python
import pandas as pd
import urllib.request
from peak_valley_detector import PeakValleyDetector

# 下載檔案
url = "https://raw.githubusercontent.com/caizongxun/crypto-reversion-system/main/peak_valley_detector.py"
urllib.request.urlretrieve(url, 'peak_valley_detector.py')

# 讀取資料
df = pd.read_csv('/content/drive/MyDrive/crypto_data/BTCUSDT_15m.csv')

# 使用檔案
detector = PeakValleyDetector(df)
result_df = detector.detect_zigzag(percentage=2.0)

print(f"頂點: {detector.get_summary()['peak_count']}")
```

---

## 方法二：多個檔案批量下載

### 當需要多個相關檔案時

```python
import urllib.request
import os

# 要下載的檔案列表
files_to_download = {
    'peak_valley_detector.py': 'https://raw.githubusercontent.com/caizongxun/crypto-reversion-system/main/peak_valley_detector.py',
    'crypto_downloader.py': 'https://raw.githubusercontent.com/caizongxun/crypto-reversion-system/main/crypto_downloader.py',
}

# 批量下載
for filename, url in files_to_download.items():
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"✓ 已下載 {filename}")
    except Exception as e:
        print(f"✗ 下載 {filename} 失敗: {e}")

print("\n所有檔案下載完成")
```

---

## 方法三：直接從 GitHub 執行代碼（不保存檔案）

### 使用 `exec()` 直接執行

```python
import urllib.request

# 直接從 GitHub 讀取代碼並執行
url = "https://raw.githubusercontent.com/caizongxun/crypto-reversion-system/main/peak_valley_detector.py"
response = urllib.request.urlopen(url)
code = response.read().decode('utf-8')

# 執行代碼
exec(code)

# 現在可以直接使用 PeakValleyDetector 類
detector = PeakValleyDetector(df)
```

**缺點**: 如果代碼中有 `if __name__ == '__main__'` 區塊，會直接執行

---

## 方法四：使用 IPython 的 `%run` 魔術指令

```python
import urllib.request

# 下載檔案
url = "https://raw.githubusercontent.com/caizongxun/crypto-reversion-system/main/peak_valley_detector.py"
urllib.request.urlretrieve(url, 'peak_valley_detector.py')

# 使用 %run 執行
%run peak_valley_detector.py
```

**缺點**: `%run` 會執行整個腳本，可能有意想不到的副作用

---

## 方法五：直接在 Colab 中定義（當代碼簡單時）

```python
# 不下載，直接在 Colab 中定義

class PeakValleyDetector:
    """簡化版本"""
    def __init__(self, df):
        self.df = df.copy()
    
    def detect_zigzag(self, percentage=2.0):
        # 在 Colab 中直接實現邏輯
        pass

# 使用
detector = PeakValleyDetector(df)
```

**適用於**: 臨時測試或代碼簡單的情況

---

## 推薦工作流程

### 適合快速實驗

```python
# 1. 安裝依賴
!pip install pandas pyarrow huggingface-hub

# 2. 掛載 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 3. 直接下載 + 導入模組
import urllib.request
from peak_valley_detector import PeakValleyDetector

url = "https://raw.githubusercontent.com/caizongxun/crypto-reversion-system/main/peak_valley_detector.py"
urllib.request.urlretrieve(url, 'peak_valley_detector.py')

# 4. 使用
import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/crypto_data/BTCUSDT_15m.csv')
detector = PeakValleyDetector(df)
result_df = detector.detect_zigzag()
```

---

## 優缺點總結

| 方法 | 優點 | 缺點 | 推薦度 |
|------|------|------|--------|
| **直接下載 + 導入** | 簡單、快速、清楚 | 檔案保存在 /content | ⭐⭐⭐⭐⭐ |
| **批量下載** | 適合多檔案 | 稍複雜 | ⭐⭐⭐⭐ |
| **exec()** | 不保存檔案 | 有副作用風險 | ⭐⭐ |
| **%run** | 簡潔 | 執行整個腳本 | ⭐⭐⭐ |
| **直接定義** | 無依賴 | 代碼重複 | ⭐⭐⭐ |

---

## 處理依賴問題

### 如果模組有內部依賴

假設 `peak_valley_detector.py` 依賴其他檔案：

```python
import urllib.request
import os

# 建立目錄
os.makedirs('modules', exist_ok=True)

# 下載相關檔案到同一目錄
files = [
    'peak_valley_detector.py',
    'utils.py',  # 如果有的話
    'config.py',  # 如果有的話
]

for file in files:
    url = f"https://raw.githubusercontent.com/caizongxun/crypto-reversion-system/main/{file}"
    try:
        urllib.request.urlretrieve(url, f'modules/{file}')
        print(f"✓ {file}")
    except:
        print(f"✗ {file} 不存在")

# 新增路徑
import sys
sys.path.append('modules')

# 現在可以導入
from peak_valley_detector import PeakValleyDetector
```

---

## 完整的一次性 Colab 工作區塊

複製整個 Cell 貼到 Colab 執行：

```python
# ============================================
# Colab 一鍵設置 + 執行
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from google.colab import drive

# 1. 掛載 Google Drive
print("正在掛載 Google Drive...")
drive.mount('/content/drive')

# 2. 安裝依賴
print("正在安裝依賴...")
!pip install -q pandas pyarrow huggingface-hub

# 3. 下載檔案
print("正在下載模組...")
url = "https://raw.githubusercontent.com/caizongxun/crypto-reversion-system/main/peak_valley_detector.py"
urllib.request.urlretrieve(url, 'peak_valley_detector.py')

# 4. 導入
from peak_valley_detector import PeakValleyDetector

# 5. 讀取資料
print("正在讀取資料...")
df = pd.read_csv('/content/drive/MyDrive/crypto_data/BTCUSDT_15m.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 6. 檢測頂底點
print("正在檢測頂底點...")
detector = PeakValleyDetector(df)
result_df = detector.detect_zigzag(percentage=2.0)
summary = detector.get_summary()

# 7. 顯示結果
print(f"\n✓ 完成！")
print(f"頂點: {summary['peak_count']} 個")
print(f"底點: {summary['valley_count']} 個")

# 8. 視覺化
fig, ax = plt.subplots(figsize=(18, 6))
ax.plot(result_df['timestamp'], result_df['close'], color='black', linewidth=1)
ax.scatter(result_df[result_df['is_peak']]['timestamp'], 
          result_df[result_df['is_peak']]['close'], 
          color='red', marker='v', s=150, label='Peak')
ax.scatter(result_df[result_df['is_valley']]['timestamp'], 
          result_df[result_df['is_valley']]['close'], 
          color='green', marker='^', s=150, label='Valley')
ax.set_title('BTC 15m - 頂點和底點')
ax.legend()
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 9. 保存
result_df.to_csv('/content/drive/MyDrive/crypto_data/BTCUSDT_15m_marked.csv', index=False)
print("\n✓ 已保存到 Google Drive")
```

---

## 不同使用場景

### 場景 1: 只需要一個簡單檔案

```python
import urllib.request
from peak_valley_detector import PeakValleyDetector

url = "https://raw.githubusercontent.com/caizongxun/crypto-reversion-system/main/peak_valley_detector.py"
urllib.request.urlretrieve(url, 'peak_valley_detector.py')

detector = PeakValleyDetector(df)
```

### 場景 2: 需要多個檔案協作

```python
import sys
sys.path.append('.')

# 下載所有需要的檔案
files = ['peak_valley_detector.py', 'crypto_downloader.py']
for file in files:
    url = f"https://raw.githubusercontent.com/caizongxun/crypto-reversion-system/main/{file}"
    urllib.request.urlretrieve(url, file)

from peak_valley_detector import PeakValleyDetector
from crypto_downloader import CryptoDataDownloader
```

### 場景 3: 頻繁迭代開發

```python
# 建議克隆倉庫一次
!git clone https://github.com/caizongxun/crypto-reversion-system.git
import sys
sys.path.append('crypto-reversion-system')

# 之後就可以直接導入
from peak_valley_detector import PeakValleyDetector
from crypto_downloader import CryptoDataDownloader
```

---

## 常見問題

### Q: 為什麼下載後還要 `from` 導入？

**A:** 下載把 `.py` 檔案放到本地，導入才能使用檔案中定義的類/函數。

### Q: 可以直接從 GitHub URL 導入嗎？

**A:** 不行，Python 預設只能導入本地檔案。需要先下載再導入。

### Q: 如何避免每次都下載？

**A:** 檔案保存在 `/content/` 中，同一個 session 內可以重複使用。session 結束後才會清除。

### Q: 如果更新了倉庫中的檔案，Colab 會自動更新嗎？

**A:** 不會。需要重新執行下載代碼。

### Q: 能否離線使用？

**A:** 不能。第一次必須聯網下載。下載後在同一 session 內可以離線使用。

---

## 未來工作流

當你在 crypto-reversion-system 倉庫中新增檔案時：

1. **新增訓練檔案** → 上傳到倉庫
2. **在 Colab 中** → 只需 3 行代碼

```python
import urllib.request
from your_new_module import YourClass

url = "https://raw.githubusercontent.com/caizongxun/crypto-reversion-system/main/your_new_module.py"
urllib.request.urlretrieve(url, 'your_new_module.py')
```

3. **使用** → 正常使用即可

```python
obj = YourClass(data)
```

---

**就是這樣簡單！** 不需要每次都克隆整個倉庫。
