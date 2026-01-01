# Advanced Reversal Probability Indicator (ARPI)

高級反轉概率指標 - 虛擬貨幣交易信號驗證系統

## 項目概述

ARPI 是一套完整的虛擬貨幣交易系統，結合以下核心功能：

1. **Pine Script v5 指標** - TradingView 上的先進反轉檢測指標
2. **機器學習驗證模型** - 使用 XGBoost 驗證 TradingView 信號可信度
3. **波動篩選機制** - 自動過濾信號過於頻繁的問題
4. **多因素確認系統** - RSI 發散、MACD 發散、Bollinger Bands 三重確認

## 創新特點

### 1. 原創反轉檢測邏輯
- RSI 發散檢測: 價格創新低而 RSI 創新高 (看漲) 或相反 (看跌)
- MACD 發散檢測: 直方圖方向與價格方向相反
- Bollinger Bands 超延: 價格觸及上下軌的極端情況
- 多因素權重: 三種指標同時確認時信號強度最高

### 2. 波動篩選系統
- ATR 最小值篩選: 波動過小（低於 1%）時不發出信號
- ATR 最大值篩選: 波動過大（超過 2%）時不發出信號
- 動態調整: 參數可根據不同交易對自由調整

### 3. 信號可信度計算

信號強度 = (確認因素數 / 3) × 100%
- 1 個因素確認: 33%
- 2 個因素確認: 66%
- 3 個因素確認: 100%

### 4. 機器學習驗證層
- 模型: XGBoost (梯度提升樹)
- 訓練數據: 歷史技術指標 + 後續 5 根蠟燭的價格走勢
- 輸出: 買入或賣出信號的推薦信心度 (0-100%)
- 用途: 對 TradingView 信號進行第二層驗證

## 安裝和使用

### Pine Script 指標安裝

1. 打開 TradingView 圖表
2. 點擊「Pine Editor」
3. 新建指標，複製 `arpi_indicator.pine` 中的代碼
4. 保存並在圖表上應用

### Python 機器學習模型

```bash
pip install -r requirements.txt
python signal_validator.py
```

## 指標參數說明

| 參數 | 預設值 | 說明 |
|------|-------|------|
| RSI 周期 | 14 | RSI 計算周期 |
| RSI 超買 | 70 | RSI 超買閾值 |
| RSI 超賣 | 30 | RSI 超賣閾值 |
| MACD 快線 | 12 | MACD 快速 EMA |
| MACD 慢線 | 26 | MACD 緩慢 EMA |
| ATR 最小% | 1.0 | 波動過小的閾值 |
| ATR 上限倍數 | 2.0 | 波動過大的閾值 |

## 文件結構

```
crypto-reversion-system/
├── arpi_indicator.pine          # Pine Script v5 指標代碼
├── signal_validator.py          # ML 驗證模型
├── requirements.txt             # Python 依賴
├── README.md                    # 本文件
└── .gitignore                   # Git 忽略文件
```

## 使用示例

```python
from signal_validator import SignalValidator
import pandas as pd

validator = SignalValidator()
df = pd.read_csv('bitcoin_data.csv')

# 訓練買入信號模型
metrics = validator.train_model(df, signal_type='buy')

# 預測新信號的可信度
confidence = validator.predict_signal_confidence(df, signal_type='buy')
print(f"買入信號可信度: {confidence:.2f}%")
```

## 風險免責聲明

1. 過去表現不代表未來結果
2. 虛擬貨幣市場高度波動，可能產生重大虧損
3. ML 模型基於歷史模式，不能預測突發事件
4. 技術指標是滯後指標
5. 始終使用止損和適當的頭寸規模

## 許可證

MIT License

## 聯繫方式

如有問題或建議，請提交 Issue 或聯繫作者。
