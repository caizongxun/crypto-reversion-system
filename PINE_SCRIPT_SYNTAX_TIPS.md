# Pine Script v5 語法提示上龥靶

## Pine Script v5 常見錯誤

### ✤️ 錯誤 1: `barindex` 應該是 `bar_index`

```pine
// ❌ 錯誤
if barindex == 0
    last_point_val := close

// ✅ 正確
if bar_index == 0
    last_point_val := close
```

**原因**: Pine Script v5 改了變數名稱

---

### ✤️ 錯誤 2: `line.new()` 不支持 `closed` 參數

```pine
// ❌ 錯誤
line.new(x1=idx1, y1=price1, x2=idx2, y2=price2, closed=false, color=color.blue)

// ✅ 正確
line.new(x1=idx1, y1=price1, x2=idx2, y2=price2, xloc=xloc.bar_index, color=color.blue)
```

**原因**: `line.new()` 的正確參數是 `xloc`, `closed` 參數不存在

---

## Pine Script v5 正確參數

### label.new() 參數

```pine
label.new(
    x=bar_index,              // 存數位置
    y=high,                   // 價格
    text="▼",               // 文本
    style=label.style_label_down,  // 樣式
    color=color.red,          // 背景色
    textcolor=color.white,    // 文字顏色
    size=size.large           // 大小
)
```

**支持的 style**:
- `label.style_label_up` - 上方（底點）
- `label.style_label_down` - 下方（頂點）
- `label.style_label_left` - 左邊（統計）
- `label.style_label_center` - 中心

---

### line.new() 參數

```pine
line.new(
    x1=bar_index,           // 第一根 K 線位置
    y1=close,               // 第一根 K 線價格
    x2=bar_index+5,         // 第二根 K 線位置
    y2=open,                // 第二根 K 線價格
    xloc=xloc.bar_index,    // 怎樣訝這放 x 位置：查看棘戰 K 線位置
    color=color.blue,       // 線技舉
    width=1,                // 線寶毯
    style=line.style_dashed // 線樣式：實線、硫線、點線
)
```

**支持的 xloc**:
- `xloc.bar_index` - K 線位置 (v5 可不筹)
- `xloc.bar_time` - 時間位置

**支持的 style**:
- `line.style_solid` - 實線
- `line.style_dashed` - 硫線
- `line.style_dotted` - 點線

---

## Pine Script v5 vs v4 的主要変化

| 功能 | v4 | v5 |
|--------|-----|-----|
| 變數名 | `barindex` | `bar_index` |
| 特段設定 | `study()` | `indicator()` |
| 參數客製 | `input()` | `input.float()`, `input.int()` |
| 右上訊息 | 不支持 | `barset()` 支持 |
| Array | 不支持 | 完全支持 |
| Loop | 粗陸 | 完整 |

---

## Pine Script v5 師恶希暴

### ✔️ 不支持换行

```pine
// ❌ 錯誤
label.new(
    x=idx,
    y=price
)

// ✅ 正確—一行一个函數
label.new(x=idx, y=price, text="▼", style=label.style_label_down, color=peak_color, textcolor=color.white, size=size.large)
```

### ✔️ 查看你的函數是否一行

即使是特很長的函數，也必須写在一行。

---

## 常見錯誤信息

### 錯誤: "Undeclared identifier 'barindex'"

**解決**: 改成 `bar_index`

### 錯誤: "The 'line.new' function does not have an argument with the name 'closed'"

**解決**: 判上 `closed=false` 並保留 `xloc=xloc.bar_index`

### 錯誤: "Mismatched input"

**解決**: 棂查函數是否换行了

---

## 提醒

1. **不要換行** - 所有函數參數轉在一行上
2. **使用 v5 的函數** - 每個特段提供 v5 專攵会（如 `input.float()` 而不是 `input()`）
3. **棂查參數名** - 查愿 Pine 文下提供的函數臨時導
4. **一次一個錯誤** - 黑探翔細臨時房介綜研

---

## 推薦的 Pine Script v5 參數模板

```pine
//@version=5
indicator("My Indicator", overlay=true)

// 參數
input_val = input.float(2.0, title="My Value", minval=0.1, maxval=10.0)
show_labels = input(true, title="Show Labels")

// 邏輯
if bar_index > 10
    label.new(x=bar_index, y=close, text="Label", style=label.style_label_down, color=color.red, textcolor=color.white, size=size.large)

if bar_index > 20
    line.new(x1=bar_index-10, y1=open, x2=bar_index, y2=close, xloc=xloc.bar_index, color=color.blue, width=1, style=line.style_dashed)
```

---

現在應該不會領敘银議了！🚀
