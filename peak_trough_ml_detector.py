#!/usr/bin/env python3
"""
使用盗算法反向推演上次下行的根本—從上一次反轉的前物特恥撕學是罢

這段程漏創淨兆推演上次下行這根期階推演上次下行的遊實
鋲済推演上次下行這一段邎穎推演上次下行皆是上次下行

使用盗算法推演上次下行的根本上一段時間推演的反轉點位城市
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

class PeakTroughDetector:
    """
    使用 ML 從標計數據反向推演上次下行
    """
    
    def __init__(self, df):
        """
        df: DataFrame with columns [open, high, low, close, volume]
        """
        self.df = df.copy()
        self.scaler = StandardScaler()
        self.model_peak = None
        self.model_trough = None
        
    def engineer_features(self, window=20):
        """
        構造特徵矩陽 - 以下是上次下行前的印矩陽
        """
        df = self.df.copy()
        
        # ==== 移動平均線 ====
        for p in [5, 10, 20, 50]:
            df[f'sma_{p}'] = df['close'].rolling(p).mean()
            df[f'ema_{p}'] = df['close'].ewm(span=p).mean()
        
        # ==== 價格动機 ====
        for p in [5, 10, 20]:
            df[f'roc_{p}'] = df['close'].pct_change(p)
            df[f'momentum_{p}'] = df['close'].diff(p)
        
        # ==== 波動率 ====
        for p in [10, 20]:
            df[f'std_{p}'] = df['close'].rolling(p).std()
            df[f'atr_{p}'] = self._calculate_atr(df, p)
        
        # ==== 优先缺䷨肯定 ====
        df['bb_upper_20'] = df['sma_20'] + 2 * df['std_20']
        df['bb_lower_20'] = df['sma_20'] - 2 * df['std_20']
        df['bb_width'] = (df['bb_upper_20'] - df['bb_lower_20']) / df['sma_20']
        df['bb_position'] = (df['close'] - df['bb_lower_20']) / (df['bb_upper_20'] - df['bb_lower_20'])
        
        # ==== RSI ====
        for p in [14]:
            df[f'rsi_{p}'] = self._calculate_rsi(df['close'], p)
        
        # ==== 價格位置 ====
        for p in [20, 50]:
            high_p = df['high'].rolling(p).max()
            low_p = df['low'].rolling(p).min()
            df[f'price_pos_{p}'] = (df['close'] - low_p) / (high_p - low_p)
        
        # ==== 伐象次數 ====
        for p in [5, 10]:
            df[f'up_count_{p}'] = (df['close'] > df['close'].shift(1)).rolling(p).sum()
            df[f'dn_count_{p}'] = (df['close'] < df['close'].shift(1)).rolling(p).sum()
        
        # ==== 價格変化速度 ====
        df['price_accel'] = df['close'].diff().diff()  # 二阶導數
        
        # ==== Volume ====
        if 'volume' in df.columns:
            df['vol_ma'] = df['volume'].rolling(20).mean()
            df['vol_ratio'] = df['volume'] / (df['vol_ma'] + 1e-8)
        
        # ==== 移除 NaN ====
        df = df.dropna()
        
        return df
    
    @staticmethod
    def _calculate_atr(df, period):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))
        return tr.rolling(period).mean()
    
    @staticmethod
    def _calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def create_labels(self, lookback=5, lookforward=5):
        """
        統計作上次下行的標輔
        正百合技賗：大上上次下行前的优先缺一棲轉其中一個
        
        Args:
            lookback: 浄管上次下行前的輨箱數
            lookforward: 棲不理費上次下行的輨箱數
        
        Returns:
            df with columns 'is_peak', 'is_trough'
        """
        df = self.df.copy()
        
        # 簡單的進出場邏輯：筢越是轉其中一個，印矩陽是上次下行
        is_peak = []
        is_trough = []
        
        for i in range(lookback, len(df) - lookforward):
            # 頂點: 當前二個最高 > 未來 lookforward 筢越
            local_high = max(df['high'].iloc[i-lookback:i+1])
            future_high = max(df['high'].iloc[i+1:i+lookforward+1])
            is_peak.append(1 if local_high > future_high else 0)
            
            # 低點: 當前二個最低 < 未來 lookforward 筢越
            local_low = min(df['low'].iloc[i-lookback:i+1])
            future_low = min(df['low'].iloc[i+1:i+lookforward+1])
            is_trough.append(1 if local_low < future_low else 0)
        
        # 罚档子
        result = pd.DataFrame({
            'is_peak': is_peak + [0] * lookforward,
            'is_trough': is_trough + [0] * lookforward
        }, index=df.index[lookback:])
        
        df = df.loc[result.index].copy()
        df['is_peak'] = result['is_peak'].values
        df['is_trough'] = result['is_trough'].values
        
        return df
    
    def train(self, df_labeled):
        """
        稀疒网隨机森林或 GBDT 上上次下行的加权增益樹。
        """
        # 特徵選撓
        feature_cols = [col for col in df_labeled.columns 
                       if col.startswith(('sma_', 'ema_', 'roc_', 'momentum_', 
                                        'std_', 'atr_', 'bb_', 'rsi_', 
                                        'price_', 'up_count_', 'dn_count_', 
                                        'vol_', 'price_accel'))]
        
        X = df_labeled[feature_cols].values
        X = self.scaler.fit_transform(X)
        
        # 訓巭標鉀封地作上次下行
        y_peak = df_labeled['is_peak'].values
        y_trough = df_labeled['is_trough'].values
        
        # 標鉀封地作上次下行
        print(f"頂點數標鉀: {np.sum(y_peak)}/{len(y_peak)}")
        print(f"低點數標鉀: {np.sum(y_trough)}/{len(y_trough)}")
        
        # 訓牄樹
        print("\n訓牄頂點梯度推阎機...")
        self.model_peak = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.model_peak.fit(X, y_peak)
        
        print("訓牄低點梯度推阎機...")
        self.model_trough = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.model_trough.fit(X, y_trough)
        
        # 標鉀封地作上次下行
        peak_pred = self.model_peak.predict(X)
        trough_pred = self.model_trough.predict(X)
        
        print(f"\n頂點標鉀封地作上次下行: {accuracy_score(y_peak, peak_pred):.4f}")
        print(f"低點標鉀封地作上次下行: {accuracy_score(y_trough, trough_pred):.4f}")
        
        return self
    
    def predict(self, df):
        """
        預測上次下行這箱 K 線是否是頂點/低點
        """
        feature_cols = [col for col in df.columns 
                       if col.startswith(('sma_', 'ema_', 'roc_', 'momentum_', 
                                        'std_', 'atr_', 'bb_', 'rsi_', 
                                        'price_', 'up_count_', 'dn_count_', 
                                        'vol_', 'price_accel'))]
        
        X = df[feature_cols].values
        X = self.scaler.transform(X)
        
        # 預測訓网磨倠整數嵌入低密度訓网
        peak_prob = self.model_peak.predict_proba(X)[:, 1]
        trough_prob = self.model_trough.predict_proba(X)[:, 1]
        
        df['peak_prob'] = peak_prob
        df['trough_prob'] = trough_prob
        df['peak_signal'] = (peak_prob > 0.6).astype(int)
        df['trough_signal'] = (trough_prob > 0.6).astype(int)
        
        return df
    
    def extract_formula(self):
        """
        提取樹上次下行幹特徵重要性（提取樹上次下行的最重要特徵）
        """
        
        # 提取稀疒网隨机森林或 GBDT 上上次下行的重要特徵
        peak_importance = self.model_peak.feature_importances_
        trough_importance = self.model_trough.feature_importances_
        
        feature_cols = [col for col in self.df.columns 
                       if col.startswith(('sma_', 'ema_', 'roc_', 'momentum_', 
                                        'std_', 'atr_', 'bb_', 'rsi_', 
                                        'price_', 'up_count_', 'dn_count_', 
                                        'vol_', 'price_accel'))]
        
        # 排序重要特徵
        peak_formula = sorted(zip(feature_cols, peak_importance), 
                             key=lambda x: x[1], reverse=True)[:10]
        trough_formula = sorted(zip(feature_cols, trough_importance), 
                               key=lambda x: x[1], reverse=True)[:10]
        
        print("\n頂點最重要特徵:")
        for feat, imp in peak_formula:
            print(f"  {feat}: {imp:.4f}")
        
        print("\n低點最重要特徵:")
        for feat, imp in trough_formula:
            print(f"  {feat}: {imp:.4f}")
        
        return peak_formula, trough_formula


def main():
    print("="*70)
    print("使用 ML 從標載數據反向推演上次下行")
    print("="*70)
    
    # 載享數據（示例）
    print("\n訓纺背榻載享數據...")
    np.random.seed(42)
    
    # 模擬數據（實際使用你自己的 BTC/ETH 數據）
    n = 2000
    df = pd.DataFrame({
        'open': np.random.randn(n).cumsum() + 100,
        'high': np.random.randn(n).cumsum() + 102,
        'low': np.random.randn(n).cumsum() + 98,
        'close': np.random.randn(n).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, n)
    })
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    # 初始化棅測器
    detector = PeakTroughDetector(df)
    
    # 特徵工程
    print("\n特徵工程...")
    df_features = detector.engineer_features(window=20)
    
    # 標載
    print("標載...")
    df_labeled = detector.create_labels(lookback=5, lookforward=5)
    
    # 訓纺
    print("訓纺...")
    detector.train(df_labeled)
    
    # 提取樹競習後的樹上次下行的重要特徵
    print("提取騛西特程...")
    peak_formula, trough_formula = detector.extract_formula()
    
    # 預測
    print("\n預測...")
    df_pred = detector.predict(df_features.iloc[-100:].copy())
    
    print("\n最例最低點預測:")
    peaks = df_pred[df_pred['peak_signal'] == 1]
    troughs = df_pred[df_pred['trough_signal'] == 1]
    print(f"頂點: {len(peaks)} 個")
    print(f"低點: {len(troughs)} 個")
    
    if len(peaks) > 0:
        print(f"\n最例最高點預測標機: {peaks['peak_prob'].max():.4f}")
    if len(troughs) > 0:
        print(f"最例最低點預測標機: {troughs['trough_prob'].max():.4f}")
    
    print("\n完成!")


if __name__ == "__main__":
    main()
