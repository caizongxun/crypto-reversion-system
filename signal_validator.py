import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SignalValidator:
    """驗證交易信號可信度的機器學習模型"""

    def __init__(self, window_size=20):
        self.window_size = window_size
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def calculate_technical_indicators(self, df):
        """計算技術指標特徵"""
        df = df.copy()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['sma20'] = df['close'].rolling(window=20).mean()
        df['std20'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['sma20'] + (df['std20'] * 2)
        df['bb_lower'] = df['sma20'] - (df['std20'] * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        df['tr'] = np.maximum(df['high'] - df['low'], np.maximum(abs(df['high'] - df['close'].shift()), abs(df['low'] - df['close'].shift())))
        df['atr'] = df['tr'].rolling(window=14).mean()
        df['atr_percent'] = (df['atr'] / df['close']) * 100
        df['returns'] = df['close'].pct_change()
        df['momentum'] = df['close'] - df['close'].shift(5)
        df['volume_sma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['price_momentum'] = df['close'].diff(5)
        df['rsi_momentum'] = df['rsi'].diff(5)
        df['macd_momentum'] = df['macd_hist'].diff(5)
        return df.dropna()

    def create_features(self, df):
        """為模型創建特徵集合"""
        feature_cols = ['rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_position', 'atr_percent', 'returns', 'momentum', 'volume_sma_ratio', 'price_momentum', 'rsi_momentum', 'macd_momentum']
        self.feature_names = feature_cols
        X = df[feature_cols].values
        return self.scaler.fit_transform(X)

    def label_signals(self, df, signal_type='buy', lookahead=5):
        """為信號標籤化"""
        y = np.zeros(len(df))
        for i in range(len(df) - lookahead):
            future_return = (df['close'].iloc[i + lookahead] - df['close'].iloc[i]) / df['close'].iloc[i]
            if signal_type == 'buy':
                y[i] = 1 if future_return > 0.005 else 0
            else:
                y[i] = 1 if future_return < -0.005 else 0
        return y

    def train_model(self, df, signal_type='buy', test_size=0.2):
        """訓練XGBoost模型"""
        print(f"開始訓練 {signal_type.upper()} 信號驗證模型...")
        df = self.calculate_technical_indicators(df)
        X = self.create_features(df)
        y = self.label_signals(df, signal_type=signal_type)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        self.model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, use_label_encoder=False, eval_metric='logloss')
        self.model.fit(X_train, y_train, verbose=False)
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"\n===== {signal_type.upper()} 信號模型性能 =====")
        print(f"準確率: {accuracy:.4f}")
        print(f"精確率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1-分數: {f1:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        print(f"特異性: {specificity:.4f}")
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc, 'specificity': specificity}

    def predict_signal_confidence(self, df, signal_type='buy'):
        """預測信號的可信度百分比"""
        if self.model is None:
            raise ValueError("模型尚未訓練")
        df = self.calculate_technical_indicators(df)
        X = self.create_features(df)
        probabilities = self.model.predict_proba(X)[:, 1]
        confidence = probabilities[-1] * 100
        return confidence

    def save_model(self, filepath):
        """保存訓練好的模型"""
        joblib.dump(self.model, filepath)
        print(f"模型已保存至: {filepath}")

    def load_model(self, filepath):
        """載入已訓練的模型"""
        self.model = joblib.load(filepath)
        print(f"模型已載入: {filepath}")
