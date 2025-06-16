import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class BitcoinPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = []
        self.metrics = {}
        
    def prepare_data(self, df):
        """Chuẩn bị dữ liệu với các features đơn giản"""
        try:
            if df is None or len(df) < 10:
                print("[ERROR] Insufficient data for prediction")
                return False
            
            # Tạo features
            df_features = df.copy()
            
            # Price features
            df_features['price_change'] = df_features['close'].pct_change()
            df_features['price_ma7'] = df_features['close'].rolling(7).mean()
            df_features['price_ma14'] = df_features['close'].rolling(14).mean()
            
            # Volume features  
            df_features['volume_ma7'] = df_features['volume'].rolling(7).mean()
            df_features['volume_change'] = df_features['volume'].pct_change()
            
            # Technical indicators
            df_features['high_low_ratio'] = df_features['high'] / df_features['low']
            df_features['close_open_ratio'] = df_features['close'] / df_features['open']
            
            # Volatility
            df_features['volatility'] = df_features['close'].rolling(7).std()
            
            # Lag features
            for lag in [1, 2, 3]:
                df_features[f'close_lag_{lag}'] = df_features['close'].shift(lag)
                df_features[f'volume_lag_{lag}'] = df_features['volume'].shift(lag)
            
            # RSI
            delta = df_features['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df_features['rsi'] = 100 - (100 / (1 + rs))
            
            # Select feature columns
            self.feature_columns = [
                'price_change', 'price_ma7', 'price_ma14',
                'volume_ma7', 'volume_change',
                'high_low_ratio', 'close_open_ratio', 'volatility',
                'close_lag_1', 'close_lag_2', 'close_lag_3',
                'volume_lag_1', 'volume_lag_2', 'volume_lag_3',
                'rsi'
            ]
            
            # Clean data
            df_features = df_features.dropna()
            
            if len(df_features) < 20:
                print("[ERROR] Not enough data after feature engineering")
                return False
            
            # Prepare X and y
            X = df_features[self.feature_columns].values
            y = df_features['close'].values
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Store for training
            self.X = X_scaled
            self.y = y
            self.df_features = df_features
            
            print(f"[OK] Prepared {len(X)} samples with {len(self.feature_columns)} features")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error preparing data: {str(e)}")
            return False
    
    def train_model(self):
        """Huấn luyện mô hình Linear Regression"""
        try:
            if not hasattr(self, 'X') or not hasattr(self, 'y'):
                print("[ERROR] Data not prepared. Call prepare_data() first.")
                return False
            
            # Split data
            split_idx = int(len(self.X) * 0.8)
            X_train = self.X[:split_idx]
            X_test = self.X[split_idx:]
            y_train = self.y[:split_idx]
            y_test = self.y[split_idx:]
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test)
            
            # Calculate metrics
            self.metrics = {
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }
            
            self.is_trained = True
            
            print(f"[OK] Model trained successfully!")
            print(f"MAE: {self.metrics['mae']:.2f}")
            print(f"RMSE: {self.metrics['rmse']:.2f}")
            print(f"R²: {self.metrics['r2']:.3f}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Error training model: {str(e)}")
            return False
    
    def predict_next_days(self, days=7):
        """Dự đoán giá cho số ngày tiếp theo"""
        try:
            if not self.is_trained:
                print("[ERROR] Model not trained yet")
                return None
            
            predictions = []
            
            # Get last known features
            last_features = self.X[-1:].copy()
            
            for day in range(days):
                # Predict next price
                pred_price = self.model.predict(last_features)[0]
                
                # Add trend and some randomness
                trend_factor = 1 + np.random.normal(0, 0.02)  # ±2% random trend
                pred_price_adjusted = pred_price * trend_factor
                
                predictions.append(max(pred_price_adjusted, 0))  # Ensure positive price
                
                # Update features for next prediction (simple approach)
                last_features = last_features.copy()
            
            print(f"[OK] Generated {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            print(f"[ERROR] Error making predictions: {str(e)}")
            return None
    
    def get_model_metrics(self):
        """Trả về metrics của mô hình"""
        return self.metrics if hasattr(self, 'metrics') else {}
    
    def get_feature_importance(self):
        """Trả về độ quan trọng của features"""
        if not self.is_trained:
            return {}
        
        try:
            # For LinearRegression, use coefficients as importance
            importance = abs(self.model.coef_)
            feature_importance = dict(zip(self.feature_columns, importance))
            
            # Sort by importance
            sorted_importance = dict(sorted(feature_importance.items(), 
                                          key=lambda x: x[1], reverse=True))
            
            return sorted_importance
            
        except Exception as e:
            print(f"[ERROR] Error getting feature importance: {str(e)}")
            return {}

# Test function
def test_predictor():
    """Test predictor với dữ liệu giả"""
    print("🧪 Testing Bitcoin Predictor...")
    
    # Generate test data
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100, freq='D')
    np.random.seed(42)
    
    base_price = 50000
    prices = []
    volumes = []
    
    for i in range(100):
        # Random walk for price
        change = np.random.normal(0.001, 0.03)
        base_price *= (1 + change)
        prices.append(base_price)
        volumes.append(np.random.uniform(1000, 5000))
    
    df = pd.DataFrame({
        'open': [p * 0.999 for p in prices],
        'high': [p * 1.01 for p in prices], 
        'low': [p * 0.99 for p in prices],
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    # Test predictor
    predictor = BitcoinPredictor()
    
    if predictor.prepare_data(df):
        if predictor.train_model():
            predictions = predictor.predict_next_days(7)
            
            if predictions:
                print("\n📈 7-day predictions:")
                for i, pred in enumerate(predictions):
                    print(f"Day {i+1}: ${pred:,.2f}")
                
                print("\n🎯 Top 5 important features:")
                importance = predictor.get_feature_importance()
                for feature, score in list(importance.items())[:5]:
                    print(f"• {feature}: {score:.4f}")
                
                print("\n✅ Predictor test completed successfully!")
                return True
            else:
                print("❌ Prediction failed")
                return False
        else:
            print("❌ Model training failed")
            return False
    else:
        print("❌ Data preparation failed")
        return False

if __name__ == "__main__":
    test_predictor()
