import requests
import pandas as pd
import time
from datetime import datetime
from pathlib import Path
from config import DATA_DIR

class BitcoinDataCollector:
    def __init__(self):
        self.data_dir = DATA_DIR
        self.bitget_symbol = "BTCUSDT_SPBL"
        # Dùng periods đã test thành công
        self.working_periods = ['1h', '4h', '6h', '12h', '1M']
        
    def get_bitget_data(self, limit=365):
        """Lấy dữ liệu từ Bitget API với periods đã test"""
        
        print(f"[INFO] Getting Bitget data for {self.bitget_symbol}...")
        
        # Test ticker trước
        if not self.test_bitget_ticker():
            print("[ERROR] Ticker failed, skipping Bitget")
            return None
        
        # Thử các period đã biết hoạt động
        for period in self.working_periods:
            print(f"[INFO] Trying period: {period}")
            
            url = "https://api.bitget.com/api/spot/v1/market/candles"
            
            # Điều chỉnh limit theo period
            if period == '1h':
                actual_limit = min(limit * 24, 500)  # 1h cần nhiều records hơn
            elif period == '4h':
                actual_limit = min(limit * 6, 500)   # 4h cần 6 records/day
            elif period == '6h':
                actual_limit = min(limit * 4, 500)   # 6h cần 4 records/day
            elif period == '12h':
                actual_limit = min(limit * 2, 500)   # 12h cần 2 records/day
            else:  # 1M
                actual_limit = min(limit // 30, 200) # 1M cần ít records
            
            params = {
                'symbol': self.bitget_symbol,
                'period': period,
                'limit': actual_limit
            }
            
            try:
                response = requests.get(url, params=params, timeout=30)
                
                print(f"Status: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"API Code: {data.get('code')}")
                    
                    if data.get('code') == '00000' and data.get('data'):
                        klines = data['data']
                        print(f"[OK] Got {len(klines)} records with period {period}")
                        
                        # Xử lý dữ liệu
                        df = self._process_bitget_data(klines, period)
                        if df is not None and len(df) > 10:
                            return df
                    else:
                        print(f"[ERROR] API Error: {data.get('msg', 'Unknown error')}")
                else:
                    print(f"[ERROR] HTTP {response.status_code}")
                    
            except Exception as e:
                print(f"[ERROR] Exception with period {period}: {str(e)[:100]}")
        
        print("[ERROR] All working periods failed")
        return None
    
    def test_bitget_ticker(self):
        """Test ticker API"""
        try:
            url = "https://api.bitget.com/api/spot/v1/market/ticker"
            params = {'symbol': self.bitget_symbol}
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000' and data.get('data'):
                    price = float(data['data']['close'])
                    print(f"[OK] Current BTC price: ${price:,.2f}")
                    return True
            
            return False
            
        except Exception as e:
            print(f"[ERROR] Ticker error: {str(e)[:100]}")
            return False
    
    def _process_bitget_data(self, data, period):
        """Xử lý dữ liệu Bitget với resampling"""
        try:
            df = pd.DataFrame(data)
            
            if len(df.columns) >= 6:
                df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume'] + [f'col_{i}' for i in range(6, len(df.columns))]
            else:
                print(f"[ERROR] Unexpected data format: {len(df.columns)} columns")
                return None
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            
            # Convert price columns
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.set_index('timestamp', inplace=True)
            df = df.dropna()
            df = df.sort_index()
            
            # Resample về daily nếu cần
            if period in ['1h', '4h', '6h', '12h']:
                print(f"[INFO] Resampling {period} data to daily...")
                df_daily = df.resample('D').agg({
                    'open': 'first',
                    'high': 'max', 
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
                df = df_daily
            
            df['price'] = df['close']
            
            print(f"[OK] Processed Bitget data: {df.shape}")
            print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
            print(f"Price range: ${df['close'].min():,.2f} - ${df['close'].max():,.2f}")
            
            return df
            
        except Exception as e:
            print(f"[ERROR] Error processing Bitget data: {str(e)[:100]}")
            return None
    
    def get_binance_data(self, symbol="BTCUSDT", interval="1d", limit=365):
        """Fallback: Binance API"""
        print(f"[INFO] Trying Binance API for {symbol}...")
        
        url = "https://api.binance.com/api/v3/klines"
        
        params = {
            'symbol': symbol,
            'interval': interval,
            'limit': limit
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                print(f"[OK] Got {len(data)} records from Binance")
                return self._process_binance_data(data)
            else:
                print(f"[ERROR] Binance error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"[ERROR] Binance error: {str(e)[:100]}")
            return None
    
    def _process_binance_data(self, data):
        """Xử lý dữ liệu Binance"""
        try:
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df.set_index('timestamp', inplace=True)
            df['price'] = df['close']
            
            print(f"[OK] Processed Binance data: {df.shape}")
            print(f"Date range: {df.index.min().date()} to {df.index.max().date()}")
            
            return df.sort_index()
            
        except Exception as e:
            print(f"[ERROR] Error processing Binance data: {str(e)[:100]}")
            return None
    
    def collect_all_data(self, days=365):
        """Thu thập dữ liệu với fallbacks"""
        print("[INFO] Starting Bitcoin data collection...")
        
        # Try 1: Bitget với periods đã test
        df = self.get_bitget_data(days)
        if df is not None and len(df) > 50:
            print("[OK] Using Bitget data")
            self.save_data(df)
            return df
        
        # Try 2: Binance
        df = self.get_binance_data("BTCUSDT", "1d", days)
        if df is not None and len(df) > 50:
            print("[OK] Using Binance data")
            self.save_data(df)
            return df
        
        # Try 3: Cached data
        df = self.load_data()
        if df is not None:
            print("[OK] Using cached data")
            return df
        
        # Try 4: Demo data
        print("[WARNING] Generating demo data...")
        return self.generate_dummy_data(days)
    
    def generate_dummy_data(self, days=365):
        """Tạo dữ liệu demo"""
        import numpy as np
        
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        
        base_price = 105000
        prices = []
        current_price = base_price
        
        for i in range(days):
            change = np.random.normal(0.001, 0.04)
            current_price *= (1 + change)
            prices.append(current_price)
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.02))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 10000, days),
            'price': prices
        }, index=dates)
        
        print(f"[INFO] Generated {len(df)} demo data points")
        return df
    
    def save_data(self, df, filename="bitcoin_data.csv"):
        try:
            filepath = self.data_dir / filename
            df.to_csv(filepath)
            print(f"[OK] Saved data to {filepath}")
        except Exception as e:
            print(f"[ERROR] Error saving: {str(e)[:100]}")
    
    def load_data(self, filename="bitcoin_data.csv"):
        try:
            filepath = self.data_dir / filename
            if filepath.exists():
                df = pd.read_csv(filepath, index_col=0, parse_dates=True)
                print(f"[OK] Loaded data from {filepath}")
                return df
            return None
        except Exception as e:
            print(f"[ERROR] Error loading: {str(e)[:100]}")
            return None

# Test
if __name__ == "__main__":
    collector = BitcoinDataCollector()
    df = collector.collect_all_data(days=100)
    if df is not None:
        print(f"\n[SUCCESS] Final data shape: {df.shape}")
        print("\nLatest 3 records:")
        print(df.tail(3))
        print(f"\nCurrent price: ${df['close'].iloc[-1]:,.2f}")
