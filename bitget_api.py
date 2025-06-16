import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import time

class BitgetAPI:
    def __init__(self):
        self.base_url = "https://api.bitget.com"
        
    def get_kline_data(self, symbol="BTCUSDT", period="1H", limit=100):
        """
        Lấy dữ liệu K-line từ Bitget
        
        Args:
            symbol: Cặp trading (BTCUSDT, ETHUSDT, etc.)
            period: Khung thời gian (1m, 5m, 15m, 30m, 1H, 4H, 6H, 12H, 1D, 1W)
            limit: Số lượng nến (max 1000)
        """
        endpoint = "/api/spot/v1/market/candles"
        
        # Tính thời gian
        end_time = int(time.time() * 1000)
        
        params = {
            'symbol': symbol,
            'period': period,
            'limit': limit,
            'endTime': end_time
        }
        
        try:
            response = requests.get(self.base_url + endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data['code'] == '00000':  # Success code
                return self._process_kline_data(data['data'])
            else:
                print(f"API Error: {data['msg']}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Request Error: {e}")
            return None
    
    def _process_kline_data(self, raw_data):
        """Xử lý dữ liệu K-line thành DataFrame"""
        if not raw_data:
            return None
            
        df = pd.DataFrame(raw_data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'
        ])
        
        # Chuyển đổi kiểu dữ liệu
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        # Sắp xếp theo thời gian
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def get_ticker_24hr(self, symbol="BTCUSDT"):
        """Lấy thông tin ticker 24h"""
        endpoint = "/api/spot/v1/market/ticker"
        
        params = {'symbol': symbol}
        
        try:
            response = requests.get(self.base_url + endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data['code'] == '00000':
                return data['data']
            else:
                print(f"API Error: {data['msg']}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Request Error: {e}")
            return None
    
    def get_orderbook(self, symbol="BTCUSDT", limit=100):
        """Lấy sổ lệnh"""
        endpoint = "/api/spot/v1/market/depth"
        
        params = {
            'symbol': symbol,
            'limit': limit,
            'type': 'step0'
        }
        
        try:
            response = requests.get(self.base_url + endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data['code'] == '00000':
                return data['data']
            else:
                print(f"API Error: {data['msg']}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Request Error: {e}")
            return None

# Test function
def test_bitget_api():
    """Test API connection"""
    api = BitgetAPI()
    
    print("🔄 Testing Bitget API...")
    
    # Test ticker
    ticker = api.get_ticker_24hr("BTCUSDT")
    if ticker:
        print(f"✅ Ticker: BTC Price = ${float(ticker['close']):,.2f}")
    
    # Test kline data
    kline_data = api.get_kline_data("BTCUSDT", "1H", 24)
    if kline_data is not None:
        print(f"✅ Kline Data: {len(kline_data)} records")
        print(f"   Latest Close: ${kline_data['close'].iloc[-1]:,.2f}")
    
    return kline_data

if __name__ == "__main__":
    test_bitget_api()
