import requests
import json

def test_kline_params():
    symbol = "BTCUSDT_SPBL"
    base_url = "https://api.bitget.com/api/spot/v1/market/candles"
    
    # Test cac period khac nhau
    periods = ['1m', '5m', '15m', '30m', '1h', '4h', '6h', '12h', '1d', '3d', '1w', '1M']
    
    print(f"Testing Kline API for {symbol}...")
    
    for period in periods:
        try:
            params = {
                'symbol': symbol,
                'period': period,
                'limit': 5
            }
            
            response = requests.get(base_url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000':
                    klines = data.get('data', [])
                    print(f"[OK] {period}: Success ({len(klines)} records)")
                    if klines:
                        latest = klines[0]
                        print(f"     Latest close: {latest[4]}")
                    return period  # Return first working period
                else:
                    print(f"[ERROR] {period}: API Error - {data.get('msg')}")
            else:
                print(f"[ERROR] {period}: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"[ERROR] {period}: Exception - {str(e)[:100]}")
    
    return None

def test_alternative_endpoints():
    """Test endpoint khac cho historical data"""
    symbol = "BTCUSDT_SPBL"
    
    # Try different endpoints
    endpoints = [
        "/api/spot/v1/market/candles",
        "/api/spot/v1/market/history-candles", 
        "/api/v2/spot/market/candles"
    ]
    
    for endpoint in endpoints:
        try:
            url = f"https://api.bitget.com{endpoint}"
            params = {
                'symbol': symbol,
                'period': '1d',
                'limit': 5
            }
            
            response = requests.get(url, params=params, timeout=10)
            print(f"\nTesting {endpoint}")
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Code: {data.get('code')}")
                
                if data.get('code') == '00000' and data.get('data'):
                    print(f"[SUCCESS] {endpoint} works!")
                    klines = data['data']
                    print(f"Got {len(klines)} records")
                    if klines:
                        print(f"Sample data: {klines[0]}")
                    return endpoint, klines
                else:
                    print(f"[ERROR] {data.get('msg')}")
            else:
                print(f"[ERROR] HTTP error")
                
        except Exception as e:
            print(f"[ERROR] Exception: {str(e)[:100]}")
    
    return None, None

def test_with_different_params():
    """Test voi cac tham so khac nhau"""
    symbol = "BTCUSDT_SPBL"
    url = "https://api.bitget.com/api/spot/v1/market/candles"
    
    # Test cac combination khac nhau
    test_cases = [
        {'period': '1d', 'limit': 5},
        {'period': '1D', 'limit': 5},
        {'period': '24h', 'limit': 5},
        {'period': 'day', 'limit': 5},
        {'period': '1d', 'limit': 10},
        {'period': '1d', 'limit': 100},
    ]
    
    print(f"\nTesting different parameters for {symbol}...")
    
    for i, params in enumerate(test_cases):
        try:
            test_params = {'symbol': symbol, **params}
            response = requests.get(url, params=test_params, timeout=10)
            
            print(f"\nTest {i+1}: {params}")
            print(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"API Code: {data.get('code')}")
                
                if data.get('code') == '00000':
                    klines = data.get('data', [])
                    print(f"[SUCCESS] Got {len(klines)} records")
                    if klines:
                        print(f"Sample: {klines[0]}")
                        return params  # Return working params
                else:
                    print(f"[ERROR] {data.get('msg')}")
            else:
                print(f"[ERROR] HTTP {response.status_code}")
                print(f"Response: {response.text[:200]}")
                
        except Exception as e:
            print(f"[ERROR] Exception: {str(e)[:100]}")
    
    return None

if __name__ == "__main__":
    print("=== Testing Bitget Kline API ===")
    
    # Test 1: Different periods
    print("\n1. Testing different periods...")
    working_period = test_kline_params()
    
    # Test 2: Different endpoints
    print("\n2. Testing different endpoints...")
    working_endpoint, sample_data = test_alternative_endpoints()
    
    # Test 3: Different parameters
    print("\n3. Testing different parameters...")
    working_params = test_with_different_params()
    
    # Summary
    print("\n=== SUMMARY ===")
    if working_period:
        print(f"Working period: {working_period}")
    if working_endpoint:
        print(f"Working endpoint: {working_endpoint}")
    if working_params:
        print(f"Working params: {working_params}")
    
    if not any([working_period, working_endpoint, working_params]):
        print("No working configuration found!")
        print("Will use Binance as fallback.")
