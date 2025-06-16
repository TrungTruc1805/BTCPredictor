import requests
import json

def test_bitget_api():
    print("Testing Bitget API...")
    
    # Test 1: Server time
    try:
        url = "https://api.bitget.com/api/spot/v1/public/time"
        response = requests.get(url, timeout=10)
        print(f"Server time status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"[OK] Server time: {data}")
        else:
            print(f"[ERROR] Server time failed: {response.text}")
    except Exception as e:
        print(f"[ERROR] Server time error: {e}")
    
    # Test 2: Ticker
    try:
        url = "https://api.bitget.com/api/spot/v1/market/ticker"
        params = {'symbol': 'BTCUSDT'}
        response = requests.get(url, params=params, timeout=10)
        print(f"\nTicker status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"[OK] Ticker response: {json.dumps(data, indent=2)}")
        else:
            print(f"[ERROR] Ticker failed: {response.text}")
    except Exception as e:
        print(f"[ERROR] Ticker error: {e}")
    
    # Test 3: Kline data
    try:
        url = "https://api.bitget.com/api/spot/v1/market/candles"
        params = {
            'symbol': 'BTCUSDT',
            'period': '1D',
            'limit': 10
        }
        response = requests.get(url, params=params, timeout=10)
        print(f"\nKline status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"[OK] Kline response: {json.dumps(data, indent=2)}")
        else:
            print(f"[ERROR] Kline failed: {response.text}")
    except Exception as e:
        print(f"[ERROR] Kline error: {e}")

if __name__ == "__main__":
    test_bitget_api()
