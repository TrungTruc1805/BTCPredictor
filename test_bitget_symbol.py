import requests
import json

def check_bitget_symbols():
    print("Checking available symbols on Bitget...")
    
    # Lấy danh sách symbols
    try:
        url = "https://api.bitget.com/api/spot/v1/public/products"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == '00000' and data.get('data'):
                symbols = data['data']
                
                # Tìm Bitcoin symbols
                btc_symbols = [s for s in symbols if 'BTC' in s.get('symbol', '')]
                
                print(f"Found {len(btc_symbols)} BTC symbols:")
                for symbol in btc_symbols[:10]:  # Show first 10
                    print(f"  - {symbol.get('symbol')} ({symbol.get('baseCoin')}/{symbol.get('quoteCoin')})")
                
                # Tìm symbol chính xác
                btc_usdt_symbols = [s for s in symbols if 'BTC' in s.get('symbol', '') and 'USDT' in s.get('symbol', '')]
                if btc_usdt_symbols:
                    correct_symbol = btc_usdt_symbols[0]['symbol']
                    print(f"\nCorrect BTC/USDT symbol: {correct_symbol}")
                    return correct_symbol
                else:
                    print("\nNo BTC/USDT symbol found")
                    return None
            else:
                print(f"API Error: {data}")
                return None
        else:
            print(f"HTTP Error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_correct_symbol(symbol):
    print(f"\nTesting with correct symbol: {symbol}")
    
    # Test ticker
    try:
        url = "https://api.bitget.com/api/spot/v1/market/ticker"
        params = {'symbol': symbol}
        response = requests.get(url, params=params, timeout=10)
        
        print(f"Ticker status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == '00000':
                ticker_data = data['data']
                print(f"[OK] Price: {ticker_data.get('close')}")
                print(f"[OK] Volume: {ticker_data.get('baseVol')}")
            else:
                print(f"[ERROR] {data}")
        else:
            print(f"[ERROR] HTTP {response.status_code}")
    except Exception as e:
        print(f"[ERROR] {e}")
    
    # Test kline
    try:
        url = "https://api.bitget.com/api/spot/v1/market/candles"
        params = {
            'symbol': symbol,
            'period': '1D',
            'limit': 5
        }
        response = requests.get(url, params=params, timeout=10)
        
        print(f"Kline status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            if data.get('code') == '00000':
                klines = data['data']
                print(f"[OK] Got {len(klines)} kline records")
                if klines:
                    latest = klines[0]
                    print(f"[OK] Latest close: {latest[4]}")
            else:
                print(f"[ERROR] {data}")
        else:
            print(f"[ERROR] HTTP {response.status_code}")
    except Exception as e:
        print(f"[ERROR] {e}")

if __name__ == "__main__":
    symbol = check_bitget_symbols()
    if symbol:
        test_correct_symbol(symbol)
