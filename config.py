import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Đường dẫn thư mục dữ liệu
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# API URLs
COINGECKO_API = "https://api.coingecko.com/api/v3"
BITGET_API = "https://api.bitget.com"

# API Keys (để trống nếu không có)
COINGECKO_API_KEY = ""
BINANCE_API_KEY = ""
BITGET_API_KEY = "bg_96c120342e858174ad85017582814cd1"


# Đường dẫn
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Tạo thư mục nếu chưa có
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


# Model parameters
LOOKBACK_DAYS = 60
PREDICTION_DAYS = 7
TRAIN_TEST_SPLIT = 0.8

# Features để sử dụng
FEATURE_COLUMNS = [
    'open', 'high', 'low', 'volume',
    'MA_7', 'MA_30', 'MA_90',
    'RSI', 'MACD', 'MACD_signal',
    'BB_upper', 'BB_lower', 'BB_middle',
    'volume_sma', 'volatility',
    'price_change_1d', 'price_change_7d',
    'day_of_week', 'month', 'quarter'
]

# Model settings
MODEL_CONFIGS = {
    'lstm': {
        'units': [50, 50, 50],
        'dropout': 0.2,
        'epochs': 100,
        'batch_size': 32
    },
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'random_state': 42
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1
    }
}
