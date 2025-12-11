"""
===========================================================================
🏦 SYNTHETIC DATA GENERATOR CHO LIGHTGBM CHỐNG GIAN LẬN & LỪA ĐẢO - VIỆT NAM
===========================================================================
Ứng dụng Streamlit tạo dữ liệu giả lập chuẩn hành vi người Việt Nam
để train mô hình LightGBM phát hiện:
- GIAN LẬN (Fraud): Account Takeover, Mule Account, Card Testing
- LỪA ĐẢO (Scam): Romance Scam, Investment Scam, Impersonation (giả công an/ngân hàng)

Author: Data Engineering Team - Vietnam Banking
Version: 2.0.0 - Optimized for 500K+ transactions
===========================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from collections import defaultdict
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ===========================================================================
# CẤU HÌNH MẶC ĐỊNH
# ===========================================================================
DEFAULT_N_TRANSACTIONS = 100_000
DEFAULT_FRAUD_RATE = 0.05
DEFAULT_N_USERS = 5000
DEFAULT_N_RECIPIENTS = 8000
RANDOM_SEED = 42

# ===========================================================================
# CONSTANTS - HÀNH VI NGƯỜI VIỆT NAM
# ===========================================================================

# Các mức tiền phổ biến tại Việt Nam (đơn vị: VND)
COMMON_AMOUNTS_VN = [
    50_000, 100_000, 150_000, 200_000, 300_000, 500_000,
    1_000_000, 2_000_000, 3_000_000, 5_000_000,
    10_000_000, 20_000_000, 50_000_000
]

# Các loại giao dịch phổ biến tại VN
TRANSACTION_TYPES = {
    'chuyen_noi_bo': 0.30,        # Chuyển khoản nội bộ ngân hàng
    'chuyen_lien_ngan_hang': 0.25, # Chuyển liên ngân hàng (Napas)
    'thanh_toan_hoa_don': 0.15,    # Điện/nước/internet
    'topup_vi': 0.12,              # Nạp ví Momo/ZaloPay/VNPay
    'rut_atm': 0.08,               # Rút ATM
    'thanh_toan_pos': 0.05,        # Quẹt thẻ POS
    'hoc_phi_vien_phi': 0.03,      # Học phí, viện phí
    'mua_hang_online': 0.02        # Mua hàng online
}

# Kênh giao dịch
CHANNELS = {
    'mobile_app': 0.55,  # Đa số người VN dùng app mobile
    'web': 0.20,
    'atm': 0.15,
    'pos': 0.10
}

# Channel risk theo quy tắc ngân hàng VN
CHANNEL_RISK_BASE = {
    'mobile_app': 0.35,
    'web': 0.25,
    'atm': 0.10,
    'pos': 0.05
}

# Transaction type risk
TX_TYPE_RISK_BASE = {
    'chuyen_lien_ngan_hang': 0.40,
    'chuyen_noi_bo': 0.20,
    'topup_vi': 0.15,
    'mua_hang_online': 0.12,
    'thanh_toan_hoa_don': 0.05,
    'rut_atm': 0.08,
    'thanh_toan_pos': 0.06,
    'hoc_phi_vien_phi': 0.04
}

# 3 vùng địa lý Việt Nam
GEO_REGIONS = {
    'bac': {'center': (21.0285, 105.8542), 'weight': 0.35},  # Hà Nội
    'trung': {'center': (16.0544, 108.2022), 'weight': 0.15},  # Đà Nẵng
    'nam': {'center': (10.8231, 106.6297), 'weight': 0.50}   # HCM
}


# ===========================================================================
# SECTION 1: GENERATE BASE TRANSACTIONS
# ===========================================================================

def generate_user_profiles(n_users, seed=RANDOM_SEED):
    """
    Tạo profile người dùng với phân phối Pareto (10% user chiếm 60% giao dịch)
    """
    np.random.seed(seed)

    users = []
    for i in range(n_users):
        user_id = f"USR_{i:06d}"

        # Phân bố vùng miền
        region = np.random.choice(
            list(GEO_REGIONS.keys()),
            p=[GEO_REGIONS[r]['weight'] for r in GEO_REGIONS.keys()]
        )

        # Thiết bị chính của user (người VN ít đổi thiết bị)
        primary_device = f"DEV_{np.random.randint(100000, 999999)}"

        # Số ngày đã mở tài khoản (30 - 3650 ngày)
        account_age = np.random.exponential(scale=500) + 30
        account_age = min(account_age, 3650)

        # Mức thu nhập ước tính (ảnh hưởng đến amount trung bình)
        # Phân phối log-normal cho thu nhập VN
        income_level = np.random.lognormal(mean=2.5, sigma=0.8)
        income_level = np.clip(income_level, 0.5, 20)

        # Giờ giao dịch quen thuộc của user (mean hour)
        preferred_hour = np.random.choice([9, 10, 14, 15, 20, 21], p=[0.15, 0.20, 0.15, 0.15, 0.20, 0.15])
        preferred_hour += np.random.uniform(-1, 1)

        # Activity level (Pareto: 10% power users)
        if np.random.random() < 0.10:
            activity_weight = np.random.uniform(5, 15)  # Power users
        elif np.random.random() < 0.30:
            activity_weight = np.random.uniform(1, 5)   # Regular users
        else:
            activity_weight = np.random.uniform(0.1, 1) # Low activity users

        users.append({
            'user_id': user_id,
            'region': region,
            'primary_device': primary_device,
            'account_age_days': int(account_age),
            'income_level': income_level,
            'preferred_hour': preferred_hour,
            'activity_weight': activity_weight
        })

    return pd.DataFrame(users)


def generate_recipient_profiles(n_recipients, seed=RANDOM_SEED):
    """
    Tạo profile người nhận tiền
    95% người nhận chỉ nhận từ 1-3 người quen
    """
    np.random.seed(seed)

    recipients = []
    for i in range(n_recipients):
        recipient_id = f"RCP_{i:06d}"

        # Số người gửi tối đa cho recipient này (95% chỉ 1-3 người)
        if np.random.random() < 0.95:
            max_senders = np.random.randint(1, 4)
        else:
            max_senders = np.random.randint(4, 50)  # Có thể là mule

        # Đánh dấu tiềm năng mule (nhận từ nhiều người)
        is_potential_mule = max_senders > 20

        recipients.append({
            'recipient_id': recipient_id,
            'max_senders': max_senders,
            'is_potential_mule': is_potential_mule
        })

    return pd.DataFrame(recipients)


def generate_amount_vn(income_level, tx_type, is_salary_period=False):
    """
    Sinh số tiền giao dịch theo hành vi người Việt Nam
    - Đa số là số chẵn nghìn
    - Phân phối theo loại giao dịch và thu nhập
    """
    # Base amount theo loại giao dịch
    if tx_type == 'thanh_toan_hoa_don':
        # Hóa đơn điện/nước/internet: 100k - 2 triệu
        base = np.random.choice([100_000, 200_000, 300_000, 500_000, 800_000, 1_000_000, 1_500_000])
    elif tx_type == 'topup_vi':
        # Nạp ví: 50k - 2 triệu
        base = np.random.choice([50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000])
    elif tx_type == 'rut_atm':
        # Rút ATM: thường chẵn 500k, 1tr, 2tr
        base = np.random.choice([500_000, 1_000_000, 2_000_000, 3_000_000, 5_000_000])
    elif tx_type == 'hoc_phi_vien_phi':
        # Học phí/viện phí: lớn hơn
        base = np.random.choice([500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000])
    elif tx_type in ['chuyen_noi_bo', 'chuyen_lien_ngan_hang']:
        # Chuyển khoản: đa dạng
        base = np.random.choice(COMMON_AMOUNTS_VN, p=[
            0.08, 0.12, 0.08, 0.15, 0.10, 0.17,  # 50k-500k (tổng: 0.70)
            0.12, 0.08, 0.04, 0.03,               # 1tr-5tr (tổng: 0.27)
            0.02, 0.007, 0.003                    # 10tr-50tr (tổng: 0.03)
        ])
    else:
        # Mua hàng online, POS
        base = np.random.choice([50_000, 100_000, 200_000, 300_000, 500_000, 1_000_000])

    # Điều chỉnh theo income level
    amount = base * (0.5 + income_level * 0.3)

    # Tăng amount trong kỳ lương
    if is_salary_period and np.random.random() < 0.3:
        amount *= np.random.uniform(1.5, 3.0)

    # Làm tròn nghìn đồng (yêu cầu bắt buộc)
    amount = int(round(amount / 1000) * 1000)

    # Giới hạn hợp lý
    amount = max(10_000, min(amount, 500_000_000))

    return amount


def generate_transaction_hour_vn(is_fraud=False, preferred_hour=None):
    """
    Sinh giờ giao dịch theo hành vi người Việt Nam
    - Cao điểm: 8-11h, 14-16h, 19:30-21:30
    - Gần như tắt sau 23h (trừ fraud)
    """
    if is_fraud and np.random.random() < 0.4:
        # Fraud thường xảy ra ban đêm 1-4 AM
        return np.random.randint(1, 5) + np.random.random()

    # Phân phối giờ bình thường của người Việt
    hour_weights = np.array([
        0.005, 0.002, 0.001, 0.001, 0.002, 0.005,  # 0-5h: rất ít
        0.02, 0.04, 0.08, 0.10, 0.10, 0.08,        # 6-11h: tăng dần, cao điểm sáng
        0.05, 0.04, 0.08, 0.09, 0.07, 0.05,        # 12-17h: cao điểm chiều
        0.04, 0.06, 0.09, 0.08, 0.04, 0.01         # 18-23h: cao điểm tối, giảm dần
    ])
    hour_weights = hour_weights / hour_weights.sum()

    hour = np.random.choice(24, p=hour_weights)
    minute = np.random.randint(0, 60)

    # Điều chỉnh theo preferred hour của user
    if preferred_hour is not None and np.random.random() < 0.3:
        hour = int(preferred_hour) % 24

    return hour + minute / 60


def is_salary_period(date):
    """
    Kiểm tra có phải kỳ lương không (ngày 25 - ngày 5 tháng sau)
    """
    day = date.day
    return day >= 25 or day <= 5


def is_bill_period(date):
    """
    Kiểm tra có phải kỳ thanh toán hóa đơn không (đầu tháng 1-10)
    """
    return date.day <= 10


def generate_base_transactions(n_transactions, n_users, n_recipients, seed=RANDOM_SEED):
    """
    Tạo dữ liệu giao dịch cơ bản theo hành vi người Việt Nam
    """
    np.random.seed(seed)
    random.seed(seed)

    # Tạo profiles
    user_profiles = generate_user_profiles(n_users, seed)
    recipient_profiles = generate_recipient_profiles(n_recipients, seed)

    # Sampling users theo activity weight (Pareto distribution)
    user_weights = user_profiles['activity_weight'].values
    user_weights = user_weights / user_weights.sum()

    transactions = []

    # Tạo mapping user -> recipients quen thuộc (95% chỉ chuyển cho 1-3 người)
    user_familiar_recipients = {}
    for user_id in user_profiles['user_id']:
        n_familiar = np.random.choice([1, 2, 3, 4, 5], p=[0.3, 0.35, 0.25, 0.07, 0.03])
        familiar_rcps = np.random.choice(recipient_profiles['recipient_id'].values, size=n_familiar, replace=False)
        user_familiar_recipients[user_id] = list(familiar_rcps)

    # Base timestamp (1 năm gần đây)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    for i in range(n_transactions):
        # Chọn user theo weight
        user_idx = np.random.choice(len(user_profiles), p=user_weights)
        user = user_profiles.iloc[user_idx]
        user_id = user['user_id']

        # Sinh timestamp
        random_days = np.random.uniform(0, 365)
        tx_date = start_date + timedelta(days=random_days)

        # Kiểm tra kỳ lương và kỳ hóa đơn
        is_salary = is_salary_period(tx_date)
        is_bill = is_bill_period(tx_date)

        # Sinh giờ giao dịch
        hour_decimal = generate_transaction_hour_vn(is_fraud=False, preferred_hour=user['preferred_hour'])
        hour = int(hour_decimal)
        minute = int((hour_decimal - hour) * 60)
        tx_datetime = tx_date.replace(hour=hour, minute=minute, second=np.random.randint(0, 60))

        # Chọn loại giao dịch
        tx_types = list(TRANSACTION_TYPES.keys())
        tx_probs = list(TRANSACTION_TYPES.values())

        # Tăng thanh toán hóa đơn trong kỳ bill
        if is_bill:
            tx_probs[tx_types.index('thanh_toan_hoa_don')] *= 2
            tx_probs = [p / sum(tx_probs) for p in tx_probs]

        tx_type = np.random.choice(tx_types, p=tx_probs)

        # Chọn kênh giao dịch
        channel = np.random.choice(list(CHANNELS.keys()), p=list(CHANNELS.values()))

        # ATM chỉ cho rút tiền
        if tx_type == 'rut_atm':
            channel = 'atm'
        elif channel == 'atm' and tx_type != 'rut_atm':
            channel = 'mobile_app'

        # Sinh số tiền
        amount = generate_amount_vn(user['income_level'], tx_type, is_salary)

        # Chọn recipient
        # 90% chuyển cho người quen, 10% người mới
        if np.random.random() < 0.90 and user_familiar_recipients[user_id]:
            recipient_id = np.random.choice(user_familiar_recipients[user_id])
            is_new_recipient = 0
        else:
            recipient_id = np.random.choice(recipient_profiles['recipient_id'].values)
            is_new_recipient = 1

        # Thiết bị: 95% dùng thiết bị chính
        if np.random.random() < 0.95:
            device_id = user['primary_device']
            is_new_device = 0
        else:
            device_id = f"DEV_{np.random.randint(100000, 999999)}"
            is_new_device = 1

        # Vị trí giao dịch (tính location_diff_km)
        user_region = user['region']
        user_center = GEO_REGIONS[user_region]['center']

        # 85% giao dịch tại vùng của mình
        if np.random.random() < 0.85:
            tx_lat = user_center[0] + np.random.normal(0, 0.1)
            tx_lon = user_center[1] + np.random.normal(0, 0.1)
        else:
            # Giao dịch ở vùng khác
            other_region = np.random.choice([r for r in GEO_REGIONS.keys() if r != user_region])
            other_center = GEO_REGIONS[other_region]['center']
            tx_lat = other_center[0] + np.random.normal(0, 0.1)
            tx_lon = other_center[1] + np.random.normal(0, 0.1)

        # Tính khoảng cách (đơn giản hóa: 1 độ ~ 111km)
        location_diff_km = np.sqrt(
            ((tx_lat - user_center[0]) * 111) ** 2 +
            ((tx_lon - user_center[1]) * 111 * np.cos(np.radians(user_center[0]))) ** 2
        )

        transactions.append({
            'transaction_id': f"TX_{i:08d}",
            'user_id': user_id,
            'recipient_id': recipient_id,
            'timestamp': tx_datetime,
            'amount': amount,
            'transaction_type': tx_type,
            'channel': channel,
            'device_id': device_id,
            'is_new_recipient': is_new_recipient,
            'is_new_device': is_new_device,
            'location_diff_km': round(location_diff_km, 2),
            'account_age_days': user['account_age_days'],
            'user_region': user_region,
            'user_preferred_hour': user['preferred_hour'],
            'is_fraud': 0  # Mặc định không phải fraud
        })

    df = pd.DataFrame(transactions)
    df = df.sort_values('timestamp').reset_index(drop=True)

    return df, user_profiles, recipient_profiles


# ===========================================================================
# SECTION 2: DERIVED FEATURES (TỐI ƯU CHO 200K+ DÒNG)
# ===========================================================================

def compute_derived_features_optimized(df, progress_callback=None):
    """
    Tính toán các feature phái sinh từ dữ liệu giao dịch
    PHIÊN BẢN TỐI ƯU: Sử dụng vectorization và numpy để xử lý nhanh hơn
    Tất cả đều là past-only (không nhìn vào tương lai)
    """
    df = df.copy()
    df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)

    if progress_callback:
        progress_callback("Đang tính amount features...")

    # 1. amount_log: Log của số tiền (vectorized)
    df['amount_log'] = np.log1p(df['amount'])

    # 2. amount_tier: Phân loại mức tiền (vectorized với np.select)
    conditions = [
        df['amount'] < 100_000,
        df['amount'] < 500_000,
        df['amount'] < 2_000_000,
        df['amount'] < 10_000_000,
    ]
    choices = ['micro', 'small', 'medium', 'large']
    df['amount_tier'] = np.select(conditions, choices, default='very_large')

    # 3. Time features (vectorized)
    df['hour_of_day'] = df['timestamp'].dt.hour + df['timestamp'].dt.minute / 60
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_night_hours'] = ((df['hour_of_day'] >= 23) | (df['hour_of_day'] < 6)).astype(int)

    # Vectorized salary/bill period
    days = df['timestamp'].dt.day
    df['is_salary_period'] = ((days >= 25) | (days <= 5)).astype(int)
    df['is_bill_period'] = (days <= 10).astype(int)

    if progress_callback:
        progress_callback("Đang tính amount_vs_avg_user...")

    # 4. amount_vs_avg_user: So sánh với trung bình user (past-only, vectorized)
    df['user_cumsum'] = df.groupby('user_id')['amount'].cumsum() - df['amount']
    df['user_cumcount'] = df.groupby('user_id').cumcount()
    df['user_avg_past'] = np.where(
        df['user_cumcount'] > 0,
        df['user_cumsum'] / df['user_cumcount'],
        df['amount']
    )
    df['amount_vs_avg_user'] = np.where(
        df['user_avg_past'] > 0,
        df['amount'] / df['user_avg_past'],
        1.0
    )
    df['amount_vs_avg_user'] = df['amount_vs_avg_user'].clip(0, 100)
    df.drop(['user_cumsum', 'user_cumcount', 'user_avg_past'], axis=1, inplace=True)

    # 5. time_gap_prev_min: Khoảng cách với giao dịch trước (phút) - vectorized
    df['prev_timestamp'] = df.groupby('user_id')['timestamp'].shift(1)
    df['time_gap_prev_min'] = (df['timestamp'] - df['prev_timestamp']).dt.total_seconds() / 60
    df['time_gap_prev_min'] = df['time_gap_prev_min'].fillna(999999).clip(0, 999999)
    df.drop('prev_timestamp', axis=1, inplace=True)

    if progress_callback:
        progress_callback("Đang tính velocity features (có thể mất vài phút)...")

    # 6-10. Các features cần tính theo user - TỐI ƯU với numba-style approach
    # Khởi tạo các cột
    df['velocity_1h'] = 0
    df['velocity_24h'] = 0
    df['recipient_count_30d'] = 0
    df['device_count_30d'] = 0
    df['is_first_large_tx'] = 0
    df['recipient_diversity'] = 0.0

    # Chuyển timestamp sang số để tính toán nhanh hơn
    df['ts_numeric'] = df['timestamp'].astype(np.int64) // 10**9  # Unix timestamp

    large_threshold = 5_000_000  # 5 triệu VND

    # Xử lý theo batch user để tối ưu
    user_groups = df.groupby('user_id')
    n_users = len(user_groups)

    for user_idx, (user_id, group) in enumerate(user_groups):
        if progress_callback and user_idx % 500 == 0:
            progress_callback(f"Đang xử lý user {user_idx}/{n_users}...")

        indices = group.index.values
        ts_values = group['ts_numeric'].values
        recipients = group['recipient_id'].values
        devices = group['device_id'].values
        amounts = group['amount'].values

        # Tính toán vectorized trong group
        seen_recipients = set()
        seen_devices_30d = {}
        seen_recipients_30d = {}
        had_large = False

        for i in range(len(indices)):
            idx = indices[i]
            current_ts = ts_values[i]

            # Velocity: đếm giao dịch trong window
            if i > 0:
                time_diffs = (current_ts - ts_values[:i]) / 3600  # Chuyển sang giờ
                df.loc[idx, 'velocity_1h'] = np.sum(time_diffs <= 1)
                df.loc[idx, 'velocity_24h'] = np.sum(time_diffs <= 24)

                # recipient_count_30d và device_count_30d
                time_diffs_days = time_diffs / 24
                mask_30d = time_diffs_days <= 30
                df.loc[idx, 'recipient_count_30d'] = len(set(recipients[:i][mask_30d]))
                df.loc[idx, 'device_count_30d'] = len(set(devices[:i][mask_30d]))

            # recipient_diversity
            if i > 0:
                df.loc[idx, 'recipient_diversity'] = len(seen_recipients) / i
            seen_recipients.add(recipients[i])

            # is_first_large_tx
            if amounts[i] >= large_threshold and not had_large:
                df.loc[idx, 'is_first_large_tx'] = 1
                had_large = True

    df.drop('ts_numeric', axis=1, inplace=True)

    return df


# ===========================================================================
# SECTION 3: FRAUD & SCAM SCENARIOS (GIAN LẬN + LỪA ĐẢO VIỆT NAM)
# ===========================================================================

def apply_fraud_scenarios(df, fraud_rate=DEFAULT_FRAUD_RATE, seed=RANDOM_SEED):
    """
    Áp dụng các kịch bản GIAN LẬN và LỪA ĐẢO theo hành vi Việt Nam

    GIAN LẬN (Fraud) - Kẻ gian chiếm đoạt tài khoản:
    1. Account Takeover - Bị hack tài khoản
    2. Mule Account - Tài khoản trung gian rửa tiền
    3. Card Testing - Test thẻ bị đánh cắp

    LỪA ĐẢO (Scam) - Nạn nhân tự nguyện chuyển tiền:
    4. Romance Scam - Lừa tình cảm
    5. Investment Scam - Lừa đầu tư/tiền ảo
    6. Impersonation - Giả mạo công an/ngân hàng
    7. Job Scam - Lừa việc làm online

    Fraud chỉ được sinh theo scenario - không dựa vào phân bố nhãn
    """
    np.random.seed(seed)
    df = df.copy()

    n_fraud_target = int(len(df) * fraud_rate)

    # Chia tỷ lệ cho các scenario (GIAN LẬN + LỪA ĐẢO)
    scenario_ratios = {
        # === GIAN LẬN (Fraud) ===
        'account_takeover': 0.20,      # Bị hack
        'mule_account': 0.15,          # Tài khoản trung gian
        'card_testing': 0.10,          # Test thẻ
        # === LỪA ĐẢO (Scam) ===
        'romance_scam': 0.15,          # Lừa tình cảm
        'investment_scam': 0.15,       # Lừa đầu tư
        'impersonation_scam': 0.15,    # Giả công an/ngân hàng
        'job_scam': 0.10               # Lừa việc làm
    }

    fraud_indices = []

    # =========================================
    # SCENARIO 1: Account Takeover (GIAN LẬN - bị hack)
    # Đặc điểm:
    # - Đổi thiết bị đột ngột
    # - Giao dịch lúc 1-4 AM
    # - Chuyển lớn đến người lạ
    # - Vị trí khác thường
    # =========================================
    n_ato = int(n_fraud_target * scenario_ratios['account_takeover'])

    ato_candidates = df[
        (df['is_new_device'] == 1) &
        (df['is_new_recipient'] == 1) &
        (df['amount'] >= 2_000_000)
    ].index.tolist()

    if ato_candidates:
        n_select = min(n_ato, len(ato_candidates))
        selected = np.random.choice(ato_candidates, size=n_select, replace=False)

        for idx in selected:
            df.loc[idx, 'hour_of_day'] = np.random.uniform(1, 4)
            df.loc[idx, 'is_night_hours'] = 1
            df.loc[idx, 'location_diff_km'] = np.random.uniform(100, 500)

            if np.random.random() < np.random.uniform(0.7, 0.9):
                df.loc[idx, 'is_fraud'] = 1
                fraud_indices.append(idx)

    # =========================================
    # SCENARIO 2: Mule Account (GIAN LẬN - tài khoản trung gian)
    # Đặc điểm:
    # - Nhiều user nhỏ chuyển nhiều khoản nhỏ
    # - Recipient nhận từ > 20 người
    # - Velocity cao bất thường
    # =========================================
    n_mule = int(n_fraud_target * scenario_ratios['mule_account'])

    recipient_sender_count = df.groupby('recipient_id')['user_id'].nunique()
    suspicious_recipients = recipient_sender_count[recipient_sender_count > 15].index.tolist()

    mule_candidates = df[
        (df['recipient_id'].isin(suspicious_recipients)) &
        (df['amount'] < 1_000_000) &
        (~df.index.isin(fraud_indices))
    ].index.tolist()

    if mule_candidates:
        n_select = min(n_mule, len(mule_candidates))
        selected = np.random.choice(mule_candidates, size=n_select, replace=False)

        for idx in selected:
            df.loc[idx, 'velocity_1h'] = np.random.randint(5, 15)
            df.loc[idx, 'velocity_24h'] = np.random.randint(20, 50)

            if np.random.random() < np.random.uniform(0.6, 0.9):
                df.loc[idx, 'is_fraud'] = 1
                fraud_indices.append(idx)

    # =========================================
    # SCENARIO 3: Card Testing (GIAN LẬN - test thẻ bị đánh cắp)
    # Đặc điểm:
    # - Nhiều giao dịch nhỏ (10k-50k)
    # - Nhiều recipient trong thời gian ngắn
    # - Test xem thẻ còn hoạt động không
    # =========================================
    n_card = int(n_fraud_target * scenario_ratios['card_testing'])

    card_candidates = df[
        (df['amount'] <= 50_000) &
        (df['velocity_1h'] >= 3) &
        (~df.index.isin(fraud_indices))
    ].index.tolist()

    if card_candidates:
        n_select = min(n_card, len(card_candidates))
        selected = np.random.choice(card_candidates, size=n_select, replace=False)

        for idx in selected:
            df.loc[idx, 'recipient_count_30d'] = np.random.randint(10, 30)

            if np.random.random() < np.random.uniform(0.5, 0.8):
                df.loc[idx, 'is_fraud'] = 1
                fraud_indices.append(idx)

    # =========================================
    # SCENARIO 4: Romance Scam (LỪA ĐẢO - lừa tình cảm)
    # Đặc điểm tại Việt Nam:
    # - Nạn nhân thường là phụ nữ trung niên, đàn ông độc thân
    # - Chuyển nhiều lần, tăng dần số tiền
    # - Giờ giao dịch: tối muộn (chat với "người yêu")
    # - Lý do: mua quà, mua vé máy bay, đầu tư chung
    # - Số tiền: từ nhỏ đến rất lớn (1tr - 50tr+)
    # =========================================
    n_romance = int(n_fraud_target * scenario_ratios['romance_scam'])

    # Romance scam: người nhận mới + số tiền tăng dần + giờ tối
    romance_candidates = df[
        (df['is_new_recipient'] == 1) &
        (df['amount'] >= 1_000_000) &
        (df['amount'] <= 50_000_000) &
        (df['hour_of_day'] >= 19) &  # Giờ tối (chat với "người yêu")
        (~df.index.isin(fraud_indices))
    ].index.tolist()

    if romance_candidates:
        n_select = min(n_romance, len(romance_candidates))
        selected = np.random.choice(romance_candidates, size=n_select, replace=False)

        for idx in selected:
            # Điều chỉnh: thường xảy ra buổi tối, số tiền tăng dần
            df.loc[idx, 'hour_of_day'] = np.random.uniform(20, 23)
            df.loc[idx, 'amount_vs_avg_user'] = np.random.uniform(2, 5)  # Cao hơn bình thường

            if np.random.random() < np.random.uniform(0.7, 0.9):
                df.loc[idx, 'is_fraud'] = 1
                fraud_indices.append(idx)

    # =========================================
    # SCENARIO 5: Investment Scam (LỪA ĐẢO - đầu tư/tiền ảo)
    # Đặc điểm tại Việt Nam:
    # - Hứa lợi nhuận cao (30-50%/tháng)
    # - Đầu tư forex, crypto, chứng khoán giả
    # - Nạp tiền qua app lừa đảo
    # - Số tiền lớn, thường là chẵn triệu
    # - Giờ giao dịch: ban ngày (sau khi đọc quảng cáo)
    # =========================================
    n_investment = int(n_fraud_target * scenario_ratios['investment_scam'])

    investment_candidates = df[
        (df['is_new_recipient'] == 1) &
        (df['amount'] >= 5_000_000) &  # Đầu tư thường số tiền lớn
        (df['hour_of_day'] >= 8) &
        (df['hour_of_day'] <= 17) &  # Giờ làm việc
        (~df.index.isin(fraud_indices))
    ].index.tolist()

    if investment_candidates:
        n_select = min(n_investment, len(investment_candidates))
        selected = np.random.choice(investment_candidates, size=n_select, replace=False)

        for idx in selected:
            # Đầu tư scam thường là số chẵn triệu
            df.loc[idx, 'hour_of_day'] = np.random.uniform(9, 16)
            df.loc[idx, 'is_first_large_tx'] = np.random.choice([0, 1], p=[0.4, 0.6])

            if np.random.random() < np.random.uniform(0.75, 0.95):
                df.loc[idx, 'is_fraud'] = 1
                fraud_indices.append(idx)

    # =========================================
    # SCENARIO 6: Impersonation Scam (LỪA ĐẢO - giả mạo công an/ngân hàng)
    # Đặc điểm tại Việt Nam:
    # - Giả công an: "dính líu rửa tiền, chuyển tiền để điều tra"
    # - Giả ngân hàng: "tài khoản bị khóa, chuyển để xác minh"
    # - Giả shipper/bưu điện: "có kiện hàng, thanh toán COD"
    # - Thường xảy ra ban ngày (giờ hành chính)
    # - Số tiền lớn, chuyển gấp trong thời gian ngắn
    # - Nạn nhân hoảng loạn, không suy nghĩ kỹ
    # =========================================
    n_impersonation = int(n_fraud_target * scenario_ratios['impersonation_scam'])

    impersonation_candidates = df[
        (df['is_new_recipient'] == 1) &
        (df['amount'] >= 10_000_000) &  # Số tiền lớn
        (df['hour_of_day'] >= 8) &
        (df['hour_of_day'] <= 17) &  # Giờ hành chính
        (df['time_gap_prev_min'] < 60) &  # Chuyển gấp
        (~df.index.isin(fraud_indices))
    ].index.tolist()

    if impersonation_candidates:
        n_select = min(n_impersonation, len(impersonation_candidates))
        selected = np.random.choice(impersonation_candidates, size=n_select, replace=False)

        for idx in selected:
            # Giả công an thường gọi vào giờ hành chính
            df.loc[idx, 'hour_of_day'] = np.random.uniform(9, 11.5)  # Sáng
            df.loc[idx, 'time_gap_prev_min'] = np.random.uniform(5, 30)  # Chuyển rất gấp

            if np.random.random() < np.random.uniform(0.8, 0.95):
                df.loc[idx, 'is_fraud'] = 1
                fraud_indices.append(idx)

    # =========================================
    # SCENARIO 7: Job Scam (LỪA ĐẢO - việc làm online)
    # Đặc điểm tại Việt Nam:
    # - "Làm task kiếm tiền online"
    # - "Nạp tiền để nhận nhiệm vụ"
    # - "Đặt cọc để nhận việc"
    # - Số tiền nhỏ ban đầu, tăng dần
    # - Nhiều giao dịch trong ngày
    # - Target: sinh viên, người thất nghiệp
    # =========================================
    n_job = int(n_fraud_target * scenario_ratios['job_scam'])

    job_candidates = df[
        (df['is_new_recipient'] == 1) &
        (df['amount'] >= 100_000) &
        (df['amount'] <= 2_000_000) &  # Số tiền vừa phải
        (df['velocity_24h'] >= 2) &  # Nhiều giao dịch trong ngày
        (~df.index.isin(fraud_indices))
    ].index.tolist()

    if job_candidates:
        n_select = min(n_job, len(job_candidates))
        selected = np.random.choice(job_candidates, size=n_select, replace=False)

        for idx in selected:
            df.loc[idx, 'velocity_24h'] = np.random.randint(3, 10)
            df.loc[idx, 'recipient_count_30d'] = np.random.randint(1, 5)

            if np.random.random() < np.random.uniform(0.6, 0.85):
                df.loc[idx, 'is_fraud'] = 1
                fraud_indices.append(idx)

    # =========================================
    # BỔ SUNG: Nếu chưa đủ fraud, thêm từ các giao dịch đáng ngờ
    # =========================================
    remaining = n_fraud_target - len(fraud_indices)
    if remaining > 0:
        suspicious = df[
            (
                (df['is_night_hours'] == 1) |
                (df['is_new_device'] == 1) |
                (df['amount'] >= 10_000_000) |
                (df['velocity_1h'] >= 5) |
                ((df['is_new_recipient'] == 1) & (df['amount'] >= 3_000_000))
            ) &
            (~df.index.isin(fraud_indices))
        ].index.tolist()

        if suspicious:
            n_add = min(remaining, len(suspicious))
            additional = np.random.choice(suspicious, size=n_add, replace=False)
            for idx in additional:
                if np.random.random() < 0.6:
                    df.loc[idx, 'is_fraud'] = 1

    return df


# ===========================================================================
# SECTION 4: RISK FEATURES (KHÔNG LEAK)
# ===========================================================================

def compute_risk_features(df, seed=RANDOM_SEED):
    """
    Tính toán các feature risk KHÔNG dựa vào label
    """
    np.random.seed(seed)
    df = df.copy()

    # 1. channel_risk: Dựa trên rule ngân hàng VN + noise
    df['channel_risk'] = df['channel'].map(CHANNEL_RISK_BASE)
    df['channel_risk'] = df['channel_risk'] + np.random.uniform(-0.05, 0.05, len(df))
    df['channel_risk'] = df['channel_risk'].clip(0, 1)

    # 2. tx_type_risk: Dựa trên rule VN + noise
    df['tx_type_risk'] = df['transaction_type'].map(TX_TYPE_RISK_BASE)
    df['tx_type_risk'] = df['tx_type_risk'] + np.random.uniform(-0.05, 0.05, len(df))
    df['tx_type_risk'] = df['tx_type_risk'].clip(0, 1)

    # 3. recipient_is_suspicious: Người nhận nhận > 20 người gửi trong 7 ngày
    # Tính số sender unique cho mỗi recipient trong 7 ngày
    df['tx_date'] = df['timestamp'].dt.date

    recipient_sender_counts = defaultdict(lambda: defaultdict(set))
    df['recipient_is_suspicious'] = 0

    for idx, row in df.iterrows():
        recipient = row['recipient_id']
        sender = row['user_id']
        tx_date = row['timestamp']

        # Đếm số sender trong 7 ngày gần đây cho recipient này
        cutoff_date = tx_date - timedelta(days=7)
        recent_senders = set()

        for date, senders in recipient_sender_counts[recipient].items():
            if date >= cutoff_date:
                recent_senders.update(senders)

        if len(recent_senders) > 20:
            df.loc[idx, 'recipient_is_suspicious'] = 1

        # Cập nhật tracking
        recipient_sender_counts[recipient][tx_date].add(sender)

    df.drop('tx_date', axis=1, inplace=True)

    # 4. behavioral_risk_score: IsolationForest (unsupervised)
    # Chỉ dùng các feature không liên quan đến label
    behavior_features = [
        'amount_log', 'hour_of_day', 'velocity_1h', 'velocity_24h',
        'time_gap_prev_min', 'location_diff_km', 'is_night_hours'
    ]

    # Chuẩn bị dữ liệu
    X_behavior = df[behavior_features].copy()
    X_behavior['time_gap_prev_min'] = X_behavior['time_gap_prev_min'].clip(0, 10000)
    X_behavior = X_behavior.fillna(0)

    # Chuẩn hóa
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_behavior)

    # Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=100,
        contamination=0.1,
        random_state=seed,
        n_jobs=-1
    )

    # Score: -1 to 1 -> chuyển về 0 to 1
    anomaly_scores = iso_forest.fit_predict(X_scaled)
    decision_scores = iso_forest.decision_function(X_scaled)

    # Chuẩn hóa về 0-1
    min_score = decision_scores.min()
    max_score = decision_scores.max()
    df['behavioral_risk_score'] = (decision_scores - min_score) / (max_score - min_score + 1e-10)
    df['behavioral_risk_score'] = 1 - df['behavioral_risk_score']  # Đảo ngược: score cao = risk cao

    # 5. time_context_risk: Độ lệch so với giờ giao dịch quen thuộc
    df['time_context_risk'] = 0.0

    for user_id, group in df.groupby('user_id'):
        indices = group.index.tolist()
        hours = group['hour_of_day'].values
        preferred = group['user_preferred_hour'].values[0] if 'user_preferred_hour' in group.columns else 12

        for i, idx in enumerate(indices):
            # Tính trung bình giờ của các giao dịch trước
            if i > 0:
                past_hours = hours[:i]
                avg_hour = np.mean(past_hours)
                current_hour = hours[i]

                # Độ lệch (circular difference cho giờ)
                diff = abs(current_hour - avg_hour)
                diff = min(diff, 24 - diff)  # Giờ là circular

                # Normalize về 0-1
                df.loc[idx, 'time_context_risk'] = diff / 12.0
            else:
                # Giao dịch đầu tiên: so với preferred hour
                diff = abs(hours[i] - preferred)
                diff = min(diff, 24 - diff)
                df.loc[idx, 'time_context_risk'] = diff / 12.0

    df['time_context_risk'] = df['time_context_risk'].clip(0, 1)

    # 6. user_activity_level: Chuẩn hóa số giao dịch 30 ngày của user
    user_tx_counts = df.groupby('user_id').size()
    max_count = user_tx_counts.max()

    df['user_activity_level'] = df['user_id'].map(user_tx_counts) / max_count

    return df


# ===========================================================================
# SECTION 5: FINAL FEATURE ENGINEERING
# ===========================================================================

def prepare_final_dataset(df):
    """
    Chuẩn bị dataset cuối cùng với đủ 31 features + label
    """
    # Encode categorical features
    df = df.copy()

    # transaction_type encoding
    tx_type_map = {
        'chuyen_noi_bo': 0,
        'chuyen_lien_ngan_hang': 1,
        'thanh_toan_hoa_don': 2,
        'topup_vi': 3,
        'rut_atm': 4,
        'thanh_toan_pos': 5,
        'hoc_phi_vien_phi': 6,
        'mua_hang_online': 7
    }
    df['transaction_type_encoded'] = df['transaction_type'].map(tx_type_map)

    # channel encoding
    channel_map = {'mobile_app': 0, 'web': 1, 'atm': 2, 'pos': 3}
    df['channel_encoded'] = df['channel'].map(channel_map)

    # amount_tier encoding
    tier_map = {'micro': 0, 'small': 1, 'medium': 2, 'large': 3, 'very_large': 4}
    df['amount_tier_encoded'] = df['amount_tier'].map(tier_map)

    # Danh sách 31 features cuối cùng
    final_features = [
        'transaction_type_encoded',  # 1. Loại giao dịch (encoded)
        'amount_log',                # 2. Log số tiền
        'amount_tier_encoded',       # 3. Mức tiền (encoded)
        'amount_vs_avg_user',        # 4. So với trung bình user
        'channel_encoded',           # 5. Kênh giao dịch (encoded)
        'channel_risk',              # 6. Risk của kênh
        'tx_type_risk',              # 7. Risk của loại giao dịch
        'hour_of_day',               # 8. Giờ trong ngày
        'day_of_week',               # 9. Ngày trong tuần
        'is_weekend',                # 10. Có phải cuối tuần
        'is_night_hours',            # 11. Giờ đêm khuya
        'is_salary_period',          # 12. Kỳ lương
        'is_bill_period',            # 13. Kỳ thanh toán hóa đơn
        'time_gap_prev_min',         # 14. Khoảng cách giao dịch trước
        'velocity_1h',               # 15. Số giao dịch trong 1h
        'velocity_24h',              # 16. Số giao dịch trong 24h
        'is_new_recipient',          # 17. Người nhận mới
        'recipient_count_30d',       # 18. Số người nhận 30 ngày
        'is_new_device',             # 19. Thiết bị mới
        'device_count_30d',          # 20. Số thiết bị 30 ngày
        'location_diff_km',          # 21. Khoảng cách vị trí
        'account_age_days',          # 22. Tuổi tài khoản
        'is_first_large_tx',         # 23. Giao dịch lớn đầu tiên
        'recipient_is_suspicious',   # 24. Người nhận đáng ngờ
        'behavioral_risk_score',     # 25. Điểm risk hành vi
        'time_context_risk',         # 26. Risk ngữ cảnh thời gian
        'user_activity_level',       # 27. Mức độ hoạt động user
        'recipient_diversity',       # 28. Đa dạng người nhận
        'amount',                    # 29. Số tiền gốc
        'velocity_ratio',            # 30. Tỷ lệ velocity 1h/24h
        'risk_score_combined'        # 31. Điểm risk tổng hợp
    ]

    # Tính thêm các features còn thiếu
    df['velocity_ratio'] = df['velocity_1h'] / (df['velocity_24h'] + 1)

    # risk_score_combined: Kết hợp các risk features (không dùng label)
    df['risk_score_combined'] = (
        df['channel_risk'] * 0.2 +
        df['tx_type_risk'] * 0.2 +
        df['behavioral_risk_score'] * 0.3 +
        df['time_context_risk'] * 0.15 +
        df['recipient_is_suspicious'] * 0.15
    )

    # Chọn các cột cần thiết
    result = df[final_features + ['is_fraud']].copy()

    # Rename để rõ ràng hơn
    result = result.rename(columns={
        'transaction_type_encoded': 'transaction_type',
        'channel_encoded': 'channel',
        'amount_tier_encoded': 'amount_tier'
    })

    return result


# ===========================================================================
# SECTION 6: SANITY CHECKS
# ===========================================================================

def run_sanity_checks(df):
    """
    Kiểm tra chất lượng dữ liệu
    """
    results = {}

    # 1. Class balance
    fraud_count = df['is_fraud'].sum()
    total_count = len(df)
    fraud_rate = fraud_count / total_count
    results['class_balance'] = {
        'fraud_count': int(fraud_count),
        'non_fraud_count': int(total_count - fraud_count),
        'fraud_rate': round(fraud_rate, 4)
    }

    # 2. Zero variance columns
    zero_var_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            zero_var_cols.append(col)
    results['zero_variance_columns'] = zero_var_cols

    # 3. Correlation với is_fraud (phát hiện leak)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'is_fraud' in numeric_cols:
        correlations = {}
        for col in numeric_cols:
            if col != 'is_fraud':
                corr = df[col].corr(df['is_fraud'])
                correlations[col] = round(corr, 4)

        # Sắp xếp theo absolute correlation
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        results['top_correlations'] = sorted_corr[:10]

        # Cảnh báo nếu correlation quá cao (potential leak)
        high_corr = [(k, v) for k, v in sorted_corr if abs(v) > 0.5]
        results['potential_leaks'] = high_corr

    # 4. Missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0].to_dict()
    results['missing_values'] = missing_cols

    # 5. Statistics summary
    results['statistics'] = {
        'total_transactions': len(df),
        'n_features': len(df.columns) - 1,
        'amount_mean': round(df['amount'].mean(), 0) if 'amount' in df.columns else None,
        'amount_median': round(df['amount'].median(), 0) if 'amount' in df.columns else None
    }

    return results


# ===========================================================================
# SECTION 7: QUICK TRAIN LIGHTGBM
# ===========================================================================

def quick_train_lightgbm(df, test_size=0.2, seed=RANDOM_SEED):
    """
    Train nhanh LightGBM với time-based split
    """
    try:
        import lightgbm as lgb
        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    except ImportError:
        return None, "LightGBM chưa được cài đặt. Hãy chạy: pip install lightgbm"

    # Chuẩn bị dữ liệu
    feature_cols = [col for col in df.columns if col != 'is_fraud']
    X = df[feature_cols].values
    y = df['is_fraud'].values

    # Time-based split (80% train, 20% test - lấy 20% cuối làm test)
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Tạo dataset
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # Parameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': seed,
        'is_unbalance': True
    }

    # Train
    model = lgb.train(
        params,
        train_data,
        num_boost_round=200,
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(stopping_rounds=20)]
    )

    # Predict
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Metrics
    metrics = {
        'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
        'f1': round(f1_score(y_test, y_pred, zero_division=0), 4),
        'auc': round(roc_auc_score(y_test, y_pred_proba), 4)
    }

    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importance()))
    importance_sorted = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    return {
        'metrics': metrics,
        'feature_importance': importance_sorted[:15],
        'train_size': len(X_train),
        'test_size': len(X_test),
        'test_fraud_rate': round(y_test.sum() / len(y_test), 4)
    }, None


# ===========================================================================
# SECTION 8: STREAMLIT APP
# ===========================================================================

def main():
    st.set_page_config(
        page_title="🏦 Synthetic Data Generator - Vietnam Banking",
        page_icon="🏦",
        layout="wide"
    )

    st.title("🏦 Synthetic Data Generator cho LightGBM Chống Gian Lận & Lừa Đảo")
    st.markdown("""
    **Ứng dụng tạo dữ liệu giả lập chuẩn hành vi người Việt Nam để train mô hình phát hiện:**
    - **GIAN LẬN (Fraud)**: Account Takeover, Mule Account, Card Testing
    - **LỪA ĐẢO (Scam)**: Romance Scam, Investment Scam, Giả công an/ngân hàng, Job Scam

    ---
    """)

    # Sidebar - Cấu hình
    st.sidebar.header("⚙️ Cấu hình")

    n_transactions = st.sidebar.number_input(
        "Số lượng giao dịch",
        min_value=1000,
        max_value=500_000,
        value=DEFAULT_N_TRANSACTIONS,
        step=10000,
        help="Số lượng giao dịch cần tạo (tối đa 500.000)"
    )

    n_users = st.sidebar.number_input(
        "Số lượng users",
        min_value=100,
        max_value=100_000,
        value=DEFAULT_N_USERS,
        step=500,
        help="Số lượng người dùng"
    )

    n_recipients = st.sidebar.number_input(
        "Số lượng recipients",
        min_value=100,
        max_value=100_000,
        value=DEFAULT_N_RECIPIENTS,
        step=500,
        help="Số lượng người nhận"
    )

    fraud_rate = st.sidebar.slider(
        "Tỷ lệ fraud",
        min_value=0.01,
        max_value=0.20,
        value=DEFAULT_FRAUD_RATE,
        step=0.01,
        help="Tỷ lệ giao dịch gian lận"
    )

    random_seed = st.sidebar.number_input(
        "Random seed",
        min_value=0,
        max_value=99999,
        value=RANDOM_SEED,
        help="Seed để tái tạo kết quả"
    )

    st.sidebar.markdown("---")

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📋 Thông tin Dataset")
        st.markdown(f"""
        - **Số giao dịch:** {n_transactions:,}
        - **Số users:** {n_users:,}
        - **Số recipients:** {n_recipients:,}
        - **Tỷ lệ fraud mục tiêu:** {fraud_rate:.1%}
        - **Số features:** 31 + 1 label
        """)

    with col2:
        st.subheader("📊 31 Features")
        with st.expander("Xem danh sách features"):
            st.markdown("""
            1. `transaction_type` - Loại giao dịch
            2. `amount_log` - Log số tiền
            3. `amount_tier` - Mức tiền
            4. `amount_vs_avg_user` - So với TB user
            5. `channel` - Kênh giao dịch
            6. `channel_risk` - Risk kênh
            7. `tx_type_risk` - Risk loại GD
            8. `hour_of_day` - Giờ trong ngày
            9. `day_of_week` - Ngày trong tuần
            10. `is_weekend` - Cuối tuần
            11. `is_night_hours` - Giờ đêm
            12. `is_salary_period` - Kỳ lương
            13. `is_bill_period` - Kỳ hóa đơn
            14. `time_gap_prev_min` - Gap GD trước
            15. `velocity_1h` - Velocity 1h
            16. `velocity_24h` - Velocity 24h
            17. `is_new_recipient` - Recipient mới
            18. `recipient_count_30d` - Số recipient 30d
            19. `is_new_device` - Device mới
            20. `device_count_30d` - Số device 30d
            21. `location_diff_km` - Khoảng cách
            22. `account_age_days` - Tuổi TK
            23. `is_first_large_tx` - GD lớn đầu tiên
            24. `recipient_is_suspicious` - Recipient nghi ngờ
            25. `behavioral_risk_score` - Risk hành vi
            26. `time_context_risk` - Risk thời gian
            27. `user_activity_level` - Mức hoạt động
            28. `recipient_diversity` - Đa dạng recipient
            29. `amount` - Số tiền gốc
            30. `velocity_ratio` - Tỷ lệ velocity
            31. `risk_score_combined` - Risk tổng hợp
            """)

        st.subheader("🎭 7 Kịch bản")
        with st.expander("Xem chi tiết kịch bản Fraud/Scam"):
            st.markdown("""
            **GIAN LẬN (Fraud) - Kẻ gian chiếm TK:**
            1. **Account Takeover** (20%) - Bị hack, đổi device, GD lúc 1-4AM
            2. **Mule Account** (15%) - TK trung gian rửa tiền
            3. **Card Testing** (10%) - Test thẻ bị cắp

            **LỪA ĐẢO (Scam) - Nạn nhân tự chuyển:**
            4. **Romance Scam** (15%) - Lừa tình cảm
            5. **Investment Scam** (15%) - Lừa đầu tư/crypto
            6. **Impersonation** (15%) - Giả công an/NH
            7. **Job Scam** (10%) - Lừa việc làm online
            """)

    st.markdown("---")

    # Generate button
    if st.button("🚀 Tạo Dataset", type="primary", use_container_width=True):

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # Step 1: Generate base transactions
            status_text.text("📝 Đang tạo giao dịch cơ bản...")
            progress_bar.progress(10)

            df, user_profiles, recipient_profiles = generate_base_transactions(
                n_transactions, n_users, n_recipients, random_seed
            )

            # Step 2: Compute derived features (TỐI ƯU cho 200K+ dòng)
            status_text.text("🔢 Đang tính toán derived features...")
            progress_bar.progress(30)

            def update_status(msg):
                status_text.text(f"🔢 {msg}")

            df = compute_derived_features_optimized(df, progress_callback=update_status)

            # Step 3: Apply fraud scenarios
            status_text.text("🎭 Đang áp dụng fraud scenarios...")
            progress_bar.progress(50)

            df = apply_fraud_scenarios(df, fraud_rate, random_seed)

            # Step 4: Compute risk features
            status_text.text("⚠️ Đang tính toán risk features...")
            progress_bar.progress(70)

            df = compute_risk_features(df, random_seed)

            # Step 5: Prepare final dataset
            status_text.text("📦 Đang chuẩn bị dataset cuối cùng...")
            progress_bar.progress(85)

            final_df = prepare_final_dataset(df)

            # Store in session state
            st.session_state['generated_data'] = final_df
            st.session_state['raw_data'] = df

            progress_bar.progress(100)
            status_text.text("✅ Hoàn thành!")

            st.success(f"✅ Đã tạo thành công {len(final_df):,} giao dịch với {len(final_df.columns)-1} features!")

        except Exception as e:
            st.error(f"❌ Lỗi: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

    # Display results if data exists
    if 'generated_data' in st.session_state:
        final_df = st.session_state['generated_data']

        st.markdown("---")
        st.subheader("📊 Kết quả")

        # Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📋 Preview Data",
            "🔍 Sanity Checks",
            "📈 Visualizations",
            "🤖 Quick Train",
            "💾 Export"
        ])

        with tab1:
            st.dataframe(final_df.head(100), use_container_width=True)
            st.markdown(f"**Shape:** {final_df.shape}")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Thống kê cơ bản:**")
                st.dataframe(final_df.describe().T, use_container_width=True)
            with col2:
                st.markdown("**Data types:**")
                st.dataframe(pd.DataFrame({
                    'Column': final_df.columns,
                    'Type': final_df.dtypes.values,
                    'Non-Null': final_df.count().values
                }), use_container_width=True)

        with tab2:
            st.markdown("### 🔍 Sanity Checks")

            if st.button("▶️ Chạy Sanity Checks"):
                checks = run_sanity_checks(final_df)

                # Class balance
                st.markdown("#### 1. Class Balance")
                col1, col2, col3 = st.columns(3)
                col1.metric("Fraud", f"{checks['class_balance']['fraud_count']:,}")
                col2.metric("Non-Fraud", f"{checks['class_balance']['non_fraud_count']:,}")
                col3.metric("Fraud Rate", f"{checks['class_balance']['fraud_rate']:.2%}")

                # Zero variance
                st.markdown("#### 2. Zero Variance Columns")
                if checks['zero_variance_columns']:
                    st.warning(f"⚠️ Các cột có variance = 0: {checks['zero_variance_columns']}")
                else:
                    st.success("✅ Không có cột nào có variance = 0")

                # Correlations
                st.markdown("#### 3. Top Correlations với is_fraud")
                if 'top_correlations' in checks:
                    corr_df = pd.DataFrame(checks['top_correlations'], columns=['Feature', 'Correlation'])
                    st.dataframe(corr_df, use_container_width=True)

                # Potential leaks
                st.markdown("#### 4. Potential Data Leaks (|corr| > 0.5)")
                if checks.get('potential_leaks'):
                    st.error(f"⚠️ Cảnh báo leak tiềm năng: {checks['potential_leaks']}")
                else:
                    st.success("✅ Không phát hiện data leak")

                # Missing values
                st.markdown("#### 5. Missing Values")
                if checks['missing_values']:
                    st.warning(f"⚠️ Các cột có missing: {checks['missing_values']}")
                else:
                    st.success("✅ Không có missing values")

        with tab3:
            st.markdown("### 📈 Visualizations")

            import matplotlib.pyplot as plt

            col1, col2 = st.columns(2)

            with col1:
                # Histogram amount
                st.markdown("#### Phân phối Amount")
                fig1, ax1 = plt.subplots(figsize=(8, 4))

                if 'amount' in final_df.columns:
                    # Log scale cho dễ nhìn
                    ax1.hist(np.log1p(final_df['amount']), bins=50, edgecolor='black', alpha=0.7)
                    ax1.set_xlabel('Log(Amount + 1)')
                    ax1.set_ylabel('Frequency')
                    ax1.set_title('Phân phối Log Amount')
                    st.pyplot(fig1)
                else:
                    st.warning("Không tìm thấy cột amount")

            with col2:
                # Histogram hour_of_day
                st.markdown("#### Phân phối Giờ giao dịch")
                fig2, ax2 = plt.subplots(figsize=(8, 4))

                if 'hour_of_day' in final_df.columns:
                    ax2.hist(final_df['hour_of_day'], bins=24, edgecolor='black', alpha=0.7, color='orange')
                    ax2.set_xlabel('Hour of Day')
                    ax2.set_ylabel('Frequency')
                    ax2.set_title('Phân phối Giờ giao dịch (Hành vi VN)')
                    ax2.set_xticks(range(0, 24, 2))
                    st.pyplot(fig2)

            # Fraud by hour
            st.markdown("#### Fraud theo Giờ")
            fig3, ax3 = plt.subplots(figsize=(12, 4))

            hour_fraud = final_df.groupby(final_df['hour_of_day'].astype(int))['is_fraud'].mean()
            ax3.bar(hour_fraud.index, hour_fraud.values, color='red', alpha=0.7)
            ax3.set_xlabel('Hour of Day')
            ax3.set_ylabel('Fraud Rate')
            ax3.set_title('Tỷ lệ Fraud theo Giờ')
            ax3.set_xticks(range(0, 24))
            st.pyplot(fig3)

            # Fraud by channel
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Fraud theo Channel")
                fig4, ax4 = plt.subplots(figsize=(6, 4))
                channel_fraud = final_df.groupby('channel')['is_fraud'].mean()
                ax4.bar(channel_fraud.index.astype(str), channel_fraud.values, color='purple', alpha=0.7)
                ax4.set_xlabel('Channel')
                ax4.set_ylabel('Fraud Rate')
                ax4.set_title('Tỷ lệ Fraud theo Kênh')
                st.pyplot(fig4)

            with col2:
                st.markdown("#### Fraud theo Amount Tier")
                fig5, ax5 = plt.subplots(figsize=(6, 4))
                tier_fraud = final_df.groupby('amount_tier')['is_fraud'].mean()
                ax5.bar(tier_fraud.index.astype(str), tier_fraud.values, color='green', alpha=0.7)
                ax5.set_xlabel('Amount Tier')
                ax5.set_ylabel('Fraud Rate')
                ax5.set_title('Tỷ lệ Fraud theo Mức tiền')
                st.pyplot(fig5)

        with tab4:
            st.markdown("### 🤖 Quick Train LightGBM")
            st.markdown("""
            Train nhanh mô hình LightGBM để kiểm tra chất lượng dữ liệu.
            Sử dụng **time-based split** (80% train / 20% test).
            """)

            if st.button("▶️ Train LightGBM"):
                with st.spinner("Đang train..."):
                    result, error = quick_train_lightgbm(final_df, seed=random_seed)

                if error:
                    st.error(error)
                    st.info("Cài đặt LightGBM: `pip install lightgbm`")
                else:
                    # Metrics
                    st.markdown("#### 📊 Metrics")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Precision", f"{result['metrics']['precision']:.4f}")
                    col2.metric("Recall", f"{result['metrics']['recall']:.4f}")
                    col3.metric("F1 Score", f"{result['metrics']['f1']:.4f}")
                    col4.metric("AUC", f"{result['metrics']['auc']:.4f}")

                    st.markdown(f"""
                    - **Train size:** {result['train_size']:,}
                    - **Test size:** {result['test_size']:,}
                    - **Test fraud rate:** {result['test_fraud_rate']:.2%}
                    """)

                    # Feature importance
                    st.markdown("#### 🏆 Top 15 Feature Importance")
                    importance_df = pd.DataFrame(
                        result['feature_importance'],
                        columns=['Feature', 'Importance']
                    )

                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
                    ax.set_xlabel('Importance')
                    ax.set_title('Feature Importance (LightGBM)')
                    ax.invert_yaxis()
                    st.pyplot(fig)

        with tab5:
            st.markdown("### 💾 Export Dataset")

            # CSV download
            csv = final_df.to_csv(index=False)

            st.download_button(
                label="📥 Download CSV (lightgbm_train_vn.csv)",
                data=csv,
                file_name="lightgbm_train_vn.csv",
                mime="text/csv",
                use_container_width=True
            )

            st.markdown(f"""
            **Thông tin file:**
            - Số dòng: {len(final_df):,}
            - Số cột: {len(final_df.columns)} (31 features + 1 label)
            - Kích thước ước tính: ~{len(csv) / 1024 / 1024:.1f} MB
            """)

            # Show columns
            st.markdown("**Danh sách cột:**")
            st.code(", ".join(final_df.columns.tolist()))

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        🏦 Synthetic Data Generator for Vietnam Banking Fraud Detection<br>
        Developed with ❤️ for Vietnamese Banking Industry
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
