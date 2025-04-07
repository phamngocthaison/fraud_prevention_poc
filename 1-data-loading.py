import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("GIAI ĐOẠN 1: TẢI VÀ KẾT HỢP DỮ LIỆU")
print("-" * 50)

# Tạo thư mục output nếu chưa tồn tại
if not os.path.exists('output'):
    os.makedirs('output')

# Bước 1: Đọc dữ liệu giao dịch và nhãn
print("Đang đọc dữ liệu giao dịch...")
transactions = pd.read_csv('data/transactions_data.csv')
print(f"Số lượng giao dịch trong tập dữ liệu: {transactions.shape[0]}")

print("\nĐang đọc nhãn gian lận từ file JSON...")
with open('data/train_fraud_labels.json', 'r') as f:
    fraud_labels = json.load(f)

# Chuyển đổi từ JSON sang DataFrame
print("\nĐang chuyển đổi từ JSON sang DataFrame...")
# Dựa trên cấu trúc thực tế của file JSON: {"target": {"id1": "No", "id2": "Yes", ...}}
if isinstance(fraud_labels, dict) and "target" in fraud_labels:
    # Chuyển đổi từ định dạng {"target": {"id1": "No", "id2": "Yes", ...}}
    target_dict = fraud_labels["target"]

    # Chuyển đổi thành danh sách các tuple (transaction_id, is_fraud)
    # Trong đó "Yes" = 1 (gian lận), "No" = 0 (không gian lận)
    records = []
    for tx_id, label in target_dict.items():
        is_fraud = 1 if label.lower() == "yes" else 0
        records.append({"transaction_id": tx_id, "is_fraud": is_fraud})

    fraud_df = pd.DataFrame(records)
else:
    # Fallback cho các định dạng khác
    fraud_df = pd.DataFrame(fraud_labels)

print(f"Số lượng nhãn gian lận: {fraud_df.shape[0]}")
print(f"Số lượng giao dịch được đánh nhãn gian lận (Yes): {(fraud_df['is_fraud'] == 1).sum()}")

# Kiểm tra cấu trúc dữ liệu
print("\nThông tin về dữ liệu giao dịch:")
print(transactions.info())
print("\nMẫu dữ liệu giao dịch:")
print(transactions.head())

print("\nThông tin về nhãn gian lận:")
print(fraud_df.info())
print("\nMẫu nhãn gian lận:")
print(fraud_df.head())

# Bước 2: Kết hợp dữ liệu giao dịch với nhãn gian lận
print("\nĐang kết hợp dữ liệu giao dịch với nhãn gian lận...")

# Chuyển đổi transaction_id sang kiểu string để đảm bảo khớp với dữ liệu JSON
if 'id' in transactions.columns:
    transactions = transactions.rename(columns={'id': 'transaction_id'})
    transactions['transaction_id'] = transactions['transaction_id'].astype(str)

if 'transaction_id' in fraud_df.columns and 'is_fraud' in fraud_df.columns:
    # Sử dụng left join để giữ lại tất cả các giao dịch
    df = transactions.merge(fraud_df, on='transaction_id', how='left')
    # Điền giá trị NaN với 0 (giả sử giao dịch không có nhãn là không gian lận)
    df['is_fraud'] = df['is_fraud'].fillna(0)
else:
    print("CẢNH BÁO: Cấu trúc nhãn gian lận không như mong đợi!")
    print("Các cột có sẵn trong fraud_df:", fraud_df.columns.tolist())
    print("Các cột có sẵn trong transactions:", transactions.columns.tolist())
    # Tạo dataframe mặc định với tất cả giao dịch được đánh là không gian lận
    df = transactions.copy()
    df['is_fraud'] = 0

print("\nKiểm tra dữ liệu sau khi kết hợp:")
print(f"Tổng số giao dịch: {df.shape[0]}")
print(f"Số giao dịch gian lận: {df['is_fraud'].sum()}")
print(f"Tỷ lệ gian lận: {df['is_fraud'].mean() * 100:.2f}%")

# Lưu dữ liệu đã kết hợp để sử dụng ở bước tiếp theo
print("\nĐang lưu dữ liệu đã kết hợp...")
df.to_csv('output/combined_data.csv', index=False)
print(f"Đã lưu dữ liệu kết hợp vào 'output/combined_data.csv'")

# Tạo biểu đồ phân bố giao dịch gian lận
print("\nĐang tạo biểu đồ phân bố gian lận...")
plt.figure(figsize=(10, 6))
sns.countplot(x='is_fraud', data=df)
plt.title('Phân bố giao dịch gian lận')
plt.xlabel('Gian lận (1) / Không gian lận (0)')
plt.ylabel('Số lượng giao dịch')
plt.xticks([0, 1], ['Hợp pháp', 'Gian lận'])
for i in range(2):
    count = df['is_fraud'].value_counts()[i]
    percent = 100 * count / df.shape[0]
    plt.text(i, count, f"{count}\n({percent:.1f}%)", ha='center', va='bottom')
plt.tight_layout()
plt.savefig('output/fraud_distribution.png')
print(f"Đã lưu biểu đồ phân bố vào 'output/fraud_distribution.png'")

print("\nHoàn thành Giai đoạn 1: Tải và kết hợp dữ liệu!")