import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

print("GIAI ĐOẠN 5: XÂY DỰNG HỆ THỐNG ĐÁNH GIÁ RỦI RO")
print("-" * 50)

# Tạo thư mục output nếu chưa tồn tại
if not os.path.exists('output5'):
    os.makedirs('output5')

# Đọc kết quả dự đoán từ bước trước
print("Đang đọc kết quả dự đoán...")
predictions = pd.read_csv('output4/test_predictions.csv')
print(f"Đã đọc {len(predictions)} kết quả dự đoán")

# Bước 1: Xây dựng hàm đánh giá rủi ro
print("\nXây dựng hàm đánh giá rủi ro...")

def calculate_risk_score(probability):
    """
    Chuyển đổi xác suất gian lận thành điểm rủi ro từ 0-100
    """
    return probability * 100

def classify_risk(risk_score):
    """
    Phân loại rủi ro dựa trên thang điểm
    """
    if risk_score < 20:
        return "Thấp"
    elif risk_score < 50:
        return "Trung bình"
    elif risk_score < 80:
        return "Cao"
    else:
        return "Rất cao"

# Bước 2: Áp dụng hàm đánh giá rủi ro vào kết quả dự đoán
print("Áp dụng đánh giá rủi ro vào kết quả dự đoán...")
predictions['risk_score'] = calculate_risk_score(predictions['probability'])
predictions['risk_category'] = predictions['risk_score'].apply(classify_risk)

# Xem phân bố danh mục rủi ro
risk_distribution = predictions['risk_category'].value_counts()
print("\nPhân bố danh mục rủi ro:")
print(risk_distribution)

# Bước 3: Phân tích mối quan hệ giữa điểm rủi ro và kết quả thực tế
print("\nPhân tích mối quan hệ giữa điểm rủi ro và kết quả thực tế...")

# Tính tỷ lệ gian lận thực tế theo danh mục rủi ro
risk_fraud_rate = predictions.groupby('risk_category')['actual'].mean() * 100
print("\nTỷ lệ gian lận thực tế theo danh mục rủi ro:")
print(risk_fraud_rate)

# Bước 4: Tạo biểu đồ phân bố rủi ro
print("\nTạo biểu đồ phân bố rủi ro...")

# Biểu đồ phân bố danh mục rủi ro
plt.figure(figsize=(10, 6))
ax = risk_distribution.plot(kind='bar', color=['green', 'orange', 'red', 'darkred'])
plt.title('Phân bố danh mục rủi ro')
plt.xlabel('Danh mục rủi ro')
plt.ylabel('Số lượng giao dịch')

# Thêm nhãn số lượng và phần trăm
for i, count in enumerate(risk_distribution):
    percent = 100 * count / len(predictions)
    ax.text(i, count, f"{count}\n({percent:.1f}%)", ha='center', va='bottom')

plt.tight_layout()
plt.savefig('output5/risk_distribution.png')
print(f"Đã lưu biểu đồ phân bố rủi ro vào 'output5/risk_distribution.png'")

# Biểu đồ điểm rủi ro dạng histogram
plt.figure(figsize=(12, 6))
sns.histplot(data=predictions, x='risk_score', hue='actual', bins=20,
             multiple='stack', palette=['skyblue', 'red'])
plt.title('Phân bố điểm rủi ro theo kết quả thực tế')
plt.xlabel('Điểm rủi ro')
plt.ylabel('Số lượng giao dịch')
plt.legend(['Hợp pháp', 'Gian lận'])
plt.tight_layout()
plt.savefig('output5/risk_score_distribution.png')
print(f"Đã lưu biểu đồ phân bố điểm rủi ro vào 'output5/risk_score_distribution.png'")

# Biểu đồ tỷ lệ gian lận theo danh mục rủi ro
plt.figure(figsize=(10, 6))
risk_fraud_rate.plot(kind='bar', color=['green', 'orange', 'red', 'darkred'])
plt.title('Tỷ lệ gian lận thực tế theo danh mục rủi ro')
plt.xlabel('Danh mục rủi ro')
plt.ylabel('Tỷ lệ gian lận (%)')
plt.tight_layout()
plt.savefig('output5/fraud_rate_by_risk.png')
print(f"Đã lưu biểu đồ tỷ lệ gian lận theo rủi ro vào 'output/fraud_rate_by_risk.png'")

# Bước 5: Đánh giá hiệu quả của hệ thống phân loại rủi ro
print("\nĐánh giá hiệu quả của hệ thống phân loại rủi ro...")

# Tính số lượng giao dịch trong mỗi danh mục
risk_counts = predictions['risk_category'].value_counts().to_dict()
total_transactions = len(predictions)

# Tính số lượng giao dịch gian lận theo danh mục
fraud_by_category = predictions[predictions['actual'] == 1]['risk_category'].value_counts().to_dict()
total_frauds = predictions['actual'].sum()

# Tỷ lệ phát hiện gian lận
detection_rate_by_category = {}
for category in risk_counts.keys():
    if category in fraud_by_category:
        detection_rate_by_category[category] = fraud_by_category[category] / total_frauds * 100
    else:
        detection_rate_by_category[category] = 0

print("\nTỷ lệ phát hiện gian lận theo danh mục rủi ro:")
for category, rate in detection_rate_by_category.items():
    print(f"{category}: {rate:.2f}%")

# Tính ROI tiềm năng
# Giả sử chi phí xem xét thủ công một giao dịch là $10
# Giả sử thiệt hại trung bình do gian lận là $1000
review_cost = 10
fraud_damage = 1000

# Với danh mục "Cao" và "Rất cao", ta xem xét thủ công
high_risk_categories = ['Cao', 'Rất cao']
reviewed_transactions = predictions[predictions['risk_category'].isin(high_risk_categories)]
review_count = len(reviewed_transactions)
caught_frauds = reviewed_transactions['actual'].sum()

# Chi phí và lợi ích
review_total_cost = review_count * review_cost
prevented_damage = caught_frauds * fraud_damage
net_benefit = prevented_damage - review_total_cost
roi = (prevented_damage / review_total_cost - 1) * 100 if review_total_cost > 0 else 0

print(f"\nVới chiến lược xem xét thủ công các giao dịch có rủi ro Cao và Rất cao:")
print(f"Số giao dịch cần xem xét: {review_count} ({review_count/total_transactions*100:.2f}% tổng số)")
print(f"Số gian lận bắt được: {caught_frauds} ({caught_frauds/total_frauds*100:.2f}% tổng số gian lận)")
print(f"Chi phí xem xét: ${review_total_cost:,.2f}")
print(f"Thiệt hại ngăn chặn: ${prevented_damage:,.2f}")
print(f"Lợi ích ròng: ${net_benefit:,.2f}")
print(f"ROI: {roi:.2f}%")

# Bước 6: Lưu hệ thống đánh giá rủi ro
print("\nLưu hệ thống đánh giá rủi ro...")

risk_system = {
    'calculate_risk_score': calculate_risk_score,
    'classify_risk': classify_risk
}

with open('output5/risk_rating_system.pkl', 'wb') as f:
    pickle.dump(risk_system, f)
print("Đã lưu hệ thống đánh giá rủi ro vào 'output/risk_rating_system.pkl'")

# Lưu kết quả có đánh giá rủi ro
predictions.to_csv('output/fraud_risk_assessment_results.csv', index=False)
print("Đã lưu kết quả đánh giá rủi ro vào 'output/fraud_risk_assessment_results.csv'")

# Bước 7: Tạo một số giao dịch mẫu cho demo
print("\nTạo một số giao dịch mẫu cho demo...")

# Lấy một số mẫu từ mỗi danh mục rủi ro
sample_size = min(5, len(predictions))
samples = []

for category in ['Thấp', 'Trung bình', 'Cao', 'Rất cao']:
    category_samples = predictions[predictions['risk_category'] == category].sample(
        min(sample_size, sum(predictions['risk_category'] == category)),
        random_state=42
    )
    samples.append(category_samples)

demo_samples = pd.concat(samples)
demo_samples = demo_samples.sample(frac=1, random_state=42)  # Trộn ngẫu nhiên

# Lưu các mẫu demo
demo_samples.to_csv('output5/demo_samples.csv', index=False)
print(f"Đã lưu {len(demo_samples)} mẫu demo vào 'output5/demo_samples.csv'")

print("\nHoàn thành Giai đoạn 5: Xây dựng hệ thống đánh giá rủi ro!")