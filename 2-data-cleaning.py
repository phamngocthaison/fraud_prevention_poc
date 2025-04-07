import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("GIAI ĐOẠN 2: LÀM SẠCH VÀ PHÂN TÍCH DỮ LIỆU")
print("-" * 50)

# Tạo thư mục output nếu chưa tồn tại
if not os.path.exists('output2'):
    os.makedirs('output2')

# Đọc dữ liệu đã kết hợp từ bước trước
print("Đang đọc dữ liệu kết hợp...")
df = pd.read_csv('output/combined_data.csv')
print(f"Đã đọc {df.shape[0]} giao dịch với {df.shape[1]} thuộc tính")

# Bước 1: Kiểm tra và hiển thị thông tin về dữ liệu
print("\nThông tin dữ liệu:")
print(df.info())

# Kiểm tra dữ liệu thiếu
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100
missing_df = pd.DataFrame({'Số giá trị thiếu': missing_values,
                           'Phần trăm': missing_percent})
missing_df = missing_df[missing_df['Số giá trị thiếu'] > 0].sort_values('Số giá trị thiếu', ascending=False)

print("\nKiểm tra dữ liệu thiếu:")
if len(missing_df) > 0:
    print(missing_df)

    # Vẽ biểu đồ hiển thị dữ liệu thiếu
    plt.figure(figsize=(12, 6))
    plt.bar(missing_df.index, missing_df['Phần trăm'])
    plt.title('Phần trăm giá trị thiếu theo cột')
    plt.xlabel('Cột')
    plt.ylabel('Phần trăm giá trị thiếu')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('output2/missing_values.png')
    print(f"Đã lưu biểu đồ giá trị thiếu vào 'output2/missing_values.png'")
else:
    print("Không có giá trị thiếu trong dữ liệu!")

# Bước 2: Phân tích phân phối giá trị và xác định giá trị ngoại lai
print("\nPhân tích phân phối giá trị số:")
numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Loại bỏ cột là nhãn và ID
if 'is_fraud' in numeric_columns:
    numeric_columns.remove('is_fraud')
if 'transaction_id' in numeric_columns:
    numeric_columns.remove('transaction_id')

# Chọn tối đa 5 cột số để hiển thị thống kê
sample_numeric = numeric_columns[:5] if len(numeric_columns) > 5 else numeric_columns
for col in sample_numeric:
    print(f"\nThống kê cho cột {col}:")
    print(df[col].describe())

# Vẽ boxplot cho các cột số để xác định outlier
if len(numeric_columns) > 0:
    # Chọn tối đa 5 cột để vẽ
    plot_columns = numeric_columns[:5] if len(numeric_columns) > 5 else numeric_columns

    plt.figure(figsize=(15, 10))
    for i, col in enumerate(plot_columns, 1):
        plt.subplot(len(plot_columns), 1, i)
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot của {col}')
    plt.tight_layout()
    plt.savefig('output2/numeric_boxplots.png')
    print(f"Đã lưu biểu đồ boxplot cho các cột số vào 'output2/numeric_boxplots.png'")

# Bước 3: Phân tích các cột phân loại
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
if 'transaction_id' in categorical_columns:
    categorical_columns.remove('transaction_id')

print("\nPhân tích các cột phân loại:")
for col in categorical_columns:
    print(f"\nGiá trị duy nhất trong cột {col}:")
    value_counts = df[col].value_counts()
    print(f"Số giá trị duy nhất: {len(value_counts)}")
    if len(value_counts) <= 10:
        print(value_counts)
    else:
        print(value_counts.head(10))
        print("...")

# Vẽ biểu đồ cho các cột phân loại
if len(categorical_columns) > 0:
    # Chọn tối đa 3 cột phân loại để vẽ
    plot_categorical = categorical_columns[:3] if len(categorical_columns) > 3 else categorical_columns

    plt.figure(figsize=(15, 5 * len(plot_categorical)))
    for i, col in enumerate(plot_categorical, 1):
        top_categories = df[col].value_counts().head(10).index
        plt.subplot(len(plot_categorical), 1, i)
        sns.countplot(y=col, data=df[df[col].isin(top_categories)],
                      order=df[col].value_counts().head(10).index)
        plt.title(f'Top 10 giá trị trong {col}')
        plt.tight_layout()
    plt.savefig('output2/categorical_plots.png')
    print(f"Đã lưu biểu đồ cho các cột phân loại vào 'output2/categorical_plots.png'")

# Bước 4: Phân tích mối tương quan giữa các đặc trưng
print("\nPhân tích tương quan giữa các đặc trưng số:")
if len(numeric_columns) > 1:
    # Thêm cột nhãn vào phân tích tương quan
    correlation_columns = numeric_columns + ['is_fraud'] if 'is_fraud' not in numeric_columns else numeric_columns

    # Tính ma trận tương quan
    correlation_matrix = df[correlation_columns].corr()

    # Vẽ heatmap tương quan
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Ma trận tương quan giữa các đặc trưng số')
    plt.tight_layout()
    plt.savefig('output2/correlation_matrix.png')
    print(f"Đã lưu ma trận tương quan vào 'output2/correlation_matrix.png'")

    # Tìm các đặc trưng có tương quan cao với is_fraud
    if 'is_fraud' in correlation_columns:
        fraud_corr = correlation_matrix['is_fraud'].sort_values(ascending=False)
        print("\nCác đặc trưng có tương quan cao nhất với is_fraud:")
        print(fraud_corr)

        # Vẽ biểu đồ tương quan với is_fraud
        plt.figure(figsize=(10, 8))
        fraud_corr = fraud_corr.drop('is_fraud')  # Loại bỏ chính nó
        sns.barplot(x=fraud_corr.values, y=fraud_corr.index)
        plt.title('Tương quan của các đặc trưng với is_fraud')
        plt.xlabel('Hệ số tương quan')
        plt.ylabel('Đặc trưng')
        plt.tight_layout()
        plt.savefig('output2/fraud_correlation.png')
        print(f"Đã lưu biểu đồ tương quan với is_fraud vào 'output2/fraud_correlation.png'")

# Bước 5: Làm sạch dữ liệu
print("\nĐang thực hiện làm sạch dữ liệu...")

# 5.1 Xử lý dữ liệu thiếu
if len(missing_df) > 0:
    # Với các cột số, điền giá trị thiếu bằng median
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        if df[col].isnull().sum() > 0:
            median_value = df[col].median()
            df[col].fillna(median_value, inplace=True)
            print(f"Đã điền giá trị thiếu trong cột {col} bằng median: {median_value}")

    # Với các cột phân loại, điền giá trị thiếu bằng mode
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].isnull().sum() > 0:
            mode_value = df[col].mode()[0]
            df[col].fillna(mode_value, inplace=True)
            print(f"Đã điền giá trị thiếu trong cột {col} bằng mode: {mode_value}")

# 5.2 Xử lý giá trị ngoại lai (outliers) cho các đặc trưng số
# Trong phát hiện gian lận, outliers có thể là dấu hiệu quan trọng, nên thay vì loại bỏ,
# chúng ta có thể xử lý bằng cách cắt ngưỡng (capping)
print("\nXử lý giá trị ngoại lai cho các cột số...")
for col in numeric_columns:
    # Tính IQR (Interquartile Range)
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    # Xác định ngưỡng trên và dưới
    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR

    # Đếm số lượng outliers
    outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
    if outliers_count > 0:
        print(f"Phát hiện {outliers_count} giá trị ngoại lai trong cột {col}")

        # Cắt ngưỡng các giá trị ngoại lai
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        print(f"Đã xử lý giá trị ngoại lai trong cột {col}")

# Kiểm tra sau khi làm sạch
print("\nKiểm tra dữ liệu sau khi làm sạch:")
print(f"Số lượng giao dịch: {df.shape[0]}")
print(f"Số lượng đặc trưng: {df.shape[1]}")
print(f"Còn lại giá trị thiếu: {df.isnull().sum().sum()}")

# Lưu dữ liệu đã làm sạch
print("\nĐang lưu dữ liệu đã làm sạch...")
df.to_csv('output2/cleaned_data.csv', index=False)
print(f"Đã lưu dữ liệu đã làm sạch vào 'output2/cleaned_data.csv'")

print("\nHoàn thành Giai đoạn 2: Làm sạch và phân tích dữ liệu!")