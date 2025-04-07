import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from scipy import sparse
import os
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing

print("GIAI ĐOẠN 3: TIỀN XỬ LÝ VÀ TẠO ĐẶC TRƯNG")
print("-" * 50)

# Tạo thư mục output nếu chưa tồn tại
if not os.path.exists('output3'):
    os.makedirs('output3')

# Đọc dữ liệu đã làm sạch từ bước trước
print("Đang đọc dữ liệu đã làm sạch...")
df = pd.read_csv('output2/cleaned_data.csv')
print(f"Đã đọc {df.shape[0]} giao dịch với {df.shape[1]} thuộc tính")

# Bước 1: Phân chia đặc trưng và nhãn
print("\nPhân chia đặc trưng và nhãn...")
X = df.drop(['is_fraud'], axis=1)
y = df['is_fraud']

# Lưu ID giao dịch nếu có để tham khảo sau này
transaction_id = None
if 'transaction_id' in X.columns:
    transaction_id = X['transaction_id']
    X = X.drop(['transaction_id'], axis=1)

print(f"Số lượng đặc trưng: {X.shape[1]}")
print(f"Phân bố nhãn: {y.value_counts().to_dict()}")

# Bước 2: Xác định các cột số và phân loại
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"\nSố lượng đặc trưng số: {len(numeric_features)}")
print(f"Các đặc trưng số: {numeric_features}")
print(f"\nSố lượng đặc trưng phân loại: {len(categorical_features)}")
print(f"Các đặc trưng phân loại: {categorical_features}")

# Bước 3: Tạo đặc trưng mới (Feature Engineering)
print("\nĐang tạo các đặc trưng mới...")

# Tạo đặc trưng mới một cách hiệu quả
if 'amount' in numeric_features:
    X['log_amount'] = np.log1p(X['amount'])
    numeric_features.append('log_amount')
    print("Đã tạo đặc trưng: log_amount = log(amount + 1)")

if 'day_of_week' in numeric_features:
    X['is_weekend'] = (X['day_of_week'] >= 5).astype(int)
    numeric_features.append('is_weekend')
    print("Đã tạo đặc trưng: is_weekend")

if 'hour' in numeric_features:
    X['night_transaction'] = ((X['hour'] >= 22) | (X['hour'] <= 5)).astype(int)
    numeric_features.append('night_transaction')
    print("Đã tạo đặc trưng: night_transaction")

# Bước 4: Phân chia dữ liệu thành tập huấn luyện và kiểm tra
print("\nPhân chia dữ liệu thành tập huấn luyện và kiểm tra...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Kích thước tập huấn luyện: {X_train.shape}")
print(f"Kích thước tập kiểm tra: {X_test.shape}")

# Bước 5: Tạo pipeline tiền xử lý với sparse matrices
print("\nTạo pipeline tiền xử lý...")

# Định nghĩa các hàm chuyển đổi an toàn
def convert_to_float(X):
    return X.astype(float)

def convert_to_str(X):
    return X.astype(str)

# Tạo bộ tiền xử lý cho đặc trưng số và phân loại
numeric_transformer = Pipeline(steps=[
    ('converter', FunctionTransformer(convert_to_float)),
    ('scaler', StandardScaler())
])

# Giới hạn các giá trị trong categorical để giảm số lượng đặc trưng
categorical_transformer = Pipeline(steps=[
    ('converter', FunctionTransformer(convert_to_str)),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True, max_categories=10))
])

# Kết hợp các bộ tiền xử lý
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'  # Bỏ qua các cột không được xử lý
)

# Kiểm tra số lượng giá trị duy nhất trong các cột categorical
print("Kiểm tra số lượng giá trị duy nhất trong các cột categorical:")
for col in categorical_features:
    unique_values = X_train[col].nunique()
    print(f"Cột {col}: {unique_values} giá trị duy nhất")

# Ước tính số lượng đặc trưng sau one-hot encoding
estimated_features = len(numeric_features)
for col in categorical_features:
    unique_values = min(X_train[col].nunique(), 10)  # Giới hạn tối đa 10 giá trị
    estimated_features += unique_values

print(f"Ước tính số lượng đặc trưng sau khi xử lý: {estimated_features}")

# Fit preprocessor trên toàn bộ dữ liệu huấn luyện trước
print("Fitting preprocessor trên toàn bộ dữ liệu huấn luyện...")
preprocessor.fit(X_train)

# Hàm xử lý batch
def process_batch(batch_data, preprocessor):
    return preprocessor.transform(batch_data)

# Áp dụng bộ tiền xử lý vào dữ liệu huấn luyện theo batch với parallel processing
print("Áp dụng tiền xử lý cho dữ liệu huấn luyện...")
batch_size = 50000  # Tăng kích thước batch
num_batches = (len(X_train) + batch_size - 1) // batch_size

# Sử dụng parallel processing
n_jobs = multiprocessing.cpu_count() - 1  # Sử dụng tất cả CPU trừ 1
print(f"Sử dụng {n_jobs} CPU cores cho parallel processing")

# Xử lý dữ liệu huấn luyện
batches = [X_train.iloc[i*batch_size:min((i+1)*batch_size, len(X_train))] 
          for i in range(num_batches)]
X_train_transformed_list = Parallel(n_jobs=n_jobs)(
    delayed(process_batch)(batch, preprocessor) for batch in tqdm(batches)
)
X_train_transformed = sparse.vstack(X_train_transformed_list)

# Xử lý dữ liệu test
batches = [X_test.iloc[i*batch_size:min((i+1)*batch_size, len(X_test))] 
          for i in range((len(X_test) + batch_size - 1) // batch_size)]
X_test_transformed_list = Parallel(n_jobs=n_jobs)(
    delayed(process_batch)(batch, preprocessor) for batch in tqdm(batches)
)
X_test_transformed = sparse.vstack(X_test_transformed_list)

# Lưu bộ tiền xử lý
print("Lưu bộ tiền xử lý...")
with open('output3/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

# Lưu dữ liệu đã biến đổi dưới dạng sparse matrix
print("\nLưu dữ liệu đã biến đổi...")
sparse.save_npz('output3/X_train_transformed.npz', X_train_transformed)
sparse.save_npz('output3/X_test_transformed.npz', X_test_transformed)
np.save('output3/y_train.npy', y_train.values)
np.save('output3/y_test.npy', y_test.values)

# Lưu thông tin đặc trưng
print("\nLưu thông tin đặc trưng...")
feature_names = {
    'numeric_features': numeric_features,
    'categorical_features': categorical_features
}
with open('output3/feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

# Lưu indices test
test_indices = X_test.index
np.save('output3/test_indices.npy', test_indices)

if transaction_id is not None:
    test_transaction_id = transaction_id.iloc[test_indices]
    test_transaction_id.to_csv('output3/test_transaction_id.csv', index=False)

print("\nHoàn thành Giai đoạn 3: Tiền xử lý và tạo đặc trưng!")