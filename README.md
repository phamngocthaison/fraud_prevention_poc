# HỆ THỐNG PHÁT HIỆN VÀ ĐÁNH GIÁ RỦI RO GIAN LẬN

Hệ thống sử dụng học máy để phát hiện giao dịch gian lận và đánh giá mức độ rủi ro. Demo được thiết kế để trình bày trong seminar chủ đề "Fraud Detection/Fraud Rating System".

## Cấu trúc dự án

Dự án được chia thành các module độc lập để dễ dàng demo từng giai đoạn:

1. **Tải và kết hợp dữ liệu** (`1-data-loading.py`): Đọc dữ liệu giao dịch và nhãn gian lận từ file JSON
2. **Làm sạch dữ liệu** (`2-data-cleaning.py`): Xử lý dữ liệu thiếu, giá trị ngoại lai và phân tích dữ liệu
3. **Tiền xử lý và tạo đặc trưng** (`3-feature-engineering.py`): Tạo đặc trưng mới và chuẩn hóa dữ liệu
4. **Huấn luyện mô hình** (`4-model-training.py`): Huấn luyện và đánh giá các mô hình phát hiện gian lận
5. **Hệ thống đánh giá rủi ro** (`5-risk-rating.py`): Xây dựng hệ thống phân loại và đánh giá mức độ rủi ro
6. **Ứng dụng demo** (`6-fraud-detection-app.py`): Giao diện trực quan để tương tác với hệ thống

## Yêu cầu hệ thống

### Môi trường Python
- Python 3.7 trở lên
- Các thư viện cần thiết được liệt kê trong `requirements.txt`

### Cài đặt

```bash
# Tạo môi trường ảo
python -m venv fraud-detection-venv

# Kích hoạt môi trường ảo
# Windows
fraud-detection-venv\Scripts\activate
# Linux/Mac
source fraud-detection-venv/bin/activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

## Chạy demo

### Chạy từng bước

Bạn có thể chạy từng bước riêng biệt để demo quy trình chi tiết:

```bash
# Bước 1: Tải và kết hợp dữ liệu
python 1-data-loading.py

# Bước 2: Làm sạch dữ liệu
python 2-data-cleaning.py

# Bước 3: Tiền xử lý và tạo đặc trưng
python 3-feature-engineering.py

# Bước 4: Huấn luyện mô hình
python 4-model-training.py

# Bước 5: Hệ thống đánh giá rủi ro
python 5-risk-rating.py

# Bước 6: Chạy ứng dụng demo
streamlit run 6-fraud-detection-app.py
```

### Chạy toàn bộ quy trình

Để chạy tất cả các bước một cách tự động, sử dụng script `run-all.py`:

```bash
# Chạy toàn bộ quy trình
python run-all.py

# Chỉ chạy ứng dụng demo (giả sử đã chạy các bước trước đó)
python run-all.py --app

# Bỏ qua bước huấn luyện mô hình (sử dụng mô hình có sẵn)
python run-all.py --skip-training

# Bắt đầu từ một bước cụ thể
python run-all.py --start-from 3

# Chỉ chạy một bước cụ thể
python run-all.py --only 4
```

## Dữ liệu

Hệ thống sử dụng bộ dữ liệu từ Kaggle: [Transactions Fraud Datasets](https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets)

### Cấu trúc dữ liệu

- `train_transaction.csv`: Dữ liệu giao dịch
- `train_fraud_labels.json`: Nhãn gian lận cho các giao dịch

### Thư mục đầu ra

Các kết quả phân tích và mô hình được lưu trong thư mục `output/`:

- `combined_data.csv`: Dữ liệu đã kết hợp
- `cleaned_data.csv`: Dữ liệu đã làm sạch
- `best_model.pkl`: Mô hình phát hiện gian lận tốt nhất
- `preprocessor.pkl`: Bộ tiền xử lý dữ liệu
- `risk_rating_system.pkl`: Hệ thống đánh giá rủi ro
- `fraud_risk_assessment_results.csv`: Kết quả đánh giá rủi ro
- Các biểu đồ và hình ảnh trực quan hóa

## Các tính năng chính của ứng dụng demo

1. **Tổng quan**: Giới thiệu về hệ thống và kiến trúc
2. **Phân tích dữ liệu**: Đánh giá hiệu suất mô hình qua các biểu đồ
3. **Mẫu giao dịch**: Xem và phân tích các giao dịch mẫu với đánh giá rủi ro
4. **Đánh giá thủ công**: Nhập thông tin giao dịch mới và nhận kết quả đánh giá
5. **Hướng dẫn sử dụng**: Chi tiết cách sử dụng hệ thống

## Danh mục rủi ro

Hệ thống phân loại giao dịch thành 4 danh mục rủi ro:

1. **Rủi ro thấp (0-20)**: Cho phép giao dịch tự động
2. **Rủi ro trung bình (21-50)**: Yêu cầu xác thực bổ sung (OTP, sinh trắc học)
3. **Rủi ro cao (51-80)**: Chuyển cho nhân viên kiểm tra thủ công
4. **Rủi ro rất cao (81-100)**: Tạm dừng giao dịch và liên hệ khách hàng

## Các file lịch sử giao dịch

Ứng dụng demo cho phép bạn xem các giao dịch mẫu với phân loại rủi ro, giúp hiểu rõ cách hệ thống hoạt động. Đồng thời, bạn có thể tạo và đánh giá các giao dịch mới để kiểm thử khả năng phát hiện gian lận của hệ thống.

## Hướng phát triển

- Thêm mô hình dựa trên học sâu (deep learning) để phát hiện các mẫu gian lận phức tạp
- Tích hợp phân tích hành vi người dùng (behavioral analytics)
- Xây dựng hệ thống học liên tục (continuous learning) để cập nhật mô hình với dữ liệu mới
- Phát triển API để tích hợp vào các hệ thống thanh toán trực tuyến

## Liên hệ

Nếu có bất kỳ câu hỏi hoặc góp ý nào, vui lòng liên hệ:
- Email: [email của bạn]

---

Tác giả: [Tên của bạn]
Ngày: [Ngày tháng năm]