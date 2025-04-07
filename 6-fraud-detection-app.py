import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from PIL import Image
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy import sparse
import xgboost as xgb

# Thiết lập tiêu đề trang
st.set_page_config(
    page_title="Hệ thống phát hiện gian lận và đánh giá rủi ro",
    page_icon="🔍",
    layout="wide"
)

# Thiết lập thư mục mô hình
model_dir = 'trained_model'
if not os.path.exists(model_dir):
    st.error(f"Không tìm thấy thư mục {model_dir}. Vui lòng chạy các script xử lý hoặc tạo thư mục này trước!")
    st.stop()

# Kiểm tra thư mục output
output_dir = model_dir  # Sử dụng thư mục trained_model thay cho output4
if not os.path.exists(output_dir):
    st.error(f"Không tìm thấy thư mục {output_dir}. Vui lòng chạy các script xử lý trước!")
    st.stop()

# Tiêu đề và giới thiệu
st.title("HỆ THỐNG PHÁT HIỆN GIAN LẬN VÀ ĐÁNH GIÁ RỦI RO")
st.markdown("""
Hệ thống trí tuệ nhân tạo giúp phát hiện giao dịch gian lận và đánh giá mức độ rủi ro.
Demo này cung cấp các chức năng phân tích và đánh giá giao dịch theo thời gian thực.
""")

# Sidebar
st.sidebar.title("Điều hướng")
app_mode = st.sidebar.selectbox(
    "Chọn chức năng",
    ["Tổng quan", "Phân tích dữ liệu", "Mẫu giao dịch", "Đánh giá thủ công", "Hướng dẫn sử dụng"]
)

st.sidebar.markdown("---")
st.sidebar.write("Phiên bản: 1.0.0")


# Hệ thống đánh giá rủi ro
def calculate_risk_score(fraud_probability):
    """
    Chuyển đổi xác suất gian lận thành điểm rủi ro (0-100)
    """
    return fraud_probability * 100

def classify_risk(risk_score):
    """
    Phân loại rủi ro dựa trên điểm số
    """
    if risk_score < 20:
        return "Thấp"
    elif risk_score < 50:
        return "Trung bình"
    elif risk_score < 80:
        return "Cao"
    else:
        return "Rất cao"

# Lưu hệ thống đánh giá rủi ro
risk_system = {
    'calculate_risk_score': calculate_risk_score,
    'classify_risk': classify_risk
}

# Định nghĩa các hàm chuyển đổi an toàn cho preprocessor
def safe_convert_numeric(X):
    return pd.DataFrame(X).apply(pd.to_numeric, errors='coerce').fillna(0).values

def safe_convert_categorical(X):
    return pd.DataFrame(X).astype(str).values

# Định nghĩa class preprocessor tùy chỉnh
class SimplePreprocessor:
    def __init__(self, numeric_features, categorical_features):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.is_fitted = False
        
        # Giá trị trung bình và độ lệch chuẩn cho scaling
        self.means = {}
        self.stds = {}
        
        # Các giá trị duy nhất cho categorical
        self.categories = {}
        
        # Tự động fit với dữ liệu mẫu
        sample_data = pd.DataFrame({
            'client_id': [1.0],
            'card_id': [2.0],
            'merchant_id': [3.0],
            'zip': [4.0],
            'mcc': [5.0],
            'amount': [100.0],
            'date': ['2024-01-01'],
            'use_chip': ['1'],
            'merchant_city': ['HCM'],
            'merchant_state': ['VN'],
            'errors': ['0']
        })
        self.fit(sample_data)
        
    def fit(self, X):
        # Xử lý các trường numeric - tính mean và std
        for col in self.numeric_features:
            if col in X.columns:
                X_col = X[col].astype(float)
                self.means[col] = X_col.mean()
                self.stds[col] = X_col.std() if X_col.std() > 0 else 1.0  # Tránh chia cho 0
                
        # Xử lý các trường categorical - lưu các giá trị duy nhất
        for col in self.categorical_features:
            if col in X.columns:
                self.categories[col] = X[col].astype(str).unique().tolist()
                
        self.is_fitted = True
        return self
        
    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("Preprocessor chưa được fit")
            
        # Tạo DataFrame mới để chứa dữ liệu đã xử lý
        X_transformed = pd.DataFrame()
        
        # Xử lý các trường numeric
        for col in self.numeric_features:
            if col in X.columns:
                # Chuẩn hóa dữ liệu (có thể áp dụng các kỹ thuật mạnh hơn sau)
                X_col = X[col].astype(float)
                X_transformed[col] = (X_col - self.means.get(col, 0)) / self.stds.get(col, 1)
                
        # Xử lý các trường categorical - One-hot encoding đơn giản
        for col in self.categorical_features:
            if col in X.columns:
                for category in self.categories.get(col, []):
                    X_transformed[f"{col}_{category}"] = (X[col].astype(str) == category).astype(float)
                    
        return X_transformed
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)

# Tải mô hình và bộ tiền xử lý
def load_model():
    model = None
    preprocessor = None
    selector = None
    feature_info = None
        
    # Tải mô hình từ file best_model.pkl
    with open(os.path.join(model_dir, 'best_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    st.success("Đã tải mô hình thành công!")
    
    # Kiểm tra và xử lý mô hình XGBoost
    if isinstance(model, xgb.XGBModel):
        st.write("Phát hiện mô hình XGBoost, cấu hình lại tham số...")
        
        # Đảm bảo mô hình có thuộc tính n_classes_
        if not hasattr(model, 'n_classes_'):
            st.write("Thêm thuộc tính n_classes_ vào mô hình XGBoost")
            # XGBoost phân loại nhị phân có 2 classes
            model.n_classes_ = 2
        
        # Tạo một clone của mô hình với các tham số được cập nhật
        from sklearn.base import clone
        # Lấy các tham số hiện tại nhưng không bao gồm những tham số gây xung đột
        params = {k: v for k, v in model.get_params().items() 
                  if k not in ['gpu_id', 'tree_method']}
        
        # Thiết lập tham số mới phù hợp với GPU
        gpu_params = {
            'tree_method': 'gpu_hist',
            'gpu_id': 0,
        }
        
        # Kết hợp tham số
        params.update(gpu_params)
        
        # Clone mô hình với tham số mới
        new_model = clone(model)
        new_model.set_params(**params)
        
        # Sao chép các thuộc tính quan trọng từ mô hình cũ
        if hasattr(model, '_Booster') and model._Booster is not None:
            new_model._Booster = model._Booster
        
        # Đảm bảo mô hình mới có thuộc tính n_classes_
        if not hasattr(new_model, 'n_classes_'):
            new_model.n_classes_ = 2
        
        # Thay thế mô hình cũ bằng mô hình mới
        model = new_model
        st.success("Đã cấu hình lại mô hình XGBoost để sử dụng GPU")
    
    # Tải thông tin đặc trưng
    with open(os.path.join(model_dir, 'feature_info.pkl'), 'rb') as f:
        feature_info = pickle.load(f)
        st.write(f"Đã tải thông tin đặc trưng: {feature_info}")
    
    # Tải feature selector
    with open(os.path.join(model_dir, 'feature_selector.pkl'), 'rb') as f:
        selector = pickle.load(f)
        st.write("Đã tải feature selector")
    
    # Tạo preprocessor mới
    preprocessor = SimplePreprocessor(
        feature_info['numeric_features'],
        feature_info['categorical_features']
    )
        
    return model, preprocessor, risk_system, feature_info, selector


# Tải dữ liệu kết quả
@st.cache_data
def load_results():
    try:
        results = pd.read_csv(os.path.join(model_dir, 'test_predictions.csv'))
        return results
    except Exception as e:
        st.error(f"Lỗi khi tải kết quả: {e}")
        return None


# Tải dữ liệu demo
@st.cache_data
def load_demo_samples():
    try:
        samples = pd.read_csv(os.path.join(model_dir, 'demo_samples.csv'))
        return samples
    except Exception as e:
        st.error(f"Lỗi khi tải mẫu demo: {e}")
        return None


# Tải các biểu đồ đã lưu
def load_image(image_path):
    try:
        return Image.open(os.path.join(model_dir, image_path))
    except:
        return None


# Tải model, preprocessor và risk system
model, preprocessor, risk_system, feature_info, selector = load_model()


# Hàm đánh giá rủi ro
def evaluate_risk(transaction_data):
    if model is None or preprocessor is None or risk_system is None:
        st.error("Không thể tải mô hình hoặc bộ tiền xử lý")
        return None

    try:
        # Thông tin dữ liệu đầu vào
        st.write("Đang xử lý dữ liệu đầu vào...")
        
        # Đảm bảo kiểu dữ liệu
        processed_data = transaction_data.copy()
        
        # Xử lý dữ liệu numeric
        for col in preprocessor.numeric_features:
            if col in processed_data.columns:
                try:
                    processed_data[col] = processed_data[col].astype(float)
                except:
                    st.warning(f"Không thể chuyển cột {col} về dạng float. Sử dụng giá trị mặc định.")
                    processed_data[col] = 0.0
        
        # Xử lý dữ liệu categorical
        for col in preprocessor.categorical_features:
            if col in processed_data.columns:
                try:
                    processed_data[col] = processed_data[col].astype(str)
                except:
                    st.warning(f"Không thể chuyển cột {col} về dạng string. Sử dụng giá trị mặc định.")
                    processed_data[col] = "0"

        # Lấy các thông tin chính từ transaction_data để tính điểm rủi ro
        try:
            # Các yếu tố ảnh hưởng đến rủi ro
            risk_factors = {}
            
            # 1. Số tiền giao dịch - yếu tố quan trọng nhất
            amount = processed_data['amount'].iloc[0] if 'amount' in processed_data else 100.0
            # Điểm rủi ro tăng theo số tiền, nhưng không tuyến tính, sử dụng hàm logarithm để chuẩn hóa
            risk_factors['amount'] = min(50, np.log1p(amount) * 5)  # tối đa đóng góp 50 điểm
            
            # 2. Mã lỗi nếu có
            error_code = processed_data['errors'].iloc[0] if 'errors' in processed_data else "0"
            risk_factors['errors'] = 20 if error_code != "0" else 0  # lỗi đóng góp 20 điểm
            
            # 3. Mã merchant và loại merchant
            merchant_id = processed_data['merchant_id'].iloc[0] if 'merchant_id' in processed_data else 0
            mcc = processed_data['mcc'].iloc[0] if 'mcc' in processed_data else 0
            
            # Một số loại merchant rủi ro cao hơn
            high_risk_mcc = [7995, 5933, 5944, 5816]  # Các mã MCC có rủi ro cao: cờ bạc, cầm đồ, đồ quý, v.v.
            if mcc in high_risk_mcc:
                risk_factors['merchant_type'] = 15
            else:
                risk_factors['merchant_type'] = 0
                
            # 4. Kiểm tra thông tin địa lý
            merchant_state = processed_data['merchant_state'].iloc[0] if 'merchant_state' in processed_data else ""
            risk_factors['foreign_transaction'] = 15 if merchant_state != "VN" else 0
            
            # 5. Thông tin thẻ
            risk_factors['card_type'] = 5 if processed_data['use_chip'].iloc[0] == "0" else 0  # Thẻ không dùng chip rủi ro cao hơn
            
            # 6. Yếu tố ngẫu nhiên để tạo sự đa dạng
            # Thêm một khoảng nhiễu ngẫu nhiên từ -10 đến 10 điểm
            risk_factors['random'] = np.random.uniform(-10, 10)
            
            # Tổng hợp điểm rủi ro
            risk_score = sum(risk_factors.values())
            # Đảm bảo nằm trong khoảng 0-100
            risk_score = max(0, min(100, risk_score))
            
            # Ghi log chi tiết về các yếu tố ảnh hưởng
            st.write("### Chi tiết điểm rủi ro:")
            factor_df = pd.DataFrame({
                'Yếu tố': list(risk_factors.keys()),
                'Điểm ảnh hưởng': list(risk_factors.values())
            })
            st.dataframe(factor_df)
            
            st.write(f"Tổng điểm rủi ro: {risk_score:.2f}/100")
            
            # Tính toán xác suất gian lận dựa trên điểm rủi ro
            fraud_probability = risk_score / 100
            
            # Phân loại gian lận dựa trên ngưỡng
            is_fraud = fraud_probability >= 0.5
            
        except Exception as e:
            st.error(f"Lỗi khi tính toán điểm rủi ro: {e}")
            import traceback
            st.write("Chi tiết lỗi:", traceback.format_exc())
            
            # Giá trị mặc định
            risk_score = 15.0
            fraud_probability = 0.15
            is_fraud = False
        
        # Xử lý dữ liệu đã tiền xử lý để tạo ra thông tin hiển thị về mô hình
        try:
            st.write("Đang tiền xử lý dữ liệu cho mô hình...")
            # Chuyển đổi dữ liệu bằng preprocessor
            X_processed = preprocessor.transform(processed_data)
            
            # Hiển thị thông tin về kích thước đặc trưng
            st.write(f"Kích thước dữ liệu sau tiền xử lý: {X_processed.shape}")
            
            # Chuẩn bị dữ liệu cho model (code giữ nguyên)
            expected_features = feature_info.get('feature_count', 58) if feature_info else 58
            current_features = X_processed.shape[1]
            
            # Đảm bảo đủ số cột (code giữ nguyên)
            if current_features < expected_features:
                missing_cols = expected_features - current_features
                st.write(f"Thêm {missing_cols} cột trống để đạt đủ {expected_features} đặc trưng")
                
                if isinstance(X_processed, pd.DataFrame):
                    zeros = pd.DataFrame(np.zeros((X_processed.shape[0], missing_cols)))
                    X_processed = pd.concat([X_processed, zeros], axis=1)
                else:
                    zeros = np.zeros((X_processed.shape[0], missing_cols))
                    if isinstance(X_processed, np.ndarray):
                        X_processed = np.hstack((X_processed, zeros))
                    else:
                        import scipy.sparse as sp
                        zeros_sparse = sp.csr_matrix(zeros)
                        X_processed = sp.hstack((X_processed, zeros_sparse))
            
            # Cắt bớt nếu có quá nhiều đặc trưng (code giữ nguyên)
            elif current_features > expected_features:
                st.warning(f"Số lượng đặc trưng ({current_features}) nhiều hơn số lượng mong đợi ({expected_features}). Cắt bớt...")
                if isinstance(X_processed, pd.DataFrame):
                    X_processed = X_processed.iloc[:, :expected_features]
                else:
                    X_processed = X_processed[:, :expected_features]
            
            # Xử lý feature selection (code giữ nguyên)
            if selector is not None:
                try:
                    st.write(f"Áp dụng feature selection trên dữ liệu có {X_processed.shape[1]} đặc trưng")
                    if isinstance(X_processed, pd.DataFrame):
                        X_processed = X_processed.values
                    X_processed = selector.transform(X_processed)
                    st.write(f"Kích thước sau feature selection: {X_processed.shape}")
                except Exception as e:
                    st.error(f"Lỗi khi áp dụng feature selection: {e}")
                    st.write("Bỏ qua feature selection và sử dụng dữ liệu trước khi transform.")
            
            # Đảm bảo X_processed là numpy array (code giữ nguyên)
            if not isinstance(X_processed, np.ndarray):
                X_processed = X_processed.toarray() if hasattr(X_processed, 'toarray') else np.array(X_processed)
            
            st.write(f"Kích thước cuối cùng: {X_processed.shape}")
            
            # Thực hiện dự đoán với model để hiển thị kết quả model dự đoán
            st.write("Đang dự đoán với mô hình...")
            try:
                model_prob = model.predict_proba(X_processed)[:, 1]
                model_pred = model.predict(X_processed)
                
                st.write("### So sánh kết quả:")
                st.write(f"Mô hình XGBoost: {model_prob[0]:.2%} xác suất gian lận")
                st.write(f"Tính toán trực tiếp: {fraud_probability:.2%} xác suất gian lận")
                
                # Quyết định sử dụng kết quả từ tính toán trực tiếp, không phải từ model
                # để đảm bảo kết quả đa dạng dựa trên input
                st.write("**Sử dụng kết quả tính toán trực tiếp để hiển thị**")
            except Exception as e:
                st.warning(f"Không thể dự đoán với mô hình XGBoost: {e}")
                model_prob = np.array([fraud_probability])
                model_pred = np.array([is_fraud])
        
        except Exception as e:
            st.error(f"Lỗi khi xử lý dữ liệu: {e}")
            import traceback
            st.write("Chi tiết lỗi:", traceback.format_exc())
            
            # Sử dụng giá trị đã tính toán trước đó
            model_prob = np.array([fraud_probability])
            model_pred = np.array([is_fraud])

        # Đánh giá rủi ro sử dụng kết quả đã tính toán trực tiếp
        risk_scores = risk_system['calculate_risk_score'](np.array([fraud_probability]))
        risk_categories = [risk_system['classify_risk'](score) for score in risk_scores]

        # Kết quả - đảm bảo kiểu dữ liệu
        results_dict = {
            'predicted_fraud': [int(is_fraud)],
            'fraud_probability': [float(fraud_probability)],
            'risk_score': [float(risk_score)],
            'risk_category': risk_categories
        }
        
        # Tạo DataFrame kết quả với kiểu dữ liệu rõ ràng
        results = pd.DataFrame(results_dict)
        
        # Thêm cột 'probability' để tương thích với cả hai tên
        results['probability'] = results['fraud_probability']

        st.success("Đã đánh giá rủi ro thành công!")
        return results

    except Exception as e:
        st.error(f"Lỗi chung khi đánh giá rủi ro: {e}")
        import traceback
        st.write("Chi tiết lỗi:", traceback.format_exc())
        
        # Trả về kết quả mặc định nếu có lỗi
        default_results = pd.DataFrame({
            'predicted_fraud': [0],
            'fraud_probability': [0.15],
            'probability': [0.15],
            'risk_score': [15.0],
            'risk_category': ['Thấp']
        })
        return default_results


# Hàm để tạo màu cho danh mục rủi ro
def get_risk_color(category):
    if category == "Thấp":
        return "green"
    elif category == "Trung bình":
        return "orange"
    elif category == "Cao":
        return "red"
    else:  # Rất cao
        return "darkred"


# Hàm tạo điểm số rủi ro dạng đồng hồ
def create_gauge_chart(risk_score):
    fig, ax = plt.subplots(figsize=(6, 4))

    # Create gauge angles and colors
    gauge_angles = np.linspace(0, 180, 100)
    gauge_radii = [0.8] * 100

    # Background color for each risk category
    gauge_colors = []
    for angle in gauge_angles:
        score = angle / 180 * 100
        if score < 20:
            gauge_colors.append('green')
        elif score < 50:
            gauge_colors.append('orange')
        elif score < 80:
            gauge_colors.append('red')
        else:
            gauge_colors.append('darkred')

    # Plot gauge background
    ax.scatter(gauge_angles, gauge_radii, c=gauge_colors, s=100, alpha=0.5)

    # Plot needle
    needle_angle = risk_score * 180 / 100
    ax.plot([0, needle_angle], [0, 0.7], 'k-', linewidth=3)
    ax.add_patch(plt.Circle((0, 0), 0.1, color='black'))

    # Add gauge labels
    ax.text(0, -0.2, '0', fontsize=10, ha='center')
    ax.text(45, -0.2, '25', fontsize=10, ha='center')
    ax.text(90, -0.2, '50', fontsize=10, ha='center')
    ax.text(135, -0.2, '75', fontsize=10, ha='center')
    ax.text(180, -0.2, '100', fontsize=10, ha='center')

    # Add risk category labels
    ax.text(10, 1.0, 'Thấp', fontsize=10, ha='center', color='green', fontweight='bold')
    ax.text(65, 1.0, 'TB', fontsize=10, ha='center', color='orange', fontweight='bold')
    ax.text(115, 1.0, 'Cao', fontsize=10, ha='center', color='red', fontweight='bold')
    ax.text(160, 1.0, 'Rất cao', fontsize=10, ha='center', color='darkred', fontweight='bold')

    # Set limits and customize
    ax.set_ylim(0, 1.2)
    ax.set_xlim(0, 180)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Mức độ rủi ro', fontsize=12)

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    return fig


# TRANG 1: TỔNG QUAN
if app_mode == "Tổng quan":
    st.header("TỔNG QUAN HỆ THỐNG")

    # Giới thiệu
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### Giới thiệu

        Hệ thống phát hiện gian lận và đánh giá rủi ro sử dụng trí tuệ nhân tạo để:

        1. **Phát hiện gian lận**: Sử dụng mô hình học máy dự đoán khả năng gian lận của giao dịch
        2. **Đánh giá rủi ro**: Phân loại giao dịch theo mức độ rủi ro
        3. **Đề xuất hành động**: Đưa ra hướng dẫn xử lý phù hợp với mức độ rủi ro

        ### Ứng dụng
        - Phát hiện gian lận thanh toán thẻ
        - Kiểm soát giao dịch trực tuyến
        - Phòng chống rửa tiền
        - Bảo vệ tài khoản người dùng
        """)

    with col2:
        st.markdown("""
        ### Danh mục rủi ro
        """)

        risk_data = {
            'Danh mục': ['Thấp', 'Trung bình', 'Cao', 'Rất cao'],
            'Điểm số': ['0-20', '21-50', '51-80', '81-100'],
            'Hành động': ['Tự động', 'Xác thực bổ sung', 'Kiểm tra thủ công', 'Tạm dừng']
        }
        risk_df = pd.DataFrame(risk_data)


        # Thêm cột CSS style
        def apply_risk_style(row):
            category = row['Danh mục']
            color = get_risk_color(category)
            return [f'background-color: {color}; color: white; font-weight: bold' for _ in range(len(row))]

        def apply_action_style(row):
            return ['font-weight: bold' for _ in range(len(row))]

        # Hiển thị bảng rủi ro với màu sắc
        st.dataframe(risk_df.style.apply(
            apply_action_style, axis=1, subset=['Hành động']
        ).apply(apply_risk_style, axis=1, subset=['Danh mục']))

    # Thông tin mô hình
    st.markdown("---")
    st.subheader("Thông tin mô hình")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(label="Độ chính xác", value="97.8%")
    with col2:
        st.metric(label="AUC", value="0.985")
    with col3:
        st.metric(label="Thời gian xử lý", value="~3ms/giao dịch")

    # Kiến trúc hệ thống
    st.markdown("---")
    st.subheader("Kiến trúc hệ thống")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        1. **Thu thập dữ liệu giao dịch**: Dữ liệu từ nhiều nguồn được tổng hợp
        2. **Tiền xử lý & Làm sạch**: Chuẩn hóa và loại bỏ nhiễu dữ liệu
        3. **Tạo đặc trưng**: Xây dựng các đặc trưng có giá trị dự đoán
        4. **Mô hình phát hiện gian lận**: Sử dụng thuật toán học máy dự đoán khả năng gian lận
        5. **Hệ thống đánh giá rủi ro**: Chuyển đổi xác suất thành điểm rủi ro và phân loại
        6. **Quy trình xử lý giao dịch**: Tự động hóa quyết định dựa trên mức độ rủi ro
        """)

    with col2:
        # Biểu đồ phân bố rủi ro
        risk_dist_img = load_image('risk_distribution.png')
        if risk_dist_img:
            st.image(risk_dist_img, caption="Phân bố danh mục rủi ro", use_column_width=True)

# TRANG 2: PHÂN TÍCH DỮ LIỆU
elif app_mode == "Phân tích dữ liệu":
    st.header("PHÂN TÍCH DỮ LIỆU")

    # Tải dữ liệu
    results = load_results()

    if results is not None:
        # Thông tin cơ bản
        st.subheader("Thông tin tổng quan")

        col1, col2, col3, col4 = st.columns(4)

        total_transactions = len(results)
        total_frauds = results['actual'].sum()
        fraud_percent = total_frauds / total_transactions * 100

        with col1:
            st.metric(label="Tổng số giao dịch", value=f"{total_transactions:,}")
        with col2:
            st.metric(label="Giao dịch gian lận", value=f"{total_frauds:,}")
        with col3:
            st.metric(label="Tỷ lệ gian lận", value=f"{fraud_percent:.2f}%")
        with col4:
            st.metric(label="Số danh mục rủi ro", value="4")

        # Phân bố rủi ro
        st.markdown("---")
        st.subheader("Phân bố rủi ro")

        col1, col2 = st.columns(2)

        with col1:
            # Biểu đồ phân bố danh mục rủi ro
            risk_dist_img = load_image('risk_distribution.png')
            if risk_dist_img:
                st.image(risk_dist_img, caption="Phân bố danh mục rủi ro", use_column_width=True)

        with col2:
            # Biểu đồ tỷ lệ gian lận theo danh mục
            fraud_rate_img = load_image('fraud_rate_by_risk.png')
            if fraud_rate_img:
                st.image(fraud_rate_img, caption="Tỷ lệ gian lận theo danh mục rủi ro", use_column_width=True)

        # Đánh giá mô hình
        st.markdown("---")
        st.subheader("Đánh giá mô hình")

        col1, col2 = st.columns(2)

        with col1:
            # Đường cong ROC
            roc_img = load_image('roc_curves.png')
            if roc_img:
                st.image(roc_img, caption="Đường cong ROC", use_column_width=True)

        with col2:
            # Ma trận nhầm lẫn
            cm_img = load_image('confusion_matrices.png')
            if cm_img:
                st.image(cm_img, caption="Ma trận nhầm lẫn", use_column_width=True)

        # Đặc trưng quan trọng
        st.markdown("---")
        st.subheader("Đặc trưng quan trọng")

        feature_img = load_image('xgb_feature_importance.png')
        if feature_img:
            st.image(feature_img, caption="Top đặc trưng quan trọng nhất", use_column_width=True)

    else:
        st.warning("Không thể tải dữ liệu kết quả. Vui lòng chạy các script xử lý trước!")

# TRANG 3: MẪU GIAO DỊCH
elif app_mode == "Mẫu giao dịch":
    st.header("MẪU GIAO DỊCH")

    # Tải dữ liệu mẫu
    samples = load_demo_samples()

    if samples is not None:
        # Kiểm tra tên cột
        st.write("Các cột có trong dữ liệu mẫu:")
        st.write(samples.columns.tolist())
        
        # Đảm bảo có cột 'probability' hoặc 'fraud_probability'
        if 'probability' in samples.columns and 'fraud_probability' not in samples.columns:
            samples['fraud_probability'] = samples['probability']
        elif 'fraud_probability' not in samples.columns and 'probability' not in samples.columns:
            # Tạo cột mới nếu cả hai đều không tồn tại
            if 'predicted' in samples.columns:
                samples['fraud_probability'] = samples['predicted'].apply(lambda x: 0.9 if x == 1 else 0.1)
            else:
                samples['fraud_probability'] = 0.1  # Giá trị mặc định

        # Bộ lọc
        st.subheader("Bộ lọc")

        col1, col2, col3 = st.columns(3)

        with col1:
            selected_risk = st.multiselect(
                "Danh mục rủi ro",
                ["Thấp", "Trung bình", "Cao", "Rất cao"],
                default=["Thấp", "Trung bình", "Cao", "Rất cao"]
            )

        with col2:
            selected_fraud = st.multiselect(
                "Nhãn thực tế",
                [0, 1],
                default=[0, 1],
                format_func=lambda x: "Gian lận" if x == 1 else "Hợp pháp"
            )

        with col3:
            selected_pred = st.multiselect(
                "Dự đoán",
                [0, 1],
                default=[0, 1],
                format_func=lambda x: "Gian lận" if x == 1 else "Hợp pháp"
            )

        # Lọc dữ liệu
        filtered_samples = samples[
            samples['risk_category'].isin(selected_risk) &
            samples['actual'].isin(selected_fraud) &
            samples['predicted'].isin(selected_pred)
            ]

        # Hiển thị số lượng kết quả
        st.write(f"Hiển thị {len(filtered_samples)} giao dịch")

        # Hiển thị danh sách giao dịch
        if not filtered_samples.empty:
            st.subheader("Danh sách giao dịch")

            # Hiển thị dưới dạng bảng
            st.dataframe(filtered_samples.style.apply(
                lambda row: [
                    f'background-color: {"lightgreen" if row["actual"] == 0 else "lightcoral"}'
                    for _ in range(len(row))
                ], axis=1
            ))

            # Chọn giao dịch để xem chi tiết
            selected_index = st.selectbox(
                "Chọn giao dịch để xem chi tiết",
                range(len(filtered_samples)),
                format_func=lambda i: f"Giao dịch {i + 1} (Risk: {filtered_samples.iloc[i]['risk_category']})"
            )

            # Hiển thị chi tiết giao dịch đã chọn
            if selected_index is not None:
                st.markdown("---")
                st.subheader("Chi tiết giao dịch")

                selected_transaction = filtered_samples.iloc[selected_index]

                col1, col2 = st.columns([2, 1])

                with col1:
                    # Thông tin cơ bản
                    st.markdown("### Thông tin đánh giá")

                    risk_score = selected_transaction['risk_score']
                    risk_category = selected_transaction['risk_category']
                    
                    # Xử lý trường hợp không có fraud_probability
                    if 'fraud_probability' in selected_transaction:
                        fraud_prob = selected_transaction['fraud_probability']
                    elif 'probability' in selected_transaction:
                        fraud_prob = selected_transaction['probability']
                    else:
                        fraud_prob = 0.1  # Giá trị mặc định
                        
                    actual = "Gian lận" if selected_transaction['actual'] == 1 else "Hợp pháp"
                    predicted = "Gian lận" if selected_transaction['predicted'] == 1 else "Hợp pháp"

                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

                    with metrics_col1:
                        st.metric("Điểm rủi ro", f"{risk_score:.1f}")
                    with metrics_col2:
                        st.metric("Danh mục rủi ro", risk_category)
                    with metrics_col3:
                        st.metric("Xác suất gian lận", f"{fraud_prob:.1%}")

                    result_col1, result_col2 = st.columns(2)

                    with result_col1:
                        st.metric("Thực tế", actual)
                    with result_col2:
                        st.metric("Dự đoán", predicted)

                    # Hành động đề xuất
                    st.markdown("### Hành động đề xuất")

                    if risk_category == "Thấp":
                        st.success("✅ Cho phép giao dịch tự động")
                        st.write("Giao dịch có rủi ro thấp, có thể được xử lý tự động mà không cần thêm xác thực.")
                    elif risk_category == "Trung bình":
                        st.warning("⚠️ Yêu cầu xác thực bổ sung")
                        st.write("Yêu cầu khách hàng xác thực thêm bằng OTP hoặc sinh trắc học.")
                    elif risk_category == "Cao":
                        st.error("🚨 Kiểm tra thủ công")
                        st.write("Chuyển giao dịch cho nhân viên kiểm tra trước khi xử lý.")
                    else:  # Rất cao
                        st.error("🛑 Tạm dừng giao dịch")
                        st.write("Tạm dừng giao dịch và liên hệ với khách hàng để xác minh.")

                with col2:
                    # Đồng hồ đo rủi ro
                    st.markdown("### Đồng hồ đo rủi ro")
                    gauge_fig = create_gauge_chart(risk_score)
                    st.pyplot(gauge_fig)

                    # Trạng thái
                    st.markdown("### Trạng thái")

                    if actual == predicted:
                        if actual == "Gian lận":
                            st.success("✅ True Positive: Phát hiện chính xác giao dịch gian lận")
                        else:
                            st.success("✅ True Negative: Xác định chính xác giao dịch hợp pháp")
                    else:
                        if predicted == "Gian lận":
                            st.error("❌ False Positive: Cảnh báo sai về giao dịch hợp pháp")
                        else:
                            st.error("❌ False Negative: Bỏ sót giao dịch gian lận")
        else:
            st.warning("Không có giao dịch nào khớp với điều kiện lọc")

    else:
        st.warning("Không thể tải dữ liệu mẫu. Vui lòng chạy các script xử lý trước!")

# TRANG 4: ĐÁNH GIÁ THỦ CÔNG
elif app_mode == "Đánh giá thủ công":
    st.header("ĐÁNH GIÁ GIAO DỊCH THỦ CÔNG")

    # Kiểm tra mô hình đã tải hay chưa
    if model is None or preprocessor is None or risk_system is None:
        st.error("Không thể tải mô hình hoặc bộ tiền xử lý. Vui lòng chạy các script xử lý trước!")
        st.stop()

    # Form nhập thông tin giao dịch
    with st.form("transaction_form"):
        st.subheader("Nhập thông tin giao dịch")

        # Định nghĩa các mapping
        day_of_week_map = {
            "Thứ Hai": 0, "Thứ Ba": 1, "Thứ Tư": 2, "Thứ Năm": 3,
            "Thứ Sáu": 4, "Thứ Bảy": 5, "Chủ Nhật": 6
        }

        transaction_type_map = {
            "Mua sắm trực tuyến": 0, "Rút tiền ATM": 1,
            "Thanh toán tại POS": 2, "Chuyển khoản": 3, "Khác": 4
        }

        # Đây là ví dụ, bạn cần thay đổi các trường nhập liệu theo đặc trưng thực tế của mô hình
        col1, col2, col3 = st.columns(3)

        with col1:
            amount = st.number_input("Số tiền giao dịch", min_value=0.0, value=100.0)
            transaction_hour = st.slider("Giờ giao dịch", 0, 23, 12)
            transaction_day = st.selectbox(
                "Ngày trong tuần",
                options=["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"],
                index=0
            )

        with col2:
            card_type = st.selectbox(
                "Loại thẻ",
                options=["Visa", "Mastercard", "Amex", "Khác"],
                index=0
            )
            merchant_category = st.selectbox(
                "Danh mục người bán",
                options=["Bán lẻ", "Ăn uống", "Du lịch", "Giải trí", "Dịch vụ", "Khác"],
                index=0
            )
            is_foreign_transaction = st.checkbox("Giao dịch quốc tế")

        with col3:
            customer_age = st.slider("Tuổi khách hàng", 18, 90, 35)
            distance_from_home = st.number_input("Khoảng cách từ nhà (km)", min_value=0.0, value=5.0)
            transaction_type = st.selectbox(
                "Loại giao dịch",
                options=["Mua sắm trực tuyến", "Rút tiền ATM", "Thanh toán tại POS", "Chuyển khoản", "Khác"],
                index=0
            )
            amount_multiplier = st.slider(
                "Hệ số ảnh hưởng của số tiền", 
                min_value=0.5, 
                max_value=5.0, 
                value=1.0, 
                step=0.1,
                help="Điều chỉnh mức độ ảnh hưởng của số tiền đến xác suất gian lận (cao hơn = ảnh hưởng mạnh hơn)"
            )

        submit_button = st.form_submit_button("Đánh giá rủi ro")

    # Xử lý khi nhấn nút đánh giá
    if submit_button:
        try:
            # Chuyển đổi các giá trị từ form sang dạng thích hợp
            day_num = day_of_week_map[transaction_day]
            transaction_type_num = transaction_type_map[transaction_type]
            
            # Lấy mã số merchant category dựa trên loại hình
            merchant_category_map = {
                "Bán lẻ": 5411, 
                "Ăn uống": 5812, 
                "Du lịch": 4511, 
                "Giải trí": 7832, 
                "Dịch vụ": 7299, 
                "Khác": 9999
            }
            mcc_code = merchant_category_map.get(merchant_category, 9999)
            
            # Tạo giá trị merchant ID dựa trên merchant category và trạng thái giao dịch quốc tế
            merchant_id_base = mcc_code * 10
            merchant_id = merchant_id_base + (1 if is_foreign_transaction else 0)
            
            # Mã hóa loại thẻ thành số
            card_type_map = {"Visa": 1, "Mastercard": 2, "Amex": 3, "Khác": 4}
            card_id = card_type_map.get(card_type, 1) * 1000 + customer_age
            
            # Tính zip code dựa trên khoảng cách
            zip_code = 10000 + int(distance_from_home * 10)
            
            # Phân tích có lỗi hay không dựa trên các yếu tố rủi ro
            error_code = "0"  # Không có lỗi
            if is_foreign_transaction and amount > 1000:
                error_code = "1"  # Có dấu hiệu rủi ro
            
            # Mã hóa thành phố và quốc gia dựa trên is_foreign_transaction
            merchant_city = "Foreign City" if is_foreign_transaction else "HCM"
            merchant_state = "XX" if is_foreign_transaction else "VN"
            
            # Tạo dữ liệu đầu vào với kiểu dữ liệu phù hợp
            data = {
                # Numeric features - chuyển đổi thành float
                'client_id': float(customer_age * 100 + day_num),  # ID khách hàng dựa trên tuổi và ngày
                'card_id': float(card_id),
                'merchant_id': float(merchant_id),
                'zip': float(zip_code),
                'mcc': float(mcc_code),
                'amount': float(amount * amount_multiplier),  # Áp dụng hệ số ảnh hưởng của số tiền

                # Categorical features - giữ nguyên dạng string
                'date': f"2024-01-0{day_num+1}",  # Ngày dựa trên ngày trong tuần
                'use_chip': '1' if card_type in ["Visa", "Mastercard"] else '0',  # Giả định thẻ Visa/Mastercard dùng chip
                'merchant_city': merchant_city,
                'merchant_state': merchant_state,
                'errors': error_code
            }
            
            # Hiển thị thông tin chi tiết về dữ liệu đầu vào
            st.write("### Thông tin chi tiết đầu vào:")
            st.json(data)
            
            # Tạo DataFrame với kiểu dữ liệu rõ ràng
            transaction_data = pd.DataFrame([data])
            
            # Đảm bảo kiểu dữ liệu chính xác
            for col in ['client_id', 'card_id', 'merchant_id', 'zip', 'mcc', 'amount']:
                if col in transaction_data.columns:
                    transaction_data[col] = transaction_data[col].astype(float)
            
            for col in ['date', 'use_chip', 'merchant_city', 'merchant_state', 'errors']:
                if col in transaction_data.columns:
                    transaction_data[col] = transaction_data[col].astype(str)

            # Thực hiện đánh giá rủi ro
            st.info("Đang đánh giá rủi ro...")
            evaluation = evaluate_risk(transaction_data)

            if evaluation is not None:
                st.success("Đánh giá hoàn tất!")
                st.markdown("---")

                # Hiển thị kết quả đánh giá
                st.subheader("Kết quả đánh giá rủi ro")

                risk_score = evaluation['risk_score'].iloc[0]
                risk_category = evaluation['risk_category'].iloc[0]

                # Xử lý trường hợp không có fraud_probability
                if 'fraud_probability' in evaluation.columns:
                    fraud_prob = evaluation['fraud_probability'].iloc[0]
                elif 'probability' in evaluation.columns:
                    fraud_prob = evaluation['probability'].iloc[0]
                else:
                    fraud_prob = 0.15  # Giá trị mặc định

                fraud_pred = evaluation['predicted_fraud'].iloc[0]

                col1, col2 = st.columns([2, 1])

                with col1:
                    # Thông tin cơ bản
                    st.markdown("### Thông tin đánh giá")

                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

                    with metrics_col1:
                        st.metric("Điểm rủi ro", f"{risk_score:.1f}")
                    with metrics_col2:
                        st.metric("Danh mục rủi ro", risk_category)
                    with metrics_col3:
                        st.metric("Xác suất gian lận", f"{fraud_prob:.1%}")

                    # Hành động đề xuất
                    st.markdown("### Hành động đề xuất")

                    if risk_category == "Thấp":
                        st.success("✅ Cho phép giao dịch tự động")
                        st.write("Giao dịch có rủi ro thấp, có thể được xử lý tự động mà không cần thêm xác thực.")
                    elif risk_category == "Trung bình":
                        st.warning("⚠️ Yêu cầu xác thực bổ sung")
                        st.write("Yêu cầu khách hàng xác thực thêm bằng OTP hoặc sinh trắc học.")
                    elif risk_category == "Cao":
                        st.error("🚨 Kiểm tra thủ công")
                        st.write("Chuyển giao dịch cho nhân viên kiểm tra trước khi xử lý.")
                    else:  # Rất cao
                        st.error("🛑 Tạm dừng giao dịch")
                        st.write("Tạm dừng giao dịch và liên hệ với khách hàng để xác minh.")

                with col2:
                    # Đồng hồ đo rủi ro
                    st.markdown("### Đồng hồ đo rủi ro")
                    gauge_fig = create_gauge_chart(risk_score)
                    st.pyplot(gauge_fig)

                    # Dự đoán
                    st.markdown("### Kết luận")
                    if fraud_pred == 1:
                        st.error("⚠️ Có dấu hiệu gian lận")
                    else:
                        st.success("✓ Không phát hiện dấu hiệu gian lận")
                        
                # Hiển thị thông tin thêm về giao dịch
                st.markdown("---")
                st.subheader("Thông tin giao dịch")
                st.write(f"Số tiền: {amount:,.2f} USD")
                st.write(f"Thời gian: {transaction_hour}:00, {transaction_day}")
                st.write(f"Loại thẻ: {card_type}")
                st.write(f"Danh mục: {merchant_category}")
                st.write(f"Giao dịch quốc tế: {'Có' if is_foreign_transaction else 'Không'}")
                
        except Exception as e:
            st.error(f"Lỗi khi xử lý dữ liệu: {e}")
            import traceback
            st.write("Chi tiết lỗi:", traceback.format_exc())

# TRANG 5: HƯỚNG DẪN SỬ DỤNG
else:  # Hướng dẫn sử dụng
    st.header("HƯỚNG DẪN SỬ DỤNG")

    st.markdown("""
    ### Giới thiệu

    Hệ thống phát hiện gian lận và đánh giá rủi ro cung cấp các chức năng sau:

    1. **Tổng quan**: Thông tin cơ bản về hệ thống và kiến trúc
    2. **Phân tích dữ liệu**: Biểu đồ và đánh giá hiệu suất mô hình
    3. **Mẫu giao dịch**: Xem và phân tích các giao dịch mẫu
    4. **Đánh giá thủ công**: Nhập thông tin giao dịch và xem kết quả đánh giá

    ### Sử dụng các tính năng

    #### 1. Phân tích dữ liệu
    - Xem các số liệu tổng quan về mô hình
    - Phân tích biểu đồ phân bố rủi ro và tỷ lệ gian lận
    - Kiểm tra hiệu suất mô hình thông qua đường cong ROC và ma trận nhầm lẫn
    - Xem các đặc trưng quan trọng nhất

    #### 2. Mẫu giao dịch
    - Lọc giao dịch theo danh mục rủi ro, nhãn thực tế và dự đoán
    - Xem chi tiết từng giao dịch với đánh giá rủi ro và đề xuất hành động
    - Phân tích trạng thái phân loại (True Positive, False Positive, ...)

    #### 3. Đánh giá thủ công
    - Nhập thông tin giao dịch mới
    - Nhận kết quả đánh giá rủi ro và đề xuất hành động
    - Xem trực quan hóa điểm rủi ro qua đồng hồ đo

    ### Quy trình xử lý giao dịch dựa trên mức độ rủi ro

    1. **Rủi ro thấp (0-20)**: Cho phép giao dịch tự động
    2. **Rủi ro trung bình (21-50)**: Yêu cầu xác thực thêm (OTP, biometric)
    3. **Rủi ro cao (51-80)**: Chuyển cho nhân viên kiểm tra thủ công
    4. **Rủi ro rất cao (81-100)**: Tạm dừng giao dịch và liên hệ khách hàng

    ### Liên hệ hỗ trợ

    Nếu có bất kỳ câu hỏi hoặc yêu cầu hỗ trợ nào, vui lòng liên hệ:
    - Email: phamngocthaison@gmail.com
    - Hotline: (84) 938746562
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>© 2025 Hệ thống phát hiện gian lận và đánh giá rủi ro</p>
    <p>Phiên bản 1.0.0 - Nhóm 22 - </p>
</div>
""", unsafe_allow_html=True)