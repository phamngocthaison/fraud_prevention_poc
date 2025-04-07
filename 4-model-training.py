import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import os
import pickle
import time
from scipy import sparse

print("GIAI ĐOẠN 4: HUẤN LUYỆN MÔ HÌNH PHÁT HIỆN GIAN LẬN")
print("-" * 50)

# Tạo thư mục output nếu chưa tồn tại
if not os.path.exists('output4'):
    os.makedirs('output4')

# Đọc dữ liệu đã xử lý từ bước trước
print("Đang đọc dữ liệu đã biến đổi...")
X_train = sparse.load_npz('output3/X_train_transformed.npz')
X_test = sparse.load_npz('output3/X_test_transformed.npz')
y_train = np.load('output3/y_train.npy')
y_test = np.load('output3/y_test.npy')

print(f"Kích thước tập huấn luyện: {X_train.shape}")
print(f"Kích thước tập kiểm tra: {X_test.shape}")
print(f"Tỷ lệ gian lận trong tập huấn luyện: {np.mean(y_train) * 100:.2f}%")
print(f"Tỷ lệ gian lận trong tập kiểm tra: {np.mean(y_test) * 100:.2f}%")

# Bước 1: Cân bằng lại dữ liệu với SMOTE
print("\nĐang cân bằng dữ liệu với SMOTE...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"Kích thước tập huấn luyện sau SMOTE: {X_train_resampled.shape}")
print(f"Phân bố lớp sau SMOTE: {np.bincount(y_train_resampled)}")
print(f"Tỷ lệ gian lận sau SMOTE: {np.mean(y_train_resampled) * 100:.2f}%")

# Bước 2: Huấn luyện mô hình Random Forest cơ bản
print("\nHuấn luyện mô hình Random Forest cơ bản...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
start_time = time.time()
rf_model.fit(X_train_resampled, y_train_resampled)
rf_train_time = time.time() - start_time
print(f"Thời gian huấn luyện Random Forest: {rf_train_time:.2f} giây")

# Đánh giá mô hình Random Forest
print("\nĐánh giá mô hình Random Forest trên tập kiểm tra...")
rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

print("\nBáo cáo phân loại Random Forest:")
print(classification_report(y_test, rf_pred))
print(f"AUC: {roc_auc_score(y_test, rf_prob):.4f}")

# Bước 3: Huấn luyện mô hình XGBoost
print("\nHuấn luyện mô hình XGBoost cơ bản...")
scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)
start_time = time.time()
xgb_model.fit(X_train_resampled, y_train_resampled)
xgb_train_time = time.time() - start_time
print(f"Thời gian huấn luyện XGBoost: {xgb_train_time:.2f} giây")

# Đánh giá mô hình XGBoost
print("\nĐánh giá mô hình XGBoost trên tập kiểm tra...")
xgb_pred = xgb_model.predict(X_test)
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

print("\nBáo cáo phân loại XGBoost:")
print(classification_report(y_test, xgb_pred))
print(f"AUC: {roc_auc_score(y_test, xgb_prob):.4f}")

# Bước 4: Tối ưu hóa mô hình XGBoost (nếu có đủ thời gian)
optimize_hyperparams = False  # Đặt thành True nếu muốn tối ưu hóa

if optimize_hyperparams:
    print("\nĐang tối ưu hóa siêu tham số cho XGBoost...")
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    grid_search = GridSearchCV(
        estimator=xgb.XGBClassifier(objective='binary:logistic', scale_pos_weight=scale_pos_weight),
        param_grid=param_grid,
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train_resampled, y_train_resampled)

    print(f"Tham số tối ưu: {grid_search.best_params_}")
    print(f"AUC tốt nhất từ CV: {grid_search.best_score_:.4f}")

    # Sử dụng mô hình tốt nhất
    best_xgb_model = grid_search.best_estimator_

    # Đánh giá mô hình tối ưu
    best_xgb_pred = best_xgb_model.predict(X_test)
    best_xgb_prob = best_xgb_model.predict_proba(X_test)[:, 1]

    print("\nBáo cáo phân loại XGBoost tối ưu:")
    print(classification_report(y_test, best_xgb_pred))
    print(f"AUC: {roc_auc_score(y_test, best_xgb_prob):.4f}")

    # Sử dụng mô hình XGBoost tối ưu cho các bước tiếp theo
    xgb_model = best_xgb_model
    xgb_pred = best_xgb_pred
    xgb_prob = best_xgb_prob

# Bước 5: Vẽ đường cong ROC cho cả hai mô hình
print("\nVẽ đường cong ROC so sánh các mô hình...")
plt.figure(figsize=(10, 8))

# ROC cho Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_prob)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_test, rf_prob):.4f})')

# ROC cho XGBoost
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_prob)
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_score(y_test, xgb_prob):.4f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Đường cong ROC')
plt.legend()
plt.savefig('output4/roc_curves.png')
print(f"Đã lưu đường cong ROC vào 'output4/roc_curves.png'")

# Bước 6: Vẽ đường cong Precision-Recall
print("\nVẽ đường cong Precision-Recall...")
plt.figure(figsize=(10, 8))

# PR curve for Random Forest
precision_rf, recall_rf, _ = precision_recall_curve(y_test, rf_prob)
plt.plot(recall_rf, precision_rf, label=f'Random Forest')

# PR curve for XGBoost
precision_xgb, recall_xgb, _ = precision_recall_curve(y_test, xgb_prob)
plt.plot(recall_xgb, precision_xgb, label=f'XGBoost')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Đường cong Precision-Recall')
plt.legend()
plt.savefig('output4/precision_recall_curves.png')
print(f"Đã lưu đường cong Precision-Recall vào 'output4/precision_recall_curves.png'")

# Bước 7: Vẽ ma trận nhầm lẫn
print("\nVẽ ma trận nhầm lẫn cho các mô hình...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Ma trận nhầm lẫn cho Random Forest
cm_rf = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_title('Ma trận nhầm lẫn - Random Forest')
ax1.set_xlabel('Dự đoán')
ax1.set_ylabel('Thực tế')
ax1.set_xticklabels(['Không gian lận', 'Gian lận'])
ax1.set_yticklabels(['Không gian lận', 'Gian lận'])

# Ma trận nhầm lẫn cho XGBoost
cm_xgb = confusion_matrix(y_test, xgb_pred)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Blues', ax=ax2)
ax2.set_title('Ma trận nhầm lẫn - XGBoost')
ax2.set_xlabel('Dự đoán')
ax2.set_ylabel('Thực tế')
ax2.set_xticklabels(['Không gian lận', 'Gian lận'])
ax2.set_yticklabels(['Không gian lận', 'Gian lận'])

plt.tight_layout()
plt.savefig('output4/confusion_matrices.png')
print(f"Đã lưu ma trận nhầm lẫn vào 'output4/confusion_matrices.png'")

# Bước 8: Phân tích đặc trưng quan trọng
print("\nPhân tích đặc trưng quan trọng...")

# Đọc tên đặc trưng
try:
    with open('output3/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
except:
    feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

# Đối với Random Forest
rf_importances = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
})
rf_importances = rf_importances.sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
top_n = 20  # Số đặc trưng quan trọng nhất để hiển thị
sns.barplot(x='importance', y='feature', data=rf_importances.head(top_n))
plt.title(f'Top {top_n} đặc trưng quan trọng nhất - Random Forest')
plt.tight_layout()
plt.savefig('output4/rf_feature_importance.png')
print(f"Đã lưu biểu đồ đặc trưng quan trọng Random Forest vào 'output4/rf_feature_importance.png'")

# Đối với XGBoost
xgb_importances = pd.DataFrame({
    'feature': feature_names,
    'importance': xgb_model.feature_importances_
})
xgb_importances = xgb_importances.sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=xgb_importances.head(top_n))
plt.title(f'Top {top_n} đặc trưng quan trọng nhất - XGBoost')
plt.tight_layout()
plt.savefig('output4/xgb_feature_importance.png')
print(f"Đã lưu biểu đồ đặc trưng quan trọng XGBoost vào 'output4/xgb_feature_importance.png'")

# Bước 9: Lưu mô hình tốt nhất
print("\nLưu các mô hình...")

# So sánh AUC của hai mô hình và chọn mô hình tốt nhất
rf_auc = roc_auc_score(y_test, rf_prob)
xgb_auc = roc_auc_score(y_test, xgb_prob)

if xgb_auc >= rf_auc:
    best_model = xgb_model
    best_model_name = "XGBoost"
    best_auc = xgb_auc
else:
    best_model = rf_model
    best_model_name = "Random Forest"
    best_auc = rf_auc

print(f"AUC Random Forest: {rf_auc:.4f}")
print(f"AUC XGBoost: {xgb_auc:.4f}")
print(f"Mô hình tốt nhất: {best_model_name} với AUC = {best_auc:.4f}")

# Lưu mô hình Random Forest
with open('output4/random_forest_model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("Đã lưu mô hình Random Forest vào 'output4/random_forest_model.pkl'")

# Lưu mô hình XGBoost
with open('output4/xgboost_model.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
print("Đã lưu mô hình XGBoost vào 'output4/xgboost_model.pkl'")

# Lưu mô hình tốt nhất
with open('output4/best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"Đã lưu mô hình tốt nhất ({best_model_name}) vào 'output4/best_model.pkl'")

# Lưu kết quả dự đoán của mô hình tốt nhất cho tập kiểm tra
if best_model_name == "XGBoost":
    best_pred = xgb_pred
    best_prob = xgb_prob
else:
    best_pred = rf_pred
    best_prob = rf_prob

# Lưu kết quả dự đoán
results = pd.DataFrame({
    'actual': y_test,
    'predicted': best_pred,
    'probability': best_prob
})

# Thêm transaction_id nếu có
try:
    test_transaction_id = pd.read_csv('output3/test_transaction_id.csv')
    if len(test_transaction_id) == len(results):
        results['transaction_id'] = test_transaction_id.values
except:
    pass

results.to_csv('output4/test_predictions.csv', index=False)
print(f"Đã lưu kết quả dự đoán vào 'output4/test_predictions.csv'")

print("\nHoàn thành Giai đoạn 4: Huấn luyện mô hình phát hiện gian lận!")