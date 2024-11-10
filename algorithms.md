# Tổng Quan Thuật Toán Machine Learning

## 1. Thuật Toán Trích Xuất Đặc Trưng

### 1.1 TF-IDF (Term Frequency-Inverse Document Frequency)

- **Mục đích**: Chuyển đổi văn bản thành vector số dựa trên tần suất từ
- **Công thức**:

  ```markdown
  TF(t,d) = số lần từ t xuất hiện trong văn bản d
  IDF(t) = log(N/DF(t))
  ```

  Trong đó:

  - N: tổng số văn bản
  - DF(t): số văn bản chứa từ t

  ```markdown
  TF-IDF(t,d) = TF(t,d) × IDF(t)
  ```

- **Tham số chính**:
  - `max_features`: 2000 (giới hạn số đặc trưng)
  - `ngram_range`: (1,3) (sử dụng unigrams đến trigrams)
  - `min_df`: 2 (loại bỏ từ hiếm)
  - `max_df`: 0.95 (loại bỏ từ quá phổ biến)

### 1.2 SVD (Singular Value Decomposition)

- **Mục đích**: Giảm chiều dữ liệu và trích xuất đặc trưng quan trọng
- **Công thức**:

  ```markdown
  A = U × Σ × V^T
  ```

  Trong đó:

  - A: ma trận dữ liệu gốc
  - U: ma trận trái trực giao
  - Σ: ma trận đường chéo các giá trị kỳ dị
  - V^T: chuyển vị của ma trận phải trực giao

- **Tham số**: `n_components = min(n_features-1, n_samples-1)`

## 2. Thuật Toán Phân Loại

### 2.1 Random Forest

- **Nguyên lý**: Tổng hợp nhiều cây quyết định
- **Công thức Gini**:
  ```markdown
  Gini = 1 - Σ(pi)²
  ```
  Trong đó:
- **Tham số tối ưu**:
  - `n_estimators`: 200-300 cây
  - `max_depth`: 20-30 levels
  - `min_samples_split`: 2-5

### 2.2 Linear SVM

- **Nguyên lý**: Tìm siêu phẳng phân tách tối ưu
- **Công thức**:

  ```markdown
  min(1/2 ||w||² + C Σ ξᵢ)
  ```

  Ràng buộc:

  ```markdown
  yᵢ(w·xᵢ + b) ≥ 1 - ξᵢ
  ξᵢ ≥ 0
  ```

  Trong đó:

  - w: vector trọng số
  - C: hệ số điều chuẩn
  - ξᵢ: biến slack

- **Tham số chính**: `C = [0.1, 1.0, 10.0]`

### 2.3 Naive Bayes

- **Nguyên lý**: Xác suất có điều kiện theo Bayes
- **Công thức**:

  ```markdown
  P(c|x) = P(x|c)P(c)/P(x)
  ```

  Trong đó:

  - P(c|x): xác suất posterior
  - P(x|c): likelihood
  - P(c): xác suất prior
  - P(x): evidence

- **Tham số**: `alpha = [0.1, 0.5, 1.0]`

## 3. Đánh Giá Mô Hình

### 3.1 Precision-Recall

```markdown
Precision = TP/(TP + FP)
Recall = TP/(TP + FN)
F1-Score = 2 × (Precision × Recall)/(Precision + Recall)
```

### 3.2 ROC-AUC

- **Công thức AUC**:
  ```markdown
  AUC = Σ(1/2 × (FPRᵢ₊₁ - FPRᵢ)(TPRᵢ₊₁ + TPRᵢ))
  ```
  Trong đó:
  - TPR = TP/(TP + FN)
  - FPR = FP/(FP + TN)

## 4. Kỹ Thuật Tối Ưu Hóa

### 4.1 Cross-Validation

- **K-fold CV công thức**:
  ```markdown
  CV Score = (1/K) × Σ Score_i
  ```
  Trong đó:
  - K: số fold
  - Score_i: điểm của fold thứ i

### 4.2 Grid Search

- **Mục đích**: Tìm tham số tối ưu
- **Công thức điểm cuối cùng**:
  ```markdown
  Final Score = (1/N) × Σ CV_Score(params)
  ```

### 4.3 Ensemble Learning

- **Voting công thức**:
  ```markdown
  Final Prediction = mode(predictions_i)
  Confidence = max(count(predictions_i)/total_models)
  ```

## 5. Pipeline Xử Lý

### 5.1 Chuẩn Hóa Dữ Liệu

- **MinMax Scaling**:

  ```markdown
  X_scaled = (X - X_min)/(X_max - X_min)
  ```

- **Z-Score Normalization**:
  ```markdown
  X_norm = (X - μ)/σ
  ```

### 5.2 Feature Selection

- **Chi-square Test**:
  ```markdown
  χ² = Σ((O - E)²/E)
  ```
  Trong đó:
  - O: giá trị quan sát
  - E: giá trị kỳ vọng

### 5.3 Class Balancing

- **Class Weights**:
  ```markdown
  weight_i = n_samples/(n_classes × n_samples_class_i)
  ```

## 6. Các Metric Đánh Giá

### 6.1 Confusion Matrix

```markdown
     Predicted
     Actual P N
     P TP FN
     N FP TN
```

### 6.2 Classification Report

- **Macro Average**:

  ```
  Macro_avg = (1/n_classes) × Σ metric_per_class
  ```

- **Weighted Average**:
  ```
  Weighted_avg = Σ(metric_per_class × support_per_class)/total_support
  ```

## 7. Mục Đích Sử Dụng Các Thành Phần

### 7.1 Thành Phần Trích Xuất Đặc Trưng

- **TfidfVectorizer**:

  - Chuyển đổi văn bản thành vector số dựa trên tần suất từ
  - Chuẩn hóa và tính toán trọng số cho các từ

- **TruncatedSVD**:
  - Giảm chiều dữ liệu vector
  - Chọn lọc đặc trưng quan trọng nhất

### 7.2 Các Mô Hình Phân Loại

#### 7.2.1 Khởi Tạo Mô Hình

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=30,
    class_weight='balanced'
)

svm = LinearSVC(
    C=1.0,
    max_iter=2000,
    class_weight='balanced'
)

nb = MultinomialNB(
    alpha=0.1,
    fit_prior=True
)
```

#### 7.2.2 Mục Đích Từng Mô Hình

- **RandomForestClassifier**: Tổng hợp dự đoán từ nhiều cây quyết định
- **LinearSVC**: Phân loại tuyến tính với SVM
- **MultinomialNB**: Phân loại xác suất cho văn bản

### 7.3 Kỹ Thuật Tối Ưu

#### 7.3.1 Code Tối Ưu

```python
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, chi2

param_grid = {
    'rf__n_estimators': [200, 300],
    'rf__max_depth': [20, 30],
    'svm__C': [0.1, 1.0, 10.0],
    'nb__alpha': [0.1, 0.5, 1.0]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='f1_weighted'
)

feature_selector = SelectKBest(
    score_func=chi2,
    k=200
)
```

#### 7.3.2 Mục Đích Tối Ưu

- **GridSearchCV**: Tìm tham số tối ưu cho mô hình
- **SelectKBest**: Chọn đặc trưng quan trọng

### 7.4 Tiền Xử Lý và Tăng Cường Dữ Liệu

#### 7.4.1 Code Xử Lý

```python
import re
from nltk.tokenize import word_tokenize

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return ' '.join(word_tokenize(text))

def augment_text(text):
    augmented = []
    tokens = text.split()
    augmented.append(' '.join(random.sample(tokens, len(tokens))))
    return augmented
```

#### 7.4.2 Mục Đích Xử Lý

- **Làm sạch văn bản**: Chuẩn hóa và xử lý nhiễu
- **Tăng cường dữ liệu**: Mở rộng tập huấn luyện
