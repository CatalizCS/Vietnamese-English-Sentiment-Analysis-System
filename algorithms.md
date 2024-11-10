# Machine Learning Algorithms Used in Sentiment Analysis Project

## 1. Text Feature Extraction

### 1.1 TF-IDF (Term Frequency-Inverse Document Frequency)

- **Purpose**: Chuyển đổi văn bản thành vector số học
- **How it works**:
  - Term Frequency (TF): Đếm số lần xuất hiện của từng từ trong văn bản
  - Inverse Document Frequency (IDF): Đánh giá tầm quan trọng của từ trong toàn bộ corpus
  - Final Score = TF \* IDF
- **Implementation**:
  ```python
  TfidfVectorizer(
          max_features=5000,
          ngram_range=(1, 2),
          min_df=2,
          max_df=0.95
  )
  ```

### 1.2 Truncated SVD (Singular Value Decomposition)

- **Purpose**: Giảm số chiều của ma trận TF-IDF
- **How it works**:
  - Phân tích ma trận thành tích của 3 ma trận U, Σ, V^T
  - Giữ lại k thành phần chính quan trọng nhất
  - Giảm nhiễu và overfit
- **Implementation**:
  ```python
  TruncatedSVD(n_components=100)
  ```

## 2. Classification Models

### 2.1 Random Forest

- **Purpose**: Phân loại sentiment bằng tập hợp nhiều cây quyết định
- **Advantages**:
  - Chống overfitting tốt
  - Xử lý được dữ liệu phi tuyến tính
  - Cho biết feature importance
- **Parameters**:
  ```python
  RandomForestClassifier(
          n_estimators=[100, 200],
          max_depth=[None, 10, 20]
  )
  ```

### 2.2 Support Vector Machine (LinearSVC)

- **Purpose**: Phân loại bằng cách tìm siêu phẳng tối ưu
- **Advantages**:
  - Hiệu quả với dữ liệu có số chiều cao
  - Phù hợp với text classification
  - Xử lý tốt với non-linear data thông qua kernel trick
- **Parameters**:
  ```python
  LinearSVC(C=[0.1, 1.0, 10.0])
  ```

### 2.3 Naive Bayes

- **Purpose**: Phân loại dựa trên xác suất có điều kiện
- **Advantages**:
  - Đơn giản, nhanh
  - Hiệu quả với dữ liệu text
  - Hoạt động tốt với ít dữ liệu
- **Parameters**:
  ```python
  MultinomialNB(alpha=[0.1, 1.0, 10.0])
  ```

## 3. Ensemble Learning

### 3.1 Voting Classifier

- **Purpose**: Kết hợp dự đoán từ nhiều mô hình
- **How it works**:
  - Hard Voting: Chọn label được dự đoán nhiều nhất
  - Soft Voting: Trung bình xác suất dự đoán
- **Implementation**:
  ```python
  VotingClassifier(
          estimators=[
                  ('rf', RandomForestClassifier()),
                  ('svm', LinearSVC()),
                  ('nb', MultinomialNB())
          ],
          voting='hard'
  )
  ```

## 4. Model Selection & Optimization

### 4.1 Grid Search with Cross Validation

- **Purpose**: Tìm hyperparameters tối ưu
- **How it works**:
  - Thử tất cả tổ hợp parameters có thể
  - Đánh giá bằng cross-validation
  - Chọn tổ hợp tốt nhất
- **Metrics**:
  - F1 Score (weighted)
  - Cross-validation score
  - Training time

## 5. Data Augmentation Techniques

### 5.1 Text Augmentation

- Synonym Replacement
- Random Swap
- Random Insertion
- Random Deletion
- Text Humanization

### 5.2 Statistical Features

- Text length
- Word count
- Average word length
- Special character ratio

## 6. Model Evaluation

### 6.1 Metrics

- Precision: Độ chính xác của các dự đoán positive
- Recall: Tỷ lệ positive cases được phát hiện
- F1 Score: Trung bình điều hòa của Precision và Recall
- Confusion Matrix: Ma trận nhầm lẫn giữa các classes

### 6.2 Visualization

- Feature Importance Plot
- Cross-validation Score Distribution
- Learning Curves
- Confusion Matrix Heatmap
