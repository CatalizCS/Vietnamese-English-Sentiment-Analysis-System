# Chi Tiết Quy Trình Training Và Thuật Toán

## 1. Tổng Quan Quy Trình

### 1.1 Luồng Xử Lý
```
Input Text -> Preprocessing -> Feature Extraction -> Model Training -> Model Evaluation -> Export Model
```

### 1.2 Các Bước Chính
1. Thu thập dữ liệu từ nhiều nguồn
2. Tiền xử lý văn bản
3. Trích xuất đặc trưng 
4. Huấn luyện mô hình ensemble
5. Đánh giá và tối ưu hóa
6. Triển khai mô hình

## 2. Thu Thập & Tiền Xử Lý Dữ Liệu

### 2.1 Nguồn Dữ Liệu
- Comments trên mạng xã hội
- Reviews trên các trang thương mại điện tử
- Dữ liệu được gán nhãn thủ công
- Dữ liệu được sinh tự động (augmentation)

### 2.2 Tiền Xử Lý Văn Bản
- Làm sạch văn bản: loại bỏ HTML, kí tự đặc biệt
- Chuẩn hóa: lowercase, unicode
- Tokenization: tách từ theo ngôn ngữ
- Loại bỏ stopwords
- Cân bằng dữ liệu giữa các classes

## 3. Trích Xuất Đặc Trưng (Feature Engineering)

### 3.1 TF-IDF Vectorization
- Chuyển văn bản thành vector số theo tần suất từ
- Tham số chính:
  - max_features: 10000 (giới hạn số đặc trưng)
  - ngram_range: (1,3) (unigrams, bigrams, trigrams)
  - min_df: 5 (loại từ hiếm)
  - max_df: 0.95 (loại từ quá phổ biến)

### 3.2 Linguistic Features (6 đặc trưng)
- Số câu trong văn bản
- Độ dài trung bình mỗi câu
- Độ dài trung bình các từ
- Tỷ lệ dấu câu
- Tỷ lệ chữ hoa
- Tỷ lệ stopwords

### 3.3 Emotion Features (15 đặc trưng)
- Điểm số cho 12 loại cảm xúc cơ bản
- Tỷ lệ emojis
- Tỷ lệ dấu cảm xúc (!,?,...)
- Số từ cảm xúc trong từ điển

## 4. Huấn Luyện Mô Hình (Model Training)

### 4.1 Ensemble Model (Kết hợp 3 mô hình)

#### a) Random Forest
- Số cây (n_estimators): 1000
- Độ sâu tối đa (max_depth): 50
- Min samples split: 10
- Class weights: balanced
- Đặc điểm:
  + Chống overfitting tốt
  + Xử lý được dữ liệu không cân bằng
  + Cho biết feature importance

#### b) Linear SVM 
- C: 0.8 (regularization strength)
- Max iterations: 2000
- Class weights: balanced
- Dual: False
- Đặc điểm:
  + Hiệu quả với văn bản
  + Tìm siêu phẳng phân tách tối ưu
  + Xử lý được high dimensional data

#### c) Multinomial Naive Bayes
- Alpha: 1.2 (smoothing parameter)
- Fit prior: True
- Đặc điểm:
  + Nhanh và nhẹ
  + Hiệu quả với text classification
  + Xử lý được sparse data

### 4.2 Cross-validation
- Stratified K-Fold với k=10
- Đảm bảo cân bằng classes trong mỗi fold
- Metrics:
  + F1-score (weighted)
  + Precision & Recall
  + Accuracy

### 4.3 Early Stopping
- Patience: 5 epochs
- Min delta: 0.001
- Monitor: validation F1-score
- Giúp tránh overfitting

### 4.4 Class Balancing
- Compute sample weights
- Undersampling class đa số
- Oversampling class thiểu số
- Class weights tự động

## 5. Đánh Giá Và Tối Ưu Hóa

### 5.1 Grid Search CV
- Tìm hyperparameters tối ưu
- Parameters được search:
  + RF: n_estimators, max_depth
  + SVM: C, max_iter
  + NB: alpha, fit_prior
- Metrics: F1-score weighted

### 5.2 Feature Selection
- Mutual Information Classifier
- Chi-square test
- SelectKBest với k=800-1200

### 5.3 Regularization
- RF: ccp_alpha=0.002 (pruning)
- SVM: C=0.8 (regularization)
- NB: alpha=1.2 (smoothing)

## 6. Lưu Trữ & Triển Khai

### 6.1 Model Checkpointing
- Lưu model sau mỗi epoch
- Lưu optimizer state
- Lưu training history
- Giới hạn 5 checkpoints gần nhất

### 6.2 Model Export
- Lưu model cuối cùng
- Lưu feature extractors
- Lưu các metrics
- Lưu config

### 6.3 Model Serving
- REST API endpoints 
- WebSocket realtime
- Batch prediction
- Load balancing

## 7. Monitoring & Logging

### 7.1 Training Metrics
- Loss curves
- Accuracy curves
- F1-score theo epoch
- Learning rate changes

### 7.2 Production Metrics
- Inference latency
- Memory usage
- Error rates
- Request throughput

### 7.3 Alerts
- High error rate
- Long response time
- Resource usage cao
- Model degradation

## 8. Kết Quả & Hiệu Năng

### 8.1 Metrics
- Accuracy: 85-90%
- F1-score: 83-88%
- ROC-AUC: 0.90-0.95

### 8.2 Performance 
- Inference time: <100ms
- Memory usage: <500MB
- Error rate: <0.1%
- Uptime: 99.9%