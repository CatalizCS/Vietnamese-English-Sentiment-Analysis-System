class TongQuanThuatToan:
    """
    Tổng quan về các thuật toán sử dụng trong hệ thống phân tích cảm xúc.
    """

    @staticmethod
    def feature_extraction_algorithms():
        """
        Các thuật toán trích xuất đặc trưng (Feature Extraction)
        """
        return {
            "TF-IDF": {
                "name": "Term Frequency-Inverse Document Frequency",
                "purpose": "Chuyển đổi văn bản thành vector số dựa trên tần suất từ",
                "implementation": "sklearn.feature_extraction.text.TfidfVectorizer",
                "parameters": {
                    "max_features": "2000 - Giới hạn số lượng đặc trưng",
                    "ngram_range": "(1,3) - Sử dụng unigrams, bigrams và trigrams",
                    "min_df": "2 - Loại bỏ các từ hiếm",
                    "max_df": "0.95 - Loại bỏ các từ phổ biến"
                }
            },
            "SVD": {
                "name": "Singular Value Decomposition",
                "purpose": "Giảm chiều dữ liệu và trích xuất các đặc trưng quan trọng",
                "implementation": "sklearn.decomposition.TruncatedSVD",
                "parameters": {
                    "n_components": "95 hoặc min(features-1, samples-1)",
                    "algorithm": "randomized"
                }
            },
            "Character N-grams": {
                "name": "Character Level Features",
                "purpose": "Phân tích mẫu ký tự, hữu ích cho lỗi chính tả và biến thể từ",
                "implementation": "Custom TfidfVectorizer với analyzer='char'",
                "parameters": {
                    "ngram_range": "(2,4)",
                    "max_features": "500"
                }
            }
        }

    @staticmethod
    def classification_algorithms():
        """
        Các thuật toán phân loại (Classification)
        """
        return {
            "RandomForest": {
                "name": "Random Forest Classifier",
                "purpose": "Phân loại tổng hợp dựa trên nhiều cây quyết định",
                "strengths": [
                    "Hiệu quả với dữ liệu có nhiều chiều",
                    "Chống overfitting tốt",
                    "Có thể xử lý dữ liệu không cân bằng"
                ],
                "parameters": {
                    "n_estimators": "200-300 cây",
                    "max_depth": "20-30 độ sâu tối đa",
                    "class_weight": "balanced"
                }
            },
            "LinearSVC": {
                "name": "Linear Support Vector Classification",
                "purpose": "Phân loại tuyến tính dựa trên Support Vector Machines",
                "strengths": [
                    "Hiệu quả với văn bản có chiều cao",
                    "Tốt cho dữ liệu phân tách tuyến tính",
                    "Nhanh và hiệu quả bộ nhớ"
                ],
                "parameters": {
                    "C": [0.1, 1.0, 10.0],
                    "max_iter": 2000,
                    "class_weight": "balanced"
                }
            },
            "MultinomialNB": {
                "name": "Multinomial Naive Bayes",
                "purpose": "Phân loại xác suất dựa trên định lý Bayes",
                "strengths": [
                    "Hiệu quả với dữ liệu văn bản",
                    "Nhanh trong huấn luyện và dự đoán",
                    "Hoạt động tốt với ít dữ liệu"
                ],
                "parameters": {
                    "alpha": [0.1, 0.5, 1.0],
                    "fit_prior": [True, False]
                }
            }
        }

    @staticmethod
    def text_augmentation_techniques():
        """
        Các kỹ thuật tăng cường dữ liệu văn bản
        """
        return {
            "Synonym Replacement": {
                "description": "Thay thế từ bằng từ đồng nghĩa",
                "implementation": "Sử dụng từ điển đồng nghĩa hoặc thay thế ký tự",
                "purpose": "Tăng đa dạng từ vựng"
            },
            "Random Swap": {
                "description": "Hoán đổi vị trí ngẫu nhiên các từ",
                "implementation": "Thuật toán hoán đổi ngẫu nhiên",
                "purpose": "Tạo biến thể cấu trúc câu"
            },
            "Random Deletion": {
                "description": "Xóa ngẫu nhiên một số từ",
                "implementation": "Xóa với xác suất cho trước",
                "purpose": "Mô phỏng câu thiếu từ"
            },
            "Text Humanization": {
                "description": "Thêm biến thể người dùng thực",
                "implementation": "Thêm typo, emojis, từ lóng",
                "purpose": "Tăng tính thực tế của dữ liệu"
            }
        }

    @staticmethod
    def evaluation_metrics():
        """
        Các độ đo đánh giá
        """
        return {
            "Classification Report": {
                "metrics": ["Precision", "Recall", "F1-score"],
                "purpose": "Đánh giá chi tiết từng lớp"
            },
            "Confusion Matrix": {
                "purpose": "Phân tích chi tiết dự đoán đúng/sai",
                "visualization": "Heatmap with seaborn"
            },
            "ROC Curve": {
                "purpose": "Đánh giá hiệu suất phân loại ở các ngưỡng khác nhau",
                "metrics": ["AUC-ROC"]
            }
        }

    @staticmethod
    def get_pipeline_overview():
        """
        Tổng quan về pipeline xử lý
        """
        return {
            "1. Data Preprocessing": {
                "steps": [
                    "Làm sạch văn bản",
                    "Chuẩn hóa ký tự",
                    "Tokenization",
                    "Loại bỏ stopwords"
                ]
            },
            "2. Feature Engineering": {
                "steps": [
                    "TF-IDF Vectorization",
                    "Character N-grams",
                    "Dimensionality Reduction (SVD)",
                    "Feature Scaling"
                ]
            },
            "3. Model Training": {
                "steps": [
                    "Ensemble Learning",
                    "Cross Validation",
                    "Hyperparameter Optimization",
                    "Model Selection"
                ]
            },
            "4. Evaluation": {
                "steps": [
                    "Performance Metrics",
                    "Error Analysis",
                    "Visualization",
                    "Model Comparison"
                ]
            }
        }

    @staticmethod
    def get_optimization_techniques():
        """
        Các kỹ thuật tối ưu hóa
        """
        return {
            "Hyperparameter Tuning": {
                "method": "Grid Search với Cross Validation",
                "parameters": {
                    "RandomForest": ["n_estimators", "max_depth", "min_samples_split"],
                    "LinearSVC": ["C", "max_iter", "tol"],
                    "MultinomialNB": ["alpha", "fit_prior"]
                }
            },
            "Feature Selection": {
                "method": "SelectKBest với chi2",
                "purpose": "Chọn đặc trưng quan trọng nhất"
            },
            "Ensemble Methods": {
                "technique": "Voting Classifier",
                "purpose": "Kết hợp dự đoán từ nhiều mô hình"
            },
            "Class Balancing": {
                "method": "Class Weights",
                "purpose": "Xử lý dữ liệu không cân bằng"
            }
        }

    @staticmethod
    def get_model_selection_criteria():
        """
        Tiêu chí lựa chọn mô hình
        """
        return {
            "Performance": {
                "metrics": ["F1-score", "Accuracy", "ROC-AUC"],
                "importance": "High"
            },
            "Training Time": {
                "consideration": "Thời gian huấn luyện hợp lý",
                "importance": "Medium"
            },
            "Memory Usage": {
                "consideration": "Sử dụng bộ nhớ hiệu quả",
                "importance": "Medium"
            },
            "Interpretability": {
                "consideration": "Khả năng giải thích kết quả",
                "importance": "Medium-High"
            }
        }
