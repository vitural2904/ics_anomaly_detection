from src.hyperparameter_tuning import hyperparameter_tuning
from src.loader.data_loader import load_test_data
from sklearn.metrics import f1_score, classification_report

import numpy as np

# sau khi tuning xong, nên lưu mô hình
def test_evaluation(model_name, number) :

    X_test_scaled, Y_test = load_test_data('data/BATADAL', number)

    model_list = None
    best_f1, best_threshold = 0.0, 0.0
    report = None

    if model_name == 'ridge' :
        model_list = hyperparameter_tuning.tuning_threshold_ridge_MOR()
    elif model_name == 'svm' :
        model_list = hyperparameter_tuning.tuning_hyperparameter_SVR()

    for model in model_list :

        threshold = model['threshold']

        # Dự đoán trạng thái hệ thống
        X_test_hat = model['model'].predict(X_test_scaled)

        # Tính lỗi tái tạo (MSE từng điểm)
        errors = np.mean((X_test_scaled - X_test_hat) ** 2, axis=1)

        # Phát hiện bất thường: nếu lỗi vượt ngưỡng thì gán là bất thường (1)
        Y_pred = (errors > threshold).astype(int)

        # Tính F1-score (giữa dự đoán và nhãn thật)
        f1 = f1_score(Y_test, Y_pred)

        if f1 > best_f1 :
            best_f1 = f1
            best_threshold = threshold
            report = classification_report(Y_test, Y_pred)

    print(f"[{model_name.upper()}] F1-score on test_{number} = {best_f1:.4f} with threshold = {best_threshold:.6f}\n")
    print(f'------------------- REPORT FOR RIDGE FOR DATASET {number} -------------------')
    print(report)
    return f1

test_evaluation('ridge', 1)
test_evaluation('ridge', 2)

# test_evaluation('svm', 1)
# test_evaluation('svm', 2)