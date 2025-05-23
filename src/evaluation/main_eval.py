import importlib.util
import os

util_path = os.path.join(os.path.dirname(__file__), "..", "data_loader", "__pycache__", "util.pyc")
util_path = os.path.abspath(util_path)
spec = importlib.util.spec_from_file_location("util", util_path)
util = importlib.util.module_from_spec(spec)
spec.loader.exec_module(util)

from src.hyperparameter_tuning import tuning_ML, tuning_DL
from src.data_loader.data_loader import load_test_data
from sklearn.metrics import f1_score, classification_report

import numpy as np
import pandas as pd

def test_evaluation(model_name, number) :

    X_test_scaled, Y_test = load_test_data('data/BATADAL', number)

    model_list = None
    best_f1, best_threshold = 0.0, 0.0
    report = None

    if model_name == 'ridge' :
        model_list = tuning_ML.tuning_threshold_ridge_MOR()
    elif model_name == 'svr' :
        model_list = tuning_ML.best_model_retrive_SVR()
    elif model_name == 'adaboost' :
        model_list = tuning_ML.best_model_retrive_adaboost()
    elif model_name == 'cnn_ae' :
        model_list = tuning_DL.best_model_retrive_cnn_ae()

    for model in model_list :

        threshold = model['threshold']

        if model_name == 'cnn_ae':
            X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

        # Dự đoán trạng thái hệ thống
        X_test_hat = model['model'].predict(X_test_scaled)

        # Đưa về cùng chiều : (sample_size, 43)
        if model_name == 'cnn_ae':
            X_test_hat = np.squeeze(X_test_hat, axis=1)
            X_test_scaled = np.squeeze(X_test_scaled, axis=1)

        # Tính lỗi tái tạo (MSE từng điểm)
        errors = np.mean((X_test_scaled - X_test_hat) ** 2, axis=1)

        # Phát hiện bất thường: nếu lỗi vượt ngưỡng thì gán là bất thường (1)
        Y_pred = (errors > threshold).astype(int)

        # Tính F1-score (giữa dự đoán và nhãn thật)
        f1 = f1_score(Y_test, Y_pred)

        if f1 > best_f1 :
            best_threshold = threshold
            report = util.classification_metric(classification_report(Y_test, Y_pred, output_dict=True))
            best_f1 = report['1.0']['f1-score']
            report_cleaned = pd.DataFrame(report).transpose().to_string(float_format="%.2f")


    if model_name == 'ridge' :
        print(f"[{model_name.upper()}] F1-score on test_{number} = {best_f1:.4f} with threshold = {best_threshold:.6f}\n")
        print(f'------------------- REPORT FOR RIDGE REGRESSION FOR DATASET {number} -------------------')
        print(report_cleaned)
    elif model_name == 'svr' :
        print(f"[{model_name.upper()}] F1-score on test_{number} = {best_f1:.4f} with threshold = {best_threshold:.6f}\n")
        print(f'------------------- REPORT FOR SVR FOR DATASET {number} -------------------')
        print(report_cleaned)
    elif model_name == 'adaboost' :
        print(f"[{model_name.upper()}] F1-score on test_{number} = {best_f1:.4f} with threshold = {best_threshold:.6f}\n")
        print(f'------------------- REPORT FOR ADABOOST FOR DATASET {number} -------------------')
        print(report_cleaned)
    elif model_name == 'cnn_ae' :
        print(f"[{model_name.upper()}] F1-score on test_{number} = {best_f1:.4f} with threshold = {best_threshold:.6f}\n")
        print(f'------------------- REPORT FOR CNN + AE FOR DATASET {number} -------------------')
        print(report_cleaned)
    
    return f1

def main_eval() :
    model_name = input("Select the model (ridge / svr / adaboost / cnn_ae): ").strip().lower()
    dataset_number = input("Number of testing dataset (1 or 2): ").strip()

    # Kiểm tra hợp lệ đầu vào
    valid_models = {'ridge', 'svr', 'adaboost', 'cnn_ae'}
    if model_name in valid_models and dataset_number in {'1', '2'}:
        test_evaluation(model_name, int(dataset_number))
    else:
        print("Invalid input. Please try again by typing the valid model name (ridge, svr, adaboost, cnn_ae) and testing dataset is 1 or 2.")


'''Program'''

main_eval()

'''-------'''