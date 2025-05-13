from src.loader.data_loader import load_attack_data
from src.train_model import train_model
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV

import numpy as np
import matplotlib.pyplot as plt
import joblib

X_attack_scaled, Y_attack = load_attack_data('data/BATADAL/test_whitebox_attack.csv')

def find_best_threshold(errors, y_true):
    precision, recall, thresholds = precision_recall_curve(y_true, errors)

    # calculate F1-score for each threshold
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)

    best_idx = f1_scores.argmax()
    best_r = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    return best_r, best_f1

def draw_histogram(errors, y_true, threshold_opt) :

    plt.hist(errors[y_true == 0], bins=100, alpha=0.6, label='Benign')
    plt.hist(errors[y_true == 1], bins=100, alpha=0.6, label='Attack')
    plt.axvline(threshold_opt, color='red', linestyle='--', label='Threshold')
    plt.legend()
    plt.title("Error distribution")
    plt.xlabel("Reconstruction Error (MSE)")
    plt.ylabel("Count")
    plt.show()

'''LinearRegression - Ridge'''
def tuning_threshold_ridge_MOR():
    
    # first try : np.logspace(-3, 4, 20) # log-scale grid,  Best Alpha = 1000.00000, Best Threshold = 0.003494, Best F1 = 0.6153
    # second try : np.logspace(3, 5, 20) # Best Alpha = 11288.37892, Best Threshold = 0.031816, Best F1 = 0.7412
    # third try : np.linspace(5000, 15000, 50) # Best Alpha = 9897.95918, Best Threshold = 0.028758, Best F1 = 0.7439
    # forth try : np.linspace(9500, 10000, 50) # Best Alpha = 9959.18367, Best Threshold = 0.028750, Best F1 = 0.7441
    alphas = np.linspace(9900, 9999, 100)

    best_f1_rounded = -1
    best_models = []

    for penalize_l2_norm in alphas:

        model = train_model.train_linear_regression_ridge_MOR(penalize_l2_norm)

        X_attack_hat = model.predict(X_attack_scaled)
        errors = np.mean((X_attack_scaled - X_attack_hat) ** 2, axis=1)

        threshold_opt, f1_opt = find_best_threshold(errors, Y_attack)
        f1_rounded = round(f1_opt, 4)

        # print(f"Alpha = {penalize_l2_norm:.5f}, Threshold = {threshold_opt:.6f}, F1 = {f1_rounded:.4f}")

        if f1_rounded > best_f1_rounded:
            best_f1_rounded = f1_rounded
            best_models = [{
                "model": model,
                "threshold": threshold_opt,
                "alpha": penalize_l2_norm,
                "f1": f1_rounded
            }]
        elif f1_rounded == best_f1_rounded:
            best_models.append({
                "model": model,
                "threshold": threshold_opt,
                "alpha": penalize_l2_norm,
                "f1": f1_rounded
            })

    # print(f"\nAll models with best F1 = {best_f1_rounded:.4f}:")
    # for item in best_models:
        # print(f"Alpha = {item['alpha']:.5f}, Threshold = {item['threshold']:.6f}")

    return best_models

# tuning_threshold_ridge_MOR()
'''--------------------------'''


'''LinearRegression - Ridge with each sensor'''
def tuning_threshold_ridge_split():

    trained_models = train_model.train_linear_regression_ridge_split()
    
    print("y_true labels:", np.unique(Y_attack, return_counts=True))

    # 1. Dự đoán từng sensor bằng mô hình riêng
    X_attack_hat = np.zeros_like(X_attack_scaled)
    for i, model in enumerate(trained_models):
        X_attack_hat[:, i] = model.predict(X_attack_scaled)

    # 2. Tính lỗi theo thời gian (mỗi dòng là một thời điểm)
    errors = np.mean((X_attack_scaled - X_attack_hat) ** 2, axis=1)

    # 3. In top 3 lỗi lớn nhất (để kiểm tra nhanh)
    top_error_idxs = np.argsort(errors)[-3:]
    print("Top 3 errors:")
    for idx in top_error_idxs:
        print(f"Index: {idx}, Error: {errors[idx]:.6f}, True label: {Y_attack[idx]}")

    # 4. Tìm threshold tốt nhất bằng precision-recall (F1)
    threshold_opt, f1_opt = find_best_threshold(errors, Y_attack)

    draw_histogram(errors, Y_attack, threshold_opt)

    print(f"Best threshold: {threshold_opt:.6f}, Best F1: {f1_opt:.4f}")
    return threshold_opt, trained_models

# tuning_threshold_ridge_split()
'''--------------------------'''


'''Suport Vector Regression - SVR'''
def tuning_hyperparameter_SVR() :
    from itertools import product

    tuning_result_SVR = ''

    C_list = np.linspace(1, 15, 15) 
    epsilon_list = np.linspace(0.005, 0.1, 20)

    best_f1 = -1
    best_threshold = None
    best_params = {}
    best_model = None

    for C_val, eps_val in product(C_list, epsilon_list):
        print(f"Training SVR with C={C_val}, epsilon={eps_val}...")
        tuning_result_SVR += f"Training SVR with C={C_val}, epsilon={eps_val}...\n"

        model = train_model.train_SVR(penalize=C_val, epsilon=eps_val)

        X_attack_hat = model.predict(X_attack_scaled)
        errors = np.mean((X_attack_scaled - X_attack_hat) ** 2, axis=1)

        threshold, f1 = find_best_threshold(errors, Y_attack)

        print(f"→ F1={f1:.4f} at threshold={threshold:.6f}")
        tuning_result_SVR += f"→ F1={f1:.4f} at threshold={threshold:.6f}"

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_params = {'C': C_val, 'epsilon': eps_val}
            best_model = model

    print("\nBest config found:")
    tuning_result_SVR += "\nBest config found:"
    print(f"  C = {best_params['C']}, epsilon = {best_params['epsilon']}")
    tuning_result_SVR += f"  C = {best_params['C']}, epsilon = {best_params['epsilon']}\n"
    print(f"  Threshold = {best_threshold:.6f}, F1 = {best_f1:.4f}")
    tuning_result_SVR += f"  Threshold = {best_threshold:.6f}, F1 = {best_f1:.4f}"

    with open('src/result/tuning/tuning_hyperparameter_SVR.txt', 'w') as f:
        f.write(tuning_result_SVR)

    save_obj = {
    'threshold': best_threshold,
    'model': best_model
    }
    joblib.dump(save_obj, 'saved_model/svr/svr_best.pkl')

    best_model_dict = {
        "model": model,
        "threshold": best_threshold,
        "best_f1" : best_f1
    }

    return best_model_dict

tuning_hyperparameter_SVR()
'''--------------------------'''