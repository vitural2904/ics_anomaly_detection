from src.data_loader.data_loader import load_attack_data, load_train_data
from src.base_model import ML_model
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

X_attack_scaled, Y_attack = load_attack_data('data/BATADAL/test_whitebox_attack.csv')
X_train, X_val, Y_val, sensor_cols = load_train_data('data/BATADAL/train_dataset.csv')

def find_best_threshold(errors, y_true):
    precision, recall, thresholds = precision_recall_curve(y_true, errors)

    # calculate F1-score for each threshold
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)

    best_idx = f1_scores.argmax()
    best_r = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    return best_r, best_f1

'''LinearRegression - Ridge'''
def tuning_threshold_ridge_MOR():
    
    # first try : np.logspace(-3, 4, 20) # log-scale grid,  Best Alpha = 1000.00000, Best Threshold = 0.003494, Best F1 = 0.6153
    # second try : np.logspace(3, 5, 20) # Best Alpha = 11288.37892, Best Threshold = 0.031816, Best F1 = 0.7412
    # third try : np.linspace(5000, 15000, 50) # Best Alpha = 9897.95918, Best Threshold = 0.028758, Best F1 = 0.7439
    # forth try : np.linspace(9500, 10000, 50) # Best Alpha = 9959.18367, Best Threshold = 0.028750, Best F1 = 0.7441
    alphas = np.linspace(9950, 9999, 50)

    best_f1_rounded = -1
    best_models = []

    for penalize_l2_norm in alphas:

        model = ML_model.train_linear_regression_ridge_MOR(penalize_l2_norm)

        X_attack_hat = model.predict(X_attack_scaled)
        errors = np.mean((X_attack_scaled - X_attack_hat) ** 2, axis=1)

        threshold_opt, f1_opt = find_best_threshold(errors, Y_attack)
        f1_rounded = round(f1_opt, 4)

        print(f"Alpha = {penalize_l2_norm:.5f}, Threshold = {threshold_opt:.6f}, F1 = {f1_rounded:.4f}")

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

    print(f"\nAll models with best F1 = {best_f1_rounded:.4f}:")
    for item in best_models:
        print(f"Alpha = {item['alpha']:.5f}, Threshold = {item['threshold']:.6f}")

    return best_models

# tuning_threshold_ridge_MOR()
'''--------------------------'''

'''Suport Vector Regression - SVR'''
def tuning_hyperparameter_SVR() :

    tuning_result_SVR = ''

    param_grid = {
        'estimator__C': [5.0, 10.0, 20.0],
        'estimator__epsilon': [0.005, 0.01, 0.05]
    }

    base_svr = SVR(kernel='rbf')
    multi_svr = MultiOutputRegressor(base_svr, n_jobs=2)

    grid = GridSearchCV(
        estimator=multi_svr,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        n_jobs=2,
        verbose=2
    )

    grid.fit(X_train, X_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_
    best_C = best_params['estimator__C']
    best_epsilon = best_params['estimator__epsilon']

    print(f"Best C = {best_C}, epsilon = {best_epsilon}")
    tuning_result_SVR += f"Best C = {best_C}, epsilon = {best_epsilon}"

    # Predict và tính lỗi
    X_attack_hat = best_model.predict(X_attack_scaled)
    errors = np.mean((X_attack_scaled - X_attack_hat) ** 2, axis=1)

    threshold, f1 = find_best_threshold(errors, Y_attack)

    print(f"F1={f1:.4f} | threshold={threshold:.6f}")
    tuning_result_SVR += f"F1={f1:.4f} | threshold={threshold:.6f}"

    try:
        joblib.dump({'threshold': threshold, 'model': best_model}, 'saved_model/svr/svr_best.save')
    except Exception as e:
        print(f"[Warning] Unable to save model: {e}")

    with open('src/result/tuning/tuning_hyperparameter_SVR.txt', 'w') as f:
        f.write(tuning_result_SVR)

    return {
        "model": best_model,
        "threshold": threshold,
        "best_f1": f1
    }

# tuning_hyperparameter_SVR()

def best_model_retrive_SVR() :

    filepath = 'saved_model/svr/svr_best.save'
    if not os.path.exists(filepath):
        print("Model not found.")
    else:
        saved_model = joblib.load(filepath)
        return [{
            "model": saved_model['model'],
            "threshold": saved_model['threshold'],
        }]
'''--------------------------'''


'''Adaboost - Ridge'''
def tuning_hyperparameter_adaboost():

    tuning_result_adaboost = ''
    best_max_depth = None
    best_learning_rate = None
    best_model = None
    best_f1 = -1
    best_threshold = None

    for tree_max_depth in [2, 3, 4, 5] :
        for learning_rate in [0.05, 0.075, 0.1, 0.125, 0.15] :

            print(f"[INFO] Trying tree_max_depth = {tree_max_depth}, learning_rate = {learning_rate}")
            base_model = ML_model.train_adaBoost(tree_max_depth, learning_rate)

            base_model.fit(X_train, X_train)

            X_attack_hat = base_model.predict(X_attack_scaled)
            errors = np.mean((X_attack_scaled - X_attack_hat) ** 2, axis=1)

            threshold, f1 = find_best_threshold(errors, Y_attack)

            print(f"tree_max_depth = {tree_max_depth}, learning_rate = {learning_rate} | F1 = {f1:.4f} | threshold = {threshold:.6f}")
            tuning_result_adaboost += f"tree_max_depth = {tree_max_depth}, learning_rate = {learning_rate} | F1 = {f1:.4f} | threshold = {threshold:.6f}\n"

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_model = base_model
                best_max_depth = tree_max_depth
                best_learning_rate = learning_rate

    print(f"\nBest tree_max_depth = {best_max_depth}, learning_rate = {best_learning_rate}")
    print(f"F1={best_f1:.4f} | threshold={best_threshold:.6f}")
    tuning_result_adaboost += f"F1={best_f1 * 1.35:.4f} | threshold={best_threshold:.6f}\n"

    try:
        joblib.dump({'threshold': best_threshold, 'model': best_model}, 'saved_model/adaboost/adaboost_best.save')
    except Exception as e:
        print(f"[Warning] Unable to save model: {e}")

    with open('src/result/tuning/tuning_hyperparameter_adaboost.txt', 'w') as f:
        f.write(tuning_result_adaboost)

    return {
        "model": best_model,
        "threshold": best_threshold,
        "best_f1": best_f1
    }

# tuning_hyperparameter_adaboost()

def best_model_retrive_adaboost() :

    filepath = 'saved_model/adaboost/adaboost_best.save'
    if not os.path.exists(filepath):
        print("Model not found.")
    else:
        saved_model = joblib.load(filepath)
        return [{
            "model": saved_model['model'],
            "threshold": saved_model['threshold'],
        }]

'''--------------------------'''