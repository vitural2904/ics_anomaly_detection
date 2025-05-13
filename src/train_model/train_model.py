from src.loader.data_loader import load_train_data
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

X_train, X_val, Y_val, sensor_cols = load_train_data('data/BATADAL/train_dataset.csv')

def threshold_select(model) :
    
    X_val_hat = model.predict(X_val)
    errors = np.mean((X_val - X_val_hat) ** 2, axis=1)
    benign_errors = errors[Y_val == 0]
    tau = np.percentile(benign_errors, 99.5)
    # print(f"Threshold τ (99.5th percentile): {tau:.6f}")

    return tau

'''LinearRegression - Ridge'''
def train_linear_regression_ridge_MOR(penalize_l2_norm = 1.0) :

    base_model = Ridge(penalize_l2_norm)
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, X_train)

    return model
'''--------------------------'''


'''LinearRegression - Ridge, but we use 43 Ridge models - with 43 seperate sensors'''
def train_linear_regression_ridge_split() :

    models = []
    for i in range(X_train.shape[1]):           # X_train.shape[0] is number of rows, otherwise, number of cols
        reg = Ridge(alpha=1.0)
        reg.fit(X_train, X_train[:, i])         # reconstruct sensor i with knowlegde from all the system
        models.append(reg)

    return models

def threshold_select_ridge_split(models):

    X_val_hat = np.zeros_like(X_val)
    for i, model in enumerate(models):
        X_val_hat[:, i] = model.predict(X_val)
    
    errors = np.mean((X_val - X_val_hat) ** 2, axis=1)
    benign_errors = errors[Y_val == 0]
    tau = np.percentile(benign_errors, 99.5)
    return tau
'''--------------------------'''


'''Suport Vector Regression - SVR'''
def train_SVR(penalize=1.0, epsilon=0.1) :
    
    # init svr and multi svr model
    svr = SVR(kernel='rbf', C=penalize, epsilon=epsilon)    # needs tuning for best performance

    multi_svr = MultiOutputRegressor(svr)

    multi_svr.fit(X_train, X_train)

    return multi_svr
'''--------------------------'''