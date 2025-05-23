import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import joblib
import os

def load_train_data(filepath, train_ratio=0.7):
    """
    Load and split BATADAL dataset.

    Args:
        filepath (str): path đến file .csv
        train_ratio (float): phần trăm dữ liệu dùng để train (mặc định 0.7)

    Returns:
        X_train: ma trận cảm biến (70% đầu) [n_train, n_features]
        X_val: ma trận cảm biến (30% cuối) [n_val, n_features]
        Y_val: vector nhãn ATT_FLAG tương ứng [n_val]
        sensor_cols: tên các cột cảm biến
    """
    df = pd.read_csv(filepath)

    df = df.iloc[:, 1:]
    sensor_cols = [col for col in df.columns if col not in ['DATETIME', 'ATT_FLAG']]

    X_all = df[sensor_cols].values
    Y_all = df['ATT_FLAG'].values

    # split benign train and validation
    n_total = len(df)
    n_train = int(train_ratio * n_total)

    X_train = X_all[:n_train]
    X_val   = X_all[n_train:]
    Y_val   = Y_all[n_train:]

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    filepath = 'scaler/training_scaler.save'
    if not os.path.exists(filepath):
        try:
            joblib.dump(scaler, filepath)
        except Exception as e:
            print(f"Error when saving scaler: {e}")
    else:
        print(f"File existed at: {filepath}.")

    return X_train_scaled, X_val_scaled, Y_val, sensor_cols

def load_attack_data(filepath) :

    df = pd.read_csv(filepath)

    df = df.iloc[:, 1:]
    sensor_cols = [col for col in df.columns if col not in ['DATETIME', 'ATT_FLAG']]

    X_attack = df[sensor_cols].values
    Y_attack = df['ATT_FLAG'].values

    attack_scaler = joblib.load('scaler/training_scaler.save')

    X_attack_scaled = attack_scaler.transform(X_attack)

    return X_attack_scaled, Y_attack

def load_test_data(filepath, number) :

    df_test = None

    if 0 <= number and number <= 2 :
        df_test = pd.read_csv(f'{filepath}/test_dataset_{number}.csv')
    else :
        print("Please type 1 or 2 for test_dataset_1 or test_dataset_2")

    df_test = df_test.iloc[:, 1:]
    sensor_cols = [col for col in df_test.columns if col not in ['DATETIME', 'ATT_FLAG']]

    X_test = df_test[sensor_cols].values
    Y_test = df_test['ATT_FLAG'].values

    test_scaler = joblib.load('scaler/training_scaler.save')

    X_test_scaled = test_scaler.transform(X_test)

    return X_test_scaled, Y_test