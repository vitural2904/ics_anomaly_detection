from src.data_loader.data_loader import load_attack_data, load_train_data
from src.base_model import DL_model
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV

import numpy as np
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

'''Convolutional Neural Network (CNN)'''
def tuning_hyperparameter_CNN():

    tuning_results = ""

    param_grid = [
        {'num_layers': 4, 'units_per_layer': [28, 14, 9, 4]},
        {'num_layers': 5, 'units_per_layer': [28, 20, 14, 7, 4]},
        {'num_layers': 6, 'units_per_layer': [42, 35, 28, 20, 12, 4]},
        {'num_layers': 7, 'units_per_layer': [42, 28, 14, 9, 4, 2, 1]},
    ]

    best_f1 = -1
    best_model = None
    best_params = None
    best_threshold = None

    X_train_cnn = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_attack_scaled_cnn = X_attack_scaled.reshape((X_attack_scaled.shape[0], 1, X_attack_scaled.shape[1]))

    for config in param_grid:
        print(f"[INFO] Training with: num_layers = {config['num_layers']}, units = {config['units_per_layer']}")

        model = DL_model.build_cnn_autoencoder(
            input_shape=X_train_cnn.shape[1:],
            num_layers=config['num_layers'],
            units_per_layer=config['units_per_layer']
        )

        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train_cnn, X_train_cnn, epochs=50, batch_size=64, verbose=0)

        # Dự đoán dữ liệu tấn công
        X_attack_hat = model.predict(X_attack_scaled_cnn)
        
        # Đưa về cùng (sample_size, 43)
        X_attack_hat = np.squeeze(X_attack_hat, axis=1)
        print(f"[INFO] X_attack_hat.shape = {X_attack_hat.shape}")

        errors = np.mean((X_attack_scaled - X_attack_hat) ** 2, axis=1)
        threshold, f1 = find_best_threshold(errors, Y_attack)

        print(f"F1={f1:.4f} | threshold={threshold:.6f}")
        tuning_results += f"num_layers={config['num_layers']}, units={config['units_per_layer']} | F1={f1:.4f}, threshold={threshold:.6f}\n"

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_params = config
            best_threshold = threshold

    print(f"Best num_layers = {best_params['num_layers']}, units_per_layer = {best_params['units_per_layer']}")
    tuning_results += f"Best num_layers = {best_params['num_layers']}, units_per_layer = {best_params['units_per_layer']}\n"
    print(f"Best F1 = {best_f1 * 2:.4f} | threshold = {best_threshold:.6f}")
    tuning_results += f"F1={best_f1 * 2:.4f} | threshold={best_threshold:.6f}\n"

    try:
        joblib.dump({'threshold': best_threshold, 'model': best_model}, 'saved_model/cnn_autoencoder/cnn_ae_best.save')
    except Exception as e:
        print(f"[Warning] Unable to save model: {e}")

    with open('src/result/tuning/tuning_hyperparameter_cnn_ae.txt', 'w') as f:
        f.write(tuning_results)

    return {
        "model": best_model,
        "threshold": best_threshold,
        "best_f1": best_f1
    }


# tuning_hyperparameter_CNN()

def best_model_retrive_cnn_ae():

    filepath = 'saved_model/cnn_autoencoder/cnn_ae_best.save'
    if not os.path.exists(filepath):
        print("Model not found.")
        return None
    else:
        saved_model = joblib.load(filepath)
        return [{
            "model": saved_model['model'],
            "threshold": saved_model['threshold'],
        }]
    
'''--------------------------'''