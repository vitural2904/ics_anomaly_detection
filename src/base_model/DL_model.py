import numpy as np

'''Convolutional Neural Network - Autoencoder (CNN - AE)'''
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Reshape, GlobalAveragePooling1D, Dense
import tensorflow as tf

def build_cnn_autoencoder(input_shape, num_layers, units_per_layer, kernel_size=3):

    assert num_layers == len(units_per_layer), "units_per_layer phải có độ dài = num_layers"

    # Dữ liệu gốc (system state)
    input_layer = Input(shape=input_shape)
    x = input_layer

    # -------- Encoder --------
    # Dữ liệu dần bị nén và trừu tượng hoá
    for i in range(num_layers):
        x = Conv1D(filters=units_per_layer[i], kernel_size=kernel_size, activation='relu', padding='same')(x)
        x = MaxPooling1D(pool_size=2, padding='same')(x)

    # Phòng trung tâm – mô hình đã học cách nén toàn bộ hiểu biết về mqh phi tuyến vào một vector đặc trưng
    shape_before_flatten = x.shape[1:]
    x = Flatten()(x) # Chuyển đổi thành vector 1 chiều

    # -------- Bottleneck --------
    # Có thể thêm Dense nếu muốn
    x = Dense(units=64, activation='relu')(x)
    x = Dense(units=int(tf.reduce_prod(shape_before_flatten)), activation='relu')(x)
    x = Reshape(target_shape=shape_before_flatten)(x)

    # -------- Decoder --------
    # Hành lang ngược, dữ liệu được giải nén và khôi phục lại kích thước ban đầu
    for i in reversed(range(num_layers)):
        x = Conv1D(filters=units_per_layer[i], kernel_size=kernel_size, activation='relu', padding='same')(x)
        x = UpSampling1D(size=2)(x)

    # Output layer (đảo ngược lại với đầu vào), đầu ra là dữ liệu đã được khôi phục (không giống nhau về giá trị)
    x = Conv1D(filters=input_shape[1], kernel_size=kernel_size, activation='sigmoid', padding='same')(x)
    x = GlobalAveragePooling1D()(x)
    output_layer = Reshape((1, input_shape[1]))(x)


    model = Model(inputs=input_layer, outputs=output_layer)

    return model

'''--------------------------'''

