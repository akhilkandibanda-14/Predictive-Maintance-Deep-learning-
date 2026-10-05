import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

def build_lstm_model(input_shape):
    """
    Builds the LSTM model for predictive maintenance binary classification.
    
    IMPORTANT DATASET LIMITATION:
    The AI4I 2020 dataset does not contain genuine longitudinal observations 
    for individual machines with timestamps or machine-specific sequences.
    This LSTM architecture is used with sequence_length=1 solely as a controlled 
    comparison against an ANN, not for predicting future degradation based on 
    temporal sequences.
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.AUC(name='prc', curve='PR')
        ]
    )
    
    return model
