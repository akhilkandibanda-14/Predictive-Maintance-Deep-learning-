import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

# Import preprocessing pipeline
from preprocessing import train_preprocessing_pipeline

# Import lstm model
from lstm_model import build_lstm_model

def evaluate_model_on_test(model, X_test, y_test, model_name="Model", is_lstm=False):
    # Predict probabilities
    y_pred_prob = model.predict(X_test).ravel()
    
    # Predict classes
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    pr_auc = average_precision_score(y_test, y_pred_prob)
    
    return {
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    }, y_pred

def train_ann_fallback(X_train, y_train, X_val, y_val, class_weights, model_path):
    print("Training a fallback ANN baseline model for comparison...")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input
    ann_model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    ann_model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy']
    )
    
    ann_model.fit(
        X_train, y_train, 
        validation_data=(X_val, y_val), 
        epochs=50, 
        batch_size=64, 
        class_weight=class_weights, 
        verbose=0
    )
    ann_model.save(model_path)
    return ann_model

def main():
    csv_file = 'dataset/predictive_maintenance.csv'
    models_dir = 'models'
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Get preprocessed data
    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test, features = train_preprocessing_pipeline(csv_file, models_dir)
    
    # Reshape for LSTM: (samples, time_steps, features)
    X_train_lstm = np.expand_dims(X_train, axis=1)
    X_val_lstm = np.expand_dims(X_val, axis=1)
    X_test_lstm = np.expand_dims(X_test, axis=1)
    
    print(f"X_train_lstm shape: {X_train_lstm.shape}")
    print(f"X_validation_lstm shape: {X_val_lstm.shape}")
    print(f"X_test_lstm shape: {X_test_lstm.shape}")
    
    # Class weights from training data ONLY
    weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(weights))
    print(f"Class weights: {class_weights}")
    
    # 2. Build LSTM
    input_shape = (X_train_lstm.shape[1], X_train_lstm.shape[2])
    model = build_lstm_model(input_shape)
    model.summary()
    
    # 3. Training
    callbacks = [
        EarlyStopping(monitor='val_prc', mode='max', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5),
        ModelCheckpoint(os.path.join(models_dir, 'lstm_model.keras'), monitor='val_prc', mode='max', save_best_only=True)
    ]
    
    print("Training LSTM...")
    history = model.fit(
        X_train_lstm, y_train,
        validation_data=(X_val_lstm, y_val),
        epochs=100,
        batch_size=64,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save training plots
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['prc'], label='Train PR-AUC')
    plt.plot(history.history['val_prc'], label='Val PR-AUC')
    plt.title('PR-AUC over Epochs')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'lstm_training_history.png'))
    plt.close()
    
    # Load the best LSTM model saved during training
    model = tf.keras.models.load_model(os.path.join(models_dir, 'lstm_model.keras'))
    
    # 4. Test Evaluation
    print("\nEvaluating LSTM on Test Set...")
    lstm_metrics, y_pred_lstm = evaluate_model_on_test(model, X_test_lstm, y_test, "LSTM", is_lstm=True)
    
    print("\n--- LSTM Confusion Matrix ---")
    cm = confusion_matrix(y_test, y_pred_lstm)
    print(cm)
    tn, fp, fn, tp = cm.ravel()
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")
    
    print("\n--- LSTM Classification Report ---")
    print(classification_report(y_test, y_pred_lstm))
    
    # 5. Compare with ANN
    ann_model_path = os.path.join(models_dir, 'ann_model.keras')
    if not os.path.exists(ann_model_path):
        print("WARNING: ann_model.keras not found.")
        ann_model = train_ann_fallback(X_train, y_train, X_val, y_val, class_weights, ann_model_path)
    else:
        ann_model = tf.keras.models.load_model(ann_model_path)
    
    print("\nEvaluating ANN on Test Set...")
    ann_metrics, _ = evaluate_model_on_test(ann_model, X_test, y_test, "ANN", is_lstm=False)
    
    df_comparison = pd.DataFrame([ann_metrics, lstm_metrics])
    df_comparison.to_csv(os.path.join(results_dir, 'model_comparison.csv'), index=False)
    
    print("\n--- Model Comparison ---")
    print(df_comparison.to_string(index=False))
    
    # Plot comparison
    metrics_to_plot = ['Precision', 'Recall', 'F1', 'PR-AUC']
    df_plot = df_comparison.set_index('Model')[metrics_to_plot]
    
    df_plot.plot(kind='bar', figsize=(10, 6), colormap='viridis')
    plt.title('ANN vs LSTM Performance Comparison')
    plt.ylabel('Score')
    plt.ylim(0, 1.05)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'model_comparison.png'))
    plt.close()

if __name__ == '__main__':
    main()
