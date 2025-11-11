import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, precision_score
from preprocess import load_training_data, load_validation_data

# -------------------------
# Пути к данным
# -------------------------
TRAIN_DIR = "data/training_data"
VAL_DIR = "data/validation_data/C-NMC_test_prelim_phase_data"
VAL_CSV = "data/validation_data/C-NMC_test_prelim_phase_data_labels.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------
# Загружаем данные
# -------------------------
print("🚀 Загружаем тренировочные данные...")
X_train, y_train = load_training_data(TRAIN_DIR)
print(f"✅ Тренировочных образцов: {len(y_train)}")

print("🚀 Загружаем валидационные данные...")
X_val, y_val = load_validation_data(VAL_DIR, VAL_CSV)
print(f"✅ Валидационных образцов: {len(y_val)}")

# -------------------------
# Определяем модели
# -------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=200, random_state=42)
}

results_summary = {}

# -------------------------
# Обучаем модели и сохраняем
# -------------------------
for name, model in models.items():
    print(f"\n🔹 Обучаем {name}...")
    model.fit(X_train, y_train)
    
    # Предсказания на валидации
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:,1] if hasattr(model, "predict_proba") else y_pred
    
    # Метрики
    accuracy = accuracy_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_prob)
    recall = recall_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    
    print(f"📊 {name}:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   ROC-AUC: {roc_auc:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   Precision: {precision:.4f}")
    
    # Сохраняем модель
    model_path = os.path.join(MODEL_DIR, f"{name.replace(' ', '_').lower()}_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    # Добавляем в results_summary
    results_summary[name] = {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "recall": recall,
        "precision": precision
    }

# -------------------------
# Сохраняем результаты для GUI
# -------------------------
summary_path = os.path.join(MODEL_DIR, "results_summary.pkl")
with open(summary_path, "wb") as f:
    pickle.dump(results_summary, f)

print(f"\n✅ Все модели обучены и сохранены в '{MODEL_DIR}/'")
print(f"✅ Результаты сохранены в '{summary_path}'")
