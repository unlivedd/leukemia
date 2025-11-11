import os
import pickle
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button
from skimage.io import imread
from skimage.transform import resize
from preprocess import extract_features  # используем только те признаки, что на обучении

# -------------------------
# Загружаем модели и результаты
# -------------------------
MODEL_DIR = "models"
model_paths = {
    "Logistic Regression": os.path.join(MODEL_DIR, "logistic_regression_model.pkl"),
    "Random Forest": os.path.join(MODEL_DIR, "random_forest_model.pkl"),
    "XGBoost": os.path.join(MODEL_DIR, "xgboost_model.pkl")
}

models = {name: pickle.load(open(path, "rb")) for name, path in model_paths.items()}

results_summary_path = os.path.join(MODEL_DIR, "results_summary.pkl")
results_summary = pickle.load(open(results_summary_path, "rb"))

# -------------------------
# Функция предсказания
# -------------------------
def predict_image(img_path):
    img = imread(img_path)
    img = resize(img, (128,128), anti_aliasing=True)
    features = extract_features(img)
    features = np.array(features).reshape(1, -1)  # reshape важно!
    
    preds = {}
    for name, model in models.items():
        pred = model.predict(features)[0]
        prob = model.predict_proba(features)[0][1] if hasattr(model, "predict_proba") else None
        preds[name] = (pred, prob)
    return preds

# -------------------------
# GUI
# -------------------------
def load_single_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.bmp *.png *.jpg")])
    if not file_path:
        return
    preds = predict_image(file_path)
    
    # Вывод результатов на экран
    result_text = f"Файл: {os.path.basename(file_path)}\n\n"
    for name, (pred, prob) in preds.items():
        cls = "Лейкемия" if pred == 1 else "Норма"
        prob_text = f"{prob:.2f}" if prob is not None else "N/A"
        result_text += f"{name}: {cls} (вероятность: {prob_text})\n"
    
    # Метрики обучения
    result_text += "\n📊 Результаты на валидации:\n"
    for name, metrics in results_summary.items():
        result_text += (f"{name}: Accuracy={metrics['accuracy']:.2f}, "
                        f"ROC-AUC={metrics['roc_auc']:.2f}, "
                        f"Recall={metrics['recall']:.2f}, "
                        f"Precision={metrics['precision']:.2f}\n")
    
    result_label.config(text=result_text)

# -------------------------
# Окно Tkinter
# -------------------------
root = tk.Tk()
root.title("Leukemia Detection GUI")

load_button = Button(root, text="Загрузить клетку", command=load_single_image)
load_button.pack(pady=10)

result_label = Label(root, text="", justify="left")
result_label.pack(padx=10, pady=10)

root.mainloop()
