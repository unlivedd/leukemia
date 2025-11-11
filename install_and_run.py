import subprocess
import sys
import os

# Установка зависимостей
subprocess.check_call([sys.executable,"-m","pip","install","--upgrade","pip"])
subprocess.check_call([sys.executable,"-m","pip","install","-r","requirements.txt"])

# Создание папки моделей
os.makedirs("models", exist_ok=True)

# Обучение моделей
print("Обучаем модели и выводим метрики...")
subprocess.check_call([sys.executable,"src/train_models.py"])

# Запуск GUI
print("Запускаем GUI...")
subprocess.check_call([sys.executable,"src/gui_app.py"])
