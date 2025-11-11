@echo off
echo Обновление pip и установка всех библиотек...
python -m pip install --upgrade pip

python -m pip install --upgrade numpy pandas scikit-learn xgboost scikit-image matplotlib openpyxl tifffile

echo Библиотеки установлены / обновлены.

if not exist "models" mkdir models

echo  Начинаем обучение моделей...
python src/train_models.py
if %ERRORLEVEL% neq 0 (
    echo  Произошла ошибка при обучении моделей!
    pause
    exit /b %ERRORLEVEL%
)
echo Модели успешно обучены и сохранены.

echo Запуск GUI...
python src/gui_app.py

pause
