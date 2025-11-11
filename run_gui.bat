@echo off
echo Установка всех необходимых библиотек...

python -m pip install --upgrade pip

python -m pip install --upgrade numpy pandas scikit-learn xgboost scikit-image matplotlib openpyxl tifffile

echo Библиотеки установлены / обновлены.

if not exist "models" mkdir models


echo Запуск GUI...
python src/gui_app.py

pause
