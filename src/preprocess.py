import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import graycomatrix, graycoprops
from skimage.transform import resize
from skimage.exposure import rescale_intensity

IMG_SIZE = (128, 128)

from skimage.feature import hog

def extract_hog_features(img, pixels_per_cell=(16,16), cells_per_block=(2,2), orientations=9):
    """
    Извлекает HOG-признаки из изображения.
    """
    gray = rgb2gray(img)
    features, _ = hog(gray,
                      orientations=orientations,
                      pixels_per_cell=pixels_per_cell,
                      cells_per_block=cells_per_block,
                      block_norm='L2-Hys',
                      visualize=True)
    return features

def load_final_data(img_path):
    """
    Загрузка одной картинки и извлечение признаков
    для предсказания модели через GUI.
    """
    from skimage.io import imread
    from skimage.transform import resize
    img = imread(img_path)
    img = resize(img, (128,128), anti_aliasing=True)
    features = extract_features(img)
    return np.array(features).reshape(1,-1)

def extract_features(img):
    """Извлекаем признаки из изображения."""
    gray = rgb2gray(img)
    gray = rescale_intensity(gray, out_range=(0, 255)).astype(np.uint8)

    glcm = graycomatrix(gray, [1], [0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]

    # Дополнительные признаки
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    brightness = np.mean(img)
    contrast_rgb = img.std()

    return [contrast, homogeneity, energy, correlation, mean_val, std_val, brightness, contrast_rgb]


def load_training_data(base_path):
    """Загрузка и подготовка данных из training_data."""
    X, y = [], []

    for fold in ["fold_0", "fold_1", "fold_2"]:
        for label, cls in enumerate(["hem", "all"]):  # 0 = нормальные, 1 = рак
            folder = os.path.join(base_path, fold, cls)
            for file in os.listdir(folder):
                if file.lower().endswith(('.bmp', '.png', '.jpg', '.jpeg')):
                    path = os.path.join(folder, file)
                    img = imread(path)
                    img = resize(img, IMG_SIZE, anti_aliasing=True)
                    features = extract_features(img)
                    X.append(features)
                    y.append(label)
    return np.array(X), np.array(y)


def load_validation_data(img_dir, labels_csv):
    """Загрузка и подготовка валидационных данных."""
    df = pd.read_csv(labels_csv)
    X, y = [], []

    for _, row in df.iterrows():
        img_name = row['new_names']
        label = row['labels']
        path = os.path.join(img_dir, img_name)

        if not os.path.exists(path):
            continue

        img = imread(path)
        img = resize(img, IMG_SIZE, anti_aliasing=True)
        features = extract_features(img)
        X.append(features)
        y.append(label)

    return np.array(X), np.array(y)
