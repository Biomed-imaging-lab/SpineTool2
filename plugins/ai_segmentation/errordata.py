import numpy as np
import glob
from tqdm import tqdm

files = glob.glob("C:/Users/Student/datasets/dataset_stage3_npz_labid/val/**/*.npz", recursive=True)
for f in tqdm(files):
    try:
        data = np.load(f, allow_pickle=True)
        _ = data['image'] # Пробуем реально прочитать данные
        _ = data['mask']
    except Exception as e:
        print(f"\nПоврежденный файл: {f}")
        print(f"Ошибка: {e}")