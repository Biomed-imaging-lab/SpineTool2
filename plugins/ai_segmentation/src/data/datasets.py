import torch
from torch.utils.data import Dataset
import numpy as np
import glob
from tifffile import imread
import os
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_multiotsu
from data_preprocessing.stage1 import normalize_minmax, normalize_image
class DendriteTrainDataset(Dataset):
	"""
	Датасет для обучения. Ожидает список словарей с путями к файлам.
	"""

	def __init__(self, data_dir):
		search_path = os.path.join(data_dir, "**", "*.npz")
		self.files = glob.glob(search_path, recursive=True)
		if not self.files:
			print(f"Warning: No files found in {data_dir}")

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):
		# Загружаем сжатый файл
		# allow_pickle=True нужен для безопасности numpy, для npz это ок
		data = np.load(self.files[idx], allow_pickle=True)

		# Данные уже имеют форму [3, 64, 64, 64] и тип float32
		image = torch.from_numpy(data['image'])

		# Маска имеет форму [64, 64, 64], добавляем канал -> [1, 64, 64, 64]
		mask = torch.from_numpy(data['mask']).float().unsqueeze(0)
		borders = torch.from_numpy(data['borders']).long()
		return {
			"image": image,
			"mask": mask,
			"borders": borders
		}


class DendriteInferenceDataset(Dataset):
	"""
	Датасет "Скользящее окно" для инференса больших Tiff файлов.
	"""

	def __init__(self, tiff_path, current_scale, area_path=None, patch_size=(64, 64, 64), overlap=16, sigma=1.0, otsu_classes=4, pad=16):
		self.patch_size = np.array(patch_size)
		self.overlap = overlap
		self.stride = (self.patch_size - self.overlap).astype(int)
		self.pad = pad

		# 1. Загрузка
		raw_image = imread(tiff_path).astype(np.float32)

		if area_path and os.path.exists(area_path):
			area = imread(area_path).astype(np.float32)
			area[area > 0] = 1.0
			if raw_image.shape == area.shape:
				raw_image = raw_image * area
			else:
				print(f"Shape mismatch {raw_image.shape} vs {area.shape}")

		I_c = normalize_image(raw_image, current_scale)

		if self.pad > 0:
			pads = ((self.pad, self.pad), 
					(self.pad, self.pad), 
					(self.pad, self.pad))
			I_c = np.pad(I_c, pads, mode='reflect')
		
		glob_norm = normalize_minmax(I_c)

		try:
			th = threshold_multiotsu(I_c, classes=otsu_classes)
			glob_otsu = np.digitize(I_c, th).astype(np.float32)
		except:
			glob_otsu = np.zeros_like(I_c)

		# Сохраняем предобработанные карты в памяти
		self.I_c = I_c
		self.glob_norm = glob_norm
		self.glob_otsu = glob_otsu
		self.shape = I_c.shape
		self.otsu_classes = otsu_classes

		# Генерация сетки координат
		self.coords = self._generate_grid()

	def _generate_grid(self):
		coords = []
		z_steps = range(0, self.shape[0], self.stride[0])
		y_steps = range(0, self.shape[1], self.stride[1])
		x_steps = range(0, self.shape[2], self.stride[2])
		for z in z_steps:
			for y in y_steps:
				for x in x_steps:
					coords.append((z, y, x))
		return coords

	def __pad_chunk(self, chunk):
		"""
		Дополняет кусок нулями до self.patch_size, если он меньше.
		"""
        # Текущий размер куска
		cz, cy, cx = chunk.shape
		dz, dy, dx = self.patch_size
        
        # Если размер совпадает, возвращаем как есть
		if cz == dz and cy == dy and cx == dx:
			return chunk
            
        # Вычисляем сколько не хватает
		pad_z = dz - cz
		pad_y = dy - cy
		pad_x = dx - cx
        
        # Делаем паддинг константой (0) справа/снизу
        # format: ((before, after), ...)
		return np.pad(chunk, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='edge')#, mode='constant', constant_values=0

	def __len__(self):
		return len(self.coords)

	def __getitem__(self, idx):
		z, y, x = self.coords[idx]
		dz, dy, dx = self.patch_size

		z_end = min(z + dz, self.shape[0])
		y_end = min(y + dy, self.shape[1])
		x_end = min(x + dx, self.shape[2])

		valid_z = z_end - z
		valid_y = y_end - y
		valid_x = x_end - x	
		# Вырезаем куски
		p_Ic = self.I_c[z:z_end, y:y_end, x:x_end]
		p_glob_norm = self.glob_norm[z:z_end, y:y_end, x:x_end]
		p_glob_otsu = self.glob_otsu[z:z_end, y:y_end, x:x_end]

		# Дополняем до 64x64x64, если кусок обрезан
		p_Ic = self.__pad_chunk(p_Ic)
		p_glob_norm = self.__pad_chunk(p_glob_norm)
		p_glob_otsu = self.__pad_chunk(p_glob_otsu)

		# Локальные вычисления (на лету для патча)
		min_v, max_v = np.min(p_Ic), np.max(p_Ic)
		if max_v > min_v:
			p_local_norm = (p_Ic - min_v) / (max_v - min_v)
		else:
			p_local_norm = np.zeros_like(p_Ic)

		try:
			th = threshold_multiotsu(p_Ic, classes=self.otsu_classes)
			p_local_otsu = np.digitize(p_Ic, th).astype(np.float32)
		except:
			p_local_otsu = np.zeros_like(p_Ic)

		# Сборка каналов
		ch3 = (p_glob_otsu + p_local_otsu) / 6.0

		input_tensor = np.stack([p_glob_norm, p_local_norm, ch3], axis=0).astype(np.float32)

		return {
			"image": torch.from_numpy(input_tensor),
			"coords": torch.tensor([z, y, x]),
			"valid_shape": torch.tensor([valid_z, valid_y, valid_x])
		}