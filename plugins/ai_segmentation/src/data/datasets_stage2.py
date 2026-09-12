import torch
from torch.utils.data import Dataset
import numpy as np
import glob
from tifffile import imread
import os
from scipy.ndimage import gaussian_filter
from skimage.filters import threshold_multiotsu
from data_preprocessing.stage2_with_skeleton import calculate_interim_image, find_optimal_j0, calculate_isi
from data_preprocessing.stage1 import normalize_minmax, normalize_image


class Stage2TrainDataset(Dataset):
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

		# Данные уже имеют форму [4, 64, 64, 64] и тип float32
		image = torch.from_numpy(data['image'])

		# Маска имеет форму [64, 64, 64], добавляем канал -> [1, 64, 64, 64]
		mask = torch.from_numpy(data['mask']).float()
		borders = torch.from_numpy(data['borders']).long()
		return {
			"image": image,
			"mask": mask,
			"borders": borders
		}


class Stage2InferenceDataset(Dataset):
	"""
	Датасет "Скользящее окно" для инференса больших Tiff файлов.
	"""

	def __init__(self, tiff_path, current_scale, l_dendrite_path, area_path=None, patch_size=(64, 64, 64), overlap=16, sigma=1.0):
		self.patch_size = np.array(patch_size)
		self.overlap = overlap
		self.stride = (self.patch_size - self.overlap).astype(int)

		# 1. Загрузка
		raw_image = imread(tiff_path).astype(np.float32)
		l_dendrite = imread(l_dendrite_path).astype(np.float32)
		if area_path and os.path.exists(area_path):
			area = imread(area_path).astype(np.float32)
			area[area > 0] = 1.0
			if raw_image.shape == area.shape:
				raw_image = raw_image * area
			else:
				print(f"Shape mismatch {raw_image.shape} vs {area.shape}")

		image = np.zeros_like(raw_image)
		# Гаусс
		image = normalize_image(raw_image, current_scale)

		glob_norm = normalize_minmax(image)
		i_interim = calculate_interim_image(raw_image, current_scale)
		l_bin = (l_dendrite > 0.5).astype(np.uint8)

		j0 = find_optimal_j0(i_interim, l_bin)
		print(f"Dataset Info: Calculated global j0 = {j0} for inference.")
        
		I_si = calculate_isi(i_interim, j0)
	
		# Сохраняем предобработанные карты в памяти
		self.I_c = image
		self.glob_norm = glob_norm
		self.L_dendrite = l_dendrite
		self.I_si = I_si

		self.shape = image.shape

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
		return np.pad(chunk, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0)#, mode='constant', constant_values=0

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
		p_ldendrite = self.L_dendrite[z:z_end, y:y_end, x:x_end]
		p_isi = self.I_si[z:z_end, y:y_end, x:x_end]

		# Дополняем до 64x64x64, если кусок обрезан
		p_Ic = self.__pad_chunk(p_Ic)
		p_glob_norm = self.__pad_chunk(p_glob_norm)
		p_ldendrite = self.__pad_chunk(p_ldendrite)
		p_isi = self.__pad_chunk(p_isi)

		# Локальные вычисления (на лету для патча)
		p_local_norm = normalize_minmax(p_Ic)

		# Сборка каналов

		input_tensor = np.stack([p_glob_norm, p_local_norm, p_ldendrite, p_isi], axis=0).astype(np.float32)

		return {
			"image": torch.from_numpy(input_tensor),
			"coords": torch.tensor([z, y, x]),
			"valid_shape": torch.tensor([valid_z, valid_y, valid_x])
		}