import torch
from torch.utils.data import Dataset
import numpy as np
import glob
from tifffile import imread
import os

from data_preprocessing.stage2 import calculate_interim_image, find_optimal_j0, calculate_isi
from data_preprocessing.stage1 import normalize_minmax
from data_preprocessing.stage3 import scale_to_shape, apply_gaussian_3d_stack

class Stage3TrainDataset(Dataset):
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

		# Данные уже имеют форму [4, 64, 128, 128] и тип float32
		image = torch.from_numpy(data['image'])

		# Маска имеет форму [64, 128, 128], добавляем канал -> [1, 64, 128, 128]
		mask = torch.from_numpy(data['mask']).float().unsqueeze(0)
		borders = torch.from_numpy(data['borders']).long()
		return {
			"image": image,
			"mask": mask,
			"borders": borders
		}


class Stage3InferenceDataset(Dataset):
	"""
	Датасет "Скользящее окно" для инференса больших Tiff файлов.
	"""

	def __init__(self, tiff_path, current_scale, l_dendrite_path, necks_path, area_path=None, patch_size=(64, 128, 128), overlap=(32, 64, 64), sigma=1.0):
		self.patch_size = np.array(patch_size)
		self.overlap = np.array(overlap)
		self.stride = (self.patch_size - self.overlap).astype(int)

		# 1. Загрузка
		raw_img = imread(tiff_path).astype(np.float32)

		if area_path and os.path.exists(area_path):
			area = imread(area_path).astype(np.float32)
			area[area > 0] = 1.0
			if raw_img.shape == area.shape:
				raw_img = raw_img * area
			else:
				print(f"Shape mismatch {raw_img.shape} vs {area.shape}")

		self.shape = raw_img.shape

		l1_raw = imread(l_dendrite_path)
		l2_raw = imread(necks_path)

		i_interim = calculate_interim_image(raw_img, current_scale)
		l1_bin = (l1_raw > 0.5).astype(np.uint8)
		j0 = find_optimal_j0(i_interim, l1_bin)
		I_si = calculate_isi(i_interim, j0)
		
		l1_up = scale_to_shape(l1_raw, self.shape)
		l2_up = scale_to_shape(l2_raw, self.shape)
		isi_up = scale_to_shape(I_si, self.shape)
		l1_s = apply_gaussian_3d_stack(l1_up, sigma=1.0)
		l2_s = apply_gaussian_3d_stack(l2_up, sigma=1.0)
		isi_s = apply_gaussian_3d_stack(isi_up, sigma=1.0)

		I_norm = normalize_minmax(raw_img)
	
		# Сохраняем предобработанные карты в памяти
		self.I_norm = I_norm
		self.L1_s = l1_s
		self.L2_s = l2_s
		self.I_si_s = isi_s

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
		return np.pad(chunk, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0)

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
		p_I_norm = self.I_norm[z:z_end, y:y_end, x:x_end]
		p_l1_s = self.L1_s[z:z_end, y:y_end, x:x_end]
		p_l2_s = self.L2_s[z:z_end, y:y_end, x:x_end]
		p_isi_s = self.I_si_s[z:z_end, y:y_end, x:x_end]

		# Дополняем до 64x64x64, если кусок обрезан
		p_I_norm = self.__pad_chunk(p_I_norm)
		p_l1_s = self.__pad_chunk(p_l1_s)
		p_l2_s = self.__pad_chunk(p_l2_s)
		p_isi_s = self.__pad_chunk(p_isi_s)

		# Сборка каналов

		input_tensor = np.stack([p_I_norm, p_isi_s, p_l1_s, p_l2_s], axis=0).astype(np.float32)

		return {
			"image": torch.from_numpy(input_tensor),
			"coords": torch.tensor([z, y, x]),
			"valid_shape": torch.tensor([valid_z, valid_y, valid_x])
		}