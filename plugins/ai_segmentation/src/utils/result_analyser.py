import numpy as np
from tifffile import imwrite
import os
import json
import numpy as np
import argparse
import random
import cv2
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count

from tifffile import imread
from scipy.ndimage import gaussian_filter, rotate
from skimage.filters import threshold_multiotsu
import os
BASE_SCALE = [0.1, 0.1]
def scale_image(image, current_scale):
    if current_scale is None: return image
    scale_y = current_scale[1] / BASE_SCALE[0]
    scale_x = current_scale[2] / BASE_SCALE[1]
    if abs(scale_y - 1.0) < 0.01 and abs(scale_x - 1.0) < 0.01: return image
    new_y = int(np.round(image.shape[1] * scale_y))
    new_x = int(np.round(image.shape[2] * scale_x))
    interp = cv2.INTER_NEAREST if np.issubdtype(image.dtype, np.integer) else cv2.INTER_AREA
    resized = np.zeros((image.shape[0], new_y, new_x), dtype=image.dtype)
    for i in range(image.shape[0]):
        resized[i] = cv2.resize(image[i], (new_x, new_y), interpolation=interp)
    return resized

def create_diff_mask_binary(predicted, label):
	"""TP=White, FN=Red, FP=Blue"""
	pred_bin = (predicted > 0.5)
	label_bin = (label > 0.5)

	rgb = np.zeros((*predicted.shape, 3), dtype=np.uint8)

	# Intersection (White)
	rgb[pred_bin & label_bin] = [255, 255, 255]
	# False Negative (Red)
	rgb[label_bin & ~pred_bin] = [250, 0, 0]
	# False Positive (Blue)
	rgb[pred_bin & ~label_bin] = [0, 0, 245]

	return rgb

if __name__ == "__main__":
    from tifffile import imread
    N="2"
    stage3_path = "C:/Users/Student/datasets/origin_scale_preds/in_vitro_2_final.tif"
    stage2_path = "C:/Users/Student/datasets/necks_scale_preds/in_vitro_2_necks.tif"
    stage1_path = "C:/Users/Student/datasets/L_dendrite_scale_preds/in_vitro_2_L_dendrite.tif"
    label_path = "C:/Users/Student/datasets/in_vitro/2/binarization.tif"
    necks_path = "C:/Users/Student/datasets/in_vitro/2/necks.tif"
    area_path = "C:/Users/Student/datasets/in_vitro/2/areas.tif"
    st3_img = imread(stage3_path)
    st2_img = imread(stage2_path)
    st1_img = imread(stage1_path)
    label_img = imread(label_path)
    necks_img = imread(necks_path)

    labels_img = scale_image(label_img, [ 0.1, 0.022, 0.022 ])
    necks_imgs = scale_image(necks_img, [ 0.1, 0.022, 0.022 ])

    if os.path.exists(area_path):
        area = imread(area_path)
        area[area > 0] = 1
        necks_img = necks_img * area
        areas = scale_image(area, [ 0.1, 0.022, 0.022 ])
        labels_img = labels_img * areas
        st1_img = st1_img * areas
        st2_img = st2_img * areas
        st3_img = st3_img * area
        necks_imgs = necks_imgs * areas


    diff_1label = create_diff_mask_binary(st1_img, labels_img)
    diff_2label = create_diff_mask_binary(st2_img, necks_imgs)
    diff_3label = create_diff_mask_binary(st3_img, necks_img)
    diff_21 = create_diff_mask_binary(st2_img, st1_img)
    
    out_1label = "2/1Sdiff_1label_vitro2.tif"
    out_2label = "2/1Sdiff_2label_vitro2.tif"
    out_3label = "2/Sdiff_3label_vitro2.tif"
    out_21 = "2/1Sdiff_21_vitro2.tif"
    imwrite(out_1label, diff_1label, photometric='rgb')
    imwrite(out_2label, diff_2label, photometric='rgb')
    imwrite(out_3label, diff_3label, photometric='rgb')
    imwrite(out_21, diff_21, photometric='rgb')