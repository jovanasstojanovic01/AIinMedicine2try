import cv2
import numpy as np
import os

test_mask_path = "./data/refuge2/test/mask"
prva_maska = os.listdir(test_mask_path)[0]
img = cv2.imread(os.path.join(test_mask_path, prva_maska), cv2.IMREAD_GRAYSCALE)
print("Jedinstvene vrednosti u test maski:", np.unique(img))