import os
import pywt
from  skimage.io import imread, imshow
from config import DATASET_PATH

def img2fv(img_path, depth=1):
  # Load img
  feacture = imread(img_path)
  # Img to feature vector
  LL = feacture
  for i in range(depth):
    LL, (LH, HL, HH) = pywt.dwt2(LL, 'haar')
  fv = LL.flatten()
  # Return
  return fv


def paths_dict(img_limit=None):
    # Init emotions dict
    emotions_dict = {}
    # Create dict of emotions dict with filepaths
    for e in os.listdir(DATASET_PATH):
        # Get list of img paths
        imgpath_ls = os.listdir(DATASET_PATH+"/"+e)
        # Save list of img paths
        if img_limit:
            emotions_dict[e] = imgpath_ls[:img_limit]
        else:
            emotions_dict[e] = imgpath_ls
    # Return
    return emotions_dict