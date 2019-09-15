#!/usr/bin/env python
from __future__ import absolute_import, division, print_function
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import sys
sys.path.append('../DeepMoon')
import input_data_gen as igen
import time
import numpy as np
import os.path
import create_data_set

def getRandomCrop(size0, size1, rawlen_range):
    # Determine image size to crop.
    rawlen_min = np.log10(rawlen_range[0])
    rawlen_max = np.log10(rawlen_range[1])
    rawlen = float('inf')
    # this is log dist
    while rawlen >= size0 or rawlen >= size1:
        rawlen = int(10**np.random.uniform(rawlen_min, rawlen_max))  
    assert rawlen < size0, "rawlen({}) < size0({})".format(rawlen , size0)
    assert rawlen < size1, "rawlen({}) < size1({})".format(rawlen , size1)
    xc = np.random.randint(0, size0 - rawlen)
    yc = np.random.randint(0, size1 - rawlen)
    box = np.array([xc, yc, xc + rawlen, yc + rawlen], dtype='int32')
    return box

def get_craters(lroc_csv_path, head_csv_path,  sub_cdim, R_km):
    craters = igen.ReadLROCHeadCombinedCraterCSV(filelroc=lroc_csv_path,
                                                 filehead=head_csv_path)
    craters = igen.ResampleCraters(craters, sub_cdim, None, arad=R_km)
    return craters 
def get_image(source_image_path, sub_cdim ,source_cdim):
    # Read source image and crater catalogs.
    assert os.path.exists(source_image_path)
    img = Image.open(source_image_path).convert("L")

    # Sample subset of image.  Co-opt igen.ResampleCraters to remove all
    # craters beyond cdim (either sub or source).
    if sub_cdim != source_cdim:
        img = igen.InitialImageCut(img, source_cdim, sub_cdim)
    return img

def get_random_crop_list(n, rawlen_range, img_size):
    box_list = []
    for i in range(n):
        box = getRandomCrop(img_size[0], img_size[1], rawlen_range)        
        box_list.append(box)
    return box_list

def create_cropped_images_set(source_image_path, lroc_csv_path, head_csv_path,outhead, amt):
    rawlen_range = [500, 6500]
    source_cdim = [-180., 180., -60., 60.]
    sub_cdim = [-18., 18., -6., 6.]
    R_km = 1737.4
    img = get_image(source_image_path, sub_cdim ,source_cdim)
    craters = get_craters(lroc_csv_path, head_csv_path,  sub_cdim, R_km)
    box_list = get_random_crop_list(amt, rawlen_range, img.size)
    create_data_set.GenDataset(box_list, img, craters, outhead, R_km, cdim=sub_cdim)

#variables    
source_image_path = "../data/Silburt/LunarLROLrocKaguya_118mperpix.png"
lroc_csv_path = "../DeepCrater/catalogues/LROCCraters.csv"
head_csv_path = "../DeepCrater/catalogues/HeadCraters.csv"
outhead = "../data/my_test_data/train"
amt = 30

start_time = time.time()
create_cropped_images_set(source_image_path, lroc_csv_path, head_csv_path,outhead, amt)
elapsed_time = time.time() - start_time
print("Time elapsed: {0:.1f} min".format(elapsed_time / 60.))
