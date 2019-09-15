import pandas as pd
import h5py
from keras.models import load_model
import sys
data_base_path = 'C:\\Users\\ariel\\workspace\\deep_moon_data'
#data_base_path = '/mnt/disks/disk0/deep_moon_working_dir/data'
deep_moon_path = 'C:\\Users\\ariel\\workspace\\DeepMoon'
#deep_moon_path = '/mnt/disks/disk0/deep_moon_working_dir/DeepCrater'

sys.path.append(deep_moon_path)
sys.path.append(deep_moon_path + "\\utils")

import score_utils as sc
import utils.processing as proc
model = load_model(data_base_path + "\\Silburt\\model_keras2.h5")

