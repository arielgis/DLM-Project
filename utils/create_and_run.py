import sys
pathstoadd = ['../find_craters_with_model/','../data_preparation/','../utils/']
[sys.path.append(dir) for dir in pathstoadd]
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import local funtions: 
import create_input
import generate_serial_input
import detect_craters_with_model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from time import time

model_path = '../../data/trained_model/Silburt/model_keras1.2.2.h5'
#model_path = '../../data/trained_model/Silburt/model_keras2.h5'
model = load_model(model_path)


def create_complete_set(source_image_path, source_cdim, sub_cdim, head_dir, crop_sizes_set, compression, catalog): 
    #default params
    R_km = 1737.4  
    deep_moon_path = os.path.abspath("../../DeepMoon")
    set_path = '../../data/model_input_images/{}_{}_{}_{}_{}'.format(head_dir, int(sub_cdim[0]), int(sub_cdim[1]), int(sub_cdim[2]), int(sub_cdim[3]))
    if not os.path.isdir(set_path):
         os.mkdir(set_path)

    img = create_input.get_image(source_image_path, sub_cdim ,source_cdim, compression)
    print("images size is {}".format(img.size))
    craters = create_input.get_craters(catalog, sub_cdim, R_km).round(6)
    craters.to_csv (r'{}/craters_table.csv'.format(set_path), index = None, header=True)
    print(set_path)
    for crop_size in crop_sizes_set:
        win_size = crop_size[0]
        overlap_size = crop_size[1]
        box_list = generate_serial_input.get_crop_list(img.size, win_size, overlap_size)
        print("generating {} images for win_size {} and overlap_size {}".format(len(box_list), win_size, overlap_size))
        tail_dir = "{}/ws_{}_ol_{}".format(set_path, win_size, overlap_size)
        if not os.path.isdir(tail_dir):
            os.mkdir(tail_dir)
        create_input.create_cropped_image_set(img, sub_cdim, R_km, box_list, craters, tail_dir+'/train', compression)
        df = create_input.create_crop_files_coordinated(box_list, sub_cdim, img)
        df.to_csv (r'{}/train_pixels.csv'.format(tail_dir), index = None, header=True)    
    return set_path

def find_all_craters(set_path, crop_sizes_set):
    all_craters = pd.read_csv('{}/craters_table.csv'.format(set_path)).round(6).sort_values(by=['Diameter (km)'])
    craters_num = len(all_craters)
#     all_folders = [x[0] for x in os.walk(data_path)][1:]
#     folders_num = len(all_folders)
    mat = np.zeros((craters_num, len(crop_sizes_set)))
    print('Scoring ...')
    for i, crop in enumerate(crop_sizes_set): 
        folder = "{}/ws_{}_ol_{}".format(set_path, crop[0], crop[1])
        tic = time()
        print(folder)
        arr = detect_craters_with_model.get_detected_craters_from_path(all_craters, folder, model)
        mat[:,i] = arr
        print('Time elapsed {:.2f} sec'.format(time()-tic))
    df = pd.DataFrame(data=mat, columns=["ws_{}_ol_{}".format(crop[0], crop[1]) for crop in crop_sizes_set])
    df['all'] = df.any(axis=1)
    df2=pd.concat([all_craters,df], axis=1)
    df2.to_csv (r'{}/craters_table.csv'.format(set_path), index = None, header=True)