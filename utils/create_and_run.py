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


def create_complete_set(source_image_path, source_cdim, head_dir, crop_sizes_set, compression): 
    #default params
    R_km = 1737.4  
    deep_moon_path = os.path.abspath("../../DeepMoon")
    lroc_csv_path = "../../data/catalogues/LROCCraters.csv".format(deep_moon_path)
    head_csv_path = "../../data/catalogues/HeadCraters.csv".format(deep_moon_path)

    #tune params
    min_lon = 108
    max_lon = 126
    min_lat = 12
    max_lat = 24
    sub_cdim = [min_lon, max_lon, min_lat, max_lat]

    coordinates_folder = '../../data/model_input_images/{}_{}_{}_{}_{}'.format(head_dir, int(sub_cdim[0]), int(sub_cdim[1]), int(sub_cdim[2]), int(sub_cdim[3]))
    if not os.path.isdir(coordinates_folder):
         os.mkdir(coordinates_folder)

    img = create_input.get_image(source_image_path, sub_cdim ,source_cdim, compression)
    print("images size is {}".format(img.size))
    craters = create_input.get_craters(lroc_csv_path, head_csv_path,  sub_cdim, R_km).round(6)
    craters.to_csv (r'{}/craters_table.csv'.format(coordinates_folder), index = None, header=True)
    print(coordinates_folder)
    for crop_size in crop_sizes_set:
        win_size = crop_size[0]
        overlap_size = crop_size[1]
        box_list = generate_serial_input.get_crop_list(img.size, win_size, overlap_size)
        print("generating {} images for win_size {} and overlap_size {}".format(len(box_list), win_size, overlap_size))
        outhead = "{}/ws_{}_ol_{}".format(coordinates_folder, win_size, overlap_size)
        if not os.path.isdir(outhead):
            os.mkdir(outhead)
        create_input.create_cropped_image_set(img, sub_cdim, R_km, box_list, craters, "{}/train".format(outhead), compression)
        df = create_input.create_crop_files_coordinated(box_list, sub_cdim, img)
        df.to_csv (r'{}/train_pixels.csv'.format(outhead), index = None, header=True)    
    return coordinates_folder

def find_all_craters(data_path, crop_sizes_set):
#     data_path = '../../data/model_input_images/'+head_dir
    all_craters = pd.read_csv('{}/craters_table.csv'.format(data_path)).round(6).sort_values(by=['Diameter (km)'])
    craters_num = len(all_craters)
    all_folders = [x[0] for x in os.walk(data_path)][1:]
    folders_num = len(all_folders)
    mat = np.zeros((craters_num, folders_num))
    print('Scoring ...')
    for i in range(folders_num):
        f = all_folders[i]
        tic = time()
        print(f)
        arr = detect_craters_with_model.get_detected_craters_from_path(all_craters, f, model)
        mat[:,i] = arr
        print('Time elapsed {:.2f} sec'.format(time()-tic))
    df = pd.DataFrame(data=mat, columns=[ws[0] for ws in crop_sizes_set])
    df2=pd.concat([all_craters,df], axis=1)
    df2.to_csv (r'{}/craters_table.csv'.format(data_path), index = None, header=True)
    
    return mat
