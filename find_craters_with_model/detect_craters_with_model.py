import os
import h5py
import sys
import pandas as pd
from keras.models import load_model
import score_utils as sc
import numpy as np

sys.path.append('../../DeepMoon/')
import utils.processing as proc
import utils.template_match_target as tmt


def get_images_data_from_path(data_path):
    images = h5py.File(data_path + '/train_images.hdf5', 'r')
    sample_data = {'imgs': [images['input_images'][...].astype('float32'),
                        images['target_masks'][...].astype('float32')]}
    proc.preprocess(sample_data)
    sd_input_images = sample_data['imgs'][0]
    sd_target_masks = sample_data['imgs'][1]
    ctrs = pd.HDFStore(data_path + '/train_craters.hdf5', 'r')
    return [sd_input_images,ctrs]


def get_craters_from_database(ctrs, image_id):
    #zeropad = int(np.log10(amt)) + 1    
    #my_craters = ctrs["img_{i:0{zp}d}".format(i=image_id, zp=zeropad)]
    my_craters = ctrs["/img_{:05d}".format(image_id)]
    labeled_craters = []
    for index, row in my_craters.iterrows():
        labeled_craters.append([int(round(row['x'])), int(round(row['y'])), int(round(row['Diameter (pix)'])/2)])
    return labeled_craters


def detect_craters_with_model(model, sd_input_images, ctrs):
    pred = model.predict(sd_input_images)    
    images_count = len(sd_input_images)     
    for i in range(images_count):
        print(i)
        labeled = get_craters_from_database(ctrs, i)
        predicted = tmt.template_match_t(pred[i].copy(), minrad=2.)
        
        
def get_image_craters_data(my_craters):
    pixel_coordinates = []
    real_coordinates = []
    for index, row in my_craters.iterrows():
        pixel_coordinates.append([int(round(row['x'])), int(round(row['y'])), int(round(row['Diameter (pix)'])/2)])
        real_coordinates.append([row['Diameter (km)'], row['Lat'],row['Long'],])
    return [np.array(real_coordinates), pixel_coordinates]


def get_index_of_detected_crater_in_the_list(all_craters, detected_crater):
    I = (all_craters['Diameter (km)'] == detected_crater[0]) & (all_craters['Lat'] == detected_crater[1]) & (all_craters['Long'] == detected_crater[2])
    ind = all_craters.index[I]
    assert len(ind)==1
    return ind
    
    
        
def get_matched_craters_indices_in_single_image(my_craters, model_prediction, all_craters):
    [real_coordinates, labeled_craters] = get_image_craters_data(my_craters)
    predicted = tmt.template_match_t(model_prediction, minrad=2.)
    [match, fn, fp] = sc.match_circle_lists(labeled_craters, predicted,0.1)
    matched_craters = [x[0] for x in match]
    matched_crater_indices = []
    for detected_crater in real_coordinates[matched_craters]:
        crater_index_in_list = get_index_of_detected_crater_in_the_list(all_craters, detected_crater)
        matched_crater_indices.append(crater_index_in_list)
    return np.array(matched_crater_indices)
    

def get_detected_craters_from_path(all_craters, f, model):
    [sd_input_images,ctrs] = get_images_data_from_path(f)
    pred = model.predict(sd_input_images)
    images_count = len(sd_input_images)   
    n = len(all_craters)
    arr = np.zeros(n,dtype='bool')
    for i in range(images_count):        
        indices_list = get_matched_craters_indices_in_single_image(ctrs["/img_{:05d}".format(i)],  pred[i].copy(), all_craters)
        arr[indices_list] = True
    return arr
    
