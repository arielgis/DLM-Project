import os
import h5py
import sys
import pandas as pd
from keras.models import load_model
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
        
        
    
    

    
data_path = '../../data/test_coordinates_-18_18_-6_6/' 
files = [x[0] for x in os.walk(data_path)] 
#for f in files[1:]:
#    print(f)
#    get_images_data_from_path(f)
    
model = load_model("/mnt/disks/disk0/deep_moon_working_dir/data/Silburt/model_keras2.h5")

f = '../../data/test_coordinates_-18_18_-6_6/ws_2000_ol_400'
[sd_input_images,ctrs] = get_images_data_from_path(f)
detect_craters_with_model(model, sd_input_images, ctrs)

