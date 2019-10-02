import sys
import pandas as pd
import numpy as np
from time import time
pathstoadd = ['../find_craters_with_model/','../data_preparation/','../utils/']
[sys.path.append(dir) for dir in pathstoadd]
import create_and_run


# Location of the set: 
# min_lon = 108
# max_lon = 126
# min_lat = 12
# max_lat = 24
min_lon = 108
max_lon = 117
min_lat = 12
max_lat = 18
sub_cdim = [min_lon, max_lon, min_lat, max_lat]



source_cdim = [-180., 180., -60., 60.]
source_image_path = "../../data/maps/LunarLROLrocKaguya_118mperpix.png"
# crop_sizes_set = [[256, 64], [512, 128], [1024, 256], [2048, 512]]
crop_sizes_set = [[512, 128], [1024, 256]]
compression = 'before'
catalog = 'new'
head_dir = 'test6_SLDEM120_' + catalog + '_catalog'
print('Producing set from SLDEM 120 m/pix map with ' + catalog + ' catalog')

tic=time()
set_dir = create_and_run.create_complete_set(source_image_path, source_cdim, sub_cdim, head_dir, crop_sizes_set, compression, catalog)
# set_dir = '../../data/model_input_images/test2_SLDEM120_new_catalog_108_117_12_18/'
create_and_run.find_all_craters(set_dir, crop_sizes_set)
print('All proccess took {:.2f} min.'.format((time()-tic)/60.))


# print('Producing set from SLDEM 60 m/pix map compressed to 8-bit before crop, with new catalog')
# source_cdim = [90., 135., 0., 30.]
# # os.system('wget -P ../../data/maps http://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/sldem2015/tiles/jp2/sldem2015_512_00n_30n_090_135.jp2')
# source_image_path = "../../data/maps/sldem2015_512_00n_30n_090_135.jp2"
# crop_sizes_set = [[512, 128], [1024, 256], [2048, 512], [4096, 1024]]
# head_dir = 'test1_SLDEM60_before_new_catalog'
# compression = 'before'
# catalog = 'new'

# tic=time()
# set_dir = create_and_run.create_complete_set(source_image_path, source_cdim, head_dir, crop_sizes_set, compression, catalog)
# results = create_and_run.find_all_craters(set_dir, crop_sizes_set)
# print('All proccess took {:.2f} min.'.format((time()-tic)/60.))
      

# print('Producing set from SLDEM 60 m/pix map compressed to 8-bit after crop, with new catalog')
# source_cdim = [90., 135., 0., 30.]
# # os.system('wget -P ../../data/maps http://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/sldem2015/tiles/jp2/sldem2015_512_00n_30n_090_135.jp2')
# source_image_path = "../../data/maps/sldem2015_512_00n_30n_090_135.jp2"
# crop_sizes_set = [[512, 128], [1024, 256], [2048, 512], [4096, 1024]]
# head_dir = 'test1_SLDEM60_after_new_catalog'
# compression = 'after'
# catalog = 'new'

# tic=time()
# set_dir = create_and_run.create_complete_set(source_image_path, source_cdim, head_dir, crop_sizes_set, compression, catalog)
# results = create_and_run.find_all_craters(set_dir, crop_sizes_set)
# print('All proccess took {:.2f} min.'.format((time()-tic)/60.))