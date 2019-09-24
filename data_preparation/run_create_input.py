import sys
import create_input
import generate_serial_input
import numpy as np
import os

#default params
R_km = 1737.4  
source_cdim = [-180., 180., -60., 60.]
source_image_path = "../../data/Silburt/LunarLROLrocKaguya_118mperpix.png"

deep_moon_path = os.path.abspath("../../DeepMoon")
lroc_csv_path = "{}/catalogues/LROCCraters.csv".format(deep_moon_path)
head_csv_path = "{}/catalogues/HeadCraters.csv".format(deep_moon_path)
#tune params
sub_cdim = [-18., 18., -6., 6.]
win_size = 1000
overlap_size = 200


    

outhead = '../../data/test_ws_{}_ol_{}_{}_{}_{}_{}'.format(win_size,overlap_size, int(sub_cdim[0]), int(sub_cdim[1]), int(sub_cdim[2]), int(sub_cdim[3]))
if os.path.isdir(outhead):
     print('{} already exists data will be rewritten'.format(outhead))
else:
    os.mkdir(outhead)   
    
img = create_input.get_image(source_image_path, sub_cdim ,source_cdim)
craters = create_input.get_craters(lroc_csv_path, head_csv_path,  sub_cdim, R_km)
box_list = generate_serial_input.get_crop_list(img.size, win_size, overlap_size)
print("generating {} images".format(len(box_list)))

create_input.create_cropped_image_set(img, sub_cdim, R_km, box_list, craters, "{}/train".format(outhead), deep_moon_path)
df = create_input.create_crop_files_coordinated(box_list, sub_cdim, img)
df.to_csv (r'{}/train_pixels.csv'.format(outhead), index = None, header=True)
craters.to_csv (r'{}/train_craters_table.csv'.format(outhead), index = None, header=True)

