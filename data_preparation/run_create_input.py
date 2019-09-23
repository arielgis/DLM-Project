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

crop_sizes_set = [[3072, 600], [2000, 400], [1000, 200], [500, 100]]
#win_size = 1000
#overlap_size = 300

coordinates_folder = '../../data/test_coordinates_{}_{}_{}_{}'.format( int(sub_cdim[0]), int(sub_cdim[1]), int(sub_cdim[2]), int(sub_cdim[3]))
if not os.path.isdir(coordinates_folder):
     os.mkdir(coordinates_folder)

img = create_input.get_image(source_image_path, sub_cdim ,source_cdim)
print("images size is {}".format(img.size))
craters = create_input.get_craters(lroc_csv_path, head_csv_path,  sub_cdim, R_km)
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
    create_input.create_cropped_image_set(img, sub_cdim, R_km, box_list, craters, "{}/train".format(outhead), deep_moon_path)
    df = create_input.create_crop_files_coordinated(box_list, sub_cdim, img)
    df.to_csv (r'{}/train_pixels.csv'.format(outhead), index = None, header=True)    



