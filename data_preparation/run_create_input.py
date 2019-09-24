import sys
import create_input
import generate_serial_input
import numpy as np
import os

#default params
R_km = 1737.4  
# source_cdim = [-180., 180., -60., 60.]
# source_image_path = "../../data/maps/LunarLROLrocKaguya_118mperpix.png"

source_cdim = [90., 135., 0., 30.]
# os.system('wget -P ../../data/maps http://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/sldem2015/tiles/jp2/sldem2015_512_00n_30n_090_135.jp2')
source_image_path = "../../data/maps/sldem2015_512_00n_30n_090_135.jp2"

deep_moon_path = os.path.abspath("../../DeepMoon")
lroc_csv_path = "../../data/catalogues/LROCCraters.csv".format(deep_moon_path)
head_csv_path = "../../data/catalogues/HeadCraters.csv".format(deep_moon_path)
#tune params
min_lon = 108
max_lon = 126
min_lat = 12
max_lat = 24
sub_cdim = [min_lon, max_lon, min_lat, max_lat]

crop_sizes_set = (np.array([[3072, 600], [2000, 400], [1000, 200], [500, 100]])*2).tolist()
#win_size = 1000
#overlap_size = 300

coordinates_folder = '../../data/model_input_images/test_SLDEM_compression_after_coordinates_{}_{}_{}_{}'.format( int(sub_cdim[0]), int(sub_cdim[1]), int(sub_cdim[2]), int(sub_cdim[3]))
if not os.path.isdir(coordinates_folder):
     os.mkdir(coordinates_folder)

img = create_input.get_image(source_image_path, sub_cdim ,source_cdim)
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
    create_input.create_cropped_image_set(img, sub_cdim, R_km, box_list, craters, "{}/train".format(outhead))
    df = create_input.create_crop_files_coordinated(box_list, sub_cdim, img)
    df.to_csv (r'{}/train_pixels.csv'.format(outhead), index = None, header=True)    



