import sys
import create_input
import numpy as np

R_km = 1737.4  
source_cdim = [-180., 180., -60., 60.]
sub_cdim = [-18., 18., -6., 6.]
source_image_path = "../data/Silburt/LunarLROLrocKaguya_118mperpix.png"
lroc_csv_path = "../DeepMoon/catalogues/LROCCraters.csv"
head_csv_path = "../DeepMoon/catalogues/HeadCraters.csv"

img = create_input.get_image(source_image_path, sub_cdim ,source_cdim)

craters = create_input.get_craters(lroc_csv_path, head_csv_path,  sub_cdim, R_km)

box_list = []
box_list.append(np.array([0, 0, 3072, 3072], dtype='int32'))
box_list.append(np.array([2048, 0, 5120, 3072], dtype='int32'))
box_list.append(np.array([4096, 0, 7168, 3072], dtype='int32'))
box_list.append(np.array([6144, 0, 9216, 3072], dtype='int32'))

outhead = '/mnt/disks/disk0/deep_moon_working_dir/data/test_images_3072/train'

create_input.create_cropped_image_set(img, sub_cdim, R_km, box_list, craters, outhead)

df = create_input.create_crop_files_coordinated(box_list, sub_cdim, img)
df.to_csv (r'{}_pixels.csv'.format(outhead), index = None, header=True)
    

        
        
