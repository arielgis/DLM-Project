#!/usr/bin/env python
#from __future__ import absolute_import, division, print_function
import numpy as np
import pandas as pd
from PIL import Image
import cartopy.crs as ccrs
import h5py
import sys
sys.path.append('../DeepMoon')
import input_data_gen as igen
import utils.transform as trf


def update_sds_box(imgs_h5_box, img_number, box):
    sds_box = imgs_h5_box.create_dataset(img_number, (4,), dtype='int32')
    sds_box[...] = box
    
def update_sds_llbd(imgs_h5_llbd, img_number, llbd):
    sds_llbd = imgs_h5_llbd.create_dataset(img_number, (4,), dtype='float')
    sds_llbd[...] = llbd
    
def update_sds_dc(imgs_h5_dc, img_number, distortion_coefficient):
    sds_dc = imgs_h5_dc.create_dataset(img_number, (1,), dtype='float')
    sds_dc[...] = np.array([distortion_coefficient])    
    
def update_sds_cll(imgs_h5_cll, img_number, clonglat_xy):
    sds_cll = imgs_h5_cll.create_dataset(img_number, (2,), dtype='float')
    sds_cll[...] = clonglat_xy.loc[:, ['x', 'y']].as_matrix().ravel()
        
def output_to_file(img_number, i, imgs_h5_inputs, imgo_arr, imgs_h5_tgts, mask, imgs_h5_box, box, imgs_h5_llbd, llbd, 
                       imgs_h5_dc,  distortion_coefficient, imgs_h5_cll, clonglat_xy, craters_h5, ctr_xy, imgs_h5):
    imgs_h5_inputs[i, ...] = imgo_arr
    imgs_h5_tgts[i, ...] = mask
    update_sds_box(imgs_h5_box, img_number, box)
    update_sds_llbd(imgs_h5_llbd, img_number, llbd)
    update_sds_dc(imgs_h5_dc, img_number, distortion_coefficient)
    update_sds_cll(imgs_h5_cll, img_number, clonglat_xy)
    craters_h5[img_number] = ctr_xy
    imgs_h5.flush()
    craters_h5.flush()
    

def init_files(outhead, amt, ilen, tglen):
    imgs_h5 = h5py.File(outhead + '_images.hdf5', 'w')
    imgs_h5_inputs = imgs_h5.create_dataset("input_images", (amt, ilen, ilen),
                                            dtype='uint8')
    imgs_h5_inputs.attrs['definition'] = "Input image dataset."
    imgs_h5_tgts = imgs_h5.create_dataset("target_masks", (amt, tglen, tglen),
                                          dtype='float32')
    imgs_h5_tgts.attrs['definition'] = "Target mask dataset."
    imgs_h5_llbd = imgs_h5.create_group("longlat_bounds")
    imgs_h5_llbd.attrs['definition'] = ("(long min, long max, lat min, "
                                        "lat max) of the cropped image.")
    imgs_h5_box = imgs_h5.create_group("pix_bounds")
    imgs_h5_box.attrs['definition'] = ("Pixel bounds of the Global DEM region"
                                       " that was cropped for the image.")
    imgs_h5_dc = imgs_h5.create_group("pix_distortion_coefficient")
    imgs_h5_dc.attrs['definition'] = ("Distortion coefficient due to "
                                      "projection transformation.")
    imgs_h5_cll = imgs_h5.create_group("cll_xy")
    imgs_h5_cll.attrs['definition'] = ("(x, y) pixel coordinates of the "
                                       "central long / lat.")
    craters_h5 = pd.HDFStore(outhead + '_craters.hdf5', 'w')
    return [imgs_h5, imgs_h5_inputs, imgs_h5_tgts, imgs_h5_llbd, imgs_h5_box, imgs_h5_dc, imgs_h5_cll, craters_h5]


def GenDataset(box_list, img, craters, outhead, arad, cdim=[-180., 180., -60., 60.]):
    
    truncate = True
    istart=0
    ringwidth = 1
    rings=True
    binary=True    
    minpix = 1.
    tglen = 256
    ilen = 256
    amt = len(box_list)
    origin = "upper"

    # Get craters.
    igen.AddPlateCarree_XY(craters, list(img.size), cdim=cdim, origin=origin)
    iglobe = ccrs.Globe(semimajor_axis=arad*1000., semiminor_axis=arad*1000.,
                        ellipse=None)

    # Initialize output hdf5s.
    [imgs_h5, imgs_h5_inputs, imgs_h5_tgts, imgs_h5_llbd, imgs_h5_box, imgs_h5_dc, imgs_h5_cll, craters_h5] = init_files(outhead, amt, ilen, tglen)

    # Zero-padding for hdf5 keys.
    zeropad = int(np.log10(amt)) + 1

    for i in range(amt):

        # Current image number.
        img_number = "img_{i:0{zp}d}".format(i=istart + i, zp=zeropad)                

        # Determine image size to crop.
        box = box_list[i]
        print("Generating {} current crop: ({})".format(img_number,box))     
       

        # Load necessary because crop may be a lazy operation; im.load() should
        # copy it.  See <http://pillow.readthedocs.io/en/3.1.x/
        # reference/Image.html>.
        im = img.crop(box)
        im.load()

        # Obtain long/lat bounds for coordinate transform.
        ix = box[::2]
        iy = box[1::2]
        llong, llat = trf.pix2coord(ix, iy, cdim, list(img.size),
                                    origin=origin)
        llbd = np.r_[llong, llat[::-1]]

        # Downsample image.
        im = im.resize([ilen, ilen], resample=Image.NEAREST)

        # Remove all craters that are too small to be seen in image.
        ctr_sub = igen.ResampleCraters(craters, llbd, im.size[1], arad=arad,
                                  minpix=minpix)

        # Convert Plate Carree to Orthographic.
        [imgo, ctr_xy, distortion_coefficient, clonglat_xy] = (
            igen.PlateCarree_to_Orthographic(
                im, llbd, ctr_sub, iglobe=iglobe, ctr_sub=True,
                arad=arad, origin=origin, rgcoeff=1.2, slivercut=0.5))

        if imgo is None:
            print("Discarding narrow image")
            continue

        imgo_arr = np.asanyarray(imgo)
        assert imgo_arr.sum() > 0, ("Sum of imgo is zero!  There likely was "
                                    "an error in projecting the cropped "
                                    "image.")

        # Make target mask.  Used Image.BILINEAR resampling because
        # Image.NEAREST creates artifacts.  Try Image.LANZCOS if BILINEAR still
        # leaves artifacts).
        tgt = np.asanyarray(imgo.resize((tglen, tglen),
                                        resample=Image.BILINEAR))
        mask = igen.make_mask(ctr_xy, tgt, binary=binary, rings=rings,
                         ringwidth=ringwidth, truncate=truncate)

        # Output everything to file.
        output_to_file(img_number, i, imgs_h5_inputs, imgo_arr, imgs_h5_tgts, mask, imgs_h5_box, box, imgs_h5_llbd, llbd, 
                       imgs_h5_dc,  distortion_coefficient, imgs_h5_cll, clonglat_xy, craters_h5, ctr_xy, imgs_h5)       
    imgs_h5.close()
    craters_h5.close()
    

    

        
        
