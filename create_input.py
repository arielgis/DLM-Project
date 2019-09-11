#!/usr/bin/env python
"""Input Image Dataset Generator

Script for generating input datasets from Lunar global digital elevation maps
(DEMs) and crater catalogs.

This script is designed to use the LRO-Kaguya DEM and a combination of the
LOLA-LROC 5 - 20 km and Head et al. 2010 >=20 km crater catalogs.  It
generates a randomized set of small (projection-corrected) images and
corresponding crater targets.  The input and target image sets are stored as
hdf5 files.  The longitude and latitude limits of each image is included in the
input set file, and tables of the craters in each image are stored in a
separate Pandas HDFStore hdf5 file.

The script's parameters are located under the Global Variables.  We recommend
making a copy of this script when generating a dataset.

MPI4py can be used to generate multiple hdf5 files simultaneously - each thread
writes `amt` number of images to its own file.
"""

########## Imports ##########

# Python 2.7 compatibility.
from __future__ import absolute_import, division, print_function

from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import sys
#sys.path.append('../DeepMoon')
import input_data_gen as igen
import time
import numpy as np

########## Global Variables ##########

# Use MPI4py?  Set this to False if it's not supposed by the system.
use_mpi4py = False

# Source image path.
source_image_path = "../data/Silburt/LunarLROLrocKaguya_118mperpix.png"

# LROC crater catalog csv path.
lroc_csv_path = "../DeepCrater/catalogues/LROCCraters.csv"

# Head et al. catalog csv path.
head_csv_path = "../DeepCrater/catalogues/HeadCraters.csv"

# Output filepath and file header.  Eg. if outhead = "./input_data/train",
# files will have extension "./out/train_inputs.hdf5" and
# "./out/train_targets.hdf5"
outhead = "../data/my_test_data/train"

# Number of images to make (if using MPI4py, number of image per thread to
# make).
amt = 30

# Range of image widths, in pixels, to crop from source image (input images
# will be scaled down to ilen). For Orthogonal projection, larger images are
# distorted at their edges, so there is some trade-off between ensuring images
# have minimal distortion, and including the largest craters in the image.
rawlen_range = [500, 6500]

# [Min long, max long, min lat, max lat] dimensions of source image.
source_cdim = [-180., 180., -60., 60.]

# [Min long, max long, min lat, max lat] dimensions of the region of the source
# to use when randomly cropping.  Used to distinguish training from test sets.
sub_cdim = [-18., 18., -6., 6.]

# Radius of the world in km (1737.4 for Moon).
R_km = 1737.4

########## Script ##########

def getRandomCrop(size0, size1, rawlen_range):
    # Determine image size to crop.
    rawlen_min = np.log10(rawlen_range[0])
    rawlen_max = np.log10(rawlen_range[1])
    rawlen = float('inf')
    # this is log dist
    while rawlen >= size0 or rawlen >= size1:
        rawlen = int(10**np.random.uniform(rawlen_min, rawlen_max))  
    assert rawlen < size0, "rawlen({}) < size0({})".format(rawlen , size0)
    assert rawlen < size1, "rawlen({}) < size1({})".format(rawlen , size1)
    xc = np.random.randint(0, size0 - rawlen)
    yc = np.random.randint(0, size1 - rawlen)
    box = np.array([xc, yc, xc + rawlen, yc + rawlen], dtype='int32')
    return box

if __name__ == '__main__':

    start_time = time.time()

# Utilize mpi4py for multithreaded processing.
    if use_mpi4py:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        print("Thread {0} of {1}".format(rank, size))
        istart = rank * amt
    else:
        istart = 0

    # Read source image and crater catalogs.
    img = Image.open(source_image_path).convert("L")
    craters = igen.ReadLROCHeadCombinedCraterCSV(filelroc=lroc_csv_path,
                                                 filehead=head_csv_path)

    # Sample subset of image.  Co-opt igen.ResampleCraters to remove all
    # craters beyond cdim (either sub or source).
    if sub_cdim != source_cdim:
        img = igen.InitialImageCut(img, source_cdim, sub_cdim)
    # This always works, since sub_cdim < source_cdim.
    craters = igen.ResampleCraters(craters, sub_cdim, None, arad=R_km)
    
    #box = getRandomCrop(img.size[0], img.size[1], rawlen_range)
    #print("box is {}".format(box))
    box_list = []
    for i in range(amt):
        box = getRandomCrop(img.size[0], img.size[1], rawlen_range)        
        box_list.append(box)
        
    # Generate input images.
    igen.GenDataset(box_list, img, craters, outhead, R_km, cdim=sub_cdim)
    
    #print("amt is {}".format(amt))
    elapsed_time = time.time() - start_time
    print("Time elapsed: {0:.1f} min".format(elapsed_time / 60.))


source_image_path = "../data/Silburt/LunarLROLrocKaguya_118mperpix.png"
import os.path
if os.path.exists(source_image_path):
    print("exists")
else:
    print("not")
