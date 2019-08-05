import numpy as np

from PIL import Image
import sys
sys.path.append('../DeepMoon/utils')
import transform as trf
cdim=[-180., 180., -60., 60.]
rawlen = 3595
xc = 74770
yc = 14157
origin = "upper"
source_image_path = "../deep_moon_data/LunarLROLrocKaguya_118mperpix.png"
img = Image.open(source_image_path).convert("L")
box = np.array([xc, yc, xc + rawlen, yc + rawlen], dtype='int32')

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
ctr_sub = ResampleCraters(craters, llbd, im.size[1], arad=arad,
                          minpix=minpix)

# Convert Plate Carree to Orthographic.
[imgo, ctr_xy, distortion_coefficient, clonglat_xy] = (
    PlateCarree_to_Orthographic(
        im, llbd, ctr_sub, iglobe=iglobe, ctr_sub=True,
        arad=arad, origin=origin, rgcoeff=1.2, slivercut=0.5))


