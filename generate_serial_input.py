import numpy as np


def get_overlapping_segments(total_size, win_size, overlap_size):
    x = list(range(0, total_size - win_size, win_size - overlap_size))
    if x[-1] < total_size - win_size:
        x.append(total_size - win_size)
    return x

def get_crop_list(total_size_vec, win_size, overlap_size):
    x = get_overlapping_segments(total_size_vec[0], win_size, overlap_size)
    y = get_overlapping_segments(total_size_vec[1], win_size, overlap_size)
    print(x)
    print(y)
    crop_list = []
    for curr_x in x:
        for curr_y in y:
            vec = np.array([curr_x, curr_y, curr_x + win_size, curr_y + win_size], dtype='int32')
            crop_list.append(vec)
            print("curr_window = [{} {}]".format(curr_x, curr_y))
    return crop_list
