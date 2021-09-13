import numpy as np
import cv2
import os


def load_velo_scan(velo_filename, dtype=np.float32, n_vec=4):
    scan = np.fromfile(velo_filename, dtype=dtype)
    scan = scan.reshape((-1, n_vec))
    return scan


def lidar_to_bird_view_img(lidar, factor=1):
    # Input:
    #   lidar: (N', 4)
    #   factor: 越大清晰度越高
    # Output:
    #   birdview: (w, l, 3)
    birdview = np.zeros(
        (INPUT_HEIGHT * factor, INPUT_WIDTH * factor, 1))
    for point in lidar:
        x, y = point[0:2]
        if X_MIN < x < X_MAX and Y_MIN < y < Y_MAX:
            x, y = int((x - X_MIN) / VOXEL_X_SIZE *
                       factor), int((y - Y_MIN) / VOXEL_Y_SIZE * factor)
            birdview[y, x] += 1
    birdview = birdview - np.min(birdview)
    divisor = np.max(birdview) - np.min(birdview)
    # TODO: adjust this factor
    birdview = np.clip((birdview / divisor * 255) *
                       5 * factor, a_min=0, a_max=255)
    for i in range(0, birdview.shape[0]):
        for j in range(0, birdview.shape[1]):
            if birdview[i][j][0] > 0:
                birdview[i][j][0] = 255
    birdview = np.tile(birdview, 3).astype(np.uint8)

    return birdview


# 限定可视化范围
INPUT_HEIGHT = 1000
INPUT_WIDTH = 1250
X_MIN = -100
X_MAX = 150
Y_MIN = -100
Y_MAX = 100
VOXEL_X_SIZE = 0.2
VOXEL_Y_SIZE = 0.2

bin_root = "D:/2020.12.16-data/label_by_Datatang_Raw_20210319/data/BD-1/RSU2/pcd/bin"
jpg_root = "D:/2020.12.16-data/label_by_Datatang_Raw_20210319/video/RSU2/pcd"
txt_files = os.listdir(bin_root)

bin_absfiles = []
jpg_absfiles = []
for file in txt_files:
    bin_absfiles.append(os.path.join(bin_root, file))
    jpg_absfiles.append(os.path.join(jpg_root, file[:-3]+'jpg'))

for i in range(0, len(bin_absfiles)):
    # print(bin_absfiles[i])
    # print(jpg_absfiles[i])
    lidar = load_velo_scan(bin_absfiles[i])
    bird_view = lidar_to_bird_view_img(lidar)

    cv2.imwrite(jpg_absfiles[i], bird_view)
    print('finished: ', bin_absfiles[i])
