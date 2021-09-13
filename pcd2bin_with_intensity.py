import os
import numpy as np


def load_pcd_data(file_path):
    pts = []
    f = open(file_path, 'r')
    data = f.readlines()

    f.close()
    line = data[9]
    # print(line)
    line = line.strip('\n')
    i = line.split(' ')
    pts_num = eval(i[-1])
    # print(pts_num)
    for line in data[11:]:
        line = line.strip('\n')
        xyzr = line.split(' ')
        # print(xyzr)
        x, y, z, r = [eval(i) for i in xyzr[:]]
        pts.append([x, y, z, r])
    assert len(pts) == pts_num
    # print(pts[0])
    res = np.zeros((pts_num, len(pts[0])), dtype=float)
    for i in range(pts_num):
        res[i] = pts[i]
    # x = np.zeros([np.array(t) for t in pts])
    return res


if __name__ == "__main__":
    pcd_file_root = "F:/IPS300/label_by_Datatang_Raw_20210319/data/BD-1/RSU1/pcd/pcd"
    bin_file_root = "F:/IPS300/label_by_Datatang_Raw_20210319/data/BD-1/RSU1/pcd/bin"
    pcd_files = os.listdir(pcd_file_root)
    pcd_absfiles = []
    bin_absfiles = []
    for file in pcd_files:
        pcd_absfiles.append(os.path.join(pcd_file_root, file))
        bin_absfiles.append(os.path.join(bin_file_root, file[:-3] + 'bin'))
    # print(pcd_absfiles[0])
    # print(bin_absfiles[0])
    for i in range(0, len(pcd_absfiles)):
        res = load_pcd_data(pcd_absfiles[i])
        res.tofile(bin_absfiles[i])
        print("finished: ", pcd_files[i])
