import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import PIL.Image as Image

def velo2img(velofile, calibfile, imgfile):
    with open(calibfile, 'r') as f:
        calib = f.readlines()

    # P2 (3 x 4) for left eye
    P2 = np.matrix([float(x) for x in calib[2].strip('\n').split(' ')[1:]]).reshape(3, 4)
    R0_rect = np.matrix([float(x) for x in calib[4].strip('\n').split(' ')[1:]]).reshape(3, 3)
    # Add a 1 in bottom-right, reshape to 4 x 4
    R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0], axis=0)
    R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0, 1], axis=1)
    Tr_velo_to_cam = np.matrix([float(x) for x in calib[5].strip('\n').split(' ')[1:]]).reshape(3, 4)
    Tr_velo_to_cam = np.insert(Tr_velo_to_cam, 3, values=[0, 0, 0, 1], axis=0)

    # read raw data from binary
    scan = np.fromfile(velofile, dtype=np.float32).reshape((-1, 4))
    points = scan[:, 0:3]  # lidar xyz (front, left, up)
    # TODO: use fov filter?
    velo = np.insert(points, 3, 1, axis=1).T
    velo = np.delete(velo, np.where(velo[0, :] < 0), axis=1)
    cam = P2 * R0_rect * Tr_velo_to_cam * velo
    cam = np.delete(cam, np.where(cam[2, :] < 0)[1], axis=1)
    # test = np.asarray(cam)
    # get u,v,z
    cam[:2] /= cam[2, :]
    # do projection staff
    # plt.figure(figsize=(12, 5), dpi=96, tight_layout=True)
    png = cv2.imread(imgfile)
    # IMG_H, IMG_W, _ = png.shape
    IMG_H, IMG_W = 375, 1242
    # restrict canvas in range
    # plt.axis([0, IMG_W, IMG_H, 0])
    # plt.imshow(png)
    # filter point out of canvas
    u, v, z = cam
    u_out = np.logical_or(u < 0, u >= IMG_W)
    v_out = np.logical_or(v < 0, v >= IMG_H)
    outlier = np.logical_or(u_out, v_out)
    cam = np.delete(cam, np.where(outlier), axis=1)
    # generate color map from depth
    u, v, z = cam
    # plt.scatter([u], [v], c=[z], cmap='rainbow_r', alpha=0.5, s=2)
    # plt.title('name')
    # plt.savefig(f'./data_object_image_2/testing/projection/{name}.png', bbox_inches='tight')
    # plt.show()
    return cam


def save_depth_png(depth_root, depth):
    IMG_H, IMG_W = 375, 1242
    img = np.zeros([IMG_H, IMG_W],dtype=np.uint16)
    w, h, z = depth
    # h = np.around(h).astype(np.uint).A.flatten()
    h = np.floor(h)
    # print(h.dtype)
    h = h.astype(np.uint).A.flatten()
    # w = np.around(w).astype(np.uint).A.flatten()
    w = np.floor(w)
    w = w.astype(np.uint).A.flatten()
    z = z.A.flatten()
    for i in range(0, h.size):
        img[h[i]][w[i]] = z[i] * 256
        # print(img[h[i]][w[i]])
    # print(img)
    img1=Image.fromarray(img)
    img1.save(depth_root)
    # cv2.imwrite(depth_root,img)
    # test = np.array(Image.open(depth_root), dtype=np.uint16)
    # test = test.astype(np.float) / 256.
    # print(test.dtype())
    

if __name__ == '__main__':
    dataset_root = "/media/idriver/0b6397fb-07e7-4be8-a61a-d25915234cde/KITTI_Object_3D"

    velo_root = os.path.join(dataset_root, 'KITTIObject_data_object_velodyne/training/velodyne')
    velo_files = sorted(os.listdir(velo_root))
    calib_root = os.path.join(dataset_root, 'KITTIObject_data_object_calib/training/calib')
    calib_files = sorted(os.listdir(calib_root))
    img_root = os.path.join(dataset_root, 'KITTIObject_data_object_image_2/training/image_2')
    img_files = sorted(os.listdir(img_root))

    assert len(velo_files) == len(calib_files), "Pointclouds and Calib files mismatch!"

    depth_root = os.path.join(dataset_root, 'lidar_as_depth/training')

    for i in range(0, len(velo_files)):
        depth = velo2img(os.path.join(velo_root, velo_files[i]), os.path.join(calib_root, calib_files[i]),
                         os.path.join(img_root, img_files[i]))
        # depth = velo2img("/media/idriver/0b6397fb-07e7-4be8-a61a-d25915234cde/depth_generate_test/000024.bin", 
                        # "/media/idriver/0b6397fb-07e7-4be8-a61a-d25915234cde/depth_generate_test/000024.txt",
                        # "/media/idriver/0b6397fb-07e7-4be8-a61a-d25915234cde/depth_generate_test/000024.png")
        
        depth_name = str(i).zfill(6) + '.png'

        save_depth_png(os.path.join(depth_root, depth_name), depth)
        print('Saved proj depth: ',depth_name)
