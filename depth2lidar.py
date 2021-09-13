import os
import numpy as np
from PIL import Image
# import open3d.cpu.pybind as o3d
import open3d


def uv2xyz(u0, v0, fx, fy, depth):

    gridy, gridx = np.mgrid[:depth.shape[0], :depth.shape[1]]
    x = (gridx - u0) / fx * depth
    x = x[156:, :]
    y = (gridy - v0) / fy * depth
    y = y[156:, :]
    z = depth
    z = z[156:, :]
    xyz = np.zeros((x.shape[0] * x.shape[1], 3))
    xyz[:, 0] = x.flatten()
    xyz[:, 1] = y.flatten()
    xyz[:, 2] = z.flatten()
    return xyz


def depth2pc_camco(depth_file, intrinsics_file):
    with Image.open(depth_file, 'r') as depth_raw:
        depth = np.array(depth_raw, dtype=float) / 256.0
        # depth = depth[156:, :]
        with open(intrinsics_file, 'r') as intrinsics_file:
            intr = intrinsics_file.readline().split()
            pc = uv2xyz(float(intr[2]), float(intr[5]), float(intr[0]), float(intr[4]), depth)
    return pc


def depth2pc_lidarco(depth_file, intrinsics_file):
    with Image.open(depth_file, 'r') as depth_raw:
        depth = np.array(depth_raw, dtype=float) / 256.0
        depth = np.pad(depth, ((11, 12), (13, 13)))
        
        with open(intrinsics_file, 'r') as f:
            calib = f.readlines()
        # P2 (3 x 4) for left eye
        P2 = np.matrix([float(x) for x in calib[2].strip('\n').split(' ')[1:]]).reshape(3, 4)
        R0_rect = np.matrix([float(x) for x in calib[4].strip('\n').split(' ')[1:]]).reshape(3, 3)
        # Add a 1 in bottom-right, reshape to 4 x 4
        R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0], axis=0)
        R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0, 1], axis=1)
        Tr_velo_to_cam = np.matrix([float(x) for x in calib[5].strip('\n').split(' ')[1:]]).reshape(3, 4)
        Tr_velo_to_cam = np.insert(Tr_velo_to_cam, 3, values=[0, 0, 0, 1], axis=0)

        
        position_encode_x = np.arange(depth.shape[1])
        position_encode_x = np.tile(position_encode_x,depth.shape[0]).reshape(depth.shape)
        position_encode_y = np.arange(depth.shape[0])
        position_encode_y = np.tile(position_encode_y,depth.shape[1]).reshape((depth.shape[1],depth.shape[0])).T
        
        depth=depth[156:,:]
        position_encode_x=position_encode_x[156:,:]
        position_encode_y=position_encode_y[156:,:]

        position_x = position_encode_x * depth
        position_y = position_encode_y * depth
        depth_x3=np.array([position_x.flatten(), position_y.flatten(), depth.flatten()])

        velo = Tr_velo_to_cam.I * R0_rect.I * P2.I *  depth_x3
        # cam = P2 * R0_rect * Tr_velo_to_cam * velo
        velo = velo[:3].A
        index=np.zeros(velo.shape[1], dtype = bool)
        for i in range(0,velo.shape[1]):
            if velo[0][i]+velo[1][i]+velo[2][i]==0:
                index[i] = True
        velo=np.delete(velo,index,axis=1)
    return velo


def depth2pcrgb(depth_files, img_files, intrinsics_files):
    pass



if __name__ == '__main__':
    # root_path = '/data/whn/kitti_depth/depth/data_depth_selection/test_depth_completion_anonymous'
    root_path = '/media/idriver/0b6397fb-07e7-4be8-a61a-d25915234cde/KITTI_Object_3D/lidar_as_depth/training'

    img_path = os.path.join(root_path, 'image_uncrop')
    depth_path = os.path.join(root_path, 'velodyne_dense_crop')
    # depth_path = os.path.join(root_path, 'velodyne_raw')
    intrinsics_path = os.path.join(root_path, 'calib')

    img_files = sorted(os.listdir(img_path))
    depth_files = sorted(os.listdir(depth_path))
    intrinsics_files = sorted(os.listdir(intrinsics_path))

    dst_path=os.path.join(root_path, 'velodyne_dense_bin')
    for i in range(0, len(depth_files)):
        # pc = depth2pc_camco(os.path.join(depth_path, depth_files[i]), os.path.join(intrinsics_path, intrinsics_files))
        pc = depth2pc_lidarco(os.path.join(depth_path, depth_files[i]), os.path.join(intrinsics_path, intrinsics_files[i]))
        # pc = depth2pc_lidarco("/media/idriver/0b6397fb-07e7-4be8-a61a-d25915234cde/KITTI_Object_3D/lidar_as_depth/training/densedepth.png", "/media/idriver/0b6397fb-07e7-4be8-a61a-d25915234cde/KITTI_Object_3D/lidar_as_depth/training/densedepth.txt")
        # save as bin
        pcbin = np.concatenate((pc, np.ones([pc.shape[0], 1])), axis=1)
        dst_name=str(i).zfill(6)+'.bin'
        pcbin.tofile(os.path.join(dst_path,dst_name))
        print("Finished ", i)
        # point_cloud = open3d.geometry.PointCloud()
        # point_cloud.points = open3d.utility.Vector3dVector(pc.T)
        # open3d.io.write_point_cloud('/home/idriver/open3d_test/test_raw.pcd', point_cloud)

        # open3d.visualization.draw_geometries([point_cloud])
        # depth2pcrgb(depth_files, img_files, intrinsics_files)
        #
        # pcd = open3d.io.read_point_cloud('/home/idriver/open3d_test/test.pcd')
        # open3d.visualization.draw_geometries([pcd])
