import numpy as np
from scipy.spatial.transform import Rotation

# BJ11_param
lidar2carR = np.array([0, 0, 0])
lidar2carT = np.array([[0.0, 0.0, 0.0]])

car2camR = np.array([0.00218161, 0.00436332, -9.57284e-06, 0.999988])
car2camT = np.array([[0.0, -0.135, -0.21]])

shortcamR_rect = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
shortcamP_rect = [1303.206437, 0.000000, 604.443674, 0.000000, 0.000000, 1300.853527, 301.882005, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000]
# shortcamP_rect = [1139.612793, 0.000000, 597.334582, 0.000000, 0.000000, 1249.264526, 297.293634, 0.000000, 0.000000, 0.000000, 1.000000, 0.000000]
axisrotate = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])

lidar2camR = np.matmul(Rotation.from_quat(car2camR).as_matrix(), axisrotate[0:3, 0:3])
lidar2camT = np.matmul(Rotation.from_quat(car2camR).as_matrix(), np.array([0.0, -0.135, -0.21])).reshape([3, 1])
lidar2cam = np.concatenate((lidar2camR, lidar2camT), axis=1)

# lidar2carR = Rotation.from_euler('zyx', lidar2carR, degrees=True)
# lidar2carR = lidar2carR.as_matrix()
# lidar2car = np.concatenate((lidar2carR, lidar2carT.T), axis=1)
# lidar2car = np.concatenate((lidar2car, [[0, 0, 0, 1]]))
# # print(lidar2car)
#
# car2camR = Rotation.from_quat(car2camR)
# car2camR = car2camR.as_matrix()
# car2cam = np.concatenate((car2camR, car2camT.T), axis=1)
# car2cam = np.concatenate((car2cam, [[0, 0, 0, 1]]))
# # print(car2cam)
#
# lidar2cam = np.matmul(car2cam, np.matmul(axisrotate, lidar2car))
# lidar2camR = lidar2cam[0:3, 0:3]
# lidar2camT = lidar2cam[0:3, 3]
# lidar2cam = lidar2cam[0:3, :]
# print(lidar2cam)

with open("./calib.txt", "w") as file:
    oneline = ""
    for Idx in range(len(shortcamP_rect)):
        oneline = oneline + "%4.12f " % shortcamP_rect[Idx]
    oneline = "P2: " + oneline + "\n"
    print(oneline)
    file.write(oneline)

    oneline = ""
    for Idx in range(len(shortcamR_rect)):
        oneline = oneline + "%4.12f " % shortcamR_rect[Idx]
    oneline = "R0_rect: " + oneline + "\n"
    print(oneline)
    file.write(oneline)

    oneline = ""
    for i in range(lidar2cam.shape[0]):
        for j in range(lidar2cam.shape[1]):
            oneline = oneline + "%4.12f " % lidar2cam[i, j]
        # oneline = oneline + "%4.12f " % lidar2cam[i, 0]
    oneline = "Tr_velo_to_cam: " + oneline + "\n"
    print(oneline)

    file.write(oneline)
