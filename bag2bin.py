import rosbag
import numpy as np
import sensor_msgs.point_cloud2 as pc2
# test=np.fromfile('/data/ftp/data_0624/xj10_20210701/lidar/1625103769100181000.bin').reshape([-1,4])
bag_file = '/data/ftp/pointcloud_label_preparation/camera_calibration/merge_bags/39_2021-07-09-11-15-26.bag'
bag = rosbag.Bag(bag_file, "r")
info = bag.get_type_and_topic_info()
# print(info)

bag_data = bag.read_messages('/driver/lidar/cloud/top_center')
for topic, msg, t in bag_data:
    lidar = pc2.read_points(msg)
    points = np.array(list(lidar))
    points = points[:, [0, 1, 2, 5]]
    # file_name='/data/ftp/data_0624/lidar0701/'+str(t)+'.bin'
    file_name = './lidar/'+str(t)+'.bin'
    points.tofile(file_name)
