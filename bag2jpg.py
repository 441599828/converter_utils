# -*- coding: utf-8 -*-
import roslib
import rosbag
import rospy
import cv2, os, sys
import numpy as np
# from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError

import argparse

all_bags = []


def toimage(bagf, video, topic_name, name):
    global all_bags
    bridge = CvBridge()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video, fourcc, 25.0, (1280 * 4, 720), True)
    ok = False
    i = 0
    with rosbag.Bag(bagf, 'r') as bag:
        for topic, msg, t in bag.read_messages():
            if topic == topic_name:
                ok = True
                try:
                    np_arr = np.fromstring(msg.data, np.uint8)
                    image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    camera1_h60 = image_np[:, 0:1280]
                    camera2_h90 = image_np[:, 1280:1280 * 2]
                    camera3_h90 = image_np[:, 1280 * 2:1280 * 3]
                    camera4_h30 = image_np[:, 1280 * 3:]
                except CvBridgeError as e:
                    print(e)
                cv2.imwrite('./images/' + name + '1_H60_' + str(t) + '.jpg', camera1_h60)
                cv2.imwrite('./images/' + name + '4_H30_' + str(t) + '.jpg', camera4_h30)
                print('./images/' + name + '1_H60_' + str(i) + '.jpg', './images/' + name + '2_H30_' + str(i) + '.jpg')
                out.write(image_np)
                i += 1
        out.release()
    return ok


def parse_all_bag(root):
    global all_bags
    childs = os.listdir(root)
    for child in childs:
        if os.path.isdir(os.path.join(root, child)):
            parse_all_bag(os.path.join(root, child))
        elif child.split(".")[-1] == "bag" and '35_perception' in child:
            all_bags.append(os.path.join(root, child))


if __name__ == '__main__':
    print("目标检测自动优化模块触发成功，即将进入数据解析功能！！！")
    root = '/data/ftp/35_perception_test_whn'
    bag_raw = '/data/ftp/bags'
    # bag_raw = '/data/ftp/pointcloud_label_preparation/camera_calibration/merge_bags'
    # root = sys.argv[1]

    parse_all_bag(bag_raw)

    # if not os.path.exists(os.path.join(root, "video")):
    #     os.mkdir(os.path.join(root, "video"))
    if not os.path.exists("./video"):
        os.mkdir("./video")
    # if not os.path.exists(os.path.join(root, "images")):
    #     os.mkdir(os.path.join(root, "images"))
    if not os.path.exists("./images"):
        os.mkdir("./images")

    # for bag in all_bags:
    for i in range(1):
        try:
            # bag='/data/ftp/data_0624/xj10_20210701094348.bag'
            bag='/data/ftp/pointcloud_label_preparation/camera_calibration/merge_bags/39_2021-07-09-11-37-47_0.bag'
            camera = "camera"
            topic = "/miivii_gmsl_ros_node_A/{}/compressed".format(camera)
            print(topic)
            name = bag.split("/")[-1].split(".")[0]
            name = name + "_" + camera
            if not os.path.exists(os.path.join("./video/", name)):
                os.mkdir(os.path.join("./video/", name))
            video = os.path.join("./video/", name, name + ".avi")
            if False == toimage(bag, video, topic, name):
                os.remove(os.path.join("./video/", name, name + ".avi"))
                os.removedirs(os.path.join("./video/", name))
            else:
                print(os.path.join("./video/", name, name + ".avi"))
        except rospy.ROSInterruptException:
            pass
© 2021 GitHub, Inc.
