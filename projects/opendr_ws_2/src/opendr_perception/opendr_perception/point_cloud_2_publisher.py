#!/usr/bin/env python3.6
# Copyright 2020-2022 OpenDR European Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import sys
import argparse
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2 as ROS_PointCloud2
from opendr_ros2_bridge import ROS2Bridge

from opendr.engine.datasets import DatasetIterator
from opendr.perception.panoptic_segmentation import SemanticKittiDataset


class PointCloud2Publisher(Node):

    def __init__(self,
                 dataset: DatasetIterator,
                 output_point_cloud_2_topic: str="/opendr/dataset_point_cloud2"
                 ):
        super().__init__("point_cloud_2_publisher")

        """
        Creates a ROS Node for publishing a PointCloud2 message from a DatasetIterator
        :param dataset: DatasetIterator from which we are reading the point cloud
        :type dataset: DatasetIterator
        :param output_point_cloud_2_topic: Topic to which we are publishing the point cloud
        :type output_point_cloud_2_topic: str
        """

        self.dataset = dataset
        self._ros2_bridge = ROS2Bridge()

        if output_point_cloud_2_topic is not None:
            self.point_cloud_2_publisher = self.create_publisher(ROS_PointCloud2, output_point_cloud_2_topic, 10)
        

    def start(self):
        """
        Starts the ROS Node
        """
        i = 0 
        print("Starting point cloud 2 publisher")
        while rclpy.ok():
            print("Publishing point cloud 2 message")
            point_cloud = self.dataset[i % len(self.dataset)][0]
            self.get_logger().info("Publishing point_cloud_2 [" + str(i) + "]")
            message = self._ros2_bridge.to_ros_point_cloud2(point_cloud, 
                                                            self.get_clock().now().to_msg(),
                                                            ROS_PointCloud2)
            self.point_cloud_2_publisher.publish(message)
            i += 1

            time.sleep(0.1)

def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser(description="Point Cloud 2 Publisher")
    parser.add_argument("--dataset_path", type=str, default='/home/canakcia/datasets/semantickitti/test_data', help="Path to the dataset")
    parser.add_argument("--output_point_cloud_2_topic", type=str, default="/opendr/dataset_point_cloud2", help="Topic to which we are publishing the point cloud")
    args = parser.parse_args()

    dataset = SemanticKittiDataset(path=os.path.join(args.dataset_path, "eval_data"), split="valid")
    point_cloud_2_publisher = PointCloud2Publisher(dataset, args.output_point_cloud_2_topic)

    point_cloud_2_publisher.start()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    point_cloud_2_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
