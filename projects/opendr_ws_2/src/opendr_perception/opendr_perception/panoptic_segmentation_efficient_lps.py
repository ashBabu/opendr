#!/usr/bin/env python
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

import sys
from pathlib import Path
import argparse
from typing import Optional, List

import numpy as  np
import rclpy
from rclpy.node import Node
import matplotlib
from sensor_msgs.msg import PointCloud as ROS_PointCloud
from sensor_msgs.msg import Image as ROS_Image

from opendr_ros2_bridge import ROS2Bridge
from opendr.perception.panoptic_segmentation import EfficientLpsLearner

# Avoid having a matplotlib GUI in a separate thread in the visualize() function
matplotlib.use('Agg')


class EfficientLpsNode(Node):

    def __init__(self,
                 input_rgb_pcl_topic: str,
                 checkpoint: str,
                 output_rgb_visualization_topic: Optional[str] = None
                 ):
        """
        Initialize the EfficientLPS ROS node and create an instance of the respective learner class.
        :param input_rgb_pcl_topic: ROS topic for the input point cloud
        :type input_rgb_pcl_topic: str
        :param checkpoint: This is either a path to a saved model or SemanticKITTI to download
            pre-trained model weights.
        :type checkpoint: str
        :param output_heatmap_topic: ROS topic for the predicted semantic and instance maps
        :type output_topic: str
        :param output_visualization_topic: ROS topic for the generated visualization of the panoptic map
        :type output_visualization_topic: str
        :param projected_output: Publish predictions as a 2D Projection.
        :type projected_output: bool
        """

        self.checkpoint = checkpoint
        


        # Initialize all ROS related things
        self._bridge = ROS2Bridge()


        # Initialize the panoptic segmentation network
        config_file = Path(sys.modules[
            EfficientLpsLearner.__module__].__file__).parent / 'configs' / 'singlegpu_semantickitti.py'
        self._learner = EfficientLpsLearner(str(config_file))

        # Other
        self._tmp_folder = Path(__file__).parent.parent / 'tmp' / 'efficientlps'
        self._tmp_folder.mkdir(exist_ok=True, parents=True)

    def _init_learner(self) -> bool:
        """
        The model can be initialized via
        1. downloading pre-trained weights for SemanticKITTI.
        2. passing a path to an existing checkpoint file.

        This has not been done in the __init__() function since logging is available only once the node is registered.
        """
        if self.checkpoint in ['semantickitti']:
            file_path = EfficientLpsLearner.download(str(self._tmp_folder),
                                                    trained_on=self.checkpoint)
            self.checkpoint = file_path

        if self._learner.load(self.checkpoint):
            self.get_logger().info('Successfully loaded the checkpoint.')
            return True
        else:
            self.get_logger().error('Failed to load the checkpoint.')
            return False

    def _init_subscriber(self):
        """
        Subscribe to all relevant topics.
        """
        self.image_subscriber = self.create_subscription(self.input_rgb_pcl_topic, ROS_PointCloud, self.callback)

    def _init_publisher(self):
        """
        Set up the publishers as requested by the user.
        """
        if self.output_heatmap_pointcloud_topic is not None:

                self._semantic_heatmap_publisher = None
        if self.output_rgb_visualization_topic is not None:
            self._visualization_publisher = self.create_publisher(ROS_Image,
                                                                  self.output_rgb_visualization_topic, 
                                                                  10)

    def _join_arrays(self, arrays: List[np.ndarray]):
        """
        Function for efficiently concatenating numpy arrays.

        :param arrays: List of numpy arrays to be concatenated
        :type arrays: List[np.ndarray]

        :return: Array comprised of the concatenated inputs.
        :rtype: np.ndarray
        """

        sizes = np.array([a.itemsize for a in arrays])
        offsets = np.r_[0, sizes.cumsum()]
        n = len(arrays[0])
        joint = np.empty((n, offsets[-1]), dtype=np.uint8)

        for a, size, offset in zip(arrays, sizes, offsets):
            joint[:, offset:offset + size] = a.view(np.uint8).reshape(n, size)

        dtype = sum((a.dtype.descr for a in arrays), [])

        return joint.ravel().view(dtype)

    def listen(self):
        """
        Start the node and begin processing input data. The order of the function calls ensures that the node does not
        try to process input images without being in a trained state.
        """
        if self._init_learner():
            self._init_publisher()
            self._init_subscriber()
            self.get_logger().info('EfficientPS node started!')
            rclpy.spin(self)

            # Destroy the node explicitly
            # (optional - otherwise it will be done automatically
            # when the garbage collector destroys the node object)
            self.destroy_node()
            rclpy.shutdown()

    def callback(self, data: ROS_PointCloud):
        """
        Predict the panoptic segmentation map from the input point cloud and publish the results.

        :param data: PointCloud data message
        :type data: sensor_msgs.msg.PointCloud
        """
        # Convert sensor_msgs.msg.Image to OpenDR Image



def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_pcl_topic', type=str, default='/usb_cam/pcl_raw',
                        help='listen to pointclouds on this topic')
    parser.add_argument('--checkpoint', type=str, default='semantickitti',
                        help='download pretrained models [semantickitti] or load from the provided path')
    parser.add_argument('--output_rgb_visualization_topic', type=str, default="/opendr/panoptic",
                        help='publish the rgb visualization on this topic')
    args = parser.parse_args()
    efficient_lps_node = EfficientLpsNode(args.input_pcl_topic, 
                                          args.checkpoint,
                                          args.output_rgb_visualization_topic)
    efficient_lps_node.listen()


if __name__ == '__main__':
    main()
