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


import rospy
import torch
import message_filters
import cv2
import numpy as np
import argparse
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image as ROS_Image
from opendr_bridge import ROSBridge
from opendr.perception.object_detection_2d import GemLearner
from opendr.perception.object_detection_2d import draw_bounding_boxes
from opendr.engine.data import Image


class ObjectDetectionGemNode:
    def __init__(
        self,
        input_rgb_image_topic="/camera/color/image_raw",
        input_infra_image_topic="/camera/infra/image_raw",
        output_rgb_image_topic="/opendr/rgb_detection_annotated",
        output_infra_image_topic="/opendr/infra_detection_annotated",
        detections_topic="/opendr/objects",
        device="cuda",
        pts_rgb=None,
        pts_infra=None,
    ):
        """
        Creates a ROS Node for object detection with GEM
        :param input_rgb_image_topic: Topic from which we are reading the input rgb image
        :type input_rgb_image_topic: str
        :param input_infra_image_topic: Topic from which we are reading the input infrared image
        :type: input_infra_image_topic: str
        :param output_rgb_image_topic: Topic to which we are publishing the annotated rgb image (if None, we are not
        publishing annotated image)
        :type output_rgb_image_topic: str
        :param output_infra_image_topic: Topic to which we are publishing the annotated infrared image (if None, we are not
        publishing annotated image)
        :type output_infra_image_topic: str
        :param detections_topic: Topic to which we are publishing the annotations (if None, we are
        not publishing annotations)
        :type detections_topic:  str
        :param device: Device on which we are running inference ('cpu' or 'cuda')
        :type device: str
        :param pts_rgb: Point on the rgb image that define alignment with the infrared image. These are camera
        specific and can be obtained using get_color_infra_alignment.py which is located in the
        opendr/perception/object_detection2d/utils module.
        :type pts_rgb: {list, numpy.ndarray}
        :param pts_infra: Points on the infrared image that define alignment with rgb image. These are camera specific
        and can be obtained using get_rgb_infra_alignment.py which is located in the
        opendr/perception/object_detection2d/utils module.
        :type pts_infra: {list, numpy.ndarray}
        """
        rospy.init_node("gem", anonymous=True)
        if output_rgb_image_topic is not None:
            self.rgb_publisher = rospy.Publisher(output_rgb_image_topic, ROS_Image, queue_size=10)
        else:
            self.rgb_publisher = None
        if output_infra_image_topic is not None:
            self.ir_publisher = rospy.Publisher(output_infra_image_topic, ROS_Image, queue_size=10)
        else:
            self.ir_publisher = None

        if detections_topic is not None:
            self.detection_publisher = rospy.Publisher(detections_topic, Detection2DArray, queue_size=10)
        else:
            self.detection_publisher = None
        if pts_infra is None:
            pts_infra = np.array(
                [
                    [478, 248],
                    [465, 338],
                    [458, 325],
                    [468, 256],
                    [341, 240],
                    [335, 310],
                    [324, 321],
                    [311, 383],
                    [434, 365],
                    [135, 384],
                    [67, 257],
                    [167, 206],
                    [124, 131],
                    [364, 276],
                    [424, 269],
                    [277, 131],
                    [41, 310],
                    [202, 320],
                    [188, 318],
                    [188, 308],
                    [196, 241],
                    [499, 317],
                    [311, 164],
                    [220, 216],
                    [435, 352],
                    [213, 363],
                    [390, 364],
                    [212, 368],
                    [390, 370],
                    [467, 324],
                    [415, 364],
                ]
            )
            rospy.logwarn(
                "\nUsing default calibration values for pts_infra!" +
                "\nThese are probably incorrect." +
                "\nThe correct values for pts_infra can be found by running get_color_infra_alignment.py." +
                "\nThis file is located in the opendr/perception/object_detection2d/utils module."
            )
        if pts_rgb is None:
            pts_rgb = np.array(
                [
                    [910, 397],
                    [889, 572],
                    [874, 552],
                    [891, 411],
                    [635, 385],
                    [619, 525],
                    [603, 544],
                    [576, 682],
                    [810, 619],
                    [216, 688],
                    [90, 423],
                    [281, 310],
                    [193, 163],
                    [684, 449],
                    [806, 431],
                    [504, 170],
                    [24, 538],
                    [353, 552],
                    [323, 550],
                    [323, 529],
                    [344, 387],
                    [961, 533],
                    [570, 233],
                    [392, 336],
                    [831, 610],
                    [378, 638],
                    [742, 630],
                    [378, 648],
                    [742, 640],
                    [895, 550],
                    [787, 630],
                ]
            )
            rospy.logwarn(
                "\nUsing default calibration values for pts_rgb!" +
                "\nThese are probably incorrect." +
                "\nThe correct values for pts_rgb can be found by running get_color_infra_alignment.py." +
                "\nThis file is located in the opendr/perception/object_detection2d/utils module."
            )
        # Object classes
        self.classes = ["N/A", "chair", "cycle", "bin", "laptop", "drill", "rocker"]

        # Estimating Homography matrix for aligning infra with RGB
        self.h, status = cv2.findHomography(pts_infra, pts_rgb)

        self.bridge = ROSBridge()

        # Initialize the detection estimation
        model_backbone = "resnet50"

        self.gem_learner = GemLearner(
            backbone=model_backbone,
            num_classes=7,
            device=device,
        )
        self.gem_learner.fusion_method = "sc_avg"
        self.gem_learner.download(path=".", verbose=True)

        # Subscribers
        msg_rgb = message_filters.Subscriber(input_rgb_image_topic, ROS_Image, queue_size=1, buff_size=10000000)
        msg_ir = message_filters.Subscriber(input_infra_image_topic, ROS_Image, queue_size=1, buff_size=10000000)

        sync = message_filters.TimeSynchronizer([msg_rgb, msg_ir], 1)
        sync.registerCallback(self.callback)
        rospy.loginfo("GEM node Initialized!")

    def listen(self):
        """
        Start the node and begin processing input data
        """
        self.fps_list = []
        rospy.loginfo("GEM node started!")
        rospy.spin()

    def callback(self, msg_rgb, msg_ir):
        """
        Callback that process the input data and publishes to the corresponding topics
        :param msg_rgb: input rgb image message
        :type msg_rgb: sensor_msgs.msg.Image
        :param msg_ir: input infrared image message
        :type msg_ir: sensor_msgs.msg.Image
        """
        # Convert images to OpenDR standard
        image_rgb = self.bridge.from_ros_image(msg_rgb).opencv()
        image_ir_raw = self.bridge.from_ros_image(msg_ir, "bgr8").opencv()
        image_ir = cv2.warpPerspective(image_ir_raw, self.h, (image_rgb.shape[1], image_rgb.shape[0]))

        # Perform inference on images
        boxes, w_sensor1, _ = self.gem_learner.infer(image_rgb, image_ir)

        #  Annotate image and publish results:
        if self.detection_publisher is not None:
            ros_detection = self.bridge.to_ros_bounding_box_list(boxes)
            self.detection_publisher.publish(ros_detection)
            # We get can the data back using self.bridge.from_ros_bounding_box_list(ros_detection)
            # e.g., opendr_detection = self.bridge.from_ros_bounding_box_list(ros_detection)

        if self.rgb_publisher is not None:
            plot_rgb = draw_bounding_boxes(image_rgb, boxes, class_names=self.classes)
            message = self.bridge.to_ros_image(Image(np.uint8(plot_rgb)))
            self.rgb_publisher.publish(message)
        if self.ir_publisher is not None:
            plot_ir = draw_bounding_boxes(image_ir, boxes, class_names=self.classes)
            message = self.bridge.to_ros_image(Image(np.uint8(plot_ir)))
            self.ir_publisher.publish(message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_rgb_image_topic", help="Topic name for input rgb image", type=str, default="/camera/color/image_raw"
    )
    parser.add_argument(
        "--output_rgb_image_topic",
        help="Topic name for output annotated rgb image",
        type=str,
        default="/opendr/rgb_image_objects_annotated",
    )
    parser.add_argument(
        "--input_infra_image_topic", help="Topic name for input infra image", type=str, default="/camera/infra/image_raw"
    )
    parser.add_argument(
        "--output_infra_image_topic",
        help="Topic name for output annotated infra image",
        type=str,
        default="/opendr/infra_image_objects_annotated",
    )
    parser.add_argument("--detections_topic", help="Topic name for detection messages", type=str, default="/opendr/objects")
    parser.add_argument(
        "--device",
        help='Device to use, either "cpu" or "cuda", defaults to "cuda"',
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
    )
    args = parser.parse_args()

    # Select the device for running the
    try:
        if args.device == "cuda" and torch.cuda.is_available():
            device = "cuda"
        elif args.device == "cuda":
            print("GPU not found. Using CPU instead.")
            device = "cpu"
        else:
            print("Using CPU.")
            device = "cpu"
    except:
        print("Using CPU.")
        device = "cpu"

    detection_estimation_node = ObjectDetectionGemNode(
        device=device,
        input_rgb_image_topic=args.input_rgb_image_topic,
        output_rgb_image_topic=args.output_rgb_image_topic,
        input_infra_image_topic=args.input_infra_image_topic,
        output_infra_image_topic=args.output_infra_image_topic,
        detections_topic=args.detections_topic,
    )
    detection_estimation_node.listen()
