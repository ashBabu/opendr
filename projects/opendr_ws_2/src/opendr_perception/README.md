# OpenDR Perception Package

This package contains ROS2 nodes related to the perception package of OpenDR.

---

## Prerequisites

Before you can run any of the toolkit's ROS2 nodes, some prerequisites need to be fulfilled:
1. First of all, you need to [set up the required packages and build your workspace.](../../README.md#first-time-setup)
2. _(Optional for nodes with [RGB input](#rgb-input-nodes))_ 

    For basic usage and testing, all the toolkit's ROS2 nodes that use RGB images are set up to expect input from a basic webcam using the default package `usb_cam` which is installed with OpenDR. You can run the webcam node in a new terminal:
    ```shell
    ros2 run usb_cam usb_cam_node_exe
    ```
    By default, the USB cam node publishes images on `/image_raw` and the RGB input nodes subscribe to this topic if not provided with an input topic argument. 
    As explained for each node below, you can modify the topics via arguments, so if you use any other node responsible for publishing images, **make sure to change the input topic accordingly.**

---

## Notes

- ### Display output images with rqt_image_view
    For any node that outputs images, `rqt_image_view` can be used to display them by running the following command:
    ```shell
    ros2 run rqt_image_view rqt_image_view &
    ```
    A window will appear, where the topic that you want to view can be selected from the drop-down menu on the top-left area of the window.
    Refer to each node's documentation below to find out the default output image topic, where applicable, and select it on the drop-down menu of rqt_image_view.

- ### Echo node output
    All OpenDR nodes publish some kind of detection message, which can be echoed by running the following command:
    ```shell
    ros2 topic echo /opendr/topic_name
    ```
    You can find out the default topic name for each node, in its documentation below.

- ### Increase performance by disabling output
    Optionally, nodes can be modified via command line arguments, which are presented for each node separately below.
    Generally, arguments give the option to change the input and output topics, the device the node runs on (CPU or GPU), etc.
    When a node publishes on several topics, where applicable, a user can opt to disable one or more of the outputs by providing `None` in the corresponding output topic.
    This disables publishing on that topic, forgoing some operations in the node, which might increase its performance.

    _An example would be to disable the output annotated image topic in a node when visualization is not needed and only use the detection message in another node, thus eliminating the OpenCV operations._

- ### An example diagram of OpenDR nodes running
    ![Face Detection ROS node running diagram](../../images/opendr_node_diagram.png)
    - On the left, the `usb_cam` node can be seen, which is using a system camera to publish images on the `/image_raw` topic.
    - In the middle, OpenDR's face detection node is running taking as input the published image. By default, the node has its input topic set to `/image_raw`.
    - To the right the two output topics of the face detection node can be seen.
    The bottom topic `/opendr/image_faces_annotated` is the annotated image which can be easily viewed with `rqt_image_view` as explained earlier.
    The other topic `/opendr/faces` is the detection message which contains the detected faces' detailed information.
    This message can be easily viewed by running `ros2 topic echo /opendr/faces` in a terminal.

<!-- - ### Other notes -->

----

## RGB input nodes

### Pose Estimation ROS2 Node

You can find the pose estimation ROS2 node python script [here](./opendr_perception/pose_estimation_node.py) to inspect the code and modify it as you wish to fit your needs.
The node makes use of the toolkit's [pose estimation tool](../../../../src/opendr/perception/pose_estimation/lightweight_open_pose/lightweight_open_pose_learner.py) whose documentation can be found [here](../../../../docs/reference/lightweight-open-pose.md).
The node publishes the detected poses in [OpenDR's 2D pose message format](../opendr_interface/msg/OpenDRPose2D.msg), which saves a list of [OpenDR's keypoint message format](../opendr_interface/msg/OpenDRPose2DKeypoint.msg).

#### Instructions for basic usage:

1. Start the node responsible for publishing images. If you have a USB camera, then you can use the `usb_cam_node` as explained in the [prerequisites above](#prerequisites).

2. You are then ready to start the pose detection node:
    ```shell
    ros2 run opendr_perception pose_estimation_node.py
    ```
    The following optional arguments are available:
   - `-h, --help`: show a help message and exit
   - `-i or --input_rgb_image_topic INPUT_RGB_IMAGE_TOPIC`: topic name for input RGB image (default=`/image_raw`)
   - `-o or --output_rgb_image_topic OUTPUT_RGB_IMAGE_TOPIC`: topic name for output annotated RGB image, `None` to stop the node from publishing on this topic (default=`/opendr/image_pose_annotated`)
   - `-d or --detections_topic DETECTIONS_TOPIC`: topic name for detection messages, `None` to stop the node from publishing on this topic (default=`/opendr/poses`)
   - `--device DEVICE`: Device to use, either `cpu` or `cuda`, falls back to `cpu` if GPU or CUDA is not found (default=`cuda`)
   - `--accelerate`: Acceleration flag that causes pose estimation to run faster but with less accuracy

3. Default output topics:
   - Output images: `/opendr/image_pose_annotated`
   - Detection messages: `/opendr/poses`

   For viewing the output, refer to the [notes above.](#notes)

### Fall Detection ROS2 Node

You can find the fall detection ROS2 node python script [here](./scripts/fall_detection.py) to inspect the code and modify it as you wish to fit your needs.
The node makes use of the toolkit's [fall detection tool](../../../../src/opendr/perception/fall_detection/fall_detector_learner.py) whose documentation can be found [here](../../../../docs/reference/fall-detection.md).
Fall detection uses the toolkit's pose estimation tool internally.

<!-- TODO Should add information about taking advantage of the pose estimation ros node when running fall detection, see issue https://github.com/opendr-eu/opendr/issues/282 -->

Instructions for basic usage and visualization of results:

1. Start the node responsible for publishing images. If you have a USB camera, then you can use the `usb_cam_node` as explained in the [prerequisites above](#prerequisites).

2. You are then ready to start the fall detection node

    ```shell
    rosrun perception fall_detection.py
    ```
    The following optional arguments are available:
   - `-h, --help`: show a help message and exit
   - `-i or --input_rgb_image_topic INPUT_RGB_IMAGE_TOPIC`: topic name for input RGB image (default=`/usb_cam/image_raw`)
   - `-o or --output_rgb_image_topic OUTPUT_RGB_IMAGE_TOPIC`: topic name for output annotated RGB image, `None` to stop the node from publishing on this topic (default=`/opendr/image_fallen_annotated`)
   - `-d or --detections_topic DETECTIONS_TOPIC`: topic name for detection messages, `None` to stop the node from publishing on this topic (default=`/opendr/fallen`)
   - `--device DEVICE`: Device to use, either `cpu` or `cuda`, falls back to `cpu` if GPU or CUDA is not found (default=`cuda`)
   - `--accelerate`: Acceleration flag that causes pose estimation that runs internally to run faster but with less accuracy

3. In a new terminal you can view the annotated image stream by running `rosrun rqt_image_view rqt_image_view` and selecting the topic `/opendr/image_fall_annotated` or by running `rostopic echo /opendr/fallen`, where the node publishes bounding boxes of detected fallen poses.

### Face Detection ROS2 Node

The face detection ROS2 node supports both the ResNet and MobileNet versions, of latter of which performs mask recognition as well.

You can find the face detection ROS2 node python script [here](./scripts/face_detection_retinaface.py) to inspect the code and modify it as you wish to fit your needs. The node makes use of the toolkit's [face detection tool](../../../../src/opendr/perception/object_detection_2d/retinaface/retinaface_learner.py) whose documentation can be found [here](../../../../docs/reference/face-detection-2d-retinaface.md).

Instructions for basic usage and visualization of results:

1. Start the node responsible for publishing images. If you have a USB camera, then you can use the `usb_cam_node` as explained in the [prerequisites above](#prerequisites).

2. You are then ready to start the face detection node

    ```shell
    rosrun perception face_detection_retinaface.py
    ```
    The following optional arguments are available:
   - `-h, --help`: show a help message and exit
   - `-i or --input_rgb_image_topic INPUT_RGB_IMAGE_TOPIC`: topic name for input RGB image (default=`/usb_cam/image_raw`)
   - `-o or --output_rgb_image_topic OUTPUT_RGB_IMAGE_TOPIC`: topic name for output annotated RGB image, `None` to stop the node from publishing on this topic (default=`/opendr/image_faces_annotated`)
   - `-d or --detections_topic DETECTIONS_TOPIC`: topic name for detection messages, `None` to stop the node from publishing on this topic (default=`/opendr/faces`)
   - `--device DEVICE`: Device to use, either `cpu` or `cuda`, falls back to `cpu` if GPU or CUDA is not found (default=`cuda`)
   - `--backbone BACKBONE`: Retinaface backbone, options are either 'mnet' or 'resnet', where 'mnet' detects masked faces as well (default=`resnet`)

3. In a new terminal you can view the annotated image stream by running `rosrun rqt_image_view rqt_image_view` and selecting the topic `/opendr/image_faces_annotated` or by running `rostopic echo /opendr/faces`, where the node publishes bounding boxes of detected faces.

### Face Recognition ROS2 Node

You can find the face recognition ROS2 node python script [here](./scripts/face_recognition.py) to inspect the code and modify it as you wish to fit your needs.
The node makes use of the toolkit's [face recognition tool](../../../../src/opendr/perception/face_recognition/face_recognition_learner.py) whose documentation can be found [here](../../../../docs/reference/face-recognition.md).

Instructions for basic usage and visualization of results:

1. Start the node responsible for publishing images. If you have a USB camera, then you can use the `usb_cam_node` as explained in the [prerequisites above](#prerequisites).

2. You are then ready to start the face recognition node

    ```shell
    rosrun perception face_recognition.py
    ```
    The following optional arguments are available:
   - `-h, --help`: show a help message and exit
   - `-i or --input_rgb_image_topic INPUT_RGB_IMAGE_TOPIC`: topic name for input RGB image (default=`/usb_cam/image_raw`)
   - `-o or --output_rgb_image_topic OUTPUT_RGB_IMAGE_TOPIC`: topic name for output annotated RGB image, `None` to stop the node from publishing on this topic (default=`/opendr/image_face_reco_annotated`)
   - `-d or --detections_topic DETECTIONS_TOPIC`: topic name for detection messages, `None` to stop the node from publishing on this topic (default=`/opendr/face_recognition`)
   - `-id or --detections_id_topic DETECTIONS_ID_TOPIC`: topic name for detection ID messages, `None` to stop the node from publishing on this topic (default=`/opendr/face_recognition_id`)
   - `--device DEVICE`: Device to use, either `cpu` or `cuda`, falls back to `cpu` if GPU or CUDA is not found (default=`cuda`)
   - `--backbone BACKBONE`: Backbone network (default=`mobilefacenet`)
   - `--dataset_path DATASET_PATH`: Path of the directory where the images of the faces to be recognized are stored (default=`./database`)

3. In a new terminal you can view the annotated image stream by running `rosrun rqt_image_view rqt_image_view` and selecting the topic `/opendr/image_face_reco_annotated` or by running `rostopic echo /opendr/face_recognition`.

**Notes**

Reference images should be placed in a defined structure like:
- imgs
    - ID1
      - image1
      - image2
    - ID2
    - ID3
    - ...

The default dataset path is `./database`. Please use the `--database_path ./your/path/` argument to define a custom one. 
Τhe name of the sub-folder, e.g. ID1, will be published under `/opendr/face_recognition_id`.

The database entry and the returned confidence is published under the topic name `/opendr/face_recognition`, and the human-readable ID
under `/opendr/face_recognition_id`.

### 2D Object Detection ROS2 Nodes

For 2D object detection, there are several ROS2 nodes implemented using various algorithms. The generic obejct detectors are SSD, YOLOv3, CenterNet and DETR.

You can find the 2D object detection ROS2 node python scripts here: [SSD node](./scripts/object_detection_2d_ssd.py), [YOLOv3 node](./scripts/object_detection_2d_yolov3.py), [CenterNet node](./scripts/object_detection_2d_centernet.py) and [DETR node](./scripts/object_detection_2d_detr.py), where you can inspect the code and modify it as you wish to fit your needs.
The nodes makes use of the toolkit's various 2D object detection tools: [SSD tool](../../../../src/opendr/perception/object_detection_2d/ssd/ssd_learner.py), [YOLOv3 tool](../../../../src/opendr/perception/object_detection_2d/yolov3/yolov3_learner.py), [CenterNet tool](../../../../src/opendr/perception/object_detection_2d/centernet/centernet_learner.py), [DETR tool](../../../../src/opendr/perception/object_detection_2d/detr/detr_learner.py), whose documentation can be found here: [SSD docs](../../../../docs/reference/object-detection-2d-ssd.md), [YOLOv3 docs](../../../../docs/reference/object-detection-2d-yolov3.md), [CenterNet docs](../../../../docs/reference/object-detection-2d-centernet.md), [DETR docs](../../../../docs/reference/detr.md).

Instructions for basic usage and visualization of results:

1. Start the node responsible for publishing images. If you have a USB camera, then you can use the `usb_cam_node` as explained in the [prerequisites above](#prerequisites).

2. You are then ready to start a 2D object detector node:
   1. SSD node
      ```shell
      rosrun perception object_detection_2d_ssd.py
      ```
      The following optional arguments are available for the SSD node:
      - `--backbone BACKBONE`: Backbone network (default=`vgg16_atrous`)
      - `--nms_type NMS_TYPE`: Non-Maximum Suppression type options are `default`, `seq2seq-nms`, `soft-nms`, `fast-nms`, `cluster-nms` (default=`default`)
      
   2. YOLOv3 node
      ```shell
      rosrun perception object_detection_2d_yolov3.py
      ```
      The following optional argument is available for the YOLOv3 node:
      - `--backbone BACKBONE`: Backbone network (default=`darknet53`)
      
   3. CenterNet node
      ```shell
      rosrun perception object_detection_2d_centernet.py
      ```
      The following optional argument is available for the YOLOv3 node:
      - `--backbone BACKBONE`: Backbone network (default=`resnet50_v1b`)
      
   4. DETR node
      ```shell
      rosrun perception object_detection_2d_detr.py
      ```

   The following optional arguments are available for all nodes above:
   - `-h, --help`: show a help message and exit
   - `-i or --input_rgb_image_topic INPUT_RGB_IMAGE_TOPIC`: topic name for input RGB image (default=`/usb_cam/image_raw`)
   - `-o or --output_rgb_image_topic OUTPUT_RGB_IMAGE_TOPIC`: topic name for output annotated RGB image, `None` to stop the node from publishing on this topic (default=`/opendr/image_objects_annotated`)
   - `-d or --detections_topic DETECTIONS_TOPIC`: topic name for detection messages, `None` to stop the node from publishing on this topic (default=`/opendr/objects`)
   - `--device DEVICE`: Device to use, either `cpu` or `cuda`, falls back to `cpu` if GPU or CUDA is not found (default=`cuda`)

3. In a new terminal you can view the annotated image stream by running `rosrun rqt_image_view rqt_image_view` and selecting the topic `/opendr/image_objects_annotated` or by running `rostopic echo /opendr/objects`, where the bounding boxes alone are published.

### 2D Object Tracking Deep Sort ROS2 Node
<!-- TODO -->
A ROS2 node for performing Object Tracking 2D using Deep Sort using either pretrained models on Market1501 dataset, or custom trained models.
This is a detection-based method, and therefore the 2D object detector is needed to provide detections, which then will be used to make associations and generate tracking ids.
The predicted tracking annotations are split into two topics with detections (default `output_detection_topic="/opendr/detection"`) and tracking ids (default `output_tracking_id_topic="/opendr/tracking_id"`).
Additionally, an annotated image is generated if the `output_image_topic` is not None (default `output_image_topic="/opendr/image_annotated"`)
Assuming the drivers have been installed and OpenDR catkin workspace has been sourced, the node can be started as:

```shell
rosrun perception object_tracking_2d_deep_sort.py
```
To get images from usb_camera, you can start the camera node as:
```shell
rosrun usb_cam usb_cam_node
```

The corresponding `input_image_topic` should be `/usb_cam/image_raw`.
If you want to use a dataset from the disk, you can start an `image_dataset.py` node as:

```shell
rosrun perception image_dataset.py
```
This will pulbish the dataset images to an `/opendr/dataset_image` topic by default, which means that the `input_image_topic` should be set to `/opendr/dataset_image`.

### Panoptic Segmentation ROS2 Node

You can find the panoptic segmentation ROS2 node python script [here](./scripts/panoptic_segmentation_efficient_ps.py) to inspect the code and modify it as you wish to fit your needs.
The node makes use of the toolkit's [panoptic segmentation tool](../../../../src/opendr/perception/panoptic_segmentation/efficient_ps/efficient_ps_learner.py) whose documentation can be found [here](../../../../docs/reference/efficient-ps.md) and additional information about Efficient PS [here](../../../../src/opendr/perception/panoptic_segmentation/README.md).

Instructions for basic usage and visualization of results:

1. Start the node responsible for publishing images. If you have a USB camera, then you can use the `usb_cam_node` as explained in the [prerequisites above](#prerequisites).

2. You are then ready to start the panoptic segmentation node

    ```shell
    rosrun perception panoptic_segmentation_efficient_ps.py
    ```
    
    The following optional arguments are available:
   - `-h, --help`: show a help message and exit
   - `--input_rgb_image_topic INPUT_RGB_IMAGE_TOPIC` : listen to RGB images on this topic (default=`/usb_cam/image_raw`)
   - `--checkpoint CHECKPOINT` : download pretrained models [cityscapes, kitti] or load from the provided path (default=`cityscapes`)
   - `--output_rgb_image_topic OUTPUT_RGB_IMAGE_TOPIC`: publish the semantic and instance maps on this topic as `OUTPUT_HEATMAP_TOPIC/semantic` and `OUTPUT_HEATMAP_TOPIC/instance`, `None` to stop the node from publishing on this topic (default=`/opendr/panoptic`)
   - `--visualization_topic VISUALIZATION_TOPIC`: publish the panoptic segmentation map as an RGB image on `VISUALIZATION_TOPIC` or a more detailed overview if using the `--detailed_visualization` flag, `None` to stop the node from publishing on this topic (default=`/opendr/panoptic/rgb_visualization`)
   - `--detailed_visualization`: generate a combined overview of the input RGB image and the semantic, instance, and panoptic segmentation maps and publish it on `OUTPUT_RGB_IMAGE_TOPIC` (default=deactivated)

3. In a new terminal you can view the annotated image stream by running `rosrun rqt_image_view rqt_image_view` and selecting the topics `/opendr/panoptic/semantic`, `/opendr/panoptic/instance` and `/opendr/panoptic/rgb_visualization` or by running `rostopic echo /opendr/panoptic/semantic`, `rostopic echo /opendr/panoptic/instance` and `rostopic echo /opendr/panoptic/rgb_visualization`.

### Semantic Segmentation ROS2 Node

You can find the semantic segmentation ROS2 node python script [here](./scripts/semantic_segmentation_bisenet.py) to inspect the code and modify it as you wish to fit your needs. The node makes use of the toolkit's [semantic segmentation tool](../../../../src/opendr/perception/semantic_segmentation/bisenet/bisenet_learner.py) whose documentation can be found [here](../../../../docs/reference/semantic-segmentation.md).

Instructions for basic usage and visualization of results:

1. Start the node responsible for publishing images. If you have a USB camera, then you can use the `usb_cam_node` as explained in the [prerequisites above](#prerequisites).

2. You are then ready to start the semantic segmentation node

    ```shell
    rosrun perception semantic_segmentation_bisenet.py
    ```
    The following optional arguments are available:
   - `-h, --help`: show a help message and exit
   - `-i or --input_rgb_image_topic INPUT_RGB_IMAGE_TOPIC`: topic name for input RGB image (default=`/usb_cam/image_raw`)
   - `-o or --output_heatmap_topic OUTPUT_HEATMAP_TOPIC`: topic to which we are publishing the heatmap in the form of a ROS2 image containing class IDs, `None` to stop the node from publishing on this topic (default=`/opendr/heatmap`)
   - `-v or --output_rgb_image_topic OUTPUT_RGB_IMAGE_TOPIC`: topic to which we are publishing the heatmap image blended with the input image and a class legend for visualization purposes, `None` to stop the node from publishing on this topic (default=`/opendr/heatmap_visualization`)
   - `--device DEVICE`: Device to use, either `cpu` or `cuda`, falls back to `cpu` if GPU or CUDA is not found (default=`cuda`)
   
3. In a new terminal you can view the annotated image stream by running `rosrun rqt_image_view rqt_image_view` and selecting the topic `/opendr/heatmap_visualization` or by running `rostopic echo /opendr/heatmap`.

**Notes**

On the table below you can find the detectable classes and their corresponding IDs:

| Class  | Bicyclist | Building | Car | Column Pole | Fence | Pedestrian | Road | Sidewalk | Sign Symbol | Sky | Tree | Unknown |
|--------|-----------|----------|-----|-------------|-------|------------|------|----------|-------------|-----|------|---------|
| **ID** | 0         | 1        | 2   | 3           | 4     | 5          | 6    | 7        | 8           | 9   | 10   | 11      |

### Landmark-based Facial Expression Recognition ROS2 Node
<!-- TODO -->
A ROS2 node for performing Landmark-based Facial Expression Recognition using the pretrained model PST-BLN on AFEW, CK+ or Oulu-CASIA datasets.
Assuming the drivers have been installed and OpenDR catkin workspace has been sourced, the node can be started as:
```shell
rosrun perception landmark_based_facial_expression_recognition.py
```
The predictied class id and confidence is published under the topic name `/opendr/landmark_based_expression_recognition`, and the human-readable class name under `/opendr/landmark_based_expression_recognition_description`.

### Skeleton-based Human Action Recognition ROS2 Node
<!-- TODO -->
A ROS2 node for performing Skeleton-based Human Action Recognition using either ST-GCN or PST-GCN models pretrained on NTU-RGBD-60 dataset. The human body poses of the image are first extracted by the light-weight Openpose method which is implemented in the toolkit, and they are passed to the skeleton-based action recognition method to be categorized.
Assuming the drivers have been installed and OpenDR catkin workspace has been sourced, the node can be started as:
```shell
rosrun perception skeleton_based_action_recognition.py
```
The predictied class id and confidence is published under the topic name `/opendr/skeleton_based_action_recognition`, and the human-readable class name under `/opendr/skeleton_based_action_recognition_description`.
Besides, the annotated image is published in `/opendr/image_pose_annotated` as well as the corresponding poses in `/opendr/poses`.

### Video Human Activity Recognition ROS2 Node

A ROS2 node for performing Human Activity Recognition using either CoX3D or X3D models pretrained on Kinetics400.

You can find the video human activity recognition ROS2 node python script [here](./scripts/video_activity_recognition.py) to inspect the code and modify it as you wish to fit your needs. 
The node makes use of the toolkit's video human activity recognition tools which can be found [here for CoX3D](../../../../src/opendr/perception/activity_recognition/cox3d/cox3d_learner.py) and 
[here for X3D](../../../../src/opendr/perception/activity_recognition/x3d/x3d_learner.py) whose documentation can be found [here](../../../../docs/reference/activity-recognition.md).

Instructions for basic usage and visualization of results:

1. Start the node responsible for publishing images. If you have a USB camera, then you can use the `usb_cam_node` as explained in the [prerequisites above](#prerequisites).

2. You are then ready to start the video human activity recognition node:

    ```shell
    rosrun perception video_activity_recognition.py
    ```
    The following optional arguments are available:
   - `-h, --help`: show a help message and exit
   - `-i or --input_rgb_image_topic INPUT_RGB_IMAGE_TOPIC`: topic name for input RGB image (default=`/usb_cam/image_raw`)
   - `-o or --output_category_topic OUTPUT_CATEGORY_TOPIC`: Topic to which we are publishing the recognized activity (default=`"/opendr/human_activity_recognition"`)
   - `-od or --output_category_description_topic OUTPUT_CATEGORY_DESRIPTION_TOPIC`: Topic to which we are publishing the ID of the recognized action (default=`/opendr/human_activity_recognition_description`)
   - `--model`: Architecture to use for human activity recognition. Choices are "cox3d-s", "cox3d-m", "cox3d-l", "x3d-xs", "x3d-s", "x3d-m", or "x3d-l" (Default: "cox3d-m").
   - `--device DEVICE`: Device to use, either `cpu` or `cuda`, falls back to `cpu` if GPU or CUDA is not found (default=`cuda`)
   
3. In a new terminal you can view predictions by running `rostopic echo /opendr/human_activity_recognition` and `rostopic echo /opendr/human_activity_recognition_description`.

---
## RGB + Infrared input

### GEM ROS2 Node
<!-- TODO -->
Assuming that you have already [built your workspace](../../README.md) and started roscore (i.e., just run `roscore`), then you can


1. Add OpenDR to `PYTHONPATH` (please make sure you do not overwrite `PYTHONPATH` ), e.g.,
```shell
export PYTHONPATH="/home/user/opendr/src:$PYTHONPATH"
```
2. First one needs to find points in the color and infrared images that correspond, in order to find the homography matrix that allows to correct for the difference in perspective between the infrared and the RGB camera.
These points can be selected using a [utility tool](../../../../src/opendr/perception/object_detection_2d/utils/get_color_infra_alignment.py) that is provided in the toolkit.

3. Pass the points you have found as *pts_color* and *pts_infra* arguments to the ROS2 gem.py node.

4. Start the node responsible for publishing images. If you have a RealSense camera, then you can use the corresponding node (assuming you have installed [realsense2_camera](http://wiki.ros.org/realsense2_camera)):

```shell
roslaunch realsense2_camera rs_camera.launch enable_color:=true enable_infra:=true enable_depth:=false enable_sync:=true infra_width:=640 infra_height:=480
```

4. You are then ready to start the pose detection node

```shell
rosrun perception object_detection_2d_gem.py
```

5. You can examine the annotated image stream using `rqt_image_view` (select one of the topics `/opendr/color_detection_annotated` or `/opendr/infra_detection_annotated`) or `rostopic echo /opendr/detections`

----
## RGBD input

### RGBD Hand Gesture Recognition ROS2 Node
<!-- TODO -->
A ROS2 node for performing hand gesture recognition using MobileNetv2 model trained on HANDS dataset.
The node has been tested with Kinectv2 for depth data acquisition with the following drivers: https://github.com/OpenKinect/libfreenect2 and https://github.com/code-iai/iai_kinect2.
Assuming that the drivers have been installed and OpenDR catkin workspace has been sourced, the node can be started as:
```shell
rosrun perception rgbd_hand_gesture_recognition.py
```
The predictied classes are published to the topic `/opendr/gestures`.

----
## Point cloud input

### 3D Object Detection Voxel ROS2 Node
<!-- TODO -->
A ROS2 node for performing Object Detection 3D using PointPillars or TANet methods with either pretrained models on KITTI dataset, or custom trained models.
The predicted detection annotations are pushed to `output_detection3d_topic` (default `output_detection3d_topic="/opendr/detection3d"`).

Assuming the drivers have been installed and OpenDR catkin workspace has been sourced, the node can be started as:
```shell
rosrun perception object_detection_3d_voxel.py
```
To get a point cloud from a dataset on the disk, you can start a `point_cloud_dataset.py` node as:
```shell
rosrun perception point_cloud_dataset.py
```
This will pulbish the dataset point clouds to a `/opendr/dataset_point_cloud` topic by default, which means that the `input_point_cloud_topic` should be set to `/opendr/dataset_point_cloud`.

### 3D Object Tracking AB3DMOT ROS2 Node
<!-- TODO -->
A ROS2 node for performing Object Tracking 3D using AB3DMOT stateless method.
This is a detection-based method, and therefore the 3D object detector is needed to provide detections, which then will be used to make associations and generate tracking ids.
The predicted tracking annotations are split into two topics with detections (default `output_detection_topic="/opendr/detection3d"`) and tracking ids (default `output_tracking_id_topic="/opendr/tracking3d_id"`).

Assuming the drivers have been installed and OpenDR catkin workspace has been sourced, the node can be started as:
```shell
rosrun perception object_tracking_3d_ab3dmot.py
```
To get a point cloud from a dataset on the disk, you can start a `point_cloud_dataset.py` node as:
```shell
rosrun perception point_cloud_dataset.py
```
This will pulbish the dataset point clouds to a `/opendr/dataset_point_cloud` topic by default, which means that the `input_point_cloud_topic` should be set to `/opendr/dataset_point_cloud`.

### 2D Object Tracking FairMOT ROS2 Node
<!-- TODO -->
A ROS2 node for performing Object Tracking 2D using FairMOT with either pretrained models on MOT dataset, or custom trained models. The predicted tracking annotations are split into two topics with detections (default `output_detection_topic="/opendr/detection"`) and tracking ids (default `output_tracking_id_topic="/opendr/tracking_id"`). Additionally, an annotated image is generated if the `output_image_topic` is not None (default `output_image_topic="/opendr/image_annotated"`)
Assuming the drivers have been installed and OpenDR catkin workspace has been sourced, the node can be started as:
```shell
rosrun perception object_tracking_2d_fair_mot.py
```
To get images from usb_camera, you can start the camera node as:
```shell
rosrun usb_cam usb_cam_node
```
The corresponding `input_image_topic` should be `/usb_cam/image_raw`.
If you want to use a dataset from the disk, you can start a `image_dataset.py` node as:
```shell
rosrun perception image_dataset.py
```
This will pulbish the dataset images to an `/opendr/dataset_image` topic by default, which means that the `input_image_topic` should be set to `/opendr/dataset_image`.

----
## Biosignal input

### Heart Anomaly Detection ROS2 Node
<!-- TODO -->
A ROS2 node for performing heart anomaly (atrial fibrillation) detection from ecg data using GRU or ANBOF models trained on AF dataset. Assuming that the OpenDR catkin workspace has been sourced, the node can be started as:
```shell
rosrun perception heart_anomaly_detection.py ECG_TOPIC MODEL
```
with `ECG_TOPIC` specifying the ROS2 topic to which the node will subscribe, and `MODEL` set to either *gru* or *anbof*. The predictied classes are published to the topic `/opendr/heartanomaly`.

----
## Audio input

### Speech Command Recognition ROS2 Node
<!-- TODO -->
A ROS2 node for recognizing speech commands from an audio stream using MatchboxNet, EdgeSpeechNets or Quadratic SelfONN models, pretrained on the Google Speech Commands dataset.
Assuming that the OpenDR catkin workspace has been sourced, the node can be started with:
```shell
rosrun perception speech_command_recognition.py INPUT_AUDIO_TOPIC
```
The following optional arguments are available:
- `--buffer_size BUFFER_SIZE`: set the size of the audio buffer (expected command duration) in seconds, default value **1.5**
- `--model MODEL`: choose the model to use: `matchboxnet` (default value), `edgespeechnets` or `quad_selfonn`
- `--model_path MODEL_PATH`: if given, the pretrained model will be loaded from the specified local path, otherwise it will be downloaded from an OpenDR FTP server

The predictions (class id and confidence) are published to the topic `/opendr/speech_recognition`.
**Note:** EdgeSpeechNets currently does not have a pretrained model available for download, only local files may be used.

----
## Dataset ROS2 Nodes

The dataset nodes can be used to publish data from the disk, which is useful to test the functionality without the use of a sensor.
Dataset nodes use a provided `DatasetIterator` object that returns a `(Data, Target)` pair.
If the type of the `Data` object is correct, the node will transform it into a corresponding ROS2 message object and publish it to a desired topic. 

### Image Dataset ROS2 Node
To get an image from a dataset on the disk, you can start a `image_dataset.py` node as:
```shell
rosrun perception image_dataset.py
```
By default, it downloads a `nano_MOT20` dataset from OpenDR's FTP server and uses it to publish data to the ROS2 topic.
You can create an instance of this node with any `DatasetIterator` object that returns `(Image, Target)` as elements.
You can inspect [the node](./scripts/image_dataset.py) and modify it to your needs for other image datasets.

### Point Cloud Dataset ROS2 Node
To get a point cloud from a dataset on the disk, you can start a `point_cloud_dataset.py` node as:
```shell
rosrun perception point_cloud_dataset.py
```
By default, it downloads a `nano_KITTI` dataset from OpenDR's FTP server and uses it to publish data to the ROS2 topic.
You can create an instance of this node with any `DatasetIterator` object that returns `(PointCloud, Target)` as elements.
You can inspect [the node](./scripts/point_cloud_dataset.py) and modify it to your needs for other point cloud datasets.
