# Copyright 2020-2021 OpenDR European Project
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
import torch
import numpy as np
import time
import math
import torchgeometry as tgm
from pathlib import Path
import torch.nn.functional as F
from opendr.perception.object_tracking_3d.single_object_tracking.voxel_bof.second_detector.core.box_np_ops import (
    box_camera_to_lidar,
    center_to_corner_box3d,
)

import opendr.perception.object_tracking_3d.single_object_tracking.voxel_bof.second_detector.data.kitti_common as kitti
from opendr.perception.object_tracking_3d.single_object_tracking.voxel_bof.second_detector.data.preprocess import (
    merge_second_batch,
)

from opendr.perception.object_tracking_3d.single_object_tracking.voxel_bof.second_detector.utils.eval import (
    get_official_eval_result,
)
from opendr.perception.object_tracking_3d.single_object_tracking.voxel_bof.second_detector.utils.progress_bar import (
    ProgressBar,
)
from opendr.perception.object_tracking_3d.single_object_tracking.voxel_bof.logger import (
    Logger,
)
from PIL import Image as PilImage


def example_convert_to_torch(
    example, dtype=torch.float32, device=None
) -> dict:
    device = device or torch.device("cuda:0")
    example_torch = {}
    float_names = [
        "voxels",
        "anchors",
        "reg_targets",
        "reg_weights",
        "bev_map",
        "rect",
        "Trv2c",
        "P2",
        "gt_boxes",
    ]

    for k, v in example.items():
        if k in float_names:
            example_torch[k] = torch.as_tensor(v, dtype=dtype, device=device)
        elif k in ["coordinates", "labels", "num_points"]:
            example_torch[k] = torch.as_tensor(
                v, dtype=torch.int32, device=device
            )
        elif k in ["anchors_mask"]:
            example_torch[k] = torch.as_tensor(
                v, dtype=torch.uint8, device=device
            )
        else:
            example_torch[k] = v
    return example_torch


def draw_pseudo_image(pseudo_image, path, targets=[], colors=[]):

    pi = pseudo_image.mean(axis=0).detach().cpu().numpy()

    grayscale_pi = (
        (pseudo_image.mean(axis=0) * 255 / pseudo_image.mean(axis=0).max())
        .detach()
        .cpu()
        .numpy()
        .astype(np.uint8)
    )
    rgb_pi = np.stack([grayscale_pi] * 3, axis=-1)
    rgb_pi[pi < 0, 0] = 0

    for target, color in zip(targets, colors):

        pos_min = (target[0] - target[1] // 2).astype(np.int32)
        pos_max = (target[0] + (target[1] + 1) // 2 - 1).astype(np.int32)

        rgb_pi[
            pos_min[0] : pos_max[0] + 1, pos_min[1] : pos_max[1] + 1, :
        ] = color

    os.makedirs(str(Path(path).parent), exist_ok=True)
    image = PilImage.fromarray(rgb_pi)
    image.save(path)

    return image


def create_targets_and_searches(
    centers, target_sizes, search_sizes, rotations, augment
):

    delta = search_sizes - target_sizes
    offsets = np.random.randint(-delta // 2, delta // 2)

    search_centers = centers + (offsets if augment else 0)

    targets = []
    searches = []

    for center, search_center, target_size, search_size, rotation in zip(
        centers, search_centers, target_sizes, search_sizes, rotations
    ):
        targets.append([center, target_size, rotation[0]])
        searches.append([search_center, search_size, rotation[0]])

    return targets, searches


def create_target_search_regions(
    bv_range,
    voxel_size,
    annos=None,
    rect=None,
    Trv2c=None,
    boxes_lidar=None,
    augment=True,
):

    bv_min = bv_range[:2]
    voxel_size_bev = voxel_size[:2]

    batch_targets = []
    batch_searches = []

    all_boxes_lidar = []

    if annos is not None:
        for i, anno_original in enumerate(annos):

            anno = kitti.remove_dontcare(anno_original)

            dims = anno["dimensions"]
            locs = anno["location"]
            rots = anno["rotation_y"]

            gt_boxes_camera = np.concatenate(
                [locs, dims, rots[..., np.newaxis]], axis=1
            )
            boxes_lidar = box_camera_to_lidar(
                gt_boxes_camera, rect[i], Trv2c[i]
            )

            all_boxes_lidar.append(boxes_lidar)
    elif boxes_lidar is not None:
        all_boxes_lidar = boxes_lidar
    else:
        raise Exception()

    for boxes_lidar in all_boxes_lidar:

        locs_lidar = boxes_lidar[:, 0:3]
        dims_lidar = boxes_lidar[:, 3:6]
        rots_lidar = boxes_lidar[:, 6:7]

        # origin = [0.5, 0.5, 0]
        # gt_corners = center_to_corner_box3d(
        #     locs_lidar,
        #     dims_lidar,
        #     rots_lidar.reshape(-1),
        #     origin=origin,
        #     axis=2,
        # )

        centers = locs_lidar
        sizes = dims_lidar
        rotations = rots_lidar

        centers_image = ((centers[:, :2] - bv_min) / voxel_size_bev).astype(
            np.int32
        )
        sizes_image = (sizes[:, :2] / voxel_size_bev).astype(np.int32) + 5
        search_sizes = sizes_image * 2 + (sizes_image < 20) * 30

        targets, searches = create_targets_and_searches(
            centers_image,
            sizes_image,
            search_sizes,
            rotations,
            augment=augment,
        )
        batch_targets.append(targets)
        batch_searches.append(searches)

    return batch_targets, batch_searches


def get_sub_image(image, center, size):
    result = torch.zeros(
        [image.shape[0], *np.floor(size + 0.5).astype(np.int32)],
        dtype=torch.float32,
        device=image.device,
    )
    image_size = image.shape[-2:]

    pos_min = np.floor(0.5 + center - size // 2).astype(np.int32)
    pos_max = np.floor(pos_min + size + 0.5).astype(np.int32) - 1

    local_min = np.array([0, 0], dtype=np.int32)
    local_max = np.floor(size + 0.5).astype(np.int32) - 1

    for i in range(2):
        if pos_min[i] < 0:
            local_min[i] -= pos_min[i]
            pos_min[i] -= pos_min[i]
        if pos_max[i] >= image_size[i]:
            delta = pos_max[i] - image_size[i] + 1
            pos_max[i] -= delta
            local_max[i] -= delta

    if np.all(pos_max > pos_min):
        result[
            :, local_min[0] : local_max[0] + 1, local_min[1] : local_max[1] + 1
        ] = image[:, pos_min[0] : pos_max[0] + 1, pos_min[1] : pos_max[1] + 1]
    else:
        print("Empty image")

    return result


def create_logisticloss_labels(
    label_size, t, r_pos, r_neg=0, ignore_label=-100, loss="bce"
):
    labels = np.zeros(label_size)

    if t[0] is None:
        return labels

    for r in range(label_size[0]):
        for c in range(label_size[1]):
            dist = np.sqrt(
                ((r - t[0]) - label_size[0] // 2) ** 2
                + ((c - t[1]) - label_size[1] // 2) ** 2,
            )
            if dist <= 0:
                labels[r, c] = 1
            elif np.all(dist <= r_pos):
                labels[r, c] = 0.7
            elif np.all(dist <= r_neg):
                labels[r, c] = ignore_label
            else:
                if loss == "bce" or loss == "focal":
                    labels[r, c] = 0
                else:
                    labels[r, c] = -1

    return labels


def get_rotated_sub_image(pseudo_image, center, size, angle):

    pi = pseudo_image.unsqueeze(0)

    M = tgm.get_rotation_matrix2d(
        torch.tensor([center[1], center[0]], dtype=torch.float32).reshape(
            1, 2
        ),
        torch.tensor(-angle / np.pi * 180).reshape(1),
        torch.tensor(1).reshape(1),
    ).to(pi.device)
    img_warped = tgm.warp_affine(pi, M, dsize=(pi.shape[2], pi.shape[3]))
    # draw_pseudo_image(img_warped[0], "./plots/rpi.png")
    # draw_pseudo_image(
    #     img_warped[0], "./plots/rpi_c.png", [[center, size]], [(255, 0, 0)]
    # )

    # theta = torch.tensor(
    #     cv2.getRotationMatrix2D(
    #         center=(int(center[1]), int(center[0])), angle=angle / np.pi * 180, scale=1
    #     ),
    #     device=pseudo_image.device,
    #     dtype=torch.float32
    # )

    # # theta = torch.tensor(
    # #     [
    # #         [math.cos(angle), math.sin(-angle), 0],
    # #         [math.sin(angle), math.cos(angle), 0],
    # #     ],
    # #     dtype=torch.float32,
    # #     device=pseudo_image.device,
    # # )

    # grid = F.affine_grid(
    #     theta.unsqueeze(0), pseudo_image.unsqueeze(0).size()
    # ).to(pseudo_image.device)
    # output = F.grid_sample(pseudo_image.unsqueeze(0), grid)

    # draw_pseudo_image(pseudo_image, "./plots/pi.png")
    # draw_pseudo_image(
    #     pseudo_image, "./plots/pi_c.png", [[center, size]], [(255, 0, 0)]
    # )
    # draw_pseudo_image(output[0], "./plots/rpi.png")

    x = int(center[1] - size[1] / 2)
    y = int(center[0] - size[0] / 2)

    image = get_sub_image(img_warped[0], center, size)

    # draw_pseudo_image(image, "./plots/im.png")
    return image


def create_pseudo_image_features(
    pseudo_image, target, net, uspcale_size, context_amount
):

    # image = get_rotated_sub_image(
    #     pseudo_image,
    #     target[0][[1, 0]],
    #     target[1][[1, 0]].astype(np.int32),
    #     target[2],
    # )
    # image_upscaled = torch.nn.functional.interpolate(
    #     image.reshape(1, *image.shape),
    #     size=(uspcale_size[0], uspcale_size[1]),
    #     mode="bicubic",
    # )

    image_upscaled, image = sub_image_with_context(
        pseudo_image,
        target,
        (uspcale_size[0], uspcale_size[1]),
        context_amount,
    )

    # if np.any(np.array(image.shape[-2:]) <= 0):
    #     image = torch.zeros((image.shape[0], 1, 1), device=image.device)

    features = net(image_upscaled)

    return features, image


def image_to_feature_coordinates(pos, feature_blocks):

    result = pos

    for _ in range(feature_blocks):
        result = (result + 1) // 2

    return result


def feature_to_image_coordinates(pos, feature_blocks):
    return pos * (2 ** feature_blocks)


def image_to_lidar_coordinates(location, size, voxel_size, bv_range):

    bv_min = bv_range[:2]
    voxel_size_bev = voxel_size[:2]

    location_lidar = location * voxel_size_bev + bv_min
    size_lidar = size * voxel_size_bev

    return location_lidar, size_lidar


def create_static_label_and_weights(
    target, search, target_size, search_size, search_size_with_context, feature_blocks, loss="bce", radius=8
):

    label_size = (
        image_to_feature_coordinates(search_size, feature_blocks)
        - image_to_feature_coordinates(target_size, feature_blocks)
        + 1
    )

    delta_position_original = target[0] - search[0]
    rotated_delta_position_original = rotate_vector(delta_position_original, search[2])

    delta_position_label_space = (
        rotated_delta_position_original / search_size_with_context * label_size
    ).astype(np.int32)

    t = delta_position_label_space[[1, 0]]
    # t = (0, 0)

    r_pos = image_to_feature_coordinates(radius, feature_blocks)

    labels = create_logisticloss_labels(label_size, t, r_pos, loss=loss)
    weights = np.zeros_like(labels)

    # pred_target_position = score_to_image_coordinates(torch.tensor(labels), target[1], search)
    # p = image_to_feature_coordinates(pred_target_position) - image_to_feature_coordinates(search[0])

    neg_label = 0 if loss == "bce" or loss == "focal" else -1

    pos_num = np.sum(labels == 1)
    neg_num = np.sum(labels == neg_label)
    if pos_num > 0:
        weights[labels == 1] = 0.5 / pos_num
    weights[labels == neg_label] = 0.5 / neg_num
    weights *= pos_num + neg_num

    labels = labels.reshape(1, 1, *labels.shape)
    weights = weights.reshape(1, 1, *weights.shape)

    return labels, weights


def size_with_context(target_size, context_amount):
    mean_size = context_amount * (np.sum(target_size))
    sub_image_size_side = np.sqrt(
        (target_size[0] + mean_size) * (target_size[1] + mean_size)
    )
    sub_image_size = np.array([sub_image_size_side, sub_image_size_side])

    return sub_image_size


def sub_image_with_context(
    pseudo_image, target, interoplation_size, context_amount
):
    target_size = target[1]
    sub_image_size = size_with_context(target_size, context_amount)

    sub_image = get_rotated_sub_image(
        pseudo_image, target[0][[1, 0]], sub_image_size, target[2],
    )
    # draw_pseudo_image(sub_image, "./plots/train/sub_image" + str(0) + ".png")

    interpolated_image = torch.nn.functional.interpolate(
        sub_image.reshape(1, *sub_image.shape),
        size=interoplation_size,
        mode="bicubic",
    )
    # draw_pseudo_image(interpolated_image.squeeze(axis=0), "./plots/train/interpolated_image" + str(0) + ".png")
    return interpolated_image, sub_image


def create_pseudo_images_and_labels(
    net,
    example_torch,
    annos=None,
    gt_boxes=None,
    target_size=np.array([127, 127]),
    search_size=np.array([255, 255]),
    context_amount=0.5,
):
    pseudo_image = net.create_pseudo_image(example_torch)
    feature_blocks = net.feature_blocks

    if annos is not None:

        batch_targets, batch_searches = create_target_search_regions(
            net.bv_range,
            net.voxel_size,
            annos,
            example_torch["rect"].cpu().numpy(),
            example_torch["Trv2c"].cpu().numpy(),
        )
    elif gt_boxes is not None:
        batch_targets, batch_searches = create_target_search_regions(
            net.bv_range, net.voxel_size, boxes_lidar=gt_boxes,
        )
    else:
        raise Exception()

    items = []

    for i, (targets, searches) in enumerate(
        zip(batch_targets, batch_searches)
    ):
        for target, search in zip(targets, searches):

            search_size_with_context = size_with_context(search[1], context_amount)

            target_image, _ = sub_image_with_context(
                pseudo_image[i],
                target,
                (target_size[0], target_size[1]),
                context_amount,
            )

            search_image, _ = sub_image_with_context(
                pseudo_image[i],
                search,
                (search_size[0], search_size[1]),
                context_amount,
            )

            # draw_pseudo_image(target_image.squeeze(axis=0), "./plots/pi_target_" + str(0) + ".png")
            # draw_pseudo_image(search_image.squeeze(axis=0), "./plots/pi_search_" + str(0) + ".png")
            # draw_pseudo_image(pseudo_image[i], "./plots/pi_" + str(0) + ".png")
            # draw_pseudo_image(pseudo_image[i], "./plots/pi_t" + str(0) + ".png", [[target[0][[1, 0]], target[1][[1, 0]]]], [(255, 0, 0)])
            # # draw_pseudo_image(pseudo_image[i], "./plots/pi_t2" + str(0) + ".png", [[target[0][[1, 0]], target[1][[1, 0]]]], [(255, 0, 0)])

            labels, weights = create_static_label_and_weights(
                # [target[0][[1, 0]], target[1][[1, 0]]],
                # [search[0][[1, 0]], search[1][[1, 0]]],
                target,
                search,
                target_size,
                search_size,
                search_size_with_context,
                feature_blocks,
            )

            labels_torch = torch.tensor(labels, device=target_image.device)
            weights_torch = torch.tensor(weights, device=target_image.device)

            items.append(
                [
                    target_image,
                    search_image,
                    labels_torch,
                    weights_torch,
                    target,
                    search,
                    search_size_with_context,
                    pseudo_image[i],
                ]
            )

    return items


def hann_window(size, device):
    hann_1 = torch.hann_window(size[0], device=device)
    hann_2 = torch.hann_window(size[1], device=device)
    window = torch.mm(hann_1.view(-1, 1), hann_2.view(1, -1))
    window = window / window.sum()
    return window


def score_to_image_coordinates(scores, target_region_size, search_region):

    max_score = torch.max(scores)
    max_idx = (scores == max_score).nonzero(as_tuple=False)[0][-2:]

    left_top_score = max_idx.cpu().numpy()
    left_top_search_features = left_top_score
    left_top_search_image = feature_to_image_coordinates(
        left_top_search_features
    )
    center_search_image = left_top_search_image + target_region_size // 2
    center_image = (
        center_search_image + search_region[0] - search_region[1] // 2
    )

    return center_image


def select_best_scores_and_search(
    multi_scale_scores_searches_penalties_and_features,
):

    (
        top_scores,
        top_search,
        first_penalty,
        top_features,
    ) = multi_scale_scores_searches_penalties_and_features[0]
    max_top_score = torch.max(top_scores) * first_penalty

    for i in range(1, len(multi_scale_scores_searches_penalties_and_features)):
        (
            scores,
            search,
            penalty,
            features,
        ) = multi_scale_scores_searches_penalties_and_features[i]
        max_score = torch.max(scores) * penalty

        if max_score > max_top_score:
            top_scores = scores
            max_top_score = max_score
            top_search = search
            top_features = features

    return top_scores, top_search, top_features


def displacement_score_to_image_coordinates(
    scores,
    score_upscale,
    search_region_size,
    search_region_rotation,
    feature_blocks,
    search_region_upscale_size=np.array([255, 255]),
):

    max_score = torch.max(scores)
    max_idx = (scores == max_score).nonzero(as_tuple=False)[0][-2:]

    final_score_size = np.array(scores.shape[-2:])

    half = (final_score_size - 1) / 2

    disp_score = max_idx.cpu().numpy() - half
    disp_search = feature_to_image_coordinates(disp_score, feature_blocks) / score_upscale
    disp_image_rotated = (
        disp_search * search_region_size / search_region_upscale_size
    )

    rot = np.array(
        [
            [
                math.cos(search_region_rotation),
                -math.sin(search_region_rotation),
            ],
            [
                math.sin(search_region_rotation),
                math.cos(search_region_rotation),
            ],
        ]
    )

    disp_image = np.dot(rot, disp_image_rotated)

    return disp_image


def create_multi_scale_searches(search, scale_penalty, delta=0.05):

    all_searches_and_penalties = []

    for delta_x in [-1, 0, 1]:
        for delta_y in [-1, 0, 1]:

            penalty = 1

            if delta_x != 0 or delta_y != 0:
                penalty *= scale_penalty

            delta_sign = np.array([delta_x, delta_y])
            delta_weight = np.round(delta * search[1]).astype(np.int32)
            delta_weight[delta_weight <= 0] = 1

            to_add = delta_weight * delta_sign
            new_target = [search[0], search[1] + to_add]

            all_searches_and_penalties.append([new_target, penalty])

    return all_searches_and_penalties


def create_multi_rotate_searches(search, rotate_penalty, delta):

    all_searches_and_penalties = []

    for delta_index in [-1, 0, 1]:
        # for delta_index in [-2, -1, 0, 1, 2]:
        penalty = 1
        if delta_index != 0:
            penalty *= rotate_penalty

        delta_angle = delta_index * delta
        new_search = [search[0], search[1], search[2] + delta_angle]
        all_searches_and_penalties.append([new_search, penalty])

    return all_searches_and_penalties


def create_scaled_scores(
    target_features, search_features, model, score_upscale, window_influence
):

    scores = model.process_features(search_features, target_features)
    scores2 = torch.nn.functional.interpolate(
        scores, scale_factor=score_upscale, mode="bicubic"
    )
    penalty = hann_window(scores2.shape[-2:], device=scores2.device)
    scores2_scaled = (
        1 - window_influence
    ) * scores2 + window_influence * penalty

    return scores2_scaled, scores, scores2


def rotate_vector(vector, angle):

    rot = np.array(
        [
            [math.cos(angle), -math.sin(angle),],
            [math.sin(angle), math.cos(angle),],
        ]
    )

    result = np.dot(rot, vector)

    return result


def train(
    siamese_model,
    input_cfg,
    train_cfg,
    eval_input_cfg,
    model_cfg,
    mixed_optimizer,
    lr_scheduler,
    model_dir,
    float_dtype,
    refine_weight,
    loss_scale,
    class_names,
    center_limit_range,
    input_dataset_iterator,
    eval_dataset_iterator,
    gt_annos,
    device,
    checkpoint_after_iter,
    checkpoints_path,
    display_step=50,
    log=print,
    auto_save=False,
    image_shape=None,
    evaluate=True,
    context_amount=0.5,
    debug=False,
):

    net = siamese_model.branch
    net.global_step -= net.global_step
    feature_blocks = net.feature_blocks

    ######################
    # PREPARE INPUT
    ######################

    take_gt_annos_from_example = False

    if gt_annos is None:
        take_gt_annos_from_example = True
        gt_annos = []

    def _worker_init_fn(worker_id):
        time_seed = np.array(time.time(), dtype=np.int32)
        np.random.seed(time_seed + worker_id)
        log(
            Logger.LOG_WHEN_VERBOSE,
            f"WORKER {worker_id} seed:",
            np.random.get_state()[1][0],
        )

    dataloader = torch.utils.data.DataLoader(
        input_dataset_iterator,
        batch_size=input_cfg.batch_size,
        shuffle=True,
        num_workers=input_cfg.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch,
        worker_init_fn=_worker_init_fn,
    )
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset_iterator,
        batch_size=eval_input_cfg.batch_size,
        shuffle=False,
        num_workers=eval_input_cfg.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch,
    )
    data_iter = iter(dataloader)

    ######################
    # TRAINING
    ######################
    total_step_elapsed = 0
    t = time.time()

    total_loop = train_cfg.steps // train_cfg.steps_per_eval + 1
    clear_metrics_every_epoch = train_cfg.clear_metrics_every_epoch

    if train_cfg.steps % train_cfg.steps_per_eval == 0:
        total_loop -= 1
    mixed_optimizer.zero_grad()

    average_loss = 0
    average_delta_error = 0

    for _ in range(total_loop):
        if total_step_elapsed + train_cfg.steps_per_eval > train_cfg.steps:
            steps = train_cfg.steps % train_cfg.steps_per_eval
        else:
            steps = train_cfg.steps_per_eval
        for step in range(steps):
            lr_scheduler.step()
            try:
                example = next(data_iter)
            except StopIteration:
                log(Logger.LOG_WHEN_NORMAL, "end epoch")
                if clear_metrics_every_epoch:
                    net.clear_metrics()
                data_iter = iter(dataloader)
                example = next(data_iter)
            example_torch = example_convert_to_torch(
                example, float_dtype, device=device
            )

            if debug:
                items = create_pseudo_images_and_labels(
                    net, example_torch, gt_boxes=example_torch["gt_boxes_2"],
                    context_amount=context_amount,
                )
            else:
                with torch.no_grad():
                    items = create_pseudo_images_and_labels(
                        net,
                        example_torch,
                        gt_boxes=example_torch["gt_boxes_2"],
                    )

            for (
                target_image,
                search_image,
                labels,
                weights,
                target,
                search,
                search_size_with_context,
                pseudo_image,
            ) in items:
                pred, feat_target, feat_search = siamese_model(
                    search_image, target_image
                )
                loss = net.criterion(pred, labels, weights)

                delta = displacement_score_to_image_coordinates(
                    pred, 1, search[1], 0, feature_blocks
                )
                true_delta = displacement_score_to_image_coordinates(
                    labels, 1, search[1], 0, feature_blocks
                )
                # predicted_label_target_coordinates = score_to_image_coordinates(
                #     labels, target[1], search
                # )
                # delta = np.abs(predicted_target_coordinates - target[0])

                if debug:
                    feat_search.register_hook(
                        lambda grad: (
                            None,
                            draw_pseudo_image(
                                grad[0], "./plots/grad/feat_search.png"
                            ),
                        )[0]
                    )
                    feat_target.register_hook(
                        lambda grad: (
                            None,
                            draw_pseudo_image(
                                grad[0], "./plots/grad/feat_target.png"
                            ),
                        )[0]
                    )
                    search_image.register_hook(
                        lambda grad: (
                            None,
                            draw_pseudo_image(
                                grad[0], "./plots/grad/search_image.png"
                            ),
                        )[0]
                    )
                    target_image.register_hook(
                        lambda grad: (
                            None,
                            draw_pseudo_image(
                                grad[0], "./plots/grad/target_image.png"
                            ),
                        )[0]
                    )
                    draw_pseudo_image(pred[0], "./plots/train/pred.png")
                    draw_pseudo_image(labels[0], "./plots/train/labels.png")
                    draw_pseudo_image(
                        feat_search[0], "./plots/train/feat_search.png"
                    )
                    draw_pseudo_image(
                        feat_target[0], "./plots/train/feat_target.png"
                    )

                    vector = target[0] - search[0]
                    rot1 = rotate_vector(vector, search[2])
                    rot2 = rotate_vector(vector, -search[2])
                    rot3 = rotate_vector(vector, search[2] + np.pi)
                    rot4 = rotate_vector(vector, -search[2] + np.pi)

                    draw_pseudo_image(
                        search_image[0],
                        "./plots/train/search_image.png",
                        [
                            [
                                np.array(search_image.shape[-2:]) / 2,
                                np.array([5, 5]),
                                0,
                            ],
                            # [
                            #     (rot1 / search[1])[[1, 0]]
                            #     * np.array(search_image.shape[-2:])
                            #     + np.array(search_image.shape[-2:]) / 2,
                            #     np.array([5, 5]),
                            #     0,
                            # ],
                            [
                                (rot1 / search_size_with_context)[[1, 0]]
                                * np.array(search_image.shape[-2:])
                                + np.array(search_image.shape[-2:]) / 2,
                                np.array([5, 5]),
                                0,
                            ],
                        ],
                        [
                            (255, 0, 0),
                            # (0, 255, 0),
                            (0, 0, 255),
                            # (0, 255, 125),
                            # (255, 255, 125),
                            # (125, 40, 215),
                            # (25, 40, 215),
                            # (225, 40, 215),
                            # (125, 0, 0),
                            # (0, 125, 0),
                            # (0, 0, 125),
                        ],
                    )

                    draw_pseudo_image(
                        target_image[0],
                        "./plots/train/target_image.png",
                        [
                            [
                                np.array(target_image.shape[-2:]) / 2,
                                np.array([5, 5]),
                                0,
                            ]
                        ],
                        [(255, 0, 0)],
                    )
                    draw_pseudo_image(
                        pseudo_image,
                        "./plots/train/pseudo_image.png",
                        [
                            [search[0][[1, 0]], np.array([5, 5]), 0],
                            [target[0][[1, 0]], np.array([4, 4]), 0],
                            [(rot1 + search[0])[[1, 0]], np.array([3, 3]), 0],
                            [(rot2 + search[0])[[1, 0]], np.array([3, 3]), 0],
                            [(rot3 + search[0])[[1, 0]], np.array([3, 3]), 0],
                            [(rot4 + search[0])[[1, 0]], np.array([3, 3]), 0],
                        ],
                        [
                            (255, 0, 0),
                            (0, 255, 255),
                            (0, 0, 255),
                            (0, 255, 0),
                            (125, 0, 255),
                            (125, 255, 0),
                        ],
                    )
                    print()
                else:
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
                    mixed_optimizer.step()
                    mixed_optimizer.zero_grad()
                net.update_global_step()
                global_step = net.get_global_step()

                average_loss += loss
                average_delta_error += np.abs(delta - true_delta)

                if global_step % display_step == 0:
                    average_loss /= display_step
                    average_delta_error /= display_step

                    print(
                        "[",
                        global_step,
                        "]",
                        "loss=" + str(float(average_loss.detach().cpu())),
                        "error_position=",
                        average_delta_error,
                    )

                    average_loss = 0
                    average_delta_error = 0

                if (
                    checkpoint_after_iter > 0
                    and global_step % checkpoint_after_iter == 0
                ):

                    save_path = (
                        checkpoints_path / f"checkpoint_{global_step}.pth"
                    )

                    torch.save(
                        {
                            "siamese_model": siamese_model.state_dict(),
                            "optimizer": mixed_optimizer.state_dict(),
                        },
                        save_path,
                    )

        total_step_elapsed += steps

        if evaluate:
            pass
            # net.eval()

        net.train()


def evaluate(
    net,
    eval_input_cfg,
    model_cfg,
    mixed_optimizer,
    model_dir,
    float_dtype,
    class_names,
    center_limit_range,
    eval_dataset_iterator,
    gt_annos,
    device,
    predict_test=False,
    log=print,
    image_shape=None,
    count=None,
):

    take_gt_annos_from_example = False

    if gt_annos is None:
        take_gt_annos_from_example = True
        gt_annos = []

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset_iterator,
        batch_size=eval_input_cfg.batch_size,
        shuffle=False,
        num_workers=eval_input_cfg.num_workers,
        pin_memory=False,
        collate_fn=merge_second_batch,
    )

    net.eval()
    t = time.time()

    if (
        model_cfg.rpn.module_class_name == "PSA"
        or model_cfg.rpn.module_class_name == "RefineDet"
    ):
        dt_annos_coarse = []
        dt_annos_refine = []
        log(Logger.LOG_WHEN_NORMAL, "Generate output labels...")
        bar = ProgressBar()
        bar.start(len(eval_dataloader) // eval_input_cfg.batch_size + 1)
        for example in iter(eval_dataloader):

            if take_gt_annos_from_example:
                gt_annos += list(example["annos"])

            example = example_convert_to_torch(
                example, float_dtype, device=device
            )
            coarse, refine = predict_kitti_to_anno(
                net,
                example,
                class_names,
                center_limit_range,
                model_cfg.lidar_input,
                use_coarse_to_fine=True,
                global_set=None,
                image_shape=image_shape,
            )
            dt_annos_coarse += coarse
            dt_annos_refine += refine

            if count is not None and len(dt_annos_refine) >= count:
                break

            bar.print_bar(
                log=lambda *x, **y: log(Logger.LOG_WHEN_NORMAL, *x, **y)
            )
    else:
        dt_annos = []
        log(Logger.LOG_WHEN_NORMAL, "Generate output labels...")
        bar = ProgressBar()
        bar.start(len(eval_dataloader) // eval_input_cfg.batch_size + 1)
        for i, example_numpy in enumerate(iter(eval_dataloader)):

            if take_gt_annos_from_example:
                gt_annos += list(example_numpy["annos"])

            example = example_convert_to_torch(
                example_numpy, float_dtype, device=device
            )

            items = create_pseudo_images_and_labels(
                net,
                example,
                gt_annos[
                    i
                    * eval_input_cfg.batch_size : (i + 1)
                    * eval_input_cfg.batch_size
                ],
            )

            dt_annos += predict_kitti_to_anno(
                net,
                example,
                class_names,
                center_limit_range,
                model_cfg.lidar_input,
                use_coarse_to_fine=False,
                global_set=None,
                image_shape=image_shape,
            )

            if count is not None and len(dt_annos) >= count:
                break

            bar.print_bar(
                log=lambda *x, **y: log(Logger.LOG_WHEN_NORMAL, *x, **y)
            )

    if count is not None:
        if (
            model_cfg.rpn.module_class_name == "PSA"
            or model_cfg.rpn.module_class_name == "RefineDet"
        ):
            gt_annos = gt_annos[: len(dt_annos_refine)]
        else:
            gt_annos = gt_annos[: len(dt_annos)]

    sec_per_example = len(eval_dataloader) / (time.time() - t)
    log(
        Logger.LOG_WHEN_NORMAL,
        f"generate label finished({sec_per_example:.2f}/s). start eval:",
    )

    log(
        Logger.LOG_WHEN_NORMAL,
        f"avg forward time per example: {net.avg_forward_time:.3f}",
    )
    log(
        Logger.LOG_WHEN_NORMAL,
        f"avg postprocess time per example: {net.avg_postprocess_time:.3f}",
    )
    if not predict_test:

        if (
            model_cfg.rpn.module_class_name == "PSA"
            or model_cfg.rpn.module_class_name == "RefineDet"
        ):
            log(Logger.LOG_WHEN_NORMAL, "Before Refine:")
            result_coarse = get_official_eval_result(
                gt_annos, dt_annos_coarse, class_names
            )
            log(Logger.LOG_WHEN_NORMAL, result_coarse)

            log(Logger.LOG_WHEN_NORMAL, "After Refine:")
            (
                result_refine,
                mAPbbox,
                mAPbev,
                mAP3d,
                mAPaos,
            ) = get_official_eval_result(
                gt_annos, dt_annos_refine, class_names, return_data=True
            )
            log(Logger.LOG_WHEN_NORMAL, result_refine)
            dt_annos = dt_annos_refine
        else:
            result, mAPbbox, mAPbev, mAP3d, mAPaos = get_official_eval_result(
                gt_annos, dt_annos, class_names, return_data=True
            )
            log(Logger.LOG_WHEN_NORMAL, result)

        return mAPbbox, mAPbev, mAP3d, mAPaos


def comput_kitti_output(
    predictions_dicts,
    batch_image_shape,
    lidar_input,
    center_limit_range,
    class_names,
    global_set,
):
    annos = []
    for i, preds_dict in enumerate(predictions_dicts):
        image_shape = (
            batch_image_shape[i] if batch_image_shape is not None else None
        )
        img_idx = (
            preds_dict["image_idx"]
            if preds_dict["image_idx"] is not None
            else 0
        )
        if preds_dict["bbox"] is not None:
            box_2d_preds = preds_dict["bbox"].detach().cpu().numpy()
            box_preds = preds_dict["box3d_camera"].detach().cpu().numpy()
            scores = preds_dict["scores"].detach().cpu().numpy()
            box_preds_lidar = preds_dict["box3d_lidar"].detach().cpu().numpy()
            # write pred to file
            label_preds = preds_dict["label_preds"].detach().cpu().numpy()
            anno = kitti.get_start_result_anno()
            num_example = 0
            for box, box_lidar, bbox, score, label in zip(
                box_preds, box_preds_lidar, box_2d_preds, scores, label_preds
            ):
                if not lidar_input:
                    if bbox[0] > image_shape[1] or bbox[1] > image_shape[0]:
                        continue
                    if bbox[2] < 0 or bbox[3] < 0:
                        continue
                if center_limit_range is not None:
                    limit_range = np.array(center_limit_range)
                    if np.any(box_lidar[:3] < limit_range[:3]) or np.any(
                        box_lidar[:3] > limit_range[3:]
                    ):
                        continue
                bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                bbox[:2] = np.maximum(bbox[:2], [0, 0])
                anno["name"].append(class_names[int(label)])
                anno["truncated"].append(0.0)
                anno["occluded"].append(0)
                anno["alpha"].append(
                    -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6]
                )
                anno["bbox"].append(bbox)
                anno["dimensions"].append(box[3:6])
                anno["location"].append(box[:3])
                anno["rotation_y"].append(box[6])
                if global_set is not None:
                    for i in range(100000):
                        if score in global_set:
                            score -= 1 / 100000
                        else:
                            global_set.add(score)
                            break
                anno["score"].append(score)

                num_example += 1
            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                annos.append(anno)
            else:
                annos.append(kitti.empty_result_anno())
        else:
            annos.append(kitti.empty_result_anno())
        num_example = annos[-1]["name"].shape[0]
        annos[-1]["image_idx"] = np.array(
            [img_idx] * num_example, dtype=np.int64
        )

    return annos


def compute_lidar_kitti_output(
    predictions_dicts, center_limit_range, class_names, global_set,
):
    annos = []
    for i, preds_dict in enumerate(predictions_dicts):
        if preds_dict["box3d_lidar"] is not None:
            scores = preds_dict["scores"].detach().cpu().numpy()
            box_preds_lidar = preds_dict["box3d_lidar"].detach().cpu().numpy()
            label_preds = preds_dict["label_preds"].detach().cpu().numpy()
            anno = kitti.get_start_result_anno()
            num_example = 0
            for box_lidar, score, label in zip(
                box_preds_lidar, scores, label_preds
            ):
                if center_limit_range is not None:
                    limit_range = np.array(center_limit_range)
                    if np.any(box_lidar[:3] < limit_range[:3]) or np.any(
                        box_lidar[:3] > limit_range[3:]
                    ):
                        continue
                anno["name"].append(class_names[int(label)])
                anno["truncated"].append(0.0)
                anno["occluded"].append(0)
                anno["alpha"].append(
                    -np.arctan2(-box_lidar[1], box_lidar[0]) + box_lidar[6]
                )
                anno["bbox"].append(None)
                anno["dimensions"].append(box_lidar[3:6])
                anno["location"].append(box_lidar[:3])
                anno["rotation_y"].append(box_lidar[6])
                if global_set is not None:
                    for i in range(100000):
                        if score in global_set:
                            score -= 1 / 100000
                        else:
                            global_set.add(score)
                            break
                anno["score"].append(score)

                num_example += 1
            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                annos.append(anno)
            else:
                annos.append(kitti.empty_result_anno())
        else:
            annos.append(kitti.empty_result_anno())
        num_example = annos[-1]["name"].shape[0]
        annos[-1]["image_idx"] = np.array([None] * num_example)

    return annos


def predict_kitti_to_anno(
    net,
    example,
    class_names,
    center_limit_range=None,
    lidar_input=False,
    use_coarse_to_fine=True,
    global_set=None,
    image_shape=None,
):
    batch_image_shape = (
        example["image_shape"]
        if "image_shape" in example
        else ([image_shape] * len(example["P2"]))
    )

    if use_coarse_to_fine:
        predictions_dicts_coarse, predictions_dicts_refine = net(example)
        annos_coarse = comput_kitti_output(
            predictions_dicts_coarse,
            batch_image_shape,
            lidar_input,
            center_limit_range,
            class_names,
            global_set,
        )
        annos_refine = comput_kitti_output(
            predictions_dicts_refine,
            batch_image_shape,
            lidar_input,
            center_limit_range,
            class_names,
            global_set,
        )
        return annos_coarse, annos_refine
    else:
        predictions_dicts_coarse = net(example)
        annos_coarse = comput_kitti_output(
            predictions_dicts_coarse,
            batch_image_shape,
            lidar_input,
            center_limit_range,
            class_names,
            global_set,
        )

        return annos_coarse


def _predict_kitti_to_file(
    net,
    example,
    result_save_path,
    class_names,
    center_limit_range=None,
    lidar_input=False,
    use_coarse_to_fine=True,
):
    batch_image_shape = example["image_shape"]
    if use_coarse_to_fine:
        _, predictions_dicts_refine = net(example)
        predictions_dicts = predictions_dicts_refine
    else:
        predictions_dicts = net(example)
    for i, preds_dict in enumerate(predictions_dicts):
        image_shape = batch_image_shape[i]
        img_idx = preds_dict["image_idx"]
        if preds_dict["bbox"] is not None:
            box_2d_preds = preds_dict["bbox"].data.cpu().numpy()
            box_preds = preds_dict["box3d_camera"].data.cpu().numpy()
            scores = preds_dict["scores"].data.cpu().numpy()
            box_preds_lidar = preds_dict["box3d_lidar"].data.cpu().numpy()
            # write pred to file
            box_preds = box_preds[
                :, [0, 1, 2, 4, 5, 3, 6]
            ]  # lhw->hwl(label file format)
            label_preds = preds_dict["label_preds"].data.cpu().numpy()
            result_lines = []
            for box, box_lidar, bbox, score, label in zip(
                box_preds, box_preds_lidar, box_2d_preds, scores, label_preds
            ):
                if not lidar_input:
                    if bbox[0] > image_shape[1] or bbox[1] > image_shape[0]:
                        continue
                    if bbox[2] < 0 or bbox[3] < 0:
                        continue
                if center_limit_range is not None:
                    limit_range = np.array(center_limit_range)
                    if np.any(box_lidar[:3] < limit_range[:3]) or np.any(
                        box_lidar[:3] > limit_range[3:]
                    ):
                        continue
                bbox[2:] = np.minimum(bbox[2:], image_shape[::-1])
                bbox[:2] = np.maximum(bbox[:2], [0, 0])
                result_dict = {
                    "name": class_names[int(label)],
                    "alpha": -np.arctan2(-box_lidar[1], box_lidar[0]) + box[6],
                    "bbox": bbox,
                    "location": box[:3],
                    "dimensions": box[3:6],
                    "rotation_y": box[6],
                    "score": score,
                }
                result_line = kitti.kitti_result_line(result_dict)
                result_lines.append(result_line)
        else:
            result_lines = []
        result_file = (
            f"{result_save_path}/{kitti.get_image_index_str(img_idx)}.txt"
        )
        result_str = "\n".join(result_lines)
        with open(result_file, "w") as f:
            f.write(result_str)


def iou_2d(center1, size1, center2, size2):

    x11, y11 = center1 - size1 / 2
    x12, y12 = center1 + size1 / 2

    x21, y21 = center2 - size2 / 2
    x22, y22 = center2 + size2 / 2

    # determine the coordinates of the intersection rectangle
    x_left = max(x11, x21)
    y_top = max(y11, y21)
    x_right = min(x12, x22)
    y_bottom = min(y12, y22)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (x12 - x11) * (y12 - y11)
    bb2_area = (x22 - x21) * (y22 - y21)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou
