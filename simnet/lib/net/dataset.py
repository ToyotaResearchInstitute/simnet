# Copyright 2019 Toyota Research Institute.  All rights reserved.

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from simnet.lib import datapoint
from simnet.lib.net.post_processing import obb_outputs, depth_outputs, segmentation_outputs


def extract_left_numpy_img(anaglyph):
  anaglyph_np = np.ascontiguousarray(anaglyph.cpu().numpy())
  anaglyph_np = anaglyph_np.transpose((1, 2, 0))
  left_img = anaglyph_np[..., 0:3] * 255.0
  return left_img


def extract_right_numpy_img(anaglyph):
  anaglyph_np = np.ascontiguousarray(anaglyph.cpu().numpy())
  anaglyph_np = anaglyph_np.transpose((1, 2, 0))
  left_img = anaglyph_np[..., 3:6] * 255.0
  return left_img


def create_anaglyph(stereo_dp):
  height, width, _ = stereo_dp.left_color.shape
  image = np.zeros([height, width, 6], dtype=np.uint8)
  cv2.normalize(stereo_dp.left_color, stereo_dp.left_color, 0, 255, cv2.NORM_MINMAX)
  cv2.normalize(stereo_dp.right_color, stereo_dp.right_color, 0, 255, cv2.NORM_MINMAX)
  image[..., 0:3] = stereo_dp.left_color
  image[..., 3:6] = stereo_dp.right_color
  image = image * 1. / 255.0
  image = image.transpose((2, 0, 1))
  return torch.from_numpy(np.ascontiguousarray(image)).float()


class Dataset(Dataset):

  def __init__(self, dataset_uri, hparams, preprocess_image_func=None, datapoint_dataset=None):
    super().__init__()

    if datapoint_dataset is None:
      datapoint_dataset = datapoint.make_dataset(dataset_uri)
    self.datapoint_handles = datapoint_dataset.list()
    # No need to shuffle, already shufled based on random uids
    self.hparams = hparams

    if preprocess_image_func is None:
      self.preprocces_image_func = create_anaglyph
    else:
      assert False
      self.preprocces_image_func = preprocess_image_func

  def __len__(self):
    return len(self.datapoint_handles)

  def __getitem__(self, idx):
    dp = self.datapoint_handles[idx].read()

    anaglyph = self.preprocces_image_func(dp.stereo)

    segmentation_target = segmentation_outputs.SegmentationOutput(dp.segmentation, self.hparams)
    segmentation_target.convert_to_torch_from_numpy()
    depth_target = depth_outputs.DepthOutput(dp.depth, self.hparams)
    depth_target.convert_to_torch_from_numpy()
    pose_target = None
    for pose_dp in dp.object_poses:
      pose_target = obb_outputs.OBBOutput(
          pose_dp.heat_map, pose_dp.vertex_target, pose_dp.z_centroid, pose_dp.cov_matrices,
          self.hparams
      )
      pose_target.convert_to_torch_from_numpy()

    # TODO(kevin): remove these unused outputs
    box_target = None
    kp_target = None

    scene_name = dp.scene_name

    return anaglyph, segmentation_target, depth_target, pose_target, box_target, kp_target, dp.detections, scene_name
