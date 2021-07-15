import numpy as np
import torch
import torch.nn as nn
from matplotlib import cm

from torch.nn import functional as F
from simnet.lib.net import losses

_mse_loss = losses.MSELoss()
_MAX_DISP = 128


class DepthOutput:

  def __init__(self, depth_pred, loss_multiplier):
    self.depth_pred = depth_pred
    self.is_numpy = False
    self.loss = nn.SmoothL1Loss()
    #self.loss = nn.MSELoss()
    self.loss_multiplier = loss_multiplier

  # Converters for torch to numpy
  def convert_to_numpy_from_torch(self):
    self.depth_pred = np.ascontiguousarray(self.depth_pred.cpu().numpy())
    self.is_numpy = True

  def convert_to_torch_from_numpy(self):
    self.depth_pred[self.depth_pred > _MAX_DISP] = _MAX_DISP - 1
    self.depth_pred = torch.from_numpy(np.ascontiguousarray(self.depth_pred)).float()
    self.is_numpy = False

  def get_visualization_img(self, left_img_np, corner_scale=1, raw_disp=True):
    if not self.is_numpy:
      self.convert_to_numpy_from_torch()
    disp = self.depth_pred[0]

    if raw_disp:
      return disp_map_visualize(disp)
    disp_scaled = disp[::corner_scale, ::corner_scale]
    left_img_np[:disp_scaled.shape[0], -disp_scaled.shape[1]:] = disp_map_visualize(disp_scaled)
    return left_img_np

  def compute_loss(self, depth_targets, log, name):
    if self.is_numpy:
      raise ValueError("Output is not in torch mode")
    depth_target_stacked = []
    for depth_target in depth_targets:
      depth_target_stacked.append(depth_target.depth_pred)
    depth_target_batch = torch.stack(depth_target_stacked)
    depth_target_batch = depth_target_batch.to(torch.device('cuda:0'))
    scale_factor = self.depth_pred.shape[2] / depth_target_batch.shape[2]
    if scale_factor != 1.0:
      depth_target_batch = F.interpolate(
          depth_target_batch[:, None, :, :], scale_factor=scale_factor
      )[:, 0, :, :]
      # scale down disparity by same factor as spatial resize
      depth_target_batch = depth_target_batch * scale_factor
    depth_loss = self.loss(self.depth_pred, depth_target_batch) / scale_factor
    log[name] = depth_loss
    return self.loss_multiplier * depth_loss


def turbo_vis(heatmap, normalize=False, uint8_output=False):
  assert len(heatmap.shape) == 2
  if normalize:
    heatmap = heatmap.astype(np.float32)
    heatmap -= np.min(heatmap)
    heatmap /= np.max(heatmap)
  assert heatmap.dtype != np.uint8

  x = heatmap
  x = x.clip(0, 1)
  a = (x * 255).astype(int)
  b = (a + 1).clip(max=255)
  f = x * 255.0 - a
  turbo_map = np.array(cm.turbo.colors)[::-1]
  pseudo_color = (turbo_map[a] + (turbo_map[b] - turbo_map[a]) * f[..., np.newaxis])
  pseudo_color[heatmap < 0.0] = 0.0
  pseudo_color[heatmap > 1.0] = 1.0
  if uint8_output:
    pseudo_color = (pseudo_color * 255).astype(np.uint8)
  return pseudo_color


def disp_map_visualize(x, max_disp=_MAX_DISP):
  assert len(x.shape) == 2
  x = x.astype(np.float64)
  valid = ((x < max_disp) & np.isfinite(x))
  if valid.sum() == 0:
    return np.zeros_like(x).astype(np.uint8)
  x -= np.min(x[valid])
  x /= np.max(x[valid])
  x = 1. - x
  x[~valid] = 0.
  x = turbo_vis(x)
  x = (x * 255).astype(np.uint8)
  return x[:, :, ::-1]
