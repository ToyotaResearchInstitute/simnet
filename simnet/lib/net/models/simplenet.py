import dataclasses

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass
class Node:
  inp: any
  module: nn.Module
  activated: bool
  stride: int
  dim: int

  def __hash__(self):
    return hash(self.module)


class NetFactory(nn.Module):

  def __init__(self):
    super().__init__()
    self.nodes = []
    self.skips = {}
    self.tags = {}

  def tag(self, node, name):
    self.tags[node] = name

  def input(self, in_dim=3, stride=1, activated=True):
    assert not self.nodes
    n = Node(inp=None, module=None, activated=activated, stride=stride, dim=in_dim)
    self.nodes.append(n)
    return n

  def _add(self, node):
    self.nodes.append(node)
    return node

  def _activate(self, node):
    if node.activated:
      return node
    return self._add(
        dataclasses.replace(
            node,
            inp=node,
            module=nn.Sequential(nn.BatchNorm2d(node.dim), nn.LeakyReLU()),
            activated=True
        )
    )

  def _conv(self, node, out_dim=None, stride=1, rate=1, kernel=3):
    node = self._activate(node)
    if out_dim is None:
      out_dim = node.dim
    padding = (kernel - 1) // 2 * rate
    return self._add(
        dataclasses.replace(
            node,
            inp=node,
            module=nn.Conv2d(
                node.dim, out_dim, kernel, stride=stride, dilation=rate, padding=padding
            ),
            activated=False,
            dim=out_dim,
            stride=node.stride * stride
        )
    )

  def _interp(self, node):
    return self._add(
        dataclasses.replace(
            node,
            inp=node,
            module=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            stride=node.stride // 2
        )
    )

  def _lateral(self, node, out_dim=None):
    if out_dim is None:
      out_dim = node.dim
    if out_dim == node.dim:
      return node
    return self._conv(node, out_dim=out_dim, kernel=1)

  def output(self, node, out_dim):
    return self._conv(node, out_dim=out_dim, kernel=1)

  def downscale(self, node, out_dim):
    return self._conv(node, out_dim, stride=2)

  def upsample(self, node, skip, out_dim):
    skip = self._lateral(skip, out_dim=out_dim)
    node = self._lateral(node, out_dim=out_dim)
    node = self._interp(node)
    self.skips[node] = skip
    return node

  def layer(self, node, out_dim=None, rate=1):
    if out_dim is None:
      out_dim = node.dim
    skip = self._lateral(node, out_dim=out_dim)
    node = self._conv(node, rate=rate)
    node = self._conv(node, rate=rate)
    self.skips[node] = skip
    return node

  def block(self, node, rates):
    for r in [int(r) for r in rates]:
      node = self.layer(node, rate=r)
    return node

  def bake(self):
    self.modules = nn.ModuleList(n.module for n in self.nodes if n.module is not None)
    return self

  def forward(self, x):
    outputs = {}
    tag_outputs = {}
    for node in self.nodes:
      if node.module is None:  # initial input
        pass
      else:
        if node in self.skips:
          x = outputs[self.skips[node]] + node.module(outputs[node.inp])
        else:
          x = node.module(outputs[node.inp])
      outputs[node] = x
      last = x
      if node in self.tags:
        tag_outputs[self.tags[node]] = x
    if tag_outputs:
      return tag_outputs
    return last


def hdrn_alpha_base(num_channels):
  net = NetFactory()
  x = net.input()
  x = net.downscale(x, num_channels)
  x = net.downscale(x, num_channels)
  x4 = x = net.block(x, '111')
  x = net.downscale(x, num_channels * 2)
  x8 = x = net.block(x, '1111')
  x = net.downscale(x, num_channels * 4)
  x = net.block(x, '12591259')
  x = net.upsample(x, x8, num_channels // 2)
  x = net.upsample(x, x4, num_channels // 2)
  return net.bake()


def make_process_cost_volume(num_disparities):
  net = NetFactory()
  x = net.input(in_dim=num_disparities, stride=4, activated=True)
  x = net.block(x, '1259')
  x = net.output(x, out_dim=num_disparities)
  return net.bake()


class HdrnAlphaStereo(nn.Module):

  def __init__(self, hparams):
    super().__init__()

    self.num_disparities = hparams.max_disparity
    self.internal_scale = hparams.cost_volume_downsample_factor
    self.internal_num_disparities = self.num_disparities // self.internal_scale
    assert self.internal_scale in [4, 8, 16]

    self.feature_extractor = hdrn_alpha_base(hparams.fe_internal_features)
    self.cost_volume = DotProductCostVolume(self.internal_num_disparities)
    self.process_cost_volume = make_process_cost_volume(self.internal_num_disparities)

    self.soft_argmin = SoftArgmin()

  def forward(self, left_image, right_image):
    left_score = self.feature_extractor(left_image)
    right_score = self.feature_extractor(right_image)

    cost_volume = self.cost_volume(left_score, right_score)
    cost_volume = self.process_cost_volume(cost_volume)

    disparity_small = self.soft_argmin(cost_volume)

    return disparity_small


class StereoBackbone(nn.Module):

  def __init__(self, hparams, in_channels=3):
    super().__init__()

    def make_rgb_stem():
      net = NetFactory()
      x = net.input(in_dim=3, stride=1, activated=True)
      x = net.downscale(x, 32)
      x = net.downscale(x, 32)
      return net.bake()

    def make_disp_features():
      net = NetFactory()
      x = net.input(in_dim=1, stride=1, activated=False)
      x = net.layer(x, 32, rate=5)
      return net.bake()

    self.rgb_stem = make_rgb_stem()
    self.stereo_stem = HdrnAlphaStereo(hparams)
    self.disp_features = make_disp_features()

    def make_rgbd_backbone(num_channels=64, out_dim=64):
      net = NetFactory()
      x = net.input(in_dim=64, activated=True, stride=4)
      x = net._lateral(x, out_dim=num_channels)
      x4 = x = net.block(x, '111')
      x = net.downscale(x, num_channels * 2)
      x8 = x = net.block(x, '1111')
      x = net.downscale(x, num_channels * 4)
      x = net.block(x, '12591259')
      net.tag(net.output(x, out_dim), 'p4')
      x = net.upsample(x, x8, out_dim)
      net.tag(x, 'p3')
      x = net.upsample(x, x4, out_dim)
      net.tag(x, 'p2')
      return net.bake()

    self.rgbd_backbone = make_rgbd_backbone()

  def forward(self, stacked_img, step, robot_joint_angles=None):
    small_disp = self.stereo_stem.forward(stacked_img[:, 0:3], stacked_img[:, 3:6])
    left_rgb_features = self.rgb_stem.forward(stacked_img[:, 0:3])
    disp_features = self.disp_features(small_disp)
    rgbd_features = torch.cat((disp_features, left_rgb_features), axis=1)
    outputs = self.rgbd_backbone.forward(rgbd_features)
    outputs['small_disp'] = small_disp
    return outputs

  @property
  def out_channels(self):
    return 32

  @property
  def stride(self):
    return 4  # = stride 2 conv -> stride 2 max pool


@torch.jit.script
def cost_volume(left, right, num_disparities: int, is_right: bool):
  batch_size, channels, height, width = left.shape

  output = torch.zeros((batch_size, channels, num_disparities, height, width),
                       dtype=left.dtype,
                       device=left.device)

  for i in range(num_disparities):
    if not is_right:
      output[:, :, i, :, i:] = left[:, :, :, i:] * right[:, :, :, :width - i]
    else:
      output[:, :, i, :, :width - i] = left[:, :, :, i:] * right[:, :, :, :width - i]

  return output


class CostVolume(nn.Module):
  """Compute cost volume using cross correlation of left and right feature maps"""

  def __init__(self, num_disparities, is_right=False):
    super().__init__()
    self.num_disparities = num_disparities
    self.is_right = is_right

  def forward(self, left, right):
    if torch.jit.is_scripting():
      return cost_volume(left, right, self.num_disparities, self.is_right)
    else:
      return self.forward_with_amp(left, right)

  @torch.jit.unused
  def forward_with_amp(self, left, right):
    """This operation is unstable at float16, so compute at float32 even when using mixed precision"""
    with torch.cuda.amp.autocast(enabled=False):
      left = left.to(torch.float32)
      right = right.to(torch.float32)
      output = cost_volume(left, right, self.num_disparities, self.is_right)
      output = torch.clamp(output, -1e3, 1e3)
      return output


@torch.jit.script
def dot_product_cost_volume(left, right, num_disparities: int, is_right: bool):
  batch_size, channels, height, width = left.shape

  output = torch.zeros((batch_size, num_disparities, height, width),
                       dtype=left.dtype,
                       device=left.device)

  for i in range(num_disparities):
    if not is_right:
      output[:, i, :, i:] = (left[:, :, :, i:] * right[:, :, :, :width - i]).mean(dim=1)
    else:
      output[:, i, :, width - i] = (left[:, :, :, i:] * right[:, :, :, :width - i]).mean(dim=1)

  return output


class DotProductCostVolume(nn.Module):
  """Compute cost volume using dot product of left and right feature maps"""

  def __init__(self, num_disparities, is_right=False):
    super().__init__()
    self.num_disparities = num_disparities
    self.is_right = is_right

  def forward(self, left, right):
    return dot_product_cost_volume(left, right, self.num_disparities, self.is_right)

  @torch.jit.unused
  def forward_with_amp(self, left, right):
    """This operation is unstable at float16, so compute at float32 even when using mixed precision"""
    with torch.cuda.amp.autocast(enabled=False):
      left = left.to(torch.float32)
      right = right.to(torch.float32)
      output = dot_product_cost_volume(left, right, self.num_disparities, self.is_right)
      output = torch.clamp(output, -1e3, 1e3)
      return output


@torch.jit.script
def soft_argmin(input):
  _, channels, _, _ = input.shape

  softmin = F.softmin(input, dim=1)
  index_tensor = torch.arange(0, channels, dtype=softmin.dtype,
                              device=softmin.device).view(1, channels, 1, 1)
  output = torch.sum(softmin * index_tensor, dim=1, keepdim=True)
  return output


class SoftArgmin(nn.Module):
  """Compute soft argmin operation for given cost volume"""

  def forward(self, input):
    return soft_argmin(input)


@torch.jit.script
def matchability(input):
  softmin = F.softmin(input, dim=1)
  log_softmin = F.log_softmax(-input, dim=1)
  output = torch.sum(softmin * log_softmin, dim=1, keepdim=True)
  return output


class Matchability(nn.Module):
  """Compute disparity matchability value from https://arxiv.org/abs/2008.04800"""

  def forward(self, input):
    if torch.jit.is_scripting():
      # Torchscript generation can't handle mixed precision, so always compute at float32.
      return matchability(input)
    else:
      return self.forward_with_amp(input)

  @torch.jit.unused
  def forward_with_amp(self, input):
    """This operation is unstable at float16, so compute at float32 even when using mixed precision"""
    with torch.cuda.amp.autocast(enabled=False):
      input = input.to(torch.float32)
      return matchability(input)


def main():
  num_channels = 32
  net = NetFactory()
  x = net.input()
  x = net.downscale(x, num_channels)
  x = net.downscale(x, num_channels)
  x4 = x = net.block(x, '111')
  x = net.downscale(x, num_channels * 2)
  x8 = x = net.block(x, '1111')
  x = net.downscale(x, num_channels * 4)
  x = net.block(x, '12591259')
  x = net.upsample(x, x8, num_channels // 2)
  x = net.upsample(x, x4, num_channels // 2)
  net.bake()

  x = torch.randn(5, 3, 512, 640)
  y = net(x)

  import torch._C as _C
  TrainingMode = _C._onnx.TrainingMode
  torch.onnx.export(
      net,
      x,
      "test_net.onnx",
      do_constant_folding=False,
      verbose=True,
      training=TrainingMode.TRAINING,
      opset_version=13
  )
  import onnx
  from onnx import shape_inference
  onnx.save(shape_inference.infer_shapes(onnx.load('test_net.onnx')), 'test_net_shapes.onnx')


if __name__ == '__main__':
  main()
