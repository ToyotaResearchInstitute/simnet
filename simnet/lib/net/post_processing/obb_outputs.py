import numpy as np
import torch

from simnet.lib import transform
from simnet.lib.net.post_processing.epnp import optimize_for_9D
from simnet.lib.net.post_processing import epnp
from simnet.lib.net.post_processing import nms
from simnet.lib.net.post_processing import pose_outputs
from simnet.lib.net.post_processing import eval3d
from simnet.lib.net import losses

_mask_l1_loss = losses.MaskedL1Loss()
_mse_loss = losses.MSELoss()


class OBBOutput:

  def __init__(self, heatmap, vertex_field, z_centroid_field, cov_field, hparams):
    self.heatmap = heatmap
    self.vertex_field = vertex_field
    self.z_centroid_field = z_centroid_field
    self.cov_field = cov_field
    self.is_numpy = False
    self.hparams = hparams

    #def _debug(name, x):
    #  print(name, x.dtype, x.shape, x.min(), x.max())
    #_debug('OBBOutput.heatmap', self.heatmap)
    #_debug('OBBOutput.z_centroid_field', self.z_centroid_field)
    #_debug('OBBOutput.vertex_field', self.vertex_field)
    #_debug('OBBOutput.cov_field', self.cov_field)
    #print('----------------------------------------------------')

  # Converters for torch to numpy
  def convert_to_numpy_from_torch(self):
    self.heatmap = np.ascontiguousarray(self.heatmap.cpu().numpy())
    self.vertex_field = np.ascontiguousarray(self.vertex_field.cpu().numpy())
    self.vertex_field = self.vertex_field.transpose((0, 2, 3, 1))
    self.vertex_field = self.vertex_field / 100.0
    self.cov_field = np.ascontiguousarray(self.cov_field.cpu().numpy())
    self.cov_field = self.cov_field.transpose((0, 2, 3, 1))
    self.cov_field = self.cov_field / 1000.0
    self.z_centroid_field = np.ascontiguousarray(self.z_centroid_field.cpu().numpy())
    self.z_centroid_field = self.z_centroid_field / 100.0 + 1.0
    self.is_numpy = True

  def convert_to_torch_from_numpy(self):
    self.vertex_field = self.vertex_field.transpose((2, 0, 1))
    self.vertex_field = 100.0 * self.vertex_field
    self.vertex_field = torch.from_numpy(np.ascontiguousarray(self.vertex_field)).float()
    self.cov_field = self.cov_field.transpose((2, 0, 1))
    self.cov_field = 1000.0 * self.cov_field
    self.cov_field = torch.from_numpy(np.ascontiguousarray(self.cov_field)).float()
    self.heatmap = torch.from_numpy(np.ascontiguousarray(self.heatmap)).float()
    # Normalize z_centroid by 1.
    self.z_centroid_field = 100.0 * (self.z_centroid_field - 1.0)
    self.z_centroid_field = torch.from_numpy(np.ascontiguousarray(self.z_centroid_field)).float()
    self.is_numpy = False

  def get_detections(self, camera_model):
    if not self.is_numpy:
      self.convert_to_numpy_from_torch()

    poses, scores = compute_oriented_bounding_boxes(
        np.copy(self.heatmap[0]),
        np.copy(self.vertex_field[0]),
        np.copy(self.z_centroid_field[0]),
        np.copy(self.cov_field[0]),
        camera_model=camera_model
    )
    detections = []
    for pose, score in zip(poses, scores):
      bbox = epnp.get_2d_bbox_of_9D_box(pose.camera_T_object, pose.scale_matrix, camera_model)
      detections.append(
          eval3d.Detection(
              camera_T_object=pose.camera_T_object,
              bbox=bbox,
              score=score,
              scale_matrix=pose.scale_matrix
          )
      )
    detections = nms.run(detections)
    return detections

  def get_visualization_img(self, left_img, camera_model=None):
    if not self.is_numpy:
      self.convert_to_numpy_from_torch()
    return draw_oriented_bounding_box_from_outputs(
        self.heatmap[0],
        self.vertex_field[0],
        self.cov_field[0],
        self.z_centroid_field[0],
        left_img,
        camera_model=camera_model
    )

  def compute_loss(self, obb_targets, log):
    if self.is_numpy:
      raise ValueError("Output is not in torch mode")
    vertex_target = torch.stack([obb_target.vertex_field for obb_target in obb_targets])
    z_centroid_field_target = torch.stack([
        obb_target.z_centroid_field for obb_target in obb_targets
    ])
    heatmap_target = torch.stack([obb_target.heatmap for obb_target in obb_targets])
    cov_target = torch.stack([obb_target.cov_field for obb_target in obb_targets])

    # Move to GPU
    heatmap_target = heatmap_target.to(torch.device('cuda:0'))
    vertex_target = vertex_target.to(torch.device('cuda:0'))
    z_centroid_field_target = z_centroid_field_target.to(torch.device('cuda:0'))
    cov_target = cov_target.to(torch.device('cuda:0'))

    cov_loss = _mask_l1_loss(cov_target, self.cov_field, heatmap_target)
    log['cov_loss'] = cov_loss
    vertex_loss = _mask_l1_loss(vertex_target, self.vertex_field, heatmap_target)
    log['vertex_loss'] = vertex_loss
    z_centroid_loss = _mask_l1_loss(z_centroid_field_target, self.z_centroid_field, heatmap_target)
    log['z_centroid'] = z_centroid_loss

    heatmap_loss = _mse_loss(heatmap_target, self.heatmap)
    log['heatmap'] = heatmap_loss
    return self.hparams.loss_vertex_mult * vertex_loss + self.hparams.loss_heatmap_mult * heatmap_loss + self.hparams.loss_z_centroid_mult * z_centroid_loss + self.hparams.loss_rotation_mult * cov_loss


def extract_cov_matrices_from_peaks(peaks, cov_matrices_output, scale_factor=8):
  assert peaks.shape[1] == 2
  cov_matrices = []
  for ii in range(peaks.shape[0]):
    index = np.zeros([2])
    index[0] = int(peaks[ii, 0] / scale_factor)
    index[1] = int(peaks[ii, 1] / scale_factor)
    index = index.astype(np.int)
    cov_mat_values = cov_matrices_output[index[0], index[1], :]
    cov_matrix = np.array([[cov_mat_values[0], cov_mat_values[3], cov_mat_values[4]],
                           [cov_mat_values[3], cov_mat_values[1], cov_mat_values[5]],
                           [cov_mat_values[4], cov_mat_values[5], cov_mat_values[2]]])
    cov_matrices.append(cov_matrix)
  return cov_matrices


def draw_oriented_bounding_box_from_outputs(
    heatmap_output, vertex_output, rotation_output, z_centroid_output, c_img, camera_model=None
):
  poses, _ = compute_oriented_bounding_boxes(
      np.copy(heatmap_output),
      np.copy(vertex_output),
      np.copy(z_centroid_output),
      np.copy(rotation_output),
      camera_model=camera_model,
      max_detections=100,
  )
  return pose_outputs.draw_9dof_cv2_boxes(c_img, poses, camera_model=camera_model)


def solve_for_rotation_from_cov_matrix(cov_matrix):
  assert cov_matrix.shape[0] == 3
  assert cov_matrix.shape[1] == 3
  U, D, Vh = np.linalg.svd(cov_matrix, full_matrices=True)
  d = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
  if d:
    D[-1] = -D[-1]
    U[:, -1] = -U[:, -1]
  # Rotation from world to points.
  rotation = np.eye(4)
  rotation[0:3, 0:3] = U
  return rotation


def compute_oriented_bounding_boxes(
    heatmap_output,
    vertex_output,
    z_centroid_output,
    cov_matrices,
    camera_model,
    ground_truth_peaks=None,
    max_detections=np.inf,
):
  peaks = pose_outputs.extract_peaks_from_centroid(
      np.copy(heatmap_output), max_peaks=max_detections
  )
  bboxes_ext = pose_outputs.extract_vertices_from_peaks(
      np.copy(peaks), np.copy(vertex_output), np.copy(heatmap_output)
  )
  z_centroids = pose_outputs.extract_z_centroid_from_peaks(
      np.copy(peaks), np.copy(z_centroid_output)
  )
  cov_matrices = pose_outputs.extract_cov_matrices_from_peaks(np.copy(peaks), np.copy(cov_matrices))
  poses = []
  scores = []
  for bbox_ext, z_centroid, cov_matrix, peak in zip(bboxes_ext, z_centroids, cov_matrices, peaks):
    bbox_ext_flipped = bbox_ext[:, ::-1]
    # Solve for pose up to a scale factor
    error, camera_T_object, scale_matrix = optimize_for_9D(
        bbox_ext_flipped.T, camera_model, solve_for_transforms=True
    )
    # Revert back to original pose.
    camera_T_object = camera_T_object @ transform.Transform.from_aa(
        axis=transform.X_AXIS, angle_deg=-45.0
    ).matrix
    # Add rotation solution to pose.
    camera_T_object = camera_T_object @ solve_for_rotation_from_cov_matrix(cov_matrix)
    # Assign correct depth factor
    abs_camera_T_object, abs_object_scale = epnp.find_absolute_scale(
        -1.0 * z_centroid, camera_T_object, scale_matrix
    )
    poses.append(transform.Pose(camera_T_object=abs_camera_T_object, scale_matrix=abs_object_scale))
    scores.append(heatmap_output[peak[0], peak[1]])

  return poses, scores
