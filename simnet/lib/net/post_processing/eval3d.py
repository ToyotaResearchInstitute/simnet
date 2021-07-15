import copy
import dataclasses

import numpy as np
import IPython

from simnet.lib.net.post_processing import epnp, nms
from simnet.lib import occlusions

EVAL_IOUS = [0.25]

_SIZES = ['small', 'large']

################################
# Precision/recall stuff


@dataclasses.dataclass
class Detection:
  camera_T_object: np.ndarray
  scale_matrix: np.ndarray
  class_label: str = None
  size_label: str = ''
  scene_name: str = ''
  ignore: bool = False
  bbox: list = None
  score: float = 1.0
  success: int = 0


def extract_objects_from_detections(detections_gt, detections):
  table_detection = detections_gt[0]
  detections_gt = detections_gt[1:]
  detections = prune_detections_2d_not_one_table(table_detection, detections)
  return table_detection, detections_gt, detections


def is_point_in_detection_box(point, bbox):
  if bbox[0][0] > point[0]:
    return False
  if bbox[0][1] > point[1]:
    return False
  if bbox[1][0] < point[0]:
    return False
  if bbox[1][1] < point[1]:
    return False
  return True


def prune_detections_2d_not_one_table(table_detection, detections):
  pruned_detections = []
  for detection in detections:
    if is_point_in_detection_box(detection.bbox[0], table_detection.bbox):
      pruned_detections.append(detection)
    elif is_point_in_detection_box(detection.bbox[1], table_detection.bbox):
      pruned_detections.append(detection)
    else:
      continue
  return pruned_detections


def get_size_predictions(pred_matches, true_matches, pred_scores, size_labels, size_name):

  pruned_true_matches = []
  pruned_pred_matches = []
  pruned_pred_scores = []

  # Prune ground truth to only contain easy samples.
  for ii in range(true_matches.shape[1]):
    if size_labels[ii] != size_name:
      continue
    pruned_true_matches.append(true_matches[:, ii])
  # Prune any predictions that intersect with an ignore class.
  for ii in range(pred_matches.shape[1]):
    gt_match = pred_matches[0, ii]
    # Check if there was a match at all
    if gt_match != -1:
      # Check if it fired on an unkown class.
      if size_labels[int(gt_match)] != size_name:
        continue
    # Remove all non-object predictions
    if gt_match == -1:
      continue
    pruned_pred_matches.append(pred_matches[:, ii])
    pruned_pred_scores.append(pred_scores[ii])

  if len(pruned_true_matches) == 0:
    pruned_true_matches = np.zeros([1, 0])
  else:
    pruned_true_matches = np.array(pruned_true_matches).T
  if len(pruned_pred_matches) == 0:
    pruned_pred_matches = np.zeros([1, 0])
  else:
    pruned_pred_matches = np.array(pruned_pred_matches).T
  pred_scores = np.array(pruned_pred_scores).T
  return pruned_true_matches, pruned_pred_matches, pruned_pred_scores


def remove_ignore_class(pred_matches, true_matches, pred_scores, ignore_labels):

  pruned_true_matches = []
  pruned_pred_matches = []
  pruned_pred_scores = []
  pruned_detections_indices = []

  # Prune ground truth for ignore classes.
  for ii in range(true_matches.shape[1]):
    if ignore_labels[ii]:
      continue
    pruned_true_matches.append(true_matches[:, ii])
  # Prune any predictions that intersect with an ignore class.
  for ii in range(pred_matches.shape[1]):
    gt_match = pred_matches[0, ii]
    # Check if there was a match at all
    if gt_match != -1:
      # Check if it fired on an unkown class.
      if ignore_labels[int(gt_match)]:
        continue
    pruned_pred_matches.append(pred_matches[:, ii])
    pruned_pred_scores.append(pred_scores[ii])
    pruned_detections_indices.append(ii)

  if len(pruned_true_matches) == 0:
    pruned_true_matches = np.zeros([1, 0])
  else:
    pruned_true_matches = np.array(pruned_true_matches).T
  if len(pruned_pred_matches) == 0:
    pruned_pred_matches = np.zeros([1, 0])
  else:
    pruned_pred_matches = np.array(pruned_pred_matches).T
  pred_scores = np.array(pruned_pred_scores).T
  return pruned_true_matches, pruned_pred_matches, pruned_pred_scores


def assign_known_depth(detections, gt_detections):
  new_detections = []
  for detection in detections:
    best_overlap_score = 0.0
    best_gt_match = None
    for gt_detection in gt_detections:
      overlap_score = nms.get_2d_one_way_iou(gt_detection, detection)
      if overlap_score > best_overlap_score:
        best_gt_match = gt_detection
        best_overlap_score = overlap_score
    if best_gt_match is not None:
      new_z = best_gt_match.camera_T_object[2, 3]
      camera_T_object, scale_matrix = epnp.find_absolute_scale(
          new_z, detection.camera_T_object, detection.scale_matrix
      )
      detection.camera_T_object = camera_T_object
      detection.scale_matrix = scale_matrix
    new_detections.append(detection)

  return new_detections


def assign_rotation_value(detection, gt_detection):
  detection.camera_T_object[0:3, 0:3] = np.eye(3)
  gt_scale_matrix = gt_detection.scale_matrix
  scale_matrix = detection.scale_matrix
  new_scale_matrix = np.eye(4)
  indices = [0, 1, 2]
  for ii in range(3):
    scale_value = scale_matrix[ii, ii]
    best_scale_index = 0
    best_match = np.inf
    for jj in indices:
      if np.abs(scale_value - gt_scale_matrix[jj, jj]) < best_match:
        best_scale_index = jj
        best_match = np.abs(scale_value - gt_scale_matrix[jj, jj])
    new_scale_matrix[best_scale_index, best_scale_index] = scale_value
    indices.remove(best_scale_index)
  detection.scale_matrix = new_scale_matrix


def assign_known_rotation(detections, gt_detections):
  new_detections = []
  #Remove rotation from ground truth.
  for ii in range(len(gt_detections)):
    gt_detections[ii].camera_T_object[0:3, 0:3] = np.eye(3)
  for detection in detections:
    best_overlap_score = 0.0
    best_gt_match = None
    for gt_detection in gt_detections:
      overlap_score = nms.get_2d_one_way_iou(gt_detection, detection)
      if overlap_score > best_overlap_score:
        best_gt_match = gt_detection
        best_overlap_score = overlap_score
    if best_gt_match is not None:
      new_z = best_gt_match.camera_T_object[2, 3]
      assign_rotation_value(detection, best_gt_match)
    new_detections.append(detection)

  return new_detections, gt_detections


def assign_size_labels(gt_detections, size_threshold=0.08):
  for ii in range(len(gt_detections)):
    scale_matrix = gt_detections[ii].scale_matrix
    if np.prod(np.diag(scale_matrix[0:3, 0:3])) < size_threshold**3:
      gt_detections[ii].size_label = 'small'
    else:
      gt_detections[ii].size_label = 'large'


def measure_3d_iou(detections, gt_detections, known_depth=False, known_rotation=False):
  gt_RTs = []
  gt_scales = []
  pred_RTs = []
  pred_scales = []
  pred_scores = []
  size_labels = []
  ignore_labels = []
  if known_depth:
    detections = assign_known_depth(detections, gt_detections)
  if known_rotation:
    detections, gt_detections = assign_known_rotation(detections, gt_detections)
  for detection in detections:
    pred_RTs.append(detection.camera_T_object)
    pred_scales.append(np.diag(detection.scale_matrix[0:3, 0:3]))
    pred_scores.append(detection.score)

  for detection in gt_detections:
    gt_RTs.append(detection.camera_T_object)
    gt_scales.append(np.diag(detection.scale_matrix[0:3, 0:3]))
    size_labels.append(detection.size_label)
    ignore_labels.append(detection.ignore)

  true_matches, pred_matches, _, indices = compute_3d_matches(
      'single_class',
      np.array(gt_RTs),
      np.array(gt_scales),
      np.array(pred_scores),
      np.array(pred_RTs),
      np.array(pred_scales),
      EVAL_IOUS,
      known_depth=known_depth
  )
  # Resort pred scores and class labels
  sorted_pred_scores = []
  sorted_class_labels = []
  sorted_detections = []
  for ii in range(pred_matches.shape[1]):
    detections[indices[ii]].success = int(pred_matches[0, ii] > -1)
    sorted_detections.append(detections[ii])
    sorted_pred_scores.append(pred_scores[ii])
  for detection in gt_detections:
    detection.success = -1
    if detection.ignore:
      detection.success = -2
    #sorted_detections.append(detection)
  # Apply ignore predictions.
  true_matches, pred_matches, sorted_pred_scores = remove_ignore_class(
      copy.deepcopy(pred_matches), copy.deepcopy(true_matches), copy.deepcopy(sorted_pred_scores),
      ignore_labels
  )
  return true_matches, pred_matches, sorted_pred_scores, size_labels, ignore_labels, sorted_detections


class Eval3d:

  def __init__(self):
    self.n = 0
    self.metrics_by_size = {cn: EvalMetrics() for cn in _SIZES}
    self.all_3d_metrics_by_scene = {}
    self.all_3d_known_depth_metrics_by_scene = {}
    self.all_3d_known_rotation_metrics_by_scene = {}

  def process_sample(self, detections, gt_detections, scene_name):
    # Mark occlusions and size labels.
    occlusions.mark_occlusions_in_detections(gt_detections)
    assign_size_labels(gt_detections)

    if scene_name not in self.all_3d_metrics_by_scene:
      self.all_3d_metrics_by_scene[scene_name] = EvalMetrics()
      self.all_3d_known_depth_metrics_by_scene[scene_name] = EvalMetrics()
      self.all_3d_known_rotation_metrics_by_scene[scene_name] = EvalMetrics()

    # Process True Positive Rate per object size
    true_matches, pred_matches, pred_scores, size_labels, _, _ = measure_3d_iou(
        copy.deepcopy(detections), copy.deepcopy(gt_detections), known_depth=True
    )
    for size_name in _SIZES:
      true_matches_per_class, pred_matches_per_class, pred_scores_per_class = get_size_predictions(
          copy.deepcopy(pred_matches), copy.deepcopy(true_matches), copy.deepcopy(pred_scores),
          copy.deepcopy(size_labels), size_name
      )
      self.metrics_by_size[size_name].process_sample(
          true_matches=true_matches_per_class,
          pred_matches=pred_matches_per_class,
          pred_scores=pred_scores_per_class
      )

    # Process 3D mAp for all classes with known depth.
    true_matches, pred_matches, pred_scores, class_labels, ignore_labels, sorted_detections = measure_3d_iou(
        copy.deepcopy(detections), copy.deepcopy(gt_detections), known_depth=True
    )
    self.all_3d_known_depth_metrics_by_scene[scene_name].process_sample(
        true_matches=true_matches, pred_matches=pred_matches, pred_scores=pred_scores
    )

    # Process 3D mAp for all classes.
    true_matches, pred_matches, pred_scores, class_labels, _, _ = measure_3d_iou(
        copy.deepcopy(detections), copy.deepcopy(gt_detections)
    )
    self.all_3d_metrics_by_scene[scene_name].process_sample(
        true_matches=true_matches, pred_matches=pred_matches, pred_scores=pred_scores
    )

    # Process 3D mAp for all classes with known depth and rotatoins.
    true_matches, pred_matches, pred_scores, class_labels, ignore_labels, _ = measure_3d_iou(
        copy.deepcopy(detections),
        copy.deepcopy(gt_detections),
        known_depth=True,
        known_rotation=True
    )
    self.all_3d_known_rotation_metrics_by_scene[scene_name].process_sample(
        true_matches=true_matches, pred_matches=pred_matches, pred_scores=pred_scores
    )

    return sorted_detections

  def process_dataset(self, results_path):

    def _print(name, x, iou):
      if final:
        prefix = '[step {self.n: 6d}] (FINAL)'
      else:
        prefix = '(incomplete)'
      if not final and name != 'MEAN':
        return
      out = f'{prefix} AP@{iou:.02f}[{name}] = {x}'
      print(out)
      if results_fh is not None:
        results_fh.write(out + '\n')

    for idx, iou in enumerate(EVAL_IOUS):
      # Plot 3d metrics
      ap_values = []
      for size_name in _SIZES:
        ap_values.append(self.metrics_by_size[size_name].process_dataset()[idx])
      # Draw the bar plot.
      graph_name = f'3D_ATP@{iou:.02f}'
      plotter.draw_standard_bar_plot(_SIZES, ap_values, results_path, y_label=graph_name)

      # Plot mAP per scene
      ap_values = []
      scene_names = []
      for scene_name in self.all_3d_known_depth_metrics_by_scene:
        ap_values.append(
            self.all_3d_known_depth_metrics_by_scene[scene_name].process_dataset()[idx]
        )
        scene_names.append(scene_name)
      # Draw the bar plot.
      graph_name = f'3D_mAP@{iou:.02f}'
      plotter.draw_standard_bar_plot(scene_names, ap_values, results_path, y_label=graph_name)

      # Plot AP curve for known depth per scene
      for scene_name in self.all_3d_known_depth_metrics_by_scene:
        precisions, recalls = self.all_3d_known_depth_metrics_by_scene[scene_name].get_pr_curve()
        plotter.draw_precision_recall_curve(
            precisions, recalls, results_path, name=scene_name + '_pr_curve'
        )

  def process_all_3D_dataset(self):
    ap_values = []
    for scene_name in self.all_3d_metrics_by_scene:
      ap_values.append(self.all_3d_metrics_by_scene[scene_name].process_dataset()[0])
    return np.average(ap_values)


class EvalMetrics:

  def __init__(self):
    self.true_matches = []
    self.pred_matches = []
    self.pred_scores = []

  def process_sample(self, true_matches, pred_matches, pred_scores):
    self.true_matches.append(true_matches)
    self.pred_matches.append(pred_matches)
    self.pred_scores.append(pred_scores)

  def process_dataset(self):
    true_matches = np.copy(np.concatenate(self.true_matches, axis=1))
    pred_matches = np.copy(np.concatenate(self.pred_matches, axis=1))
    pred_scores = np.copy(np.concatenate(self.pred_scores, axis=0))
    assert true_matches.shape[0] == pred_matches.shape[0]
    num_ious = true_matches.shape[0]
    ap_per_iou = []
    for i in range(num_ious):
      ap_per_iou.append(
          compute_ap_from_matches_scores(pred_matches[i, :], pred_scores, true_matches[i, :])
      )
    return ap_per_iou

  def get_pr_curve(self):
    true_matches = np.concatenate(self.true_matches, axis=1)
    pred_matches = np.concatenate(self.pred_matches, axis=1)
    pred_scores = np.concatenate(self.pred_scores, axis=0)
    assert true_matches.shape[0] == pred_matches.shape[0]
    num_ious = true_matches.shape[0]
    ap_per_iou = []
    return compute_pr_curve_from_matches_scores(pred_matches[0, :], pred_scores, true_matches[0, :])


def compute_3d_iou_new(RT_1, RT_2, scales_1, scales_2, debug=False, known_depth=True):
  '''Computes IoU overlaps between two 3d bboxes.
     bbox_3d_1, bbox_3d_1: [3, 8]
  '''

  # flatten masks
  def asymmetric_3d_iou(RT_1, RT_2, scales_1, scales_2):
    assert RT_1.shape == (4, 4)
    assert RT_2.shape == (4, 4)
    assert scales_1.shape == (3,)
    assert scales_2.shape == (3,)
    noc_cube_1 = get_3d_bbox(scales_1, 0)
    bbox_3d_1 = transform_coordinates_3d(noc_cube_1, RT_1)
    assert bbox_3d_1.shape == (3, 8)

    noc_cube_2 = get_3d_bbox(scales_2, 0)
    bbox_3d_2 = transform_coordinates_3d(noc_cube_2, RT_2)
    assert bbox_3d_2.shape == (3, 8)

    bbox_1_max = np.amax(bbox_3d_1, axis=1)
    bbox_1_min = np.amin(bbox_3d_1, axis=1)
    bbox_2_max = np.amax(bbox_3d_2, axis=1)
    bbox_2_min = np.amin(bbox_3d_2, axis=1)

    assert bbox_1_min.shape == (3,)
    assert bbox_1_max.shape == (3,)
    assert bbox_2_min.shape == (3,)
    assert bbox_2_max.shape == (3,)

    overlap_min = np.maximum(bbox_1_min, bbox_2_min)
    overlap_max = np.minimum(bbox_1_max, bbox_2_max)

    # intersections and union
    if np.amin(overlap_max - overlap_min) < 0:
      intersections = 0
    else:
      intersections = np.prod(overlap_max - overlap_min)
    union = np.prod(bbox_1_max - bbox_1_min) + np.prod(bbox_2_max - bbox_2_min) - intersections
    overlaps = intersections / union

    if False:
      print('bbox_3d_1:', bbox_3d_1)
      print('bbox_3d_2:', bbox_3d_2)
      print('bbox_1_min:', bbox_1_min)
      print('bbox_1_max:', bbox_1_max)
      print('bbox_2_min:', bbox_2_min)
      print('bbox_2_max:', bbox_2_max)
      print('overlap_min', overlap_min)
      print('overlap_max', overlap_max)
      print('intersections:', intersections)
      print('union:', union)
      print('overlaps:', overlaps)

    assert not np.isnan(overlaps)
    return overlaps

  if RT_1 is None or RT_2 is None:
    return -1

  max_iou = asymmetric_3d_iou(RT_1, RT_2, scales_1, scales_2)

  return max_iou


def get_3d_bbox(scale, shift=0):
  """
  Input: 
    scale: [3] or scalar
    shift: [3] or scalar
  Return 
    bbox_3d: [3, N]

  """
  bbox_3d = np.array(
      [[scale[0] / 2, +scale[1] / 2, scale[2] / 2], [scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
       [-scale[0] / 2, +scale[1] / 2, scale[2] / 2], [-scale[0] / 2, +scale[1] / 2, -scale[2] / 2],
       [+scale[0] / 2, -scale[1] / 2, scale[2] / 2], [+scale[0] / 2, -scale[1] / 2, -scale[2] / 2],
       [-scale[0] / 2, -scale[1] / 2, scale[2] / 2], [-scale[0] / 2, -scale[1] / 2, -scale[2] / 2]]
  ) + shift

  bbox_3d = bbox_3d.transpose()
  return bbox_3d


def transform_coordinates_3d(coordinates, RT):
  """
  Input: 
    coordinates: [3, N]
    RT: [4, 4]
  Return 
    new_coordinates: [3, N]

  """
  assert RT.shape == (4, 4)
  assert len(coordinates.shape) == 2
  assert coordinates.shape[0] == 3
  coordinates = np.vstack([coordinates, np.ones((1, coordinates.shape[1]), dtype=np.float32)])
  new_coordinates = RT @ coordinates
  new_coordinates = new_coordinates[:3, :] / new_coordinates[3, :]
  assert new_coordinates.shape[0] == 3
  return new_coordinates


def resolve_z_ambiguity(RT, scale, gt_RT):
  new_z = gt_RT[2, 3]
  object_scale = np.eye(4)
  object_scale[0:3, 0:3] = np.diag(scale)
  scaled_RT, scaled_object_scale = epnp.find_absolute_scale(new_z, RT, object_scale)
  return scaled_RT, np.diag(scaled_object_scale[0:3, 0:3])


def compute_3d_matches(
    class_name,
    gt_RTs,
    gt_scales,
    pred_scores,
    pred_RTs,
    pred_scales,
    iou_3d_thresholds,
    score_threshold=0,
    known_depth=False,
    debug=False
):
  """Finds matches between prediction and ground truth instances.
  Returns:
    gt_matches: 2-D array. For each GT box it has the index of the matched
          predicted box.
    pred_matches: 2-D array. For each predicted box, it has the index of
          the matched ground truth box.
    overlaps: [pred_boxes, gt_boxes] IoU overlaps.
  """
  #assert gt_scales.shape[1] == 3
  #assert pred_scales.shape[1] == 3

  num_pred = pred_scales.shape[0]
  num_gt = gt_scales.shape[0]
  indices = np.zeros(0)

  if num_pred:
    # Sort predictions by score from high to low
    indices = np.argsort(pred_scores)[::-1]

    pred_scores = pred_scores[indices].copy()
    pred_scales = pred_scales[indices, :].copy()
    pred_RTs = pred_RTs[indices, :, :].copy()

  # Compute IoU overlaps [pred_bboxs gt_bboxs]
  overlaps = np.zeros((num_pred, num_gt), dtype=np.float32)
  for i in range(num_pred):
    for j in range(num_gt):
      overlaps[i, j] = compute_3d_iou_new(
          pred_RTs[i, :, :], gt_RTs[j, :, :], pred_scales[i, :], gt_scales[j, :], class_name
      )
  # Loop through predictions and find matching ground truth boxes
  num_iou_3d_thres = len(iou_3d_thresholds)
  pred_matches = -1 * np.ones([num_iou_3d_thres, num_pred])
  gt_matches = -1 * np.ones([num_iou_3d_thres, num_gt])

  for s, iou_thres in enumerate(iou_3d_thresholds):
    for i in range(num_pred):
      # Find best matching ground truth box
      # 1. Sort matches by score
      sorted_ixs = np.argsort(overlaps[i])[::-1]
      # 2. Remove low scores
      low_score_idx = np.where(overlaps[i, sorted_ixs] < score_threshold)[0]
      if low_score_idx.size > 0:
        sorted_ixs = sorted_ixs[:low_score_idx[0]]
      # 3. Find the match
      for j in sorted_ixs:
        # If ground truth box is already matched, go to next one
        #print('gt_match: ', gt_match[j])
        if gt_matches[s, j] > -1:
          continue
        # If we reach IoU smaller than the threshold, end the loop
        iou = overlaps[i, j]
        #print('iou: ', iou)
        if iou < iou_thres:
          break

        if iou > iou_thres:
          gt_matches[s, j] = i
          pred_matches[s, i] = j
          break

  if debug:
    IPython.embed()

  return gt_matches, pred_matches, overlaps, indices


def compute_ap_from_matches_scores(pred_match, pred_scores, gt_match, debug=False):
  # sort the scores from high to low
  # print(pred_match.shape, pred_scores.shape)
  assert len(pred_match.shape) == 1
  assert len(gt_match.shape) == 1
  assert pred_match.shape[0] == pred_scores.shape[0]

  if gt_match.shape[0] == 0:
    return 0.

  score_indices = np.argsort(pred_scores)[::-1]
  pred_scores = pred_scores[score_indices]
  pred_match = pred_match[score_indices]

  precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
  recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

  # Pad with start and end values to simplify the math
  precisions = np.concatenate([[0], precisions, [0]])
  recalls = np.concatenate([[0], recalls, [1]])

  # Ensure precision values decrease but don't increase. This way, the
  # precision value at each recall threshold is the maximum it can be
  # for all following recall thresholds, as specified by the VOC paper.
  #precisions2 = precisions
  for i in range(len(precisions) - 2, -1, -1):
    precisions[i] = np.maximum(precisions[i], precisions[i + 1])

  # Compute mean AP over recall range
  indices = np.where(recalls[:-1] != recalls[1:])[0] + 1
  ap = np.sum((recalls[indices] - recalls[indices - 1]) * precisions[indices])
  #assert 0 <= ap <= 1.

  if debug:
    IPython.embed()

  return ap


def compute_pr_curve_from_matches_scores(pred_match, pred_scores, gt_match, debug=False):
  # sort the scores from high to low
  # print(pred_match.shape, pred_scores.shape)
  assert len(pred_match.shape) == 1
  assert len(gt_match.shape) == 1
  assert pred_match.shape[0] == pred_scores.shape[0]

  if gt_match.shape[0] == 0:
    return 0.

  score_indices = np.argsort(pred_scores)[::-1]
  pred_scores = pred_scores[score_indices]
  pred_match = pred_match[score_indices]
  precisions = np.cumsum(pred_match > -1) / (np.arange(len(pred_match)) + 1)
  recalls = np.cumsum(pred_match > -1).astype(np.float32) / len(gt_match)

  return precisions, recalls
