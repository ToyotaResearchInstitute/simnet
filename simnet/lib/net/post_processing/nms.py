import numpy as np


def run(detections, overlap_thresh=0.75, order_mode='confidence'):
  # initialize the list of picked detections
  pruned_detections = []

  # sort the indexes
  if order_mode == 'lower_y':
    idxs = create_order_by_lower_y(detections)
  elif order_mode == 'confidence':
    idxs = create_order_by_score(detections)

  overlap_function = get_2d_one_way_iou

  # keep looping while some indexes still remain in the indexes list
  while len(idxs) > 0:
    # grab the last index in the indexes list and add the index value
    # to the list of picked indexes
    last = len(idxs) - 1
    ii = idxs[last]
    indices_to_suppress = []
    for index, index_of_index in zip(idxs[:last], range(last)):
      detection_proposal = detections[index]
      overlap = overlap_function(detections[ii], detection_proposal)
      if overlap > overlap_thresh:
        indices_to_suppress.append(index_of_index)
    # Add the the pruned_detections.
    pruned_detections.append(detections[ii])
    indices_to_suppress.append(last)
    idxs = np.delete(idxs, indices_to_suppress)

  # return only the bounding boxes that were picked
  return prune_by_min_height(pruned_detections)


def prune_by_min_height(detections):
  pruned_detections = []
  for detection in detections:
    if detection.bbox[1][0] - detection.bbox[0][0] < 12:
      continue
    pruned_detections.append(detection)
  return pruned_detections


def create_order_by_lower_y(detections):
  idxs = []
  for detection in detections:
    idxs.append(detection.bbox[1][1])
  idxs = np.argsort(idxs)
  return idxs


def create_order_by_score(detections):
  idxs = []
  for detection in detections:
    idxs.append(detection.score)
  idxs = np.argsort(idxs)
  return idxs


def get_2d_one_way_iou(detection_one, detection_two):
  box_one = np.array([
      detection_one.bbox[0][0], detection_one.bbox[0][1], detection_one.bbox[1][0],
      detection_one.bbox[1][1]
  ])
  box_two = np.array([
      detection_two.bbox[0][0], detection_two.bbox[0][1], detection_two.bbox[1][0],
      detection_two.bbox[1][1]
  ])
  # determine the (x, y)-coordinates of the intersection rectangle
  xA = max(box_one[0], box_two[0])
  yA = max(box_one[1], box_two[1])
  xB = min(box_one[2], box_two[2])
  yB = min(box_one[3], box_two[3])
  # compute the area of intersection rectangle
  inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
  # compute the area of both the prediction and ground-truth
  # rectangles
  box_one_area = (box_one[2] - box_one[0] + 1) * (box_one[3] - box_one[1] + 1)
  box_two_area = (box_two[2] - box_two[0] + 1) * (box_two[3] - box_two[1] + 1)
  # compute the intersection over union by taking the intersection
  # area and dividing it by the sum of prediction + ground-truth
  # areas - the interesection area
  if float(box_one_area) == 0.0:
    return 0
  return inter_area / float(box_one_area + box_two_area - inter_area)
