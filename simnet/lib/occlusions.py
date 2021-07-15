import numpy as np

from simnet.lib import camera
from simnet.lib.net.post_processing import epnp


### 3D Occlusions
def object_is_outside_image(detection, camera_model):
  bbox = epnp.get_2d_bbox_of_9D_box(detection.camera_T_object, detection.scale_matrix, camera_model)
  width = camera_model.width
  height = camera_model.height

  if bbox[0][0] < 0 or bbox[0][0] < 0:
    return True
  if bbox[1][0] > width or bbox[1][1] > height:
    return True
  return False


def get_bbox_image(detection, camera_model):
  bbox = epnp.get_2d_bbox_of_9D_box(detection.camera_T_object, detection.scale_matrix, camera_model)
  width = camera_model.width
  height = camera_model.height
  img = np.zeros([height, width])
  img[int(bbox[0][1]):int(bbox[1][1]), int(bbox[0][0]):int(bbox[1][0])] = 1.0
  return img


def mark_occlusions_in_detections(
    detections, occlusion_score=0.5, camera_model=None, allow_outside_of_image=False
):
  if camera_model is None:
    camera_model = camera.FMKCamera()
  for ii in range(len(detections)):
    if object_is_outside_image(detections[ii], camera_model):
      detections[ii].ignore = True
      continue
    bbox_unocc = get_bbox_image(detections[ii], camera_model)
    bbox_occ = np.copy(bbox_unocc)
    bbox_prop = np.copy(bbox_occ)
    for detection_proposal in detections:
      # Check if the object is behind the target object.
      if detection_proposal.camera_T_object[2, 3] >= detections[ii].camera_T_object[2, 3]:
        continue
      bbox_proposal = get_bbox_image(detection_proposal, camera_model)
      bbox_occ = bbox_occ - bbox_proposal
      bbox_prop = bbox_prop + bbox_proposal
    occlusion_level = np.sum(bbox_occ > 0) / np.sum(bbox_unocc > 0)
    if occlusion_level < occlusion_score:
      detections[ii].ignore = True
