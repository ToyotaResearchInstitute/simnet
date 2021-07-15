import numpy as np

from simnet.lib import camera
# Definition of unit cube centered at the orign
x_width = 1.0
y_depth = 1.0
z_height = 1.0

# Assigned based on ordering of vertices in mesh from unit cube primitive.
# Note this is the internal trimesh ordering not what is specified in primitives
_WORLD_T_POINTS = np.array([
    [0, 0, 0],  #0
    [0, 0, z_height],  #1
    [0, y_depth, 0],  #2
    [0, y_depth, z_height],  #3
    [x_width, 0, 0],  #4
    [x_width, 0, z_height],  #5
    [x_width, y_depth, 0],  #6
    [x_width, y_depth, z_height],  #7
]) - 0.5


def get_2d_bbox_of_9D_box(camera_T_object, scale_matrix, camera_model):
  unit_box_homopoints = camera.convert_points_to_homopoints(_WORLD_T_POINTS.T)
  morphed_homopoints = camera_T_object @ (scale_matrix @ unit_box_homopoints)
  morphed_pixels = camera.convert_homopixels_to_pixels(camera_model.K_matrix @ morphed_homopoints).T
  bbox = [
      np.array([np.min(morphed_pixels[:, 0]),
                np.min(morphed_pixels[:, 1])]),
      np.array([np.max(morphed_pixels[:, 0]),
                np.max(morphed_pixels[:, 1])])
  ]
  return bbox


def project_pose_onto_image(pose, camera_model):
  unit_box_homopoints = camera.convert_points_to_homopoints(_WORLD_T_POINTS.T)
  morphed_homopoints = pose.camera_T_object @ (pose.scale_matrix @ unit_box_homopoints)
  morphed_pixels = camera.convert_homopixels_to_pixels(camera_model.project(morphed_homopoints)).T
  morphed_pixels = morphed_pixels[:, ::-1]
  return morphed_pixels


def get_2d_bbox_of_projection(bbox_ext):
  bbox = [
      np.array([np.min(bbox_ext[:, 0]), np.min(bbox_ext[:, 1])]),
      np.array([np.max(bbox_ext[:, 0]), np.max(bbox_ext[:, 1])])
  ]
  return bbox


def define_control_points():
  return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])


def compute_alphas(Xw, Cw):
  X = np.concatenate((Xw, np.array([np.ones((8))])), axis=0)
  C = Cw.T
  C = np.concatenate((C, np.array([np.ones((4))])), axis=0)
  Alpha = np.matmul(np.linalg.inv(C), X)
  return Alpha.T


def construct_M_matrix(bbox_pixels, alphas, K_matrix):
  M = np.zeros([16, 12])
  f_x = K_matrix[0, 0]
  f_y = K_matrix[1, 1]
  c_x = K_matrix[0, 2]
  c_y = K_matrix[1, 2]
  for ii in range(8):
    u = bbox_pixels[0, ii]
    v = bbox_pixels[1, ii]
    for jj in range(4):
      alpha = alphas[ii, jj]
      M[ii * 2, jj * 3] = f_x * alpha
      M[ii * 2, jj * 3 + 2] = (c_x - u) * alpha
      M[ii * 2 + 1, jj * 3 + 1] = f_y * alpha
      M[ii * 2 + 1, jj * 3 + 2] = (c_y - v) * alpha
  return M


def convert_control_to_box_vertices(control_points, alphas):
  bbox_vertices = np.zeros([8, 3])
  for i in range(8):
    for j in range(4):
      alpha = alphas[i, j]
      bbox_vertices[i] = bbox_vertices[i] + alpha * control_points[j]

  return bbox_vertices


def solve_for_control_points(M):
  e_vals, e_vecs = np.linalg.eig(M.T @ M)
  control_points = e_vecs[:, np.argmin(e_vals)]
  control_points = control_points.reshape([4, 3])
  return control_points


def compute_homopoints_from_control_points(camera_control_points, alphas, K_matrix):
  camera_points = convert_control_to_box_vertices(camera_control_points, alphas)
  camera_homopoints = camera.convert_points_to_homopoints(camera_points.T)
  return camera_homopoints
  unit_box_homopoints = camera.convert_points_to_homopoints(_WORLD_T_POINTS.T)


def optimize_for_9D(bbox_pixels, camera_model, solve_for_transforms=False):
  K_matrix = camera_model.K_matrix
  Cw = define_control_points()
  Xw = _WORLD_T_POINTS
  alphas = compute_alphas(Xw.T, Cw)
  M = construct_M_matrix(bbox_pixels, alphas, np.copy(K_matrix))
  camera_control_points = solve_for_control_points(M)
  camera_points = convert_control_to_box_vertices(camera_control_points, alphas)
  camera_homopoints = camera.convert_points_to_homopoints(camera_points.T)
  if solve_for_transforms:
    unit_box_homopoints = camera.convert_points_to_homopoints(_WORLD_T_POINTS.T)
    # Test both the negative and positive solutions of the control points and pick the best one. Taken from the Google MediaPipe Code base.
    error_one, camera_T_object_one, scale_matrix_one = estimateSimilarityUmeyama(
        unit_box_homopoints, camera_homopoints
    )
    camera_homopoints = compute_homopoints_from_control_points(
        -1 * camera_control_points, alphas, K_matrix
    )
    error_two, camera_T_object_two, scale_matrix_two = estimateSimilarityUmeyama(
        unit_box_homopoints, camera_homopoints
    )
    if error_one < error_two:
      camera_T_object = camera_T_object_one
      scale_matrix = scale_matrix_one
    else:
      camera_T_object = camera_T_object_two
      scale_matrix = scale_matrix_two

    # Compute Fit to original pixles:
    morphed_points = camera_T_object @ (scale_matrix @ unit_box_homopoints)
    morphed_pixels = points_to_camera(morphed_points, K_matrix)
    confidence = np.linalg.norm(bbox_pixels - morphed_pixels)
    return confidence, camera_T_object, scale_matrix
  camera_homopixels = K_matrix @ camera_homopoints
  return camera.convert_homopixels_to_pixels(camera_homopixels).T


def estimateSimilarityUmeyama(source_hom, TargetHom):
  # Copy of original paper is at: http://web.stanford.edu/class/cs273/refs/umeyama.pdf
  assert source_hom.shape[0] == 4
  assert TargetHom.shape[0] == 4
  SourceCentroid = np.mean(source_hom[:3, :], axis=1)
  TargetCentroid = np.mean(TargetHom[:3, :], axis=1)
  nPoints = source_hom.shape[1]

  CenteredSource = source_hom[:3, :] - np.tile(SourceCentroid, (nPoints, 1)).transpose()
  CenteredTarget = TargetHom[:3, :] - np.tile(TargetCentroid, (nPoints, 1)).transpose()

  CovMatrix = np.matmul(CenteredTarget, np.transpose(CenteredSource)) / nPoints

  if np.isnan(CovMatrix).any():
    print('nPoints:', nPoints)
    print('source_hom', source_hom.shape)
    print('TargetHom', TargetHom.shape)
    raise RuntimeError('There are NANs in the input.')

  U, D, Vh = np.linalg.svd(CovMatrix, full_matrices=True)
  d = (np.linalg.det(U) * np.linalg.det(Vh)) < 0.0
  if d:
    D[-1] = -D[-1]
    U[:, -1] = -U[:, -1]

  Rotation = np.matmul(U, Vh)
  var_source = np.std(CenteredSource[:3, :], axis=1)
  var_target_aligned = np.std(np.linalg.inv(Rotation) @ CenteredTarget[:3, :], axis=1)
  ScaleMatrix = np.diag(var_target_aligned / var_source)

  Translation = TargetHom[:3, :].mean(axis=1) - source_hom[:3, :].mean(axis=1).dot(
      ScaleMatrix @ Rotation.T
  )

  source_T_target = np.identity(4)
  source_T_target[:3, :3] = Rotation
  source_T_target[:3, 3] = Translation
  scale_matrix = np.eye(4)
  scale_matrix[0:3, 0:3] = ScaleMatrix
  # Measure error fit
  morphed_points = source_T_target @ (scale_matrix @ source_hom)
  error = np.linalg.norm(morphed_points - TargetHom)
  return error, source_T_target, scale_matrix


def points_to_camera(world_T_homopoints, K_matrix):
  homopixels = K_matrix @ world_T_homopoints
  return camera.convert_homopixels_to_pixels(homopixels)


def find_absolute_scale(new_z, camera_T_object, object_scale, debug=True):
  old_z = camera_T_object[2, 3]
  abs_camera_T_object = np.copy(camera_T_object)
  abs_camera_T_object[0:3, 3] = (new_z / old_z) * abs_camera_T_object[0:3, 3]
  abs_object_scale = np.eye(4)
  abs_object_scale[0:3, 0:3] = (new_z / old_z) * object_scale[0:3, 0:3]

  return abs_camera_T_object, abs_object_scale


def test_pose_solver():
  world_t_points = np.copy(_WORLD_T_POINTS)
  world_t_points = camera.convert_points_to_homopoints(world_t_points.T)
  R_t = np.eye(4)
  R_t[0:3, 0:3] = euler_to_rotation_matrix([1.7, 3.8, 5.2])
  R_t[2, 3] = -0.01
  R_t[1, 3] = -0.00001
  S = np.eye(4)
  S[0, 0] = 0.5
  S[1, 1] = 0.05
  S[2, 2] = 0.5
  target_t_points = R_t @ (S @ world_t_points)
  target_t_points = np.array([[
      0.61674494, 0.61674494, 0.61674494, 0.61674494, 1.93547767, 1.93547767, 1.93547767, 1.93547767
  ], [
      0.40753347, 0.40753347, 0.42753347, 0.42753347, 0.40753347, 0.40753347, 0.42753347, 0.42753347
  ], [
      2.56231278, 3.84837313, 2.56231278, 3.84837313, 2.56231278, 3.84837313, 2.56231278, 3.84837313
  ], [1., 1., 1., 1., 1., 1., 1., 1.]])
  world_T_camera = np.array([[0.99376416, -0.05495078, 0.0970217, 1.43237753],
                             [0.00736171, 0.90056662, 0.43465568, 0.72350959],
                             [-0.11125918, -0.43123099, 0.89535536, 3.94788388], [0., 0., 0., 1.]])
  target_t_points = np.linalg.inv(world_T_camera) @ target_t_points
  pixels_target = camera.convert_homopixels_to_pixels(camera.FMKCamera().project(target_t_points))
  pixels_target_gt = np.array([[198.60924037, 181.08880028], [-384.55689665, 452.51438988],
                               [197.90794419, 176.84519486], [-405.05431808, 439.75925094],
                               [451.6781963, 190.51153042], [1088.99607868, 718.53052576],
                               [452.57505678, 185.91876331], [1128.80169693, 709.80385466]])
  #target_t_points = camera.FMKCamera().RT_matrix @ target_t_points
  #pixels_target = points_to_camera(target_t_points)
  #print("Custom projection ",np.round(pixels_target,3))
  _, camera_T_object, scale_matrix = optimize_for_9D(pixels_target, solve_for_transforms=True)

  camera_T_object[0:3, 0:3] = np.eye(3)
  #camera_T_object[0:2,3] = 0.0
  #camera_T_object[0,3] = 0.1
  print(camera_T_object)
  scale_matrix = np.eye(4)
  abs_camera_T_object = np.copy(camera_T_object)
  abs_camera_T_object[2, 3] = 1.0
  find_absolute_scale(1.0, camera_T_object, scale_matrix)
  assert False
  #camera_T_object,scale_matrix = estimateSimilarityUmeyama(world_t_points,target_t_points)
  print("Found matrices")
  print(np.round(camera_T_object, 3))
  print(np.round(scale_matrix, 3))
  print("Gt matrices")
  print(np.round(S, 3))
  print(np.round(R_t, 3))
  print("Pixel Projections")
  pixels_found = optimize_for_9D(pixels_target)
  print(np.round(pixels_target.T, 3))
  print(np.round(pixels_found, 3))
  print("Checking transform projections")
  target_t_points = camera_T_object @ (scale_matrix @ world_t_points)
  camera_homopixels = camera.FMKCamera().K_matrix @ target_t_points
  print(np.round(camera.convert_homopixels_to_pixels(camera_homopixels).T, 3))


if __name__ == "__main__":
  test_pose_solver()
