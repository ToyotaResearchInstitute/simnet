import numpy as np

from simnet.lib import transform

CLIPPING_PLANE_NEAR = 0.4

SCALE_FACTOR = 4


class FMKCamera:

  def __init__(
      self,
      hfov_deg=100.,
      vfov_deg=80.,
      height=2048,
      width=2560,
      stereo_baseline=0.10,
      enable_noise=False,
      override_intrinsics=None
  ):
    """ The default camera model to match the Basler's rectified implementation at TOT """

    # This is to go from mmt to pyrender frame
    self.RT_matrix = transform.Transform.from_aa(axis=transform.X_AXIS, angle_deg=180.0).matrix
    if override_intrinsics is not None:
      self._set_intrinsics(override_intrinsics)
      return
    height = height // SCALE_FACTOR
    width = width // SCALE_FACTOR
    assert height % 64 == 0
    assert width % 64 == 0

    self.height = height
    self.width = width

    self.stereo_baseline = stereo_baseline
    self.is_left = True

    hfov = np.deg2rad(hfov_deg)
    vfov = np.deg2rad(vfov_deg)
    focal_length_x = 0.5 * width / np.tan(0.5 * hfov)
    focal_length_y = 0.5 * height / np.tan(0.5 * vfov)

    focal_length = focal_length_x
    focal_length_ar = focal_length_y / focal_length_x

    self._set_intrinsics(
        np.array([
            [focal_length, 0., width / 2., 0.],
            [0., focal_length * focal_length_ar, height / 2., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., 1.],
        ])
    )

  def add_camera_noise(self, img):
    return camera_noise.add(img)

  def make_datapoint(self):
    k_matrix = self.K_matrix[:3, :3]
    if params.ENABLE_STEREO:
      assert self.stereo_baseline is not None
      return datapoint.StereoCameraDataPoint(
          k_matrix=k_matrix,
          baseline=self.stereo_baseline,
      )
    return datapoint.CameraDataPoint(k_matrix=k_matrix,)

  def _set_intrinsics(self, intrinsics_matrix):
    assert intrinsics_matrix.shape[0] == 4
    assert intrinsics_matrix.shape[1] == 4

    self.K_matrix = intrinsics_matrix
    self.proj_matrix = self.K_matrix @ self.RT_matrix

  def project(self, points):
    """Project 4d homogenous points (4xN) to 4d homogenous pixels (4xN)"""
    assert len(points.shape) == 2
    assert points.shape[0] == 4
    return self.proj_matrix @ points

  def deproject(self, pixels):
    """Deproject 4d homogenous pixels (4xN) to 4d homogenous points (4xN)"""
    assert len(pixels.shape) == 2
    assert pixels.shape[0] == 4
    return np.linalg.inv(self.proj_matrix) @ pixels

  def splat_points(self, hpoints_camera):
    """Project 4d homogenous points (4xN) to 4d homogenous points (4xN)"""
    assert len(hpoints_camera.shape) == 2
    assert hpoints_camera.shape[0] == 4
    hpixels = self.project(hpoints_camera)
    pixels = convert_homopixels_to_pixels(hpixels)
    depths_camera = convert_homopoints_to_points(hpoints_camera)[2, :]
    image = np.zeros((self.height, self.width))
    pixel_cols = np.clip(np.round(pixels[0, :]).astype(np.int32), 0, self.width - 1)
    pixel_rows = np.clip(np.round(pixels[1, :]).astype(np.int32), 0, self.height - 1)
    image[pixel_rows, pixel_cols] = depths_camera < CLIPPING_PLANE_NEAR
    return image

  def deproject_depth_image(self, depth_image):
    assert depth_image.shape == (self.height, self.width)
    v, u = np.indices(depth_image.shape).astype(np.float32)
    z = depth_image.reshape((1, -1))
    pixels = np.stack([u.flatten(), v.flatten()], axis=0)
    hpixels = convert_pixels_to_homopixels(pixels, z)
    hpoints = self.deproject(hpixels)
    return hpoints


def convert_homopixels_to_pixels(pixels):
  """Project 4d homogenous pixels (4xN) to 2d pixels (2xN)"""
  assert len(pixels.shape) == 2
  assert pixels.shape[0] == 4
  pixels_3d = pixels[:3, :] / pixels[3:4, :]
  pixels_2d = pixels_3d[:2, :] / pixels_3d[2:3, :]
  assert pixels_2d.shape[1] == pixels.shape[1]
  assert pixels_2d.shape[0] == 2
  return pixels_2d


def convert_pixels_to_homopixels(pixels, depths):
  """Project 2d pixels (2xN) and depths (meters, 1xN) to 4d pixels (4xN)"""
  assert len(pixels.shape) == 2
  assert pixels.shape[0] == 2
  assert len(depths.shape) == 2
  assert depths.shape[1] == pixels.shape[1]
  assert depths.shape[0] == 1
  pixels_4d = np.concatenate([
      depths * pixels,
      depths,
      np.ones_like(depths),
  ], axis=0)
  assert pixels_4d.shape[0] == 4
  assert pixels_4d.shape[1] == pixels.shape[1]
  return pixels_4d


def convert_points_to_homopoints(points):
  """Project 3d points (3xN) to 4d homogenous points (4xN)"""
  assert len(points.shape) == 2
  assert points.shape[0] == 3
  points_4d = np.concatenate([
      points,
      np.ones((1, points.shape[1])),
  ], axis=0)
  assert points_4d.shape[1] == points.shape[1]
  assert points_4d.shape[0] == 4
  return points_4d


def convert_homopoints_to_points(points_4d):
  """Project 4d homogenous points (4xN) to 3d points (3xN)"""
  assert len(points_4d.shape) == 2
  assert points_4d.shape[0] == 4
  points_3d = points_4d[:3, :] / points_4d[3:4, :]
  assert points_3d.shape[1] == points_3d.shape[1]
  assert points_3d.shape[0] == 3
  return points_3d
