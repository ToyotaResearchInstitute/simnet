import numpy as np
import colour


def get_colors(num_colors):
  assert num_colors > 0

  colors = list(colour.Color("purple").range_to(colour.Color("green"), num_colors))
  color_rgb = 255 * np.array([np.array(a.get_rgb()) for a in colors])
  color_rgb = [a.astype(np.int) for a in color_rgb]
  return color_rgb


def get_panoptic_colors():
  colors = [
      colour.Color("yellow"),
      colour.Color("blue"),
      colour.Color("green"),
      colour.Color("red"),
      colour.Color("purple")
  ]
  color_rgb = 255 * np.array([np.array(a.get_rgb()) for a in colors])
  color_rgb = [a.astype(np.int) for a in color_rgb]
  return color_rgb


def get_unique_colors(num_colors):
  '''
  Gives a the specified number of unique colors
  Args:
     num_colors: an int specifying the number of colors
  Returs:
     A list of  rgb colors in the range of (0,255)
  '''
  color_rgb = get_colors(num_colors)

  if (num_colors != len(np.unique(color_rgb, axis=0))):
    raise ValueError('Colors returned are not unique.')

  return color_rgb
