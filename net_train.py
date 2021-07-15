import os

os.environ['PYTHONHASHSEED'] = str(1)

import argparse
from importlib.machinery import SourceFileLoader
import sys

import random

random.seed(12345)
import numpy as np

np.random.seed(12345)
import torch

torch.manual_seed(12345)

import wandb

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers

from simnet.lib.net import common
from simnet.lib import datapoint, camera
from simnet.lib.net.post_processing.eval3d import Eval3d, extract_objects_from_detections
from simnet.lib.net import panoptic_trainer

_GPU_TO_USE = 0


class EvalMethod():

  def __init__(self):

    self.eval_3d = Eval3d()
    self.camera_model = camera.FMKCamera()

  def process_sample(self, pose_outputs, box_outputs, seg_outputs, detections_gt, scene_name):
    detections = pose_outputs.get_detections(self.camera_model)
    if scene_name != 'sim':
      table_detection, detections_gt, detections = extract_objects_from_detections(
          detections_gt, detections
      )
    self.eval_3d.process_sample(detections, detections_gt, scene_name)
    return True

  def process_all_dataset(self, log):
    log['all 3Dmap'] = self.eval_3d.process_all_3D_dataset()

  def draw_detections(
      self, pose_outputs, box_outputs, seg_outputs, keypoint_outputs, left_image_np, llog, prefix
  ):
    pose_vis = pose_outputs.get_visualization_img(
        np.copy(left_image_np), camera_model=self.camera_model
    )
    llog[f'{prefix}/pose'] = wandb.Image(pose_vis, caption=prefix)
    seg_vis = seg_outputs.get_visualization_img(np.copy(left_image_np))
    llog[f'{prefix}/seg'] = wandb.Image(seg_vis, caption=prefix)

  def reset(self):
    self.eval_3d = Eval3d()


if __name__ == "__main__":
  parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
  common.add_train_args(parser)
  hparams = parser.parse_args()
  train_ds = datapoint.make_dataset(hparams.train_path)
  samples_per_epoch = len(train_ds.list())
  samples_per_step = hparams.train_batch_size
  steps = hparams.max_steps
  steps_per_epoch = samples_per_epoch // samples_per_step
  epochs = int(np.ceil(steps / steps_per_epoch))
  actual_steps = epochs * steps_per_epoch
  print('Samples per epoch', samples_per_epoch)
  print('Steps per epoch', steps_per_epoch)
  print('Target steps:', steps)
  print('Actual steps:', actual_steps)
  print('Epochs:', epochs)

  model = panoptic_trainer.PanopticModel(hparams, epochs, train_ds, EvalMethod())
  model_checkpoint = ModelCheckpoint(filepath=hparams.output, save_top_k=-1, period=1, mode='max')
  wandb_logger = loggers.WandbLogger(name=hparams.wandb_name, project='simnet')
  trainer = pl.Trainer(
      max_nb_epochs=epochs,
      early_stop_callback=None,
      gpus=[_GPU_TO_USE],
      checkpoint_callback=model_checkpoint,
      #val_check_interval=0.7,
      check_val_every_n_epoch=1,
      logger=wandb_logger,
      default_save_path=hparams.output,
      use_amp=False,
      print_nan_grads=True
  )
  trainer.fit(model)
