import os
import copy

os.environ['PYTHONHASHSEED'] = str(1)

import random

random.seed(123456)
import numpy as np

np.random.seed(123456)
import torch

torch.manual_seed(123456)

import wandb

import pytorch_lightning as pl

from simnet.lib.net import common
from simnet.lib.net.dataset import extract_left_numpy_img
from simnet.lib.net.functions.learning_rate import lambda_learning_rate_poly, lambda_warmup

_GPU_TO_USE = 0


class PanopticModel(pl.LightningModule):

  def __init__(
      self, hparams, epochs=None, train_dataset=None, eval_metric=None, preprocess_func=None
  ):
    super().__init__()

    self.hparams = hparams
    self.epochs = epochs
    self.train_dataset = train_dataset

    self.model = common.get_model(hparams)
    self.eval_metrics = eval_metric
    self.preprocess_func = preprocess_func

  def forward(self, image):
    seg_output, depth_output, small_depth_output, pose_output, box_output, keypoint_output = self.model(
        image, self.global_step
    )
    return seg_output, depth_output, small_depth_output, pose_output, box_output, keypoint_output

  def optimizer_step(self, epoch_nb, batch_nb, optimizer, optimizer_i, second_order_closure=None):
    super().optimizer_step(epoch_nb, batch_nb, optimizer, optimizer_i, second_order_closure)

    learning_rate = 0.0
    for param_group in optimizer.param_groups:
      learning_rate = param_group['lr']
      break
    self.logger.experiment.log({'learning_rate': learning_rate})

  def training_step(self, batch, batch_idx):
    image, seg_target, depth_target, pose_targets, box_targets, keypoint_targets, _, _ = batch
    seg_output, depth_output, small_depth_output, pose_outputs, box_outputs, keypoint_outputs = self.forward(
        image
    )

    log = {}
    loss = depth_output.compute_loss(copy.deepcopy(depth_target), log, 'refined_disp')
    if self.hparams.frozen_stereo_checkpoint is None:
      loss = loss + small_depth_output.compute_loss(depth_target, log, 'cost_volume_disp')
    else:
      assert False
    loss = loss + seg_output.compute_loss(seg_target, log)
    if pose_targets[0] is not None:
      loss = loss + pose_outputs.compute_loss(pose_targets, log)
    if box_targets[0] is not None:
      loss = loss + box_outputs.compute_loss(box_targets, log)
    if keypoint_targets[0] is not None:
      loss += keypoint_outputs.compute_loss(keypoint_targets, log)
    log['train/loss/total'] = loss

    if (batch_idx % 500) == 0:
      with torch.no_grad():
        llog = {}
        prefix = 'train'
        left_image_np = extract_left_numpy_img(image[0])
        logger = self.logger.experiment
        seg_pred_vis = seg_output.get_visualization_img(np.copy(left_image_np))
        llog[f'{prefix}/seg'] = wandb.Image(seg_pred_vis, caption=prefix)
        if pose_targets[0] is not None:
          pose_vis = pose_outputs.get_visualization_img(
              np.copy(left_image_np), camera_model=self.eval_metrics.camera_model
          )
          llog[f'{prefix}/pose'] = wandb.Image(pose_vis, caption=prefix)
        if box_targets[0] is not None:
          box_vis = box_outputs.get_visualization_img(np.copy(left_image_np))
          llog[f'{prefix}/box'] = wandb.Image(box_vis, caption=prefix)
        if keypoint_targets[0] is not None:
          kp_vis = keypoint_outputs.get_visualization_img(np.copy(left_image_np))
          kp_pred_vis = keypoint_outputs.get_detections(np.copy(left_image_np))
          for idx, kp_vis_img in enumerate(kp_vis):
            llog[f'{prefix}/keypoints_{idx}'] = wandb.Image(kp_vis_img, caption=prefix)
          llog[f'{prefix}/keypoints_pred'] = wandb.Image(kp_pred_vis, caption=prefix)
        depth_vis = depth_output.get_visualization_img(np.copy(left_image_np))
        llog[f'{prefix}/disparity'] = wandb.Image(depth_vis, caption=prefix)
        small_depth_vis = small_depth_output.get_visualization_img(np.copy(left_image_np))
        llog[f'{prefix}/small_disparity'] = wandb.Image(small_depth_vis, caption=prefix)
        logger.log(llog)
    return {'loss': loss, 'log': log}

  def validation_step(self, batch, batch_idx):
    #if corl.sim_on_sim_overfit:
    #  # If we are overfitting on sim data set batch size to 1 and enable batch norm for val to make
    #  # it match train. this doesn't make sense unless trying to get val and train to match
    #  # perfectly on a single sample for an overfit test
    #  self.model.train()

    image, seg_target, depth_target, pose_targets, box_targets, keypoint_targets, detections_gt, scene_name = batch
    seg_output, depth_output, small_depth_output, pose_outputs, box_outputs, keypoint_outputs = self.forward(
        image
    )
    log = {}
    with torch.no_grad():
      # Compute mAP score
      if scene_name[0] != 'fmk':
        self.eval_metrics.process_sample(
            pose_outputs, box_outputs, seg_output, detections_gt[0], scene_name[0]
        )
      logger = self.logger.experiment
      if batch_idx < 5 or scene_name[0] == 'fmk':
        llog = {}
        left_image_np = extract_left_numpy_img(image[0])
        prefix = f'val/{batch_idx}'
        logger = self.logger.experiment
        depth_vis = depth_output.get_visualization_img(np.copy(left_image_np))
        llog[f'{prefix}/disparity'] = wandb.Image(depth_vis, caption=prefix)
        small_depth_vis = small_depth_output.get_visualization_img(np.copy(left_image_np))
        llog[f'{prefix}/small_disparity'] = wandb.Image(small_depth_vis, caption=prefix)
        self.eval_metrics.draw_detections(
            pose_outputs, box_outputs, seg_output, keypoint_outputs, left_image_np, llog, prefix
        )

        logger.log(llog)
    return log

  def validation_epoch_end(self, outputs):
    self.trainer.checkpoint_callback.save_best_only = False
    log = {}
    self.eval_metrics.process_all_dataset(log)
    self.eval_metrics.reset()
    return {'log': log}

  @pl.data_loader
  def train_dataloader(self):
    return common.get_loader(
        self.hparams,
        "train",
        preprocess_func=self.preprocess_func,
        datapoint_dataset=self.train_dataset
    )

  @pl.data_loader
  def val_dataloader(self):
    return common.get_loader(self.hparams, "val", preprocess_func=self.preprocess_func)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optim_learning_rate)
    lr_lambda = lambda_learning_rate_poly(self.epochs, self.hparams.optim_poly_exp)
    if self.hparams.optim_warmup_epochs is not None and self.hparams.optim_warmup_epochs > 0:
      lr_lambda = lambda_warmup(self.hparams.optim_warmup_epochs, 0.2, lr_lambda)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return [optimizer], [scheduler]
