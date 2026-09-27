import logging
import math
import os
import threading
import faulthandler

import cv2
import numpy as np
from omegaconf import OmegaConf
from tools.notify_tool import push_notify
from tools.system_monitor import NvitopSystemMonitor
from data_provider.nocs_dataset import batch_to_device
from modules.center_anchor import project_translation_to_image
from utils.rotation_utils import *
from utils.amp_utils import autocast_disabled
from utils.config_utils import load_config
from utils.e2e_pose_loss import compute_e2e_pose_aux_loss
from utils.joint_geometry_loss import compute_joint_geometry_loss
from utils.keypoint_utils import (
    canonical_bbox_axis_keypoints,
    inverse_transform_points,
    metric_keypoints_from_size,
    transform_keypoints,
    solve_rotation_from_keypoints,
    solve_weighted_rotation_from_points,
)
import pathlib
import pickle as cPickle
import time
from utils.evaluation_utils import compute_3d_matches_for_each_gt
from utils.vis_utils import calculate_2d_projections, draw_bboxes, get_3d_bbox, transform_coordinates_3d
from torch.cuda.amp import autocast, GradScaler
from utils.logger_utils import *
from utils.checkpoint_utils import *
from utils.scheduler import BNMomentumScheduler
from utils.octree_utils import *
import torch.optim
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm


class TrainingSolver:
    def __init__(self, model, loss, dataloaders, logger, cfg):
        self.best_loss = 1e6
        self.model = model
        self.loss = loss
        if isinstance(dataloaders, dict):
            self.dataloaders = dataloaders["train"]
            self.val_dataloader = dataloaders.get("val")
        else:
            self.dataloaders = dataloaders
            self.val_dataloader = None
        self.logger = logger
        self.cfg = cfg
        self.device = cfg.device

        self.epoch = 0
        self.iter = 0
        self.image_size = cfg.train_dataset.img_size
        self.per_val = cfg.solver.per_val
        self.per_write = cfg.solver.per_write
        self.per_save_pth = int(
            OmegaConf.select(cfg, "train.per_save_pth", default=1)
        )
        self.manual_save_pth = list(
            OmegaConf.select(cfg, "train.manual_save_pth", default=[])
        )
        self.num_mini_batch = cfg.train_dataloader.num_mini_batch
        self.cfg_path = cfg.config_path
        self.checkpoint = cfg.checkpoint
        self.resume = bool(OmegaConf.select(cfg, "train.resume", default=False))
        self.log_buffer = LogBuffer()
        optimizer_lr = float(OmegaConf.select(cfg, "optimizer.lr", default=1.0e-4))
        self.base_lr = float(
            OmegaConf.select(cfg, "cyclic_lr.base_lr", default=optimizer_lr)
        )
        self.max_lr = float(
            OmegaConf.select(cfg, "cyclic_lr.max_lr", default=optimizer_lr)
        )
        if self.base_lr <= 0.0 or self.max_lr <= 0.0:
            raise ValueError("cyclic_lr base_lr and max_lr must be positive")
        if self.base_lr > self.max_lr:
            raise ValueError("cyclic_lr.base_lr must not exceed cyclic_lr.max_lr")
        amp_init_scale = float(OmegaConf.select(
            cfg,
            "amp.init_scale",
            default=65536.0,
        ))
        amp_growth_interval = int(OmegaConf.select(
            cfg,
            "amp.growth_interval",
            default=2000,
        ))
        if amp_init_scale <= 0.0:
            raise ValueError("amp.init_scale must be positive")
        if amp_growth_interval <= 0:
            raise ValueError("amp.growth_interval must be positive")
        self.scaler = GradScaler(
            init_scale=amp_init_scale,
            growth_interval=amp_growth_interval,
        )
        self._last_nonfinite_log_iter = -10**9
        self.step_log_interval = max(1, int(OmegaConf.select(cfg, "debug.hang_debug.step_log_interval", default=50)))
        self.watchdog_enabled = bool(OmegaConf.select(cfg, "debug.hang_debug.enable", default=True))
        self.watchdog_timeout_sec = int(OmegaConf.select(cfg, "debug.hang_debug.timeout_sec", default=600))
        self.watchdog_check_interval_sec = int(
            OmegaConf.select(cfg, "debug.hang_debug.check_interval_sec", default=20)
        )
        self.watchdog_dump_cooldown_sec = int(
            OmegaConf.select(cfg, "debug.hang_debug.dump_cooldown_sec", default=180)
        )
        self._last_progress_ts = time.time()
        self._last_progress_tag = "init"
        self._watchdog_stop_event = threading.Event()
        self._watchdog_thread = None
        self._watchdog_last_dump_ts = 0.0
        self._hang_dump_file = None
        self._hang_dump_path = os.path.join(self.cfg.log_dir, "hang_dumps.log")
        self._metric_center_z_invalid_batches = 0
        self._metric_center_z_zero_grad_batches = 0
        self._joint_geometry_zero_grad_batches = 0
        self._joint_geometry_nonfinite_grad_batches = 0

        if cfg.solver.model_arch == "da2_metric_center_z_pose":
            metric_depth_source = str(OmegaConf.select(
                cfg,
                "metric_center_z.depth_source",
                default="online_hf",
            ))
            try:
                import transformers

                transformers_version = transformers.__version__
            except (ImportError, AttributeError):
                transformers_version = "unavailable"
            self.logger.warning(
                "[MetricCenterZRuntime] source=%s torch=%s transformers=%s revision=%s "
                "depth_fp32=true mask_fallback=%s mask_min_pixels=%d",
                metric_depth_source,
                torch.__version__,
                transformers_version,
                str(OmegaConf.select(cfg, "metric_center_z.revision", default="unpinned")),
                str(bool(OmegaConf.select(
                    cfg,
                    "train_dataset.scene_mask_fallback_to_bbox",
                    default=False,
                ))),
                int(OmegaConf.select(
                    cfg,
                    "train_dataset.scene_mask_min_pixels",
                    default=1,
                )),
            )

        # Build optimizer based on configs
        OPTIMIZERS = {
            'Adam': torch.optim.Adam,
            'AdamW': torch.optim.AdamW,
            'SGD': torch.optim.SGD,
            'Adagrad': torch.optim.Adagrad,
            # 'AdamW8bit': bnb.optim.AdamW8bit
        }
        if cfg.optimizer.type not in OPTIMIZERS:
            raise ValueError(f"Optimizer type {cfg.optimizer.type} not found.")
        OptimizerClass = OPTIMIZERS[cfg.optimizer.type]
        trainable_only = bool(
            OmegaConf.select(cfg, "optimizer.trainable_only", default=False)
        )
        optimizer_parameters = (
            [parameter for parameter in self.model.parameters() if parameter.requires_grad]
            if trainable_only
            else self.model.parameters()
        )
        if trainable_only and not optimizer_parameters:
            raise ValueError("optimizer.trainable_only found no trainable parameters")
        self.optimizer = OptimizerClass(optimizer_parameters, lr=cfg.optimizer.lr,
                                        weight_decay=cfg.optimizer.weight_decay)

        if self.checkpoint != "":
            if self.resume:
                self.epoch, self.iter = load_checkpoint(
                    self.model,
                    self.checkpoint,
                    optimizer=self.optimizer,
                    device=self.device,
                )
                # Ensure initial_lr is set in param_groups
                for param_group in self.optimizer.param_groups:
                    if 'initial_lr' not in param_group:
                        param_group['initial_lr'] = self.base_lr
                self.logger.warning(
                    '\n>>>> Model resumed at epoch:{}, iter:{}! <<<<\n'.format(self.epoch, self.iter)
                )
            else:
                checkpoint_config = (
                    self.cfg
                    if str(
                        OmegaConf.select(
                            self.cfg,
                            "checkpoint_loading.mode",
                            default="standard",
                        )
                    ).lower()
                    == "selective"
                    else None
                )
                load_checkpoint(
                    self.model,
                    self.checkpoint,
                    optimizer=None,
                    config=checkpoint_config,
                    device=self.device,
                )
                self.epoch = 0
                self.iter = 0
                self.logger.warning(
                    '\n>>>> Model initialized from checkpoint and starts a new run: {} <<<<\n'.format(
                        self.checkpoint
                    )
                )

        self.cyclic_step_size_up = int(
            OmegaConf.select(cfg, "cyclic_lr.step_size_up", default=self.num_mini_batch * 2)
        )
        self.lr_scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.base_lr,
                                                        max_lr=self.max_lr,
                                                        step_size_up=self.cyclic_step_size_up,
                                                        mode='triangular', cycle_momentum=False,
                                                        last_epoch=self.iter - 1)

        bnm_lmbd = lambda it: max(cfg.bn.bn_momentum * cfg.bn.bn_decay ** (int(it / cfg.bn.decay_step)),
                                  cfg.bn.bnm_clip)
        self.bnm_scheduler = BNMomentumScheduler(self.model, bn_lambda=bnm_lmbd, last_epoch=self.iter)

        tb_start_step = self.iter // self.per_write
        self.tb_writer = tools_writer(dir_project=cfg.log_dir, num_counter=2, get_sum=True, start_step=tb_start_step)
        self.system_monitor = NvitopSystemMonitor(
            enabled=bool(OmegaConf.select(cfg, "system_monitor.enable", default=True)),
            logger=self.logger,
        )
        self._start_hang_watchdog()

    def _model_ref(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _model_requires_octree(self):
        return bool(getattr(self._model_ref(), "requires_octree", True))

    def _maybe_reset_cyclic_lr(self):
        reset_every = int(OmegaConf.select(self.cfg, "cyclic_lr.reset_every_epoch", default=0))
        if reset_every <= 0 or self.epoch <= 0 or self.epoch % reset_every != 0:
            return
        decay_gamma = float(OmegaConf.select(self.cfg, "cyclic_lr.reset_decay_gamma", default=1.0))
        decay = decay_gamma ** (self.epoch / reset_every)
        base_lr = self.base_lr * decay
        max_lr = self.max_lr * decay
        step_size_up = int(OmegaConf.select(self.cfg, "cyclic_lr.reset_step_size_up", default=self.cyclic_step_size_up))
        self.lr_scheduler = optim.lr_scheduler.CyclicLR(
            self.optimizer,
            base_lr=base_lr,
            max_lr=max_lr,
            step_size_up=step_size_up,
            mode='triangular',
            cycle_momentum=False,
        )
        self.logger.warning(
            "[CyclicLRReset] epoch=%d base_lr=%.8g max_lr=%.8g step_size_up=%d",
            self.epoch,
            base_lr,
            max_lr,
            step_size_up,
        )

    def solve(self):
        # Enable training acceleration options
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        try:
            while self.epoch <= self.cfg.train.max_epoch:
                self.dynamic_load_config()
                one_based_checkpoints = bool(
                    OmegaConf.select(
                        self.cfg,
                        "train.checkpoint_epoch_one_based",
                        default=False,
                    )
                )
                checkpoint_epoch = (
                    self.epoch + 1 if one_based_checkpoints else self.epoch
                )
                self._mark_progress(f"epoch_start:{self.epoch}")
                self.logger.warning('\n>>>> Epoch {} <<<<'.format(self.epoch))
                self.logger.warning("[EpochStart] epoch=%d iter=%d", self.epoch, self.iter)

                end = time.time()
                dict_info_train = self.train()
                train_time = time.time() - end
                self._mark_progress(f"epoch_train_done:{self.epoch}")

                dict_info = {'train_time(min)': train_time / 60.0}
                if OmegaConf.select(
                    self.cfg, "logging.train_metrics", default=None
                ) is None:
                    for key, value in dict_info_train.items():
                        if 'loss' in key:
                            dict_info[key] = value
                else:
                    dict_info.update(dict_info_train)

                validation_metrics = None
                validation_every = max(
                    1,
                    int(OmegaConf.select(
                        self.cfg, "validation.every_epochs", default=self.per_val
                    )),
                )
                if (
                    self.val_dataloader is not None
                    and checkpoint_epoch % validation_every == 0
                ):
                    validation_metrics = self.validate()
                    self.write_summary(validation_metrics, "val")
                    for key, value in validation_metrics.items():
                        dict_info[f"val/{key}"] = value

                periodic_due = (
                    self.per_save_pth > 0
                    and checkpoint_epoch % self.per_save_pth == 0
                )
                if periodic_due or checkpoint_epoch in self.manual_save_pth:
                    ckpt_path = os.path.join(self.cfg.log_dir, 'epoch_' + str(checkpoint_epoch) + '.pth')
                    self.logger.warning("[CheckpointStart] type=periodic path=%s", ckpt_path)
                    self._mark_progress(f"checkpoint_periodic_start:{self.epoch}")
                    save_checkpoint(
                        self.model,
                        ckpt_path,
                        checkpoint_epoch,
                        self.iter,
                        optimizer=self.optimizer.state_dict(),
                        config=self.cfg,
                    )
                    self._mark_progress(f"checkpoint_periodic_done:{self.epoch}")
                    self.logger.warning("[CheckpointDone] type=periodic path=%s", ckpt_path)

                selection_loss = None
                if validation_metrics is not None:
                    selection_loss = validation_metrics["loss/loss_all"]
                elif self.val_dataloader is None:
                    selection_loss = dict_info['loss/loss_all']
                if selection_loss is not None and selection_loss < self.best_loss:
                    path_obj = pathlib.Path(self.cfg.log_dir)
                    self.best_loss = selection_loss
                    ckpt_path = os.path.join(
                        self.cfg.log_dir,
                        'best_epoch_' + str(checkpoint_epoch) + '_' + str(self.best_loss) + '.pth',
                    )
                    self.logger.warning("[CheckpointStart] type=best path=%s", ckpt_path)
                    self._mark_progress(f"checkpoint_best_start:{self.epoch}")
                    save_checkpoint(
                        self.model,
                        ckpt_path,
                        checkpoint_epoch,
                        self.iter,
                        optimizer=self.optimizer.state_dict(),
                        config=self.cfg,
                    )
                    self._mark_progress(f"checkpoint_best_done:{self.epoch}")
                    self.logger.warning("[CheckpointDone] type=best path=%s", ckpt_path)
                    for file_path in path_obj.glob("best*.pth"):
                        if os.path.abspath(file_path) != os.path.abspath(ckpt_path):
                            file_path.unlink()

                self._maybe_reset_cyclic_lr()

                path_obj = pathlib.Path(self.cfg.log_dir)
                last_loss = dict_info['loss/loss_all']
                ckpt_path = os.path.join(
                    self.cfg.log_dir,
                    'last_epoch_' + str(checkpoint_epoch) + '_' + str(last_loss) + '.pth',
                )
                self.logger.warning("[CheckpointStart] type=last path=%s", ckpt_path)
                self._mark_progress(f"checkpoint_last_start:{self.epoch}")
                save_checkpoint(
                    self.model,
                    ckpt_path,
                    checkpoint_epoch,
                    self.iter,
                    optimizer=self.optimizer.state_dict(),
                    config=self.cfg,
                )
                self._mark_progress(f"checkpoint_last_done:{self.epoch}")
                self.logger.warning("[CheckpointDone] type=last path=%s", ckpt_path)
                for file_path in path_obj.glob("last*.pth"):
                    if os.path.abspath(file_path) != os.path.abspath(ckpt_path):
                        file_path.unlink()

                prefix = 'Epoch {} - '.format(self.epoch)
                write_info = self.get_logger_info(prefix, dict_info=dict_info)
                self.logger.warning(write_info + "\n")
                self.logger.warning(
                    "[EpochDone] epoch=%d iter=%d train_time_min=%.3f",
                    self.epoch,
                    self.iter,
                    train_time / 60.0,
                )
                self.epoch += 1
        finally:
            self._stop_hang_watchdog()

    def train(self):
        mode = 'train'
        self.model.train()
        end = time.time()
        self._mark_progress(f"train_start:{self.epoch}")

        # Reset datasets
        real_loader = self.dataloaders
        if hasattr(real_loader.dataset, "set_epoch"):
            self._mark_progress(f"set_epoch_start:{self.epoch}")
            real_loader.dataset.set_epoch(self.epoch)
            self._mark_progress(f"set_epoch_done:{self.epoch}")
        real_bs = len(real_loader)

        self._mark_progress(f"dataloader_iter_start:{self.epoch}")
        real_iter = iter(real_loader)
        self._mark_progress(f"dataloader_iter_ready:{self.epoch}")

        for i, real_data in enumerate(real_iter):
            self._mark_progress(f"batch_fetch_done:e{self.epoch}:b{i}:iter{self.iter}")
            data_time = time.time() - end
            # 1. Forward pass with AMP autocast
            self._mark_progress(f"forward_start:e{self.epoch}:b{i}:iter{self.iter}")
            with autocast():
            # with autocast(self.device):
                # Keep loss computation inside autocast context
                loss, dict_info_step = self.step(real_data)
            self._check_metric_center_z_forward_health(dict_info_step)
            self._mark_progress(f"forward_done:e{self.epoch}:b{i}:iter{self.iter}")

            if not torch.isfinite(loss):
                if self.iter - self._last_nonfinite_log_iter >= 20:
                    self._last_nonfinite_log_iter = self.iter
                    self.logger.error(
                        "[NonFiniteLoss] epoch=%d iter=%d batch=%d loss=%s octree=%s x0=%s x0_size=%s x0_t=%s x0_r=%s x0_geo=%s geo_valid_ratio=%s",
                        self.epoch,
                        self.iter,
                        i,
                        str(float(loss)),
                        str(dict_info_step.get('loss/octree_loss', float('nan'))),
                        str(dict_info_step.get('loss/x0', float('nan'))),
                        str(dict_info_step.get('loss/x0_size', float('nan'))),
                        str(dict_info_step.get('loss/x0_translation', float('nan'))),
                        str(dict_info_step.get('loss/x0_rotation', float('nan'))),
                        str(dict_info_step.get('loss/x0_rotation_geo', float('nan'))),
                        str(dict_info_step.get('loss/x0_rotation_geo_valid_ratio', float('nan'))),
                    )
                self.optimizer.zero_grad(set_to_none=True)
                if not self._params_are_finite():
                    raise RuntimeError(
                        "Model parameters are already non-finite at epoch={}, iter={}, batch={}. "
                        "Resume from the previous finite checkpoint; continuing will keep producing NaN losses.".format(
                            self.epoch, self.iter, i)
                    )
                end = time.time()
                self.iter += 1
                continue

            forward_time = time.time() - end - data_time
            # 2. Backward pass with gradient scaling
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            backward_time = time.time() - end - forward_time - data_time
            self._mark_progress(f"backward_done:e{self.epoch}:b{i}:iter{self.iter}")
            # 3. Unscale/clip gradients and step optimizer
            # scaler.step() only applies optimizer.step() when gradients are finite
            grad_clip_norm = float(OmegaConf.select(self.cfg, "optimizer.grad_clip_norm", default=0.0))
            grad_norm = -1.0
            has_finite_grads = True
            self.scaler.unscale_(self.optimizer)
            if grad_clip_norm > 0.0:
                grad_norm_tensor = torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                grad_norm = float(grad_norm_tensor)
                has_finite_grads = bool(torch.isfinite(grad_norm_tensor))
            else:
                has_finite_grads = self._grads_are_finite()

            if has_finite_grads:
                self._check_metric_center_z_gradient_health(dict_info_step)
                self._check_joint_geometry_gradient_health(dict_info_step)

            optimizer_stepped = False
            if has_finite_grads:
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                optimizer_stepped = self.scaler.get_scale() >= scale_before
            else:
                self._check_joint_geometry_nonfinite_gradients()
                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.update()
                self.logger.error(
                    "[NonFiniteGrad] epoch=%d iter=%d batch=%d grad_norm=%s; skipped optimizer/scheduler step",
                    self.epoch,
                    self.iter,
                    i,
                    str(grad_norm),
                )

            if optimizer_stepped and not self._params_are_finite():
                raise RuntimeError(
                    "Model parameters became non-finite after optimizer step at "
                    "epoch={}, iter={}, batch={}. Resume from the previous finite checkpoint "
                    "and lower the learning rate or enable grad clipping.".format(self.epoch, self.iter, i)
                )

            if optimizer_stepped and self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if optimizer_stepped and self.bnm_scheduler is not None:
                self.bnm_scheduler.step()
            self._mark_progress(f"optim_done:e{self.epoch}:b{i}:iter{self.iter}")
            # 5. Keep zero_grad before backward for more stable behavior

            dict_info_step.update({
                'timing/T_data': data_time,
                'timing/T_forward': forward_time,
                'timing/T_backward': backward_time,
                'train/learning_rate': (
                    self.optimizer.param_groups[0]['lr']
                    if self.optimizer.param_groups
                    else 0.0
                ),
                'train/grad_norm': grad_norm,
            })
            # if i % self.step_log_interval == 0:
            #     self.logger.warning(
            #         "[StepTrace] epoch=%d iter=%d batch=%d/%d data=%.3fs fwd=%.3fs bwd=%.3fs loss=%.6f",
            #         self.epoch,
            #         self.iter,
            #         i,
            #         real_bs,
            #         data_time,
            #         forward_time,
            #         backward_time,
            #         float(dict_info_step.get('loss/loss_all', float('nan'))),
            #     )

            self.log_buffer.update(dict_info_step)

            if i % self.per_write == 0:
                push_notify(msg="[{}/{}][{}/{}][{}]".format(self.epoch, self.cfg.train.max_epoch, i, real_bs, self.iter))
                self.log_buffer.average(self.per_write)
                prefix = '[{:>4}/{:>4}][{:>4}/{:>4}][{:>6}] Train - '.format(
                    self.epoch, self.cfg.train.max_epoch, i, real_bs, self.iter)
                write_info = self.get_logger_info(
                    prefix, dict_info=self.log_buffer._output)
                self.logger.warning(write_info)
                summary_step = self.tb_writer.list_couter[0]
                self.write_summary(self.log_buffer._output, mode)
                self.system_monitor.record(self.tb_writer.writer, summary_step)
            end = time.time()
            self._mark_progress(f"batch_done:e{self.epoch}:b{i}:iter{self.iter}")

            self.iter += 1
            i += 1

        dict_info_epoch = self.log_buffer.avg
        self.log_buffer.clear()

        return dict_info_epoch

    def validate(self):
        """Evaluate the held-out REAL-train manifest split without updating state."""
        if self.val_dataloader is None:
            return {}
        was_training = self.model.training
        self.model.eval()
        buffer = LogBuffer()
        max_batches = int(
            OmegaConf.select(self.cfg, "validation.max_batches", default=0)
        )
        try:
            with torch.no_grad():
                for batch_index, real_data in enumerate(self.val_dataloader):
                    if max_batches > 0 and batch_index >= max_batches:
                        break
                    with autocast():
                        loss, metrics = self.step(real_data)
                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            f"Non-finite validation loss at batch {batch_index}"
                        )
                    buffer.update(metrics)
        finally:
            if was_training:
                self.model.train()
        averaged = buffer.avg
        self.logger.warning(
            "[IndependentValidationDone] epoch=%d batches=%d loss=%.6f "
            "class_acc=%.4f bottle_can_acc=%.4f translation_mae_cm=%.3f",
            self.epoch,
            min(len(self.val_dataloader), max_batches)
            if max_batches > 0 else len(self.val_dataloader),
            float(averaged.get("loss/loss_all", float("nan"))),
            float(averaged.get("joint/class_accuracy", 0.0)),
            float(averaged.get("joint/bottle_can_accuracy", 0.0)),
            float(averaged.get("joint/translation_mae_cm", 0.0)),
        )
        return averaged

    def _check_metric_center_z_forward_health(self, metrics):
        if self.cfg.solver.model_arch != "da2_metric_center_z_pose":
            return
        enabled_ratio = float(metrics.get("metric_center_z/enabled_ratio", 0.0))
        route_ratio = float(metrics.get("metric_center_z/route_ratio", 0.0))
        if enabled_ratio <= 0.0:
            return
        if route_ratio > 0.0:
            self._metric_center_z_invalid_batches = 0
            return

        self._metric_center_z_invalid_batches = (
            getattr(self, "_metric_center_z_invalid_batches", 0) + 1
        )
        patience = int(OmegaConf.select(
            self.cfg,
            "metric_center_z.fail_fast_invalid_batches",
            default=0,
        ))
        if patience <= 0 or self._metric_center_z_invalid_batches < patience:
            return
        raise RuntimeError(
            "Metric Center-Z has no routed samples for {} consecutive eligible "
            "batches (iter={}). Diagnostics: mask_pixels={:.1f}, "
            "finite_ratio={:.6f}, in_range_ratio={:.6f}, depth_range=[{:.6g}, "
            "{:.6g}]m, mask_fallback_ratio={:.6f}. Check the training machine's "
            "SAM masks and metric-depth runtime before continuing.".format(
                self._metric_center_z_invalid_batches,
                self.iter,
                float(metrics.get("metric_center_z/mask_pixels", 0.0)),
                float(metrics.get("metric_center_z/finite_pixel_ratio", 0.0)),
                float(metrics.get("metric_center_z/valid_pixel_ratio", 0.0)),
                float(metrics.get("metric_center_z/depth_min_m", 0.0)),
                float(metrics.get("metric_center_z/depth_max_m", 0.0)),
                float(metrics.get("metric_center_z/mask_fallback_ratio", 0.0)),
            )
        )

    def _check_metric_center_z_gradient_health(self, metrics):
        if self.cfg.solver.model_arch != "da2_metric_center_z_pose":
            return
        route_ratio = float(metrics.get("metric_center_z/route_ratio", 0.0))
        if route_ratio <= 0.0:
            metrics["metric_center_z/head_grad_norm"] = 0.0
            return

        model = self._model_ref()
        head = getattr(model, "metric_center_z_head", None)
        squared_norm = 0.0
        if head is not None:
            for parameter in head.parameters():
                if parameter.grad is not None:
                    squared_norm += float(parameter.grad.float().square().sum())
        grad_norm = math.sqrt(squared_norm)
        metrics["metric_center_z/head_grad_norm"] = grad_norm
        if grad_norm > 1.0e-12:
            self._metric_center_z_zero_grad_batches = 0
            return

        self._metric_center_z_zero_grad_batches = (
            getattr(self, "_metric_center_z_zero_grad_batches", 0) + 1
        )
        patience = int(OmegaConf.select(
            self.cfg,
            "metric_center_z.fail_fast_zero_grad_batches",
            default=0,
        ))
        if patience > 0 and self._metric_center_z_zero_grad_batches >= patience:
            raise RuntimeError(
                "Metric Center-Z head gradient stayed zero for {} consecutive "
                "routed batches (iter={}). Stop instead of producing an identity "
                "checkpoint.".format(
                    self._metric_center_z_zero_grad_batches,
                    self.iter,
                )
            )

    def _check_joint_geometry_gradient_health(self, metrics):
        if self.cfg.solver.model_arch != "da2_joint_geometry_pose":
            return
        self._joint_geometry_nonfinite_grad_batches = 0
        model = self._model_ref()
        groups = {
            "category": getattr(model, "joint_category_head", None),
            "nocs": getattr(model, "joint_nocs_head", None),
            "geometry": getattr(model, "joint_geometry_head", None),
        }
        total_squared = 0.0
        for name, module in groups.items():
            squared = 0.0
            if module is not None:
                for parameter in module.parameters():
                    if parameter.grad is not None:
                        squared += float(parameter.grad.float().square().sum())
            metrics[f"joint/grad_norm_{name}"] = math.sqrt(squared)
            total_squared += squared
        total_norm = math.sqrt(total_squared)
        metrics["joint/grad_norm_total"] = total_norm
        if total_norm > 1.0e-12:
            self._joint_geometry_zero_grad_batches = 0
            return
        self._joint_geometry_zero_grad_batches = (
            getattr(self, "_joint_geometry_zero_grad_batches", 0) + 1
        )
        patience = int(OmegaConf.select(
            self.cfg, "joint_geometry.fail_fast_zero_grad_batches", default=0
        ))
        if patience > 0 and self._joint_geometry_zero_grad_batches >= patience:
            raise RuntimeError(
                "Joint geometry heads received zero gradient for {} consecutive "
                "batches (iter={}).".format(
                    self._joint_geometry_zero_grad_batches, self.iter
                )
            )

    def _check_joint_geometry_nonfinite_gradients(self):
        if self.cfg.solver.model_arch != "da2_joint_geometry_pose":
            return
        model = self._model_ref()
        groups = {
            "category": getattr(model, "joint_category_head", None),
            "nocs": getattr(model, "joint_nocs_head", None),
            "geometry": getattr(model, "joint_geometry_head", None),
        }
        counts = {}
        for name, module in groups.items():
            count = 0
            if module is not None:
                for parameter in module.parameters():
                    if parameter.grad is not None:
                        count += int((~torch.isfinite(parameter.grad)).sum())
            counts[name] = count
        self._joint_geometry_nonfinite_grad_batches = (
            getattr(self, "_joint_geometry_nonfinite_grad_batches", 0) + 1
        )
        self.logger.error(
            "[JointNonFiniteGrad] consecutive=%d category=%d nocs=%d geometry=%d",
            self._joint_geometry_nonfinite_grad_batches,
            counts["category"],
            counts["nocs"],
            counts["geometry"],
        )
        patience = int(OmegaConf.select(
            self.cfg,
            "joint_geometry.fail_fast_nonfinite_grad_batches",
            default=3,
        ))
        if patience > 0 and self._joint_geometry_nonfinite_grad_batches >= patience:
            raise RuntimeError(
                "Joint geometry produced non-finite gradients for {} consecutive "
                "batches. Counts: category={}, nocs={}, geometry={}.".format(
                    self._joint_geometry_nonfinite_grad_batches,
                    counts["category"],
                    counts["nocs"],
                    counts["geometry"],
                )
            )

    def step(self, real_data):
        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        dict_info = {}
        batch_octree = None
        if self._model_requires_octree():
            batch_points = init_batch_points(real_data)
            batch_octree = build_batch_octree(batch_points, self.cfg.octree.depth, self.cfg.octree.full_depth)
            real_data['batch_octree'] = batch_octree

        real_data = batch_to_device(real_data, self.device)
        end_points = self.model(real_data)

        loss_pose = self._compute_pose_noise_loss(end_points)
        weight_pose_rotation_noise = float(
            OmegaConf.select(self.cfg, "loss.weight_pose_rotation_noise", default=1.0)
        )
        if (
            weight_pose_rotation_noise != 1.0
            and end_points.get('pose_prediction_mode') != 'structured_direct_rs'
            and 'delta_rotation' in end_points
        ):
            loss_pose = loss_pose + (weight_pose_rotation_noise - 1.0) * F.mse_loss(
                end_points['pred_rotation'],
                end_points['delta_rotation'],
            )
        zero = loss_pose.new_zeros(())
        if batch_octree is not None and 'octree_out' in end_points:
            loss_octree, octree_dict = self.loss['octree'](
                batch_octree, end_points['octree_out'], real_data["category_label"])
        else:
            loss_octree = zero
            octree_dict = {
                'loss/octree_struc': 0.0,
                'loss/octree_reg': 0.0,
                'loss/octree_cls': 0.0,
                'loss/octree_loss': 0.0,
            }
        if 'pred_latent_code' in end_points and 'shape_code' in real_data:
            loss_shape = self.loss['shape'](end_points['pred_latent_code'], real_data["shape_code"])
        else:
            loss_shape = zero
        loss_diff9d_shape, diff9d_shape_dict = self._compute_diff9d_shape_losses(end_points, real_data)
        loss_x0, x0_dict = self._compute_x0_pose_loss(end_points, real_data)
        loss_rot_bin, rot_bin_dict = self._compute_rotation_bin_loss(end_points, real_data)
        loss_direct_rot, direct_rot_dict = self._compute_direct_rotation_loss(end_points, real_data)
        loss_metric_depth, metric_depth_dict = self._compute_metric_depth_loss(end_points, real_data)
        loss_e2e_pose, e2e_pose_dict = self._compute_e2e_pose_aux_loss(
            end_points,
            real_data,
        )
        loss_metric_size, metric_size_dict = self._compute_metric_size_loss(end_points, real_data)
        loss_structured_rs, structured_rs_dict = self._compute_structured_rs_loss(
            end_points,
            real_data,
        )
        loss_rotation_v6, rotation_v6_dict = self._compute_rotation_v6_loss(
            end_points,
            real_data,
        )
        loss_rotation_v9, rotation_v9_dict = (
            self._compute_rotation_correspondence_v9_loss(
                end_points,
                real_data,
            )
        )
        loss_absolute_center, absolute_center_dict = (
            self._compute_absolute_center_translation_loss(
                end_points,
                real_data,
            )
        )
        loss_guarded_center_z, guarded_center_z_dict = (
            self._compute_guarded_center_z_v11_loss(end_points, real_data)
        )
        loss_translation_anchor, translation_anchor_dict = self._compute_translation_anchor_loss(
            end_points, real_data
        )
        loss_translation_vote, translation_vote_dict = self._compute_translation_vote_loss(
            end_points,
            real_data,
        )
        loss_center_z_v4, center_z_v4_dict = self._compute_center_z_v4_loss(
            end_points,
            real_data,
        )
        loss_metric_depth_v5, metric_depth_v5_dict = (
            self._compute_metric_depth_v5_loss(end_points, real_data)
        )
        loss_final_t, final_t_dict = self._compute_final_translation_loss(end_points, real_data)
        loss_center_anchor, center_anchor_dict = self._compute_center_anchor_losses(end_points, real_data)
        loss_keypoint, keypoint_dict = self._compute_canonical_keypoint_loss(end_points, real_data)
        loss_nocs, nocs_dict = self._compute_nocs_correspondence_loss(end_points, real_data)
        loss_joint_geometry, joint_geometry_dict = compute_joint_geometry_loss(
            self.cfg, end_points, real_data
        )

        loss_all = (
            loss_pose + loss_octree + loss_shape + loss_diff9d_shape + loss_x0 + loss_rot_bin
            + loss_direct_rot + loss_metric_depth + loss_e2e_pose
            + loss_metric_size + loss_final_t
            + loss_structured_rs + loss_rotation_v6
            + loss_rotation_v9
            + loss_absolute_center
            + loss_guarded_center_z
            + loss_translation_anchor + loss_center_anchor
            + loss_translation_vote + loss_center_z_v4 + loss_metric_depth_v5
            + loss_keypoint + loss_nocs
            + loss_joint_geometry
        )

        dict_info.update(octree_dict)
        dict_info.update(diff9d_shape_dict)
        dict_info.update(x0_dict)
        dict_info.update(rot_bin_dict)
        dict_info.update(direct_rot_dict)
        dict_info.update(metric_depth_dict)
        dict_info.update(e2e_pose_dict)
        dict_info.update(metric_size_dict)
        dict_info.update(structured_rs_dict)
        dict_info.update(rotation_v6_dict)
        dict_info.update(rotation_v9_dict)
        dict_info.update(absolute_center_dict)
        dict_info.update(guarded_center_z_dict)
        dict_info.update(translation_anchor_dict)
        dict_info.update(translation_vote_dict)
        dict_info.update(center_z_v4_dict)
        dict_info.update(metric_depth_v5_dict)
        dict_info.update(final_t_dict)
        dict_info.update(center_anchor_dict)
        dict_info.update(keypoint_dict)
        dict_info.update(nocs_dict)
        dict_info.update(joint_geometry_dict)
        dict_info.update({
            'loss/pose': float(loss_pose),
            'loss/shape': float(loss_shape),
            'loss/diff9d_shape_weighted': float(loss_diff9d_shape),
            'loss/x0_weighted': float(loss_x0),
            'loss/rotation_bin_weighted': float(loss_rot_bin),
            'loss/direct_rotation_weighted': float(loss_direct_rot),
            'loss/metric_depth_weighted': float(loss_metric_depth),
            'loss/e2e_pose_weighted': float(loss_e2e_pose),
            'loss/metric_size_weighted': float(loss_metric_size),
            'loss/structured_rs_weighted': float(loss_structured_rs),
            'loss/rotation_v6_weighted': float(loss_rotation_v6),
            'loss/rotation_v9_weighted': float(loss_rotation_v9),
            'loss/absolute_center_weighted': float(loss_absolute_center),
            'loss/guarded_center_z_weighted': float(loss_guarded_center_z),
            'loss/translation_anchor_weighted': float(loss_translation_anchor),
            'loss/translation_vote_weighted': float(loss_translation_vote),
            'loss/center_z_v4_weighted': float(loss_center_z_v4),
            'loss/metric_depth_v5_weighted': float(loss_metric_depth_v5),
            'loss/final_translation_weighted': float(loss_final_t),
            'loss/center_anchor_weighted': float(loss_center_anchor),
            'loss/canonical_keypoint_weighted': float(loss_keypoint),
            'loss/nocs_correspondence_weighted': float(loss_nocs),
            'loss/joint_geometry_weighted': float(loss_joint_geometry),
            'loss/loss_all': float(loss_all),
            'timing/resnet_forward_time'   : end_points.get('resnet_forward_time', 0.0),
            'timing/octree_forward_time'   : end_points.get('octree_forward_time', 0.0),
            'timing/shapenet_forward_time' : end_points.get('shapenet_forward_time', 0.0),
            'timing/diffusion_forward_time': end_points.get('diffusion_forward_time', 0.0),
            'timing/denoiser_time'         : end_points.get('denoiser_time', 0.0)
        })

        return loss_all, dict_info

    def _compute_pose_noise_loss(self, end_points):
        if end_points.get('pose_prediction_mode') in {
            'structured_direct_rs',
            'e2e_direct_pose',
            'joint_geometry_pose',
        }:
            return end_points['pred_size'].new_zeros(())
        pred_r = end_points['pred_rotation']
        pred_s = end_points['pred_size']
        target_r = end_points['delta_rotation']
        target_s = end_points['delta_size']
        pose_loss_type = str(OmegaConf.select(self.cfg, "loss.pose_loss_type", default="mse")).lower()
        if end_points.get('translation_prediction_mode') in {
            'deterministic_center_depth',
            'ray_depth_v1',
            'ray_depth_v2',
            'ray_depth_v3',
            'ray_depth_v4_center_z',
            'metric_depth_v5_center_z',
        }:
            if pose_loss_type == "diff9d":
                loss_r = torch.mean(torch.norm(pred_r - target_r, dim=1))
                loss_s = torch.mean(torch.norm(pred_s - target_s, dim=1))
                return loss_r + loss_s
            return F.mse_loss(pred_r, target_r) + F.mse_loss(pred_s, target_s)

        pred_t = end_points['pred_translation']
        target_t = end_points['delta_translation']
        if pose_loss_type == "diff9d":
            loss_r = torch.mean(torch.norm(pred_r - target_r, dim=1))
            loss_t = torch.mean(torch.norm(pred_t - target_t, dim=1))
            loss_s = torch.mean(torch.norm(pred_s - target_s, dim=1))
            return loss_r + loss_t + loss_s
        return self.loss['pose'](pred_t, pred_r, pred_s, target_t, target_r, target_s)

    def _compute_diff9d_shape_losses(self, end_points, real_data):
        weight_shape = float(OmegaConf.select(self.cfg, "loss.weight_diff9d_shape_chamfer", default=0.0))
        weight_nocs = float(OmegaConf.select(self.cfg, "loss.weight_diff9d_nocs_shape", default=0.0))
        zero = end_points['pred_size'].new_zeros(())
        empty = {
            'loss/diff9d_shape_chamfer': 0.0,
            'loss/diff9d_nocs_shape': 0.0,
        }
        if weight_shape <= 0.0 and weight_nocs <= 0.0:
            return zero, empty

        loss_shape = zero
        if weight_shape > 0.0 and 'pred_shape' in end_points and 'model' in real_data:
            pred_shape = end_points['pred_shape']
            target_shape = real_data['model'].to(device=pred_shape.device, dtype=pred_shape.dtype)
            dist = torch.cdist(pred_shape, target_shape, p=2)
            loss_shape = 0.5 * dist.min(dim=2).values.mean(dim=1) + 0.5 * dist.min(dim=1).values.mean(dim=1)
            loss_shape = loss_shape.mean()
            empty['loss/diff9d_shape_chamfer'] = float(loss_shape)

        loss_nocs = zero
        if weight_nocs > 0.0 and 'pred_nocs_shape' in end_points and 'qo' in real_data:
            pred_nocs = end_points['pred_nocs_shape']
            target_nocs = real_data['qo'].to(device=pred_nocs.device, dtype=pred_nocs.dtype)
            threshold = float(OmegaConf.select(self.cfg, "loss.diff9d_nocs_smooth_l1_threshold", default=0.1))
            diff = torch.abs(pred_nocs - target_nocs)
            less = diff.pow(2) / (2.0 * threshold)
            higher = diff - threshold / 2.0
            loss_nocs = torch.where(diff > threshold, higher, less)
            loss_nocs = torch.mean(torch.sum(loss_nocs, dim=2))
            empty['loss/diff9d_nocs_shape'] = float(loss_nocs)

        return weight_shape * loss_shape + weight_nocs * loss_nocs, empty

    def _compute_rotation_bin_loss(self, end_points, real_data):
        weight = float(OmegaConf.select(self.cfg, "loss.weight_rotation_bin", default=0.0))
        logits = end_points.get('pred_rotation_bin_logits')
        zero = end_points['pred_size'].new_zeros(())
        if weight <= 0.0 or logits is None:
            return zero, {
                'loss/rotation_bin': 0.0,
                'loss/rotation_bin_acc': 0.0,
                'loss/rotation_bin_valid_ratio': 0.0,
            }

        bin_count = int(OmegaConf.select(self.cfg, "loss.rotation_bin_count", default=24))
        bin_count = max(1, bin_count)
        if logits.shape[1] != bin_count:
            raise ValueError(f"rotation bin logits shape {logits.shape} does not match bin_count={bin_count}")

        gt_rot = real_data['rotation_label']
        finite_rot = torch.isfinite(gt_rot.reshape(gt_rot.shape[0], -1)).all(dim=1)
        yaw = torch.atan2(gt_rot[:, 0, 2], gt_rot[:, 2, 2])
        yaw = torch.remainder(yaw + 2.0 * math.pi, 2.0 * math.pi)
        target = torch.floor(yaw / (2.0 * math.pi / float(bin_count))).long().clamp(0, bin_count - 1)

        category = real_data.get('category_label')
        valid = finite_rot
        if bool(OmegaConf.select(self.cfg, "loss.rotation_bin_skip_symmetric", default=True)):
            symmetric = self._symmetric_mask(category, gt_rot.shape[0], gt_rot.device)
            valid = valid & ~symmetric

        if not torch.any(valid):
            return zero, {
                'loss/rotation_bin': 0.0,
                'loss/rotation_bin_acc': 0.0,
                'loss/rotation_bin_valid_ratio': 0.0,
            }

        loss = F.cross_entropy(logits[valid], target[valid])
        pred = logits[valid].argmax(dim=1)
        acc = (pred == target[valid]).float().mean()
        return loss * weight, {
            'loss/rotation_bin': float(loss),
            'loss/rotation_bin_acc': float(acc),
            'loss/rotation_bin_valid_ratio': float(valid.float().mean()),
        }

    def _compute_direct_rotation_loss(self, end_points, real_data):
        weight = float(OmegaConf.select(self.cfg, "loss.weight_direct_rotation", default=0.0))
        weight_geo = float(OmegaConf.select(self.cfg, "loss.weight_direct_rotation_geo", default=0.0))
        asym_geo_weight = float(OmegaConf.select(self.cfg, "loss.direct_rotation_asym_geo_weight", default=1.0))
        sym_geo_weight = float(OmegaConf.select(self.cfg, "loss.direct_rotation_sym_geo_weight", default=1.0))
        pred_rotation_6d = end_points.get('pred_direct_rotation_6d')
        zero = end_points['pred_size'].new_zeros(())
        empty = {
            'loss/direct_rotation': 0.0,
            'loss/direct_rotation_geo': 0.0,
            'loss/direct_rotation_asym_geo': 0.0,
            'loss/direct_rotation_sym_geo': 0.0,
            'loss/direct_rotation_valid_ratio': 0.0,
        }
        if weight <= 0.0 or pred_rotation_6d is None:
            return zero, empty

        gt_rot = real_data['rotation_label']
        gt_rotation_6d = rotation_matrix_to_6d(gt_rot)
        valid = (
            torch.isfinite(pred_rotation_6d).all(dim=1)
            & torch.isfinite(gt_rotation_6d).all(dim=1)
            & torch.isfinite(gt_rot.reshape(gt_rot.shape[0], -1)).all(dim=1)
        )
        if not torch.any(valid):
            return zero, empty

        rotation_target_6d = gt_rotation_6d
        if bool(OmegaConf.select(self.cfg, "loss.direct_rotation_symmetry_aware", default=True)):
            rotation_target_6d = self._select_symmetry_rotation_6d_target(
                pred_rotation_6d,
                gt_rot,
                real_data.get('category_label'),
            )
        per_sample_direct = F.mse_loss(
            pred_rotation_6d[valid],
            rotation_target_6d[valid],
            reduction='none',
        ).mean(dim=1)
        categories_valid = real_data.get('category_label')
        categories_valid = categories_valid[valid] if categories_valid is not None else None
        asym_mask = self._asymmetric_direct_rotation_mask(
            categories_valid,
            per_sample_direct.shape[0],
            pred_rotation_6d.device,
        )
        sample_weights = torch.where(
            asym_mask,
            per_sample_direct.new_tensor(asym_geo_weight),
            per_sample_direct.new_tensor(sym_geo_weight),
        )
        configured_class_weights = OmegaConf.select(
            self.cfg, "loss.direct_rotation_class_weights", default=None)
        if configured_class_weights is not None and categories_valid is not None:
            class_weights = torch.as_tensor(
                configured_class_weights,
                device=per_sample_direct.device,
                dtype=per_sample_direct.dtype,
            )
            if class_weights.ndim != 1 or class_weights.numel() == 0:
                raise ValueError(
                    "loss.direct_rotation_class_weights must be a non-empty list")
            category_vector = categories_valid.to(
                device=per_sample_direct.device).reshape(-1).long()
            if torch.any(
                (category_vector < 0) | (category_vector >= class_weights.numel())
            ):
                raise ValueError(
                    "category_label lies outside loss.direct_rotation_class_weights")
            sample_weights = (
                sample_weights * class_weights[category_vector].clamp_min(0.0)
            )
        loss_direct = (per_sample_direct * sample_weights).sum() / sample_weights.sum().clamp_min(1.0e-6)

        loss_geo = zero
        loss_geo_asym = zero
        loss_geo_sym = zero
        if weight_geo > 0.0:
            with autocast(False):
                pred_rot_mat = six_d_to_rotation_matrix(pred_rotation_6d[valid].float())
                gt_rot_mat = gt_rot[valid].float()
                if bool(OmegaConf.select(self.cfg, "loss.direct_rotation_symmetry_aware", default=True)):
                    geo_losses = self._symmetry_aware_geodesic_losses(
                        pred_rot_mat, gt_rot_mat, categories_valid).to(dtype=zero.dtype)
                else:
                    rel_rot = torch.matmul(pred_rot_mat.transpose(1, 2), gt_rot_mat)
                    trace = rel_rot[:, 0, 0] + rel_rot[:, 1, 1] + rel_rot[:, 2, 2]
                    cos_theta = ((trace - 1.0) * 0.5).clamp(min=-1.0 + 1.0e-4, max=1.0 - 1.0e-4)
                    geo_losses = torch.acos(cos_theta).to(dtype=zero.dtype)
                geo_weights = sample_weights.to(
                    device=geo_losses.device, dtype=geo_losses.dtype)
                loss_geo = (geo_losses * geo_weights).sum() / geo_weights.sum().clamp_min(1.0e-6)
                if torch.any(asym_mask):
                    loss_geo_asym = geo_losses[asym_mask.to(device=geo_losses.device)].mean()
                if torch.any(~asym_mask):
                    loss_geo_sym = geo_losses[(~asym_mask).to(device=geo_losses.device)].mean()

        loss = loss_direct + weight_geo * loss_geo
        return loss * weight, {
            'loss/direct_rotation': float(loss_direct),
            'loss/direct_rotation_geo': float(loss_geo),
            'loss/direct_rotation_asym_geo': float(loss_geo_asym),
            'loss/direct_rotation_sym_geo': float(loss_geo_sym),
            'loss/direct_rotation_valid_ratio': float(valid.float().mean()),
        }

    def _compute_metric_depth_loss(self, end_points, real_data):
        weight_center = float(OmegaConf.select(self.cfg, "loss.weight_metric_depth_center", default=0.0))
        weight_points = float(OmegaConf.select(self.cfg, "loss.weight_metric_depth_points", default=0.0))
        pred_z = end_points.get('pred_metric_depth_z')
        pred_scale = end_points.get('pred_metric_depth_scale')
        zero = end_points['pred_size'].new_zeros(())
        empty = {
            'loss/metric_depth_center': 0.0,
            'loss/metric_depth_points': 0.0,
            'loss/metric_depth_point_valid_ratio': 0.0,
        }
        if (weight_center <= 0.0 and weight_points <= 0.0) or pred_z is None:
            return zero, empty

        beta = float(OmegaConf.select(self.cfg, "loss.metric_depth_beta", default=0.02))
        gt_center_z = real_data['translation_label'][:, 2].to(dtype=pred_z.dtype)
        valid_center = torch.isfinite(pred_z) & torch.isfinite(gt_center_z) & (gt_center_z > 0.0)
        if torch.any(valid_center):
            loss_center = F.smooth_l1_loss(pred_z[valid_center], gt_center_z[valid_center], beta=beta)
        else:
            loss_center = zero

        loss_points = zero
        point_valid_ratio = 0.0
        if weight_points > 0.0 and pred_scale is not None and 'metric_pts' in real_data:
            est_pts = real_data['pts'].to(dtype=pred_scale.dtype)
            metric_pts = real_data['metric_pts'].to(device=est_pts.device, dtype=pred_scale.dtype)
            valid_points = (
                torch.isfinite(est_pts[:, :, 2])
                & torch.isfinite(metric_pts[:, :, 2])
                & (est_pts[:, :, 2] > 0.0)
                & (metric_pts[:, :, 2] > 0.0)
            )
            if 'metric_depth_valid' in real_data:
                valid_points = valid_points & real_data['metric_depth_valid'].to(device=est_pts.device).bool()
            if torch.any(valid_points):
                pred_metric_z = pred_scale.view(-1, 1) * est_pts[:, :, 2]
                loss_points = F.smooth_l1_loss(
                    pred_metric_z[valid_points],
                    metric_pts[:, :, 2][valid_points],
                    beta=beta,
                )
                point_valid_ratio = float(valid_points.float().mean())

        loss = weight_center * loss_center + weight_points * loss_points
        return loss, {
            'loss/metric_depth_center': float(loss_center),
            'loss/metric_depth_points': float(loss_points),
            'loss/metric_depth_point_valid_ratio': point_valid_ratio,
        }

    def _compute_metric_size_loss(self, end_points, real_data):
        weight_log = float(OmegaConf.select(self.cfg, "loss.weight_metric_size_log", default=0.0))
        weight_l1 = float(OmegaConf.select(self.cfg, "loss.weight_metric_size_l1", default=0.0))
        pred_size = end_points.get('pred_metric_size')
        zero = end_points['pred_size'].new_zeros(())
        empty = {
            'loss/metric_size_log': 0.0,
            'loss/metric_size_l1': 0.0,
            'loss/metric_size_valid_ratio': 0.0,
        }
        if (weight_log <= 0.0 and weight_l1 <= 0.0) or pred_size is None:
            return zero, empty

        gt_size = real_data['size_label'].to(device=pred_size.device, dtype=pred_size.dtype)
        valid = (
            torch.isfinite(pred_size).all(dim=1)
            & torch.isfinite(gt_size).all(dim=1)
            & (pred_size > 0.0).all(dim=1)
            & (gt_size > 0.0).all(dim=1)
        )
        if not torch.any(valid):
            return zero, empty

        beta = float(OmegaConf.select(self.cfg, "loss.metric_size_beta", default=0.02))
        log_ratio = torch.log(pred_size[valid].clamp_min(1.0e-6)) - torch.log(gt_size[valid].clamp_min(1.0e-6))
        loss_log = F.smooth_l1_loss(log_ratio, torch.zeros_like(log_ratio), beta=beta)
        loss_l1 = F.smooth_l1_loss(pred_size[valid], gt_size[valid], beta=beta)
        loss = weight_log * loss_log + weight_l1 * loss_l1
        return loss, {
            'loss/metric_size_log': float(loss_log),
            'loss/metric_size_l1': float(loss_l1),
            'loss/metric_size_valid_ratio': float(valid.float().mean()),
        }

    def _compute_structured_rs_loss(self, end_points, real_data):
        pred_size = end_points.get('pred_structured_size')
        pred_rotation = end_points.get('pred_structured_rotation')
        zero = end_points['pred_size'].new_zeros(())
        empty = {
            'loss/structured_axis': 0.0,
            'loss/structured_asym_geodesic': 0.0,
            'loss/structured_size_log': 0.0,
            'loss/structured_size_l1': 0.0,
            'loss/structured_axis_extent': 0.0,
            'loss/structured_radial': 0.0,
            'loss/structured_corner': 0.0,
            'loss/structured_joint_ramp': 0.0,
            'loss/structured_valid_ratio': 0.0,
        }
        if pred_size is None or pred_rotation is None:
            return zero, empty

        # The structured R/S head runs in FP32 and its complete loss must do the
        # same. In particular, an FP16 clamp rounds 1 - 1e-4 back to 1 before
        # acos, which leaves a finite forward value but an infinite derivative.
        with autocast_disabled(pred_size.device):
            return self._compute_structured_rs_loss_fp32(
                pred_size=pred_size.float(),
                pred_rotation=pred_rotation.float(),
                real_data=real_data,
                empty=empty,
            )

    def _compute_structured_rs_loss_fp32(
        self,
        pred_size,
        pred_rotation,
        real_data,
        empty,
    ):
        zero = pred_size.new_zeros(())

        gt_size = real_data['size_label'].to(
            device=pred_size.device,
            dtype=torch.float32,
        )
        gt_rotation = real_data['rotation_label'].to(
            device=pred_rotation.device,
            dtype=torch.float32,
        )
        valid = (
            torch.isfinite(pred_size).all(dim=1)
            & torch.isfinite(gt_size).all(dim=1)
            & torch.isfinite(pred_rotation.reshape(pred_rotation.shape[0], -1)).all(dim=1)
            & torch.isfinite(gt_rotation.reshape(gt_rotation.shape[0], -1)).all(dim=1)
            & (pred_size > 0.0).all(dim=1)
            & (gt_size > 0.0).all(dim=1)
        )
        if not torch.any(valid):
            return zero, empty

        pred_size = pred_size[valid]
        gt_size = gt_size[valid]
        pred_rotation = pred_rotation[valid]
        gt_rotation = gt_rotation[valid]
        categories = real_data.get('category_label')
        categories = categories[valid] if categories is not None else None
        beta = float(
            OmegaConf.select(self.cfg, 'loss.structured_rs_beta', default=0.02)
        )

        pred_axis = pred_rotation[:, :, 1]
        gt_axis = gt_rotation[:, :, 1]
        axis_cosine = (pred_axis * gt_axis).sum(dim=1).clamp(-1.0, 1.0)
        loss_axis = (1.0 - axis_cosine).mean()

        asym_mask = self._asymmetric_direct_rotation_mask(
            categories,
            pred_rotation.shape[0],
            pred_rotation.device,
        )
        loss_asym_geo = zero
        if torch.any(asym_mask):
            rel_rotation = torch.matmul(
                pred_rotation[asym_mask].transpose(1, 2),
                gt_rotation[asym_mask],
            )
            trace = (
                rel_rotation[:, 0, 0]
                + rel_rotation[:, 1, 1]
                + rel_rotation[:, 2, 2]
            )
            cosine = ((trace - 1.0) * 0.5).clamp(
                min=-1.0 + 1.0e-4,
                max=1.0 - 1.0e-4,
            )
            loss_asym_geo = torch.acos(cosine).mean()

        log_size_error = torch.log(pred_size.clamp_min(1.0e-6)) - torch.log(
            gt_size.clamp_min(1.0e-6)
        )
        loss_size_log = F.smooth_l1_loss(
            log_size_error,
            torch.zeros_like(log_size_error),
            beta=beta,
        )
        loss_size_l1 = F.smooth_l1_loss(pred_size, gt_size, beta=beta)

        pred_axis_extent = pred_axis * pred_size[:, 1:2]
        gt_axis_extent = gt_axis * gt_size[:, 1:2]
        loss_axis_extent = F.smooth_l1_loss(
            pred_axis_extent,
            gt_axis_extent,
            beta=beta,
        )

        sym_mask = self._symmetric_mask(
            categories,
            pred_rotation.shape[0],
            pred_rotation.device,
        )
        loss_radial = zero
        if torch.any(sym_mask):
            pred_log = torch.log(pred_size[sym_mask].clamp_min(1.0e-6))
            gt_log = torch.log(gt_size[sym_mask].clamp_min(1.0e-6))
            pred_radius = 0.5 * (pred_log[:, 0] + pred_log[:, 2])
            gt_radius = 0.5 * (gt_log[:, 0] + gt_log[:, 2])
            pred_anisotropy = pred_log[:, 0] - pred_log[:, 2]
            gt_anisotropy = gt_log[:, 0] - gt_log[:, 2]
            loss_radial = F.smooth_l1_loss(
                pred_radius,
                gt_radius,
                beta=beta,
            ) + 0.5 * F.smooth_l1_loss(
                pred_anisotropy,
                gt_anisotropy,
                beta=beta,
            )

        loss_corner = zero
        if torch.any(asym_mask):
            canonical = pred_size.new_tensor(
                [
                    [-0.5, -0.5, -0.5],
                    [-0.5, -0.5, 0.5],
                    [-0.5, 0.5, -0.5],
                    [-0.5, 0.5, 0.5],
                    [0.5, -0.5, -0.5],
                    [0.5, -0.5, 0.5],
                    [0.5, 0.5, -0.5],
                    [0.5, 0.5, 0.5],
                ]
            )
            pred_object_corners = (
                canonical.unsqueeze(0) * pred_size[asym_mask].unsqueeze(1)
            )
            gt_object_corners = (
                canonical.unsqueeze(0) * gt_size[asym_mask].unsqueeze(1)
            )
            pred_corners = torch.matmul(
                pred_object_corners,
                pred_rotation[asym_mask].transpose(1, 2),
            )
            gt_corners = torch.matmul(
                gt_object_corners,
                gt_rotation[asym_mask].transpose(1, 2),
            )
            loss_corner = F.smooth_l1_loss(
                pred_corners,
                gt_corners,
                beta=beta,
            )

        ramp_start = int(
            OmegaConf.select(
                self.cfg,
                'loss.structured_rs_joint_ramp_start_epoch',
                default=10,
            )
        )
        ramp_end = int(
            OmegaConf.select(
                self.cfg,
                'loss.structured_rs_joint_ramp_end_epoch',
                default=20,
            )
        )
        epoch = int(getattr(self, 'epoch', 0))
        if epoch < ramp_start:
            joint_ramp = 0.0
        elif ramp_end <= ramp_start or epoch >= ramp_end:
            joint_ramp = 1.0
        else:
            joint_ramp = float(epoch - ramp_start) / float(ramp_end - ramp_start)

        weighted = (
            float(OmegaConf.select(self.cfg, 'loss.weight_structured_axis', default=0.0))
            * loss_axis
            + float(OmegaConf.select(self.cfg, 'loss.weight_structured_asym_geodesic', default=0.0))
            * loss_asym_geo
            + float(OmegaConf.select(self.cfg, 'loss.weight_structured_size_log', default=0.0))
            * loss_size_log
            + float(OmegaConf.select(self.cfg, 'loss.weight_structured_size_l1', default=0.0))
            * loss_size_l1
            + joint_ramp
            * (
                float(OmegaConf.select(self.cfg, 'loss.weight_structured_axis_extent', default=0.0))
                * loss_axis_extent
                + float(OmegaConf.select(self.cfg, 'loss.weight_structured_radial', default=0.0))
                * loss_radial
                + float(OmegaConf.select(self.cfg, 'loss.weight_structured_corner', default=0.0))
                * loss_corner
            )
        )
        return weighted, {
            'loss/structured_axis': float(loss_axis),
            'loss/structured_asym_geodesic': float(loss_asym_geo),
            'loss/structured_size_log': float(loss_size_log),
            'loss/structured_size_l1': float(loss_size_l1),
            'loss/structured_axis_extent': float(loss_axis_extent),
            'loss/structured_radial': float(loss_radial),
            'loss/structured_corner': float(loss_corner),
            'loss/structured_joint_ramp': float(joint_ramp),
            'loss/structured_valid_ratio': float(valid.float().mean()),
        }

    def _compute_rotation_v6_loss(self, end_points, real_data):
        hypotheses = end_points.get('pred_rotation_v6_hypotheses')
        base_rotation = end_points.get('pred_rotation_v6_base')
        logits = end_points.get('pred_rotation_v6_logits')
        residuals = end_points.get('pred_rotation_v6_axis_angle')
        zero = end_points['pred_size'].new_zeros(())
        empty = {
            'loss/rotation_v6_geodesic': 0.0,
            'loss/rotation_v6_improvement': 0.0,
            'loss/rotation_v6_confidence': 0.0,
            'loss/rotation_v6_delta_regularize': 0.0,
            'loss/rotation_v6_valid_ratio': 0.0,
            'rotation/rotation_v6_base_error_deg': 0.0,
            'rotation/rotation_v6_best_error_deg': 0.0,
            'rotation/rotation_v6_selected_error_deg': 0.0,
            'rotation/rotation_v6_improved_ratio': 0.0,
            'rotation/rotation_v6_identity_ratio': 0.0,
            'rotation/rotation_v6_hard_ratio': 0.0,
            'rotation/rotation_v6_residual_deg': 0.0,
        }
        weights_cfg = {
            'geo': float(
                OmegaConf.select(
                    self.cfg,
                    'loss.weight_rotation_v6_geodesic',
                    default=0.0,
                )
            ),
            'improvement': float(
                OmegaConf.select(
                    self.cfg,
                    'loss.weight_rotation_v6_improvement',
                    default=0.0,
                )
            ),
            'confidence': float(
                OmegaConf.select(
                    self.cfg,
                    'loss.weight_rotation_v6_confidence',
                    default=0.0,
                )
            ),
            'delta': float(
                OmegaConf.select(
                    self.cfg,
                    'loss.weight_rotation_v6_delta_regularize',
                    default=0.0,
                )
            ),
        }
        if (
            hypotheses is None
            or base_rotation is None
            or logits is None
            or residuals is None
            or max(weights_cfg.values()) <= 0.0
        ):
            return zero, empty

        gt_rotation = real_data['rotation_label'].to(
            device=hypotheses.device,
            dtype=torch.float32,
        )
        batch_size, num_hypotheses = hypotheses.shape[:2]
        if num_hypotheses < 2:
            raise ValueError('rotation_v6 requires identity plus a learned hypothesis')
        categories = self._category_vector(
            real_data.get('category_label'),
            batch_size,
            hypotheses.device,
        )
        valid = (
            torch.isfinite(hypotheses.reshape(batch_size, -1)).all(dim=1)
            & torch.isfinite(base_rotation.reshape(batch_size, -1)).all(dim=1)
            & torch.isfinite(gt_rotation.reshape(batch_size, -1)).all(dim=1)
            & torch.isfinite(logits).all(dim=1)
            & torch.isfinite(residuals.reshape(batch_size, -1)).all(dim=1)
        )
        if not torch.any(valid):
            return zero, empty

        with autocast_disabled(hypotheses.device):
            hypotheses = hypotheses[valid].float()
            base_rotation = base_rotation[valid].float()
            gt_rotation = gt_rotation[valid]
            logits = logits[valid].float()
            residuals = residuals[valid].float()
            categories = categories[valid]
            valid_count = hypotheses.shape[0]

            repeated_gt = gt_rotation.unsqueeze(1).expand(
                -1, num_hypotheses, -1, -1
            ).reshape(-1, 3, 3)
            repeated_categories = categories.unsqueeze(1).expand(
                -1, num_hypotheses
            ).reshape(-1)
            all_errors = self._symmetry_aware_geodesic_losses(
                hypotheses.reshape(-1, 3, 3),
                repeated_gt,
                repeated_categories,
            ).reshape(valid_count, num_hypotheses)
            base_errors = all_errors[:, 0]
            learned_errors = all_errors[:, 1:]
            best_learned_errors, best_learned_index = learned_errors.min(dim=1)

            sample_weights = torch.ones_like(best_learned_errors)
            hard_mask = torch.zeros_like(best_learned_errors, dtype=torch.bool)
            pred_translation = end_points.get('pred_translation')
            gt_translation = real_data.get('translation_label')
            if pred_translation is not None and gt_translation is not None:
                pred_translation = pred_translation[valid].float()
                gt_translation = gt_translation.to(
                    device=pred_translation.device,
                    dtype=torch.float32,
                )[valid]
                translation_error_cm = torch.linalg.vector_norm(
                    pred_translation - gt_translation,
                    dim=1,
                ) * 100.0
                base_error_deg = torch.rad2deg(base_errors)
                hard_mask = (
                    (translation_error_cm <= float(
                        OmegaConf.select(
                            self.cfg,
                            'loss.rotation_v6_hard_translation_cm',
                            default=10.0,
                        )
                    ))
                    & (base_error_deg >= float(
                        OmegaConf.select(
                            self.cfg,
                            'loss.rotation_v6_hard_min_deg',
                            default=8.0,
                        )
                    ))
                    & (base_error_deg <= float(
                        OmegaConf.select(
                            self.cfg,
                            'loss.rotation_v6_hard_max_deg',
                            default=25.0,
                        )
                    ))
                )
                hard_weight = float(
                    OmegaConf.select(
                        self.cfg,
                        'loss.rotation_v6_hard_weight',
                        default=2.0,
                    )
                )
                sample_weights = torch.where(
                    hard_mask,
                    sample_weights.new_full((), hard_weight),
                    sample_weights,
                )

            hypothesis_temperature = math.radians(
                max(
                    float(
                        OmegaConf.select(
                            self.cfg,
                            'loss.rotation_v6_hypothesis_temperature_deg',
                            default=5.0,
                        )
                    ),
                    0.1,
                )
            )
            hypothesis_weights = torch.softmax(
                -learned_errors.detach() / hypothesis_temperature,
                dim=1,
            )
            per_sample_geo = (
                hypothesis_weights * learned_errors
            ).sum(dim=1)
            loss_geo = (
                per_sample_geo * sample_weights
            ).sum() / sample_weights.sum().clamp_min(1.0)
            margin = math.radians(
                float(
                    OmegaConf.select(
                        self.cfg,
                        'loss.rotation_v6_improvement_margin_deg',
                        default=0.0,
                    )
                )
            )
            loss_improvement = (
                F.relu(best_learned_errors - base_errors.detach() + margin)
                * sample_weights
            ).sum() / sample_weights.sum().clamp_min(1.0)

            confidence_temperature = math.radians(
                max(
                    float(
                        OmegaConf.select(
                            self.cfg,
                            'loss.rotation_v6_confidence_temperature_deg',
                            default=5.0,
                        )
                    ),
                    0.1,
                )
            )
            target_probabilities = torch.softmax(
                -all_errors.detach() / confidence_temperature,
                dim=1,
            )
            loss_confidence = F.kl_div(
                F.log_softmax(logits, dim=1),
                target_probabilities,
                reduction='batchmean',
            )
            learned_residual_norm = torch.linalg.vector_norm(
                residuals[:, 1:], dim=-1
            )
            loss_delta = learned_residual_norm.square().mean()

            selected_index = logits.argmax(dim=1)
            selected_errors = all_errors.gather(
                1, selected_index.unsqueeze(1)
            ).squeeze(1)
            weighted = (
                weights_cfg['geo'] * loss_geo
                + weights_cfg['improvement'] * loss_improvement
                + weights_cfg['confidence'] * loss_confidence
                + weights_cfg['delta'] * loss_delta
            )

        radians_to_degrees = 180.0 / math.pi
        return weighted.to(dtype=zero.dtype), {
            'loss/rotation_v6_geodesic': float(loss_geo),
            'loss/rotation_v6_improvement': float(loss_improvement),
            'loss/rotation_v6_confidence': float(loss_confidence),
            'loss/rotation_v6_delta_regularize': float(loss_delta),
            'loss/rotation_v6_valid_ratio': float(valid.float().mean()),
            'rotation/rotation_v6_base_error_deg': float(
                base_errors.mean() * radians_to_degrees
            ),
            'rotation/rotation_v6_best_error_deg': float(
                best_learned_errors.mean() * radians_to_degrees
            ),
            'rotation/rotation_v6_selected_error_deg': float(
                selected_errors.mean() * radians_to_degrees
            ),
            'rotation/rotation_v6_improved_ratio': float(
                (best_learned_errors < base_errors).float().mean()
            ),
            'rotation/rotation_v6_identity_ratio': float(
                (selected_index == 0).float().mean()
            ),
            'rotation/rotation_v6_hard_ratio': float(hard_mask.float().mean()),
            'rotation/rotation_v6_residual_deg': float(
                learned_residual_norm.mean() * radians_to_degrees
            ),
        }

    def _compute_e2e_pose_aux_loss(self, end_points, real_data):
        return compute_e2e_pose_aux_loss(self.cfg, end_points, real_data)

    def _compute_rotation_correspondence_v9_loss(self, end_points, real_data):
        pred_rotation = end_points.get('pred_rotation_v9')
        pred_nocs = end_points.get('pred_rotation_v9_nocs')
        confidence_logits = end_points.get(
            'pred_rotation_v9_confidence_logits'
        )
        point_valid_output = end_points.get('pred_rotation_v9_point_valid')
        zero = end_points['pred_size'].new_zeros(())
        empty = {
            'loss/rotation_v9_nocs': 0.0,
            'loss/rotation_v9_geodesic': 0.0,
            'loss/rotation_v9_axis': 0.0,
            'loss/rotation_v9_confidence': 0.0,
            'loss/rotation_v9_valid_ratio': 0.0,
            'rotation/rotation_v9_error_deg': 0.0,
            'rotation/rotation_v9_symmetric_axis_error_deg': 0.0,
            'rotation/rotation_v9_confidence_mean': 0.0,
        }
        weights = {
            'nocs': float(OmegaConf.select(
                self.cfg,
                'loss.weight_rotation_v9_nocs',
                default=0.0,
            )),
            'geodesic': float(OmegaConf.select(
                self.cfg,
                'loss.weight_rotation_v9_geodesic',
                default=0.0,
            )),
            'axis': float(OmegaConf.select(
                self.cfg,
                'loss.weight_rotation_v9_axis',
                default=0.0,
            )),
            'confidence': float(OmegaConf.select(
                self.cfg,
                'loss.weight_rotation_v9_confidence',
                default=0.0,
            )),
        }
        if weights['geodesic'] != 0.0 or weights['axis'] != 0.0:
            raise ValueError(
                'V9 Kabsch geodesic/axis terms are monitoring-only because '
                'SVD backward is unstable; set '
                'loss.weight_rotation_v9_geodesic=0 and '
                'loss.weight_rotation_v9_axis=0'
            )
        if (
            pred_rotation is None
            or pred_nocs is None
            or max(weights.values()) <= 0.0
        ):
            return zero, empty

        pts = real_data.get('pts_metric', real_data['pts']).to(
            device=pred_nocs.device,
            dtype=pred_nocs.dtype,
        )
        gt_rotation = real_data['rotation_label'].to(
            device=pred_rotation.device,
            dtype=pred_rotation.dtype,
        )
        gt_translation = real_data['translation_label'].to(
            device=pred_nocs.device,
            dtype=pred_nocs.dtype,
        )
        gt_size = real_data['size_label'].to(
            device=pred_nocs.device,
            dtype=pred_nocs.dtype,
        ).clamp_min(1.0e-4)
        categories = self._category_vector(
            real_data.get('category_label'),
            pred_nocs.shape[0],
            pred_nocs.device,
        )

        valid = (
            torch.isfinite(pred_nocs).all(dim=-1)
            & torch.isfinite(pts).all(dim=-1)
            & (pts[..., 2] > 0.0)
        )
        if 'pts_metric_valid' in real_data:
            valid = valid & real_data['pts_metric_valid'].to(
                device=valid.device
            ).bool()
        if point_valid_output is not None:
            valid = valid & point_valid_output.to(device=valid.device).bool()
        finite_sample = (
            (valid.sum(dim=1) >= 3)
            & torch.isfinite(gt_rotation).flatten(1).all(dim=1)
            & torch.isfinite(gt_translation).all(dim=1)
            & torch.isfinite(gt_size).all(dim=1)
            & torch.isfinite(pred_rotation).flatten(1).all(dim=1)
        )
        if not torch.any(finite_sample):
            return zero, empty

        gt_nocs = inverse_transform_points(
            pts,
            gt_rotation.to(dtype=pts.dtype),
            gt_translation,
            gt_size,
        )
        if bool(OmegaConf.select(
            self.cfg,
            'loss.rotation_v9_symmetry_aware',
            default=True,
        )):
            gt_nocs = self._select_symmetry_nocs_target(
                pred_nocs,
                pts,
                gt_size,
                gt_rotation.to(dtype=pts.dtype),
                gt_translation,
                real_data.get('category_label'),
            )

        def balanced_mean(values, value_categories):
            present = torch.unique(value_categories)
            per_class = [
                values[value_categories == class_id].mean()
                for class_id in present
                if torch.any(value_categories == class_id)
            ]
            return torch.stack(per_class).mean() if per_class else values.mean()

        beta = float(OmegaConf.select(
            self.cfg,
            'loss.rotation_v9_nocs_beta',
            default=0.02,
        ))
        point_nocs_loss = F.smooth_l1_loss(
            pred_nocs,
            gt_nocs,
            beta=beta,
            reduction='none',
        ).mean(dim=-1)
        sample_nocs_loss = (
            (point_nocs_loss * valid.to(dtype=point_nocs_loss.dtype)).sum(dim=1)
            / valid.sum(dim=1).clamp_min(1).to(dtype=point_nocs_loss.dtype)
        )
        loss_nocs = balanced_mean(
            sample_nocs_loss[finite_sample],
            categories[finite_sample],
        )

        # Weighted Kabsch uses an SVD.  Its forward result is appropriate for
        # inference and monitoring, but SVD backward is undefined when two
        # singular values coincide and is numerically unstable near that
        # boundary.  Real object point sets hit that boundary often enough to
        # poison the fresh head with NaN gradients.  V9 therefore learns the
        # absolute correspondences directly; the analytic Kabsch rotation is
        # detached and remains a metric/output only.
        monitored_rotation = pred_rotation.detach().float()
        rotation_errors = self._symmetry_aware_geodesic_losses(
            monitored_rotation[finite_sample],
            gt_rotation[finite_sample].float(),
            categories[finite_sample],
        )
        monitored_geodesic = balanced_mean(
            rotation_errors,
            categories[finite_sample],
        ).to(dtype=zero.dtype)

        symmetric_mask = self._symmetric_mask(
            categories,
            monitored_rotation.shape[0],
            monitored_rotation.device,
        ) & finite_sample
        monitored_axis = zero
        axis_error_mean = 0.0
        if torch.any(symmetric_mask):
            pred_axis = F.normalize(
                monitored_rotation[symmetric_mask, :, 1],
                dim=-1,
                eps=1.0e-6,
            )
            gt_axis = F.normalize(
                gt_rotation[symmetric_mask, :, 1].float(),
                dim=-1,
                eps=1.0e-6,
            )
            axis_cos = (pred_axis * gt_axis).sum(dim=-1).clamp(
                min=-1.0 + 1.0e-4,
                max=1.0 - 1.0e-4,
            )
            axis_errors = torch.acos(axis_cos)
            monitored_axis = balanced_mean(
                axis_errors,
                categories[symmetric_mask],
            ).to(dtype=zero.dtype)
            axis_error_mean = float(torch.rad2deg(axis_errors).mean())

        loss_confidence = zero
        confidence_mean = 0.0
        if confidence_logits is not None and torch.any(valid):
            confidence_tau = float(OmegaConf.select(
                self.cfg,
                'loss.rotation_v9_confidence_tau',
                default=0.08,
            ))
            point_error = torch.linalg.vector_norm(
                pred_nocs.detach() - gt_nocs.detach(),
                dim=-1,
            )
            confidence_target = torch.exp(
                -point_error / max(confidence_tau, 1.0e-4)
            ).clamp(0.0, 1.0)
            confidence_loss_points = F.binary_cross_entropy_with_logits(
                confidence_logits[valid],
                confidence_target[valid],
                reduction='none',
            )
            loss_confidence = confidence_loss_points.mean().to(dtype=zero.dtype)
            confidence_mean = float(
                torch.sigmoid(confidence_logits[valid]).mean()
            )

        weighted = (
            weights['nocs'] * loss_nocs
            + weights['confidence'] * loss_confidence
        )
        return weighted.to(dtype=zero.dtype), {
            'loss/rotation_v9_nocs': float(loss_nocs),
            'loss/rotation_v9_geodesic': float(monitored_geodesic),
            'loss/rotation_v9_axis': float(monitored_axis),
            'loss/rotation_v9_confidence': float(loss_confidence),
            'loss/rotation_v9_valid_ratio': float(valid.float().mean()),
            'rotation/rotation_v9_error_deg': float(
                torch.rad2deg(rotation_errors).mean()
            ),
            'rotation/rotation_v9_symmetric_axis_error_deg': axis_error_mean,
            'rotation/rotation_v9_confidence_mean': confidence_mean,
        }

    def _compute_translation_anchor_loss(self, end_points, real_data):
        weight = float(
            OmegaConf.select(
                self.cfg,
                "loss.weight_translation_anchor",
                default=0.0,
            )
        )
        pred_anchor = end_points.get('pred_translation_anchor')
        zero = end_points['pred_size'].new_zeros(())
        empty = {
            'loss/translation_anchor': 0.0,
            'loss/translation_anchor_valid_ratio': 0.0,
        }
        if weight <= 0.0 or pred_anchor is None:
            return zero, empty

        target = real_data['translation_label'].to(
            device=pred_anchor.device,
            dtype=pred_anchor.dtype,
        )
        valid = (
            torch.isfinite(pred_anchor).all(dim=1)
            & torch.isfinite(target).all(dim=1)
        )
        if not torch.any(valid):
            return zero, empty

        beta = float(
            OmegaConf.select(
                self.cfg,
                "loss.translation_anchor_beta",
                default=0.02,
            )
        )
        loss = F.smooth_l1_loss(
            pred_anchor[valid],
            target[valid],
            beta=beta,
        )
        return weight * loss, {
            'loss/translation_anchor': float(loss),
            'loss/translation_anchor_valid_ratio': float(valid.float().mean()),
        }

    def _compute_translation_vote_loss(self, end_points, real_data):
        """Supervise every valid point vote against the GT object center."""
        weight_uv = float(
            OmegaConf.select(
                self.cfg,
                "loss.weight_translation_vote_center_uv",
                default=0.0,
            )
        )
        weight_log_z = float(
            OmegaConf.select(
                self.cfg,
                "loss.weight_translation_vote_log_z",
                default=0.0,
            )
        )
        weight_uncertainty = float(
            OmegaConf.select(
                self.cfg,
                "loss.weight_translation_vote_uncertainty",
                default=0.0,
            )
        )
        weight_fallback = float(
            OmegaConf.select(
                self.cfg,
                "loss.weight_translation_fallback_regularize",
                default=0.0,
            )
        )
        zero = end_points['pred_size'].new_zeros(())
        empty = {
            'loss/translation_vote_center_uv': 0.0,
            'loss/translation_vote_log_z': 0.0,
            'loss/translation_vote_uncertainty': 0.0,
            'loss/translation_fallback_regularize': 0.0,
            'translation/fallback_weight_mean': 0.0,
            'translation/vote_confidence_uv_mean': 0.0,
            'translation/vote_confidence_z_mean': 0.0,
            'loss/translation_vote_valid_ratio': 0.0,
        }
        if (
            weight_uv <= 0.0
            and weight_log_z <= 0.0
            and weight_uncertainty <= 0.0
            and weight_fallback <= 0.0
        ):
            return zero, empty
        required = {
            'pred_translation_vote_uv',
            'pred_translation_vote_log_depth',
            'pred_translation_vote_valid_mask',
            'translation_cam_k',
            'translation_bbox_wh',
        }
        if not required.issubset(end_points):
            return zero, empty

        vote_uv = end_points['pred_translation_vote_uv']
        vote_log_z = end_points['pred_translation_vote_log_depth']
        vote_valid = end_points['pred_translation_vote_valid_mask'].to(
            device=vote_uv.device,
        ).bool()
        gt_translation = real_data['translation_label'].to(
            device=vote_uv.device,
            dtype=vote_uv.dtype,
        )
        cam_k = end_points['translation_cam_k'].to(
            device=vote_uv.device,
            dtype=vote_uv.dtype,
        )
        bbox_wh = end_points['translation_bbox_wh'].to(
            device=vote_uv.device,
            dtype=vote_uv.dtype,
        ).clamp_min(1.0)
        sample_valid = (
            torch.isfinite(gt_translation).all(dim=1)
            & (gt_translation[:, 2] > 1.0e-6)
            & torch.isfinite(cam_k).all(dim=1)
            & torch.isfinite(bbox_wh).all(dim=1)
        )
        valid = (
            vote_valid
            & sample_valid.unsqueeze(1)
            & torch.isfinite(vote_uv).all(dim=-1)
            & torch.isfinite(vote_log_z)
        )
        if not torch.any(valid):
            return zero, empty

        gt_uv = project_translation_to_image(gt_translation, cam_k)
        uv_error = (
            vote_uv - gt_uv.unsqueeze(1)
        ) / bbox_wh.unsqueeze(1)
        loss_uv = F.smooth_l1_loss(
            uv_error[valid],
            torch.zeros_like(uv_error[valid]),
            beta=float(
                OmegaConf.select(
                    self.cfg,
                    "loss.translation_vote_center_beta",
                    default=0.05,
                )
            ),
        )
        target_log_z = torch.log(
            gt_translation[:, 2].clamp_min(1.0e-6)
        ).unsqueeze(1)
        log_z_error = vote_log_z - target_log_z
        loss_log_z = F.smooth_l1_loss(
            log_z_error[valid],
            torch.zeros_like(log_z_error[valid]),
            beta=float(
                OmegaConf.select(
                    self.cfg,
                    "loss.translation_vote_log_z_beta",
                    default=0.02,
                )
            ),
        )
        loss_uncertainty = zero
        confidence_uv_mean = zero
        confidence_z_mean = zero
        vote_log_variance = end_points.get(
            'pred_translation_vote_log_variance'
        )
        if weight_uncertainty > 0.0:
            if vote_log_variance is None:
                raise KeyError(
                    "translation vote uncertainty loss requires "
                    "pred_translation_vote_log_variance"
                )
            vote_log_variance = vote_log_variance.to(
                device=vote_uv.device,
                dtype=vote_uv.dtype,
            )
            if vote_log_variance.shape != (*vote_uv.shape[:2], 2):
                raise ValueError(
                    "pred_translation_vote_log_variance must have shape [B,N,2]"
                )
            log_var_uv = vote_log_variance[:, :, 0]
            log_var_z = vote_log_variance[:, :, 1]
            uv_cost = F.smooth_l1_loss(
                uv_error,
                torch.zeros_like(uv_error),
                beta=float(
                    OmegaConf.select(
                        self.cfg,
                        "loss.translation_vote_center_beta",
                        default=0.05,
                    )
                ),
                reduction='none',
            ).mean(dim=-1)
            log_z_cost = F.smooth_l1_loss(
                log_z_error,
                torch.zeros_like(log_z_error),
                beta=float(
                    OmegaConf.select(
                        self.cfg,
                        "loss.translation_vote_log_z_beta",
                        default=0.02,
                    )
                ),
                reduction='none',
            )
            uncertainty_cost = (
                torch.exp(-log_var_uv) * uv_cost
                + 0.5 * log_var_uv
                + torch.exp(-log_var_z) * log_z_cost
                + 0.5 * log_var_z
            )
            loss_uncertainty = uncertainty_cost[valid].mean()
            confidence_uv_mean = torch.sigmoid(-log_var_uv[valid]).mean()
            confidence_z_mean = torch.sigmoid(-log_var_z[valid]).mean()

        fallback_weight = end_points.get('pred_translation_fallback_weight')
        loss_fallback = zero
        fallback_weight_mean = zero
        if fallback_weight is not None:
            fallback_weight = fallback_weight.to(
                device=vote_uv.device,
                dtype=vote_uv.dtype,
            )
            fallback_weight_mean = fallback_weight.mean()
            loss_fallback = fallback_weight_mean

        loss = (
            weight_uv * loss_uv
            + weight_log_z * loss_log_z
            + weight_uncertainty * loss_uncertainty
            + weight_fallback * loss_fallback
        )
        return loss, {
            'loss/translation_vote_center_uv': float(loss_uv),
            'loss/translation_vote_log_z': float(loss_log_z),
            'loss/translation_vote_uncertainty': float(loss_uncertainty),
            'loss/translation_fallback_regularize': float(loss_fallback),
            'translation/fallback_weight_mean': float(fallback_weight_mean),
            'translation/vote_confidence_uv_mean': float(confidence_uv_mean),
            'translation/vote_confidence_z_mean': float(confidence_z_mean),
            'loss/translation_vote_valid_ratio': float(valid.float().mean()),
        }

    def _compute_center_z_v4_loss(self, end_points, real_data):
        """Directly supervise the V4 object-center metric depth correction."""
        weight_metric = float(
            OmegaConf.select(
                self.cfg,
                "loss.weight_center_z_v4_metric_z",
                default=0.0,
            )
        )
        weight_log = float(
            OmegaConf.select(
                self.cfg,
                "loss.weight_center_z_v4_log_z",
                default=0.0,
            )
        )
        weight_improvement = float(
            OmegaConf.select(
                self.cfg,
                "loss.weight_center_z_v4_improvement",
                default=0.0,
            )
        )
        weight_delta = float(
            OmegaConf.select(
                self.cfg,
                "loss.weight_center_z_v4_delta_regularize",
                default=0.0,
            )
        )
        zero = end_points['pred_size'].new_zeros(())
        empty = {
            'loss/center_z_v4_metric_z': 0.0,
            'loss/center_z_v4_log_z': 0.0,
            'loss/center_z_v4_improvement': 0.0,
            'loss/center_z_v4_delta_regularize': 0.0,
            'loss/center_z_v4_valid_ratio': 0.0,
            'translation/center_z_v4_mae_cm': 0.0,
            'translation/center_z_v4_base_mae_cm': 0.0,
            'translation/center_z_v4_improved_ratio': 0.0,
            'translation/center_z_v4_delta_log_z_mean': 0.0,
        }
        if max(weight_metric, weight_log, weight_improvement, weight_delta) <= 0.0:
            return zero, empty

        required = {
            'pred_center_z_v4_depth',
            'pred_center_z_v4_base_depth',
            'pred_center_z_v4_delta_log_depth',
        }
        if not required.issubset(end_points):
            if end_points.get('translation_prediction_mode') == 'ray_depth_v4_center_z':
                missing = sorted(required.difference(end_points))
                raise KeyError(f"V4 Center-Z loss missing outputs: {missing}")
            return zero, empty

        pred_z = end_points['pred_center_z_v4_depth'].view(-1)
        base_z = end_points['pred_center_z_v4_base_depth'].to(
            device=pred_z.device,
            dtype=pred_z.dtype,
        ).view(-1)
        delta_log_z = end_points['pred_center_z_v4_delta_log_depth'].to(
            device=pred_z.device,
            dtype=pred_z.dtype,
        ).view(-1)
        gt_translation = real_data['translation_label'].to(
            device=pred_z.device,
            dtype=pred_z.dtype,
        )
        gt_z = gt_translation[:, 2].view(-1)
        valid = (
            torch.isfinite(pred_z)
            & torch.isfinite(base_z)
            & torch.isfinite(delta_log_z)
            & torch.isfinite(gt_z)
            & (pred_z > 1.0e-6)
            & (base_z > 1.0e-6)
            & (gt_z > 1.0e-6)
        )
        if not torch.any(valid):
            return zero, empty

        beta = float(
            OmegaConf.select(
                self.cfg,
                "loss.center_z_v4_beta",
                default=0.02,
            )
        )
        pred_valid = pred_z[valid]
        base_valid = base_z[valid]
        gt_valid = gt_z[valid]
        delta_valid = delta_log_z[valid]
        loss_metric = F.smooth_l1_loss(
            pred_valid,
            gt_valid,
            beta=beta,
        )
        loss_log = F.smooth_l1_loss(
            torch.log(pred_valid.clamp_min(1.0e-6)),
            torch.log(gt_valid.clamp_min(1.0e-6)),
            beta=beta,
        )
        pred_abs = (pred_valid - gt_valid).abs()
        base_abs = (base_valid - gt_valid).abs()
        margin = float(
            OmegaConf.select(
                self.cfg,
                "loss.center_z_v4_improvement_margin",
                default=0.0,
            )
        )
        loss_improvement = F.relu(pred_abs - base_abs + margin).mean()
        loss_delta = delta_valid.abs().mean()
        loss = (
            weight_metric * loss_metric
            + weight_log * loss_log
            + weight_improvement * loss_improvement
            + weight_delta * loss_delta
        )
        return loss, {
            'loss/center_z_v4_metric_z': float(loss_metric),
            'loss/center_z_v4_log_z': float(loss_log),
            'loss/center_z_v4_improvement': float(loss_improvement),
            'loss/center_z_v4_delta_regularize': float(loss_delta),
            'loss/center_z_v4_valid_ratio': float(valid.float().mean()),
            'translation/center_z_v4_mae_cm': float(pred_abs.mean() * 100.0),
            'translation/center_z_v4_base_mae_cm': float(base_abs.mean() * 100.0),
            'translation/center_z_v4_improved_ratio': float(
                (pred_abs < base_abs).float().mean()
            ),
            'translation/center_z_v4_delta_log_z_mean': float(delta_valid.mean()),
        }

    @staticmethod
    def _category_source_balanced_mean(values, categories, sources):
        """Average sources within each class, then average present classes."""
        category_means = []
        for category_id in torch.unique(categories):
            category_mask = categories == category_id
            source_means = []
            for source_id in torch.unique(sources[category_mask]):
                group_mask = category_mask & (sources == source_id)
                if torch.any(group_mask):
                    source_means.append(values[group_mask].mean())
            if source_means:
                category_means.append(torch.stack(source_means).mean())
        if not category_means:
            return values.new_zeros(())
        return torch.stack(category_means).mean()

    @staticmethod
    def _source_balanced_mean(values, sources):
        source_means = []
        for source_id in torch.unique(sources):
            source_mask = sources == source_id
            if torch.any(source_mask):
                source_means.append(values[source_mask].mean())
        if not source_means:
            return values.new_zeros(())
        return torch.stack(source_means).mean()

    def _compute_absolute_center_translation_loss(self, end_points, real_data):
        """Supervise amodal center reconstruction in metric and canonical space."""
        zero = end_points['pred_size'].new_zeros(())
        empty = {
            'loss/absolute_center_xyz': 0.0,
            'loss/absolute_center_z': 0.0,
            'loss/absolute_center_canonical': 0.0,
            'loss/absolute_center_point_vote': 0.0,
            'loss/absolute_center_query_vote': 0.0,
            'loss/absolute_center_size_normalized': 0.0,
            'loss/absolute_center_confidence': 0.0,
            'loss/absolute_center_entropy': 0.0,
            'loss/absolute_center_valid_ratio': 0.0,
            'translation/absolute_center_mae_cm': 0.0,
            'translation/absolute_center_balanced_mae_cm': 0.0,
            'translation/absolute_center_x_mae_cm': 0.0,
            'translation/absolute_center_y_mae_cm': 0.0,
            'translation/absolute_center_z_mae_cm': 0.0,
            'translation/absolute_center_under_10cm': 0.0,
            'translation/absolute_center_base_mae_cm': 0.0,
            'translation/absolute_center_improved_ratio': 0.0,
            'translation/absolute_center_fallback_ratio': 0.0,
            'translation/absolute_center_point_entropy': 0.0,
            'translation/absolute_center_query_entropy': 0.0,
        }
        weights = {
            'xyz': float(OmegaConf.select(
                self.cfg, 'loss.weight_absolute_center_xyz', default=0.0
            )),
            'z': float(OmegaConf.select(
                self.cfg, 'loss.weight_absolute_center_z', default=0.0
            )),
            'canonical': float(OmegaConf.select(
                self.cfg, 'loss.weight_absolute_center_canonical', default=0.0
            )),
            'point_vote': float(OmegaConf.select(
                self.cfg, 'loss.weight_absolute_center_point_vote', default=0.0
            )),
            'query_vote': float(OmegaConf.select(
                self.cfg, 'loss.weight_absolute_center_query_vote', default=0.0
            )),
            'size_normalized': float(OmegaConf.select(
                self.cfg,
                'loss.weight_absolute_center_size_normalized',
                default=0.0,
            )),
            'confidence': float(OmegaConf.select(
                self.cfg,
                'loss.weight_absolute_center_confidence',
                default=0.0,
            )),
            'entropy': float(OmegaConf.select(
                self.cfg, 'loss.weight_absolute_center_entropy', default=0.0
            )),
        }
        pred_translation = end_points.get(
            'pred_absolute_center_translation_raw'
        )
        surface_center = end_points.get('pred_absolute_center_surface_center')
        pred_canonical = end_points.get(
            'pred_absolute_center_canonical_offset'
        )
        point_offsets = end_points.get('pred_absolute_center_point_offsets')
        point_votes = end_points.get('pred_absolute_center_point_votes')
        query_offsets = end_points.get('pred_absolute_center_query_offsets')
        point_weights = end_points.get('pred_absolute_center_point_weights')
        query_weights = end_points.get('pred_absolute_center_query_weights')
        point_valid = end_points.get('pred_absolute_center_point_valid')
        if (
            pred_translation is None
            or surface_center is None
            or pred_canonical is None
            or point_offsets is None
            or point_votes is None
            or query_offsets is None
            or point_valid is None
            or max(weights.values()) <= 0.0
        ):
            return zero, empty

        with autocast_disabled(pred_translation.device):
            pred_translation = pred_translation.float()
            surface_center = surface_center.float()
            pred_canonical = pred_canonical.float()
            point_offsets = point_offsets.float()
            point_votes = point_votes.float()
            query_offsets = query_offsets.float()
            point_valid = point_valid.to(device=pred_translation.device).bool()
            gt_translation = real_data['translation_label'].to(
                device=pred_translation.device,
                dtype=torch.float32,
            )
            rotation = end_points['pred_rotation_v6'].detach().float()
            size = end_points['pred_structured_size'].detach().float().clamp_min(
                1.0e-4
            )
            batch_size = pred_translation.shape[0]
            categories = self._category_vector(
                real_data.get('category_label'),
                batch_size,
                pred_translation.device,
            )
            source_value = real_data.get('source_id')
            if source_value is None:
                sources = torch.zeros(
                    batch_size,
                    device=pred_translation.device,
                    dtype=torch.long,
                )
            else:
                sources = source_value.to(
                    device=pred_translation.device
                ).long().view(batch_size, -1)[:, 0]

            sample_valid = (
                point_valid.any(dim=1)
                & torch.isfinite(pred_translation).all(dim=1)
                & torch.isfinite(surface_center).all(dim=1)
                & torch.isfinite(pred_canonical).all(dim=1)
                & torch.isfinite(gt_translation).all(dim=1)
                & torch.isfinite(rotation).flatten(1).all(dim=1)
                & torch.isfinite(size).all(dim=1)
                & (gt_translation[:, 2] > 1.0e-6)
            )
            if not torch.any(sample_valid):
                return zero, empty

            target_canonical = torch.bmm(
                (gt_translation - surface_center).unsqueeze(1),
                rotation,
            ).squeeze(1) / size
            beta_metric = float(OmegaConf.select(
                self.cfg, 'loss.absolute_center_metric_beta', default=0.02
            ))
            beta_canonical = float(OmegaConf.select(
                self.cfg, 'loss.absolute_center_canonical_beta', default=0.05
            ))

            metric_per_sample = F.smooth_l1_loss(
                pred_translation,
                gt_translation,
                beta=beta_metric,
                reduction='none',
            ).mean(dim=1)
            z_per_sample = F.smooth_l1_loss(
                pred_translation[:, 2],
                gt_translation[:, 2],
                beta=beta_metric,
                reduction='none',
            )
            canonical_per_sample = F.smooth_l1_loss(
                pred_canonical,
                target_canonical,
                beta=beta_canonical,
                reduction='none',
            ).mean(dim=1)
            object_scale = torch.linalg.vector_norm(size, dim=1).clamp_min(0.05)
            normalized_per_sample = F.smooth_l1_loss(
                (pred_translation - gt_translation) / object_scale.unsqueeze(1),
                torch.zeros_like(pred_translation),
                beta=float(OmegaConf.select(
                    self.cfg,
                    'loss.absolute_center_size_normalized_beta',
                    default=0.10,
                )),
                reduction='none',
            ).mean(dim=1)

            point_error = F.smooth_l1_loss(
                point_votes,
                target_canonical.unsqueeze(1).expand_as(point_votes),
                beta=beta_canonical,
                reduction='none',
            ).mean(dim=-1)
            point_loss_per_sample = (
                point_error * point_valid.to(dtype=point_error.dtype)
            ).sum(dim=1) / point_valid.sum(dim=1).clamp_min(1).to(
                dtype=point_error.dtype
            )
            query_loss_per_sample = F.smooth_l1_loss(
                query_offsets,
                target_canonical.unsqueeze(1).expand_as(query_offsets),
                beta=beta_canonical,
                reduction='none',
            ).mean(dim=(1, 2))

            def balanced(values):
                return self._category_source_balanced_mean(
                    values[sample_valid],
                    categories[sample_valid],
                    sources[sample_valid],
                )

            loss_xyz = balanced(metric_per_sample)
            loss_z = balanced(z_per_sample)
            loss_canonical = balanced(canonical_per_sample)
            loss_point_vote = balanced(point_loss_per_sample)
            loss_query_vote = balanced(query_loss_per_sample)
            loss_size_normalized = balanced(normalized_per_sample)

            loss_confidence = zero.float()
            confidence_tau = max(float(OmegaConf.select(
                self.cfg,
                'loss.absolute_center_confidence_tau',
                default=0.10,
            )), 1.0e-4)
            if point_weights is not None and query_weights is not None:
                point_weights = point_weights.float().clamp_min(1.0e-12)
                query_weights = query_weights.float().clamp_min(1.0e-12)
                point_distance = torch.linalg.vector_norm(
                    point_votes.detach() - target_canonical.unsqueeze(1),
                    dim=-1,
                )
                point_target_logits = (-point_distance / confidence_tau).masked_fill(
                    ~point_valid,
                    -1.0e4,
                )
                point_target_weights = torch.softmax(
                    point_target_logits,
                    dim=1,
                )
                point_confidence = (
                    point_target_weights
                    * (
                        point_target_weights.clamp_min(1.0e-12).log()
                        - point_weights.log()
                    )
                ).sum(dim=1)
                query_distance = torch.linalg.vector_norm(
                    query_offsets.detach() - target_canonical.unsqueeze(1),
                    dim=-1,
                )
                query_target_weights = torch.softmax(
                    -query_distance / confidence_tau,
                    dim=1,
                )
                query_confidence = (
                    query_target_weights
                    * (
                        query_target_weights.clamp_min(1.0e-12).log()
                        - query_weights.log()
                    )
                ).sum(dim=1)
                loss_confidence = balanced(
                    0.5 * (point_confidence + query_confidence)
                )

            point_entropy = end_points.get('pred_absolute_center_point_entropy')
            query_entropy = end_points.get('pred_absolute_center_query_entropy')
            loss_entropy = zero.float()
            if point_entropy is not None and query_entropy is not None:
                entropy_per_sample = 1.0 - 0.5 * (
                    point_entropy.float() + query_entropy.float()
                )
                loss_entropy = balanced(entropy_per_sample)

            weighted = (
                weights['xyz'] * loss_xyz
                + weights['z'] * loss_z
                + weights['canonical'] * loss_canonical
                + weights['point_vote'] * loss_point_vote
                + weights['query_vote'] * loss_query_vote
                + weights['size_normalized'] * loss_size_normalized
                + weights['confidence'] * loss_confidence
                + weights['entropy'] * loss_entropy
            )

            absolute_error = torch.abs(pred_translation - gt_translation)
            translation_error = torch.linalg.vector_norm(
                pred_translation - gt_translation,
                dim=1,
            )
            valid_error = translation_error[sample_valid]
            base_translation = end_points.get(
                'pred_absolute_center_base_translation'
            )
            base_error = torch.zeros_like(valid_error)
            if base_translation is not None:
                base_error = torch.linalg.vector_norm(
                    base_translation.detach().float()[sample_valid]
                    - gt_translation[sample_valid],
                    dim=1,
                )
            balanced_mae = balanced(translation_error)
            fallback_mask = end_points.get('pred_absolute_center_fallback_mask')
            fallback_ratio = (
                float(fallback_mask.float().mean())
                if fallback_mask is not None
                else 0.0
            )

        valid_abs = absolute_error[sample_valid]
        return weighted.to(dtype=zero.dtype), {
            'loss/absolute_center_xyz': float(loss_xyz),
            'loss/absolute_center_z': float(loss_z),
            'loss/absolute_center_canonical': float(loss_canonical),
            'loss/absolute_center_point_vote': float(loss_point_vote),
            'loss/absolute_center_query_vote': float(loss_query_vote),
            'loss/absolute_center_size_normalized': float(loss_size_normalized),
            'loss/absolute_center_confidence': float(loss_confidence),
            'loss/absolute_center_entropy': float(loss_entropy),
            'loss/absolute_center_valid_ratio': float(sample_valid.float().mean()),
            'translation/absolute_center_mae_cm': float(valid_error.mean() * 100.0),
            'translation/absolute_center_balanced_mae_cm': float(
                balanced_mae * 100.0
            ),
            'translation/absolute_center_x_mae_cm': float(
                valid_abs[:, 0].mean() * 100.0
            ),
            'translation/absolute_center_y_mae_cm': float(
                valid_abs[:, 1].mean() * 100.0
            ),
            'translation/absolute_center_z_mae_cm': float(
                valid_abs[:, 2].mean() * 100.0
            ),
            'translation/absolute_center_under_10cm': float(
                (valid_error <= 0.10).float().mean()
            ),
            'translation/absolute_center_base_mae_cm': float(
                base_error.mean() * 100.0
            ),
            'translation/absolute_center_improved_ratio': float(
                (valid_error < base_error).float().mean()
            ),
            'translation/absolute_center_fallback_ratio': fallback_ratio,
            'translation/absolute_center_point_entropy': float(
                point_entropy[sample_valid].mean()
            ) if point_entropy is not None else 0.0,
            'translation/absolute_center_query_entropy': float(
                query_entropy[sample_valid].mean()
            ) if query_entropy is not None else 0.0,
        }

    def _compute_guarded_center_z_v11_loss(self, end_points, real_data):
        """Train a bounded Z candidate and, when requested, its V11 gate."""
        zero = end_points['pred_size'].new_zeros(())
        empty = {
            'loss/guarded_center_z_metric': 0.0,
            'loss/guarded_center_z_log': 0.0,
            'loss/guarded_center_z_residual_target': 0.0,
            'loss/guarded_center_z_improvement': 0.0,
            'loss/guarded_center_z_confidence': 0.0,
            'loss/guarded_center_z_group_bias': 0.0,
            'loss/guarded_center_z_delta_regularize': 0.0,
            'loss/guarded_center_z_valid_ratio': 0.0,
            'translation/guarded_center_z_candidate_mae_cm': 0.0,
            'translation/guarded_center_z_final_mae_cm': 0.0,
            'translation/guarded_center_z_base_mae_cm': 0.0,
            'translation/guarded_center_z_candidate_signed_cm': 0.0,
            'translation/guarded_center_z_final_signed_cm': 0.0,
            'translation/guarded_center_z_candidate_improved_ratio': 0.0,
            'translation/guarded_center_z_applied_ratio': 0.0,
            'translation/guarded_center_z_applied_precision': 0.0,
            'translation/guarded_center_z_regression_ratio': 0.0,
            'translation/guarded_center_z_confidence_mean': 0.0,
            'translation/guarded_center_z_confidence_target_mean': 0.0,
            'translation/guarded_center_z_confidence_positive_ratio': 0.0,
            'translation/guarded_center_z_candidate_gain_cm': 0.0,
            'translation/guarded_center_z_residual_target_mae': 0.0,
            'translation/guarded_center_z_fallback_ratio': 0.0,
        }
        weights = {
            'metric': float(OmegaConf.select(
                self.cfg, 'loss.weight_guarded_center_z_metric', default=0.0
            )),
            'log': float(OmegaConf.select(
                self.cfg, 'loss.weight_guarded_center_z_log', default=0.0
            )),
            'residual_target': float(OmegaConf.select(
                self.cfg,
                'loss.weight_guarded_center_z_residual_target',
                default=0.0,
            )),
            'improvement': float(OmegaConf.select(
                self.cfg, 'loss.weight_guarded_center_z_improvement', default=0.0
            )),
            'confidence': float(OmegaConf.select(
                self.cfg, 'loss.weight_guarded_center_z_confidence', default=0.0
            )),
            'group_bias': float(OmegaConf.select(
                self.cfg, 'loss.weight_guarded_center_z_group_bias', default=0.0
            )),
            'delta': float(OmegaConf.select(
                self.cfg,
                'loss.weight_guarded_center_z_delta_regularize',
                default=0.0,
            )),
        }
        if max(weights.values()) <= 0.0:
            return zero, empty

        required = {
            'pred_guarded_center_z_candidate_depth',
            'pred_guarded_center_z_depth',
            'pred_guarded_center_z_base_depth',
            'pred_guarded_center_z_delta_log_depth',
            'pred_guarded_center_z_fallback_mask',
        }
        if weights['confidence'] > 0.0:
            required.add('pred_guarded_center_z_improve_logit')
        if not required.issubset(end_points):
            if end_points.get('translation_prediction_mode') == 'guarded_center_z_v11':
                missing = sorted(required.difference(end_points))
                raise KeyError(f"V11 guarded Center-Z loss missing outputs: {missing}")
            return zero, empty

        candidate_z = end_points['pred_guarded_center_z_candidate_depth'].float().view(-1)
        final_z = end_points['pred_guarded_center_z_depth'].float().view(-1)
        base_z = end_points['pred_guarded_center_z_base_depth'].float().view(-1)
        delta_log_z = end_points['pred_guarded_center_z_delta_log_depth'].float().view(-1)
        improve_logit = end_points.get('pred_guarded_center_z_improve_logit')
        if improve_logit is None:
            improve_logit = torch.full_like(candidate_z, 20.0)
        else:
            improve_logit = improve_logit.float().view(-1)
        fallback_mask = end_points['pred_guarded_center_z_fallback_mask'].to(
            device=candidate_z.device,
            dtype=torch.bool,
        ).view(-1)
        gt_z = real_data['translation_label'][:, 2].to(
            device=candidate_z.device,
            dtype=torch.float32,
        ).view(-1)
        categories = self._category_vector(
            real_data.get('category_label'),
            candidate_z.shape[0],
            candidate_z.device,
        )
        source_value = real_data.get('source_id')
        if source_value is None:
            sources = torch.zeros_like(categories)
        else:
            sources = source_value.to(candidate_z.device).long().view(-1)

        valid = (
            torch.isfinite(candidate_z)
            & torch.isfinite(final_z)
            & torch.isfinite(base_z)
            & torch.isfinite(delta_log_z)
            & torch.isfinite(improve_logit)
            & torch.isfinite(gt_z)
            & (candidate_z > 1.0e-6)
            & (final_z > 1.0e-6)
            & (base_z > 1.0e-6)
            & (gt_z > 1.0e-6)
        )
        if not torch.any(valid):
            return zero, empty

        candidate_valid = candidate_z[valid]
        final_valid = final_z[valid]
        base_valid = base_z[valid]
        gt_valid = gt_z[valid]
        delta_valid = delta_log_z[valid]
        logits_valid = improve_logit[valid]
        categories_valid = categories[valid]
        sources_valid = sources[valid]
        fallback_valid = fallback_mask[valid]

        beta = float(OmegaConf.select(
            self.cfg, 'loss.guarded_center_z_beta', default=0.02
        ))
        candidate_abs = (candidate_valid - gt_valid).abs()
        final_abs = (final_valid - gt_valid).abs()
        base_abs = (base_valid - gt_valid).abs()
        metric_per_item = F.smooth_l1_loss(
            candidate_valid,
            gt_valid,
            beta=beta,
            reduction='none',
        )
        log_per_item = F.smooth_l1_loss(
            torch.log(candidate_valid),
            torch.log(gt_valid),
            beta=beta,
            reduction='none',
        )
        margin = float(OmegaConf.select(
            self.cfg,
            'loss.guarded_center_z_improvement_margin_m',
            default=0.005,
        ))
        improvement_per_item = F.relu(candidate_abs - base_abs + margin)

        target_source_id = int(OmegaConf.select(
            self.cfg,
            'loss.guarded_center_z_target_source_id',
            default=0,
        ))
        target_source_weight = float(OmegaConf.select(
            self.cfg,
            'loss.guarded_center_z_target_source_weight',
            default=1.0,
        ))
        if target_source_weight <= 0.0:
            raise ValueError(
                'loss.guarded_center_z_target_source_weight must be positive'
            )

        def balanced(values):
            """Balance classes/sources while optionally emphasizing REAL."""
            category_means = []
            for category_id in torch.unique(categories_valid):
                category_mask = categories_valid == category_id
                source_means = []
                source_weights = []
                for source_id in torch.unique(sources_valid[category_mask]):
                    group_mask = category_mask & (sources_valid == source_id)
                    if torch.any(group_mask):
                        source_means.append(values[group_mask].mean())
                        source_weights.append(
                            target_source_weight
                            if int(source_id) == target_source_id
                            else 1.0
                        )
                if source_means:
                    stacked = torch.stack(source_means)
                    group_weights = stacked.new_tensor(source_weights)
                    category_means.append(
                        (stacked * group_weights).sum()
                        / group_weights.sum().clamp_min(1.0e-12)
                    )
            if not category_means:
                return values.new_zeros(())
            return torch.stack(category_means).mean()

        loss_metric = balanced(metric_per_item)
        loss_log = balanced(log_per_item)
        loss_improvement = balanced(improvement_per_item)
        loss_delta = balanced(delta_valid.abs())

        max_target_log_residual = float(OmegaConf.select(
            self.cfg,
            'loss.guarded_center_z_max_target_log_residual',
            default=0.20,
        ))
        if max_target_log_residual <= 0.0:
            raise ValueError(
                'loss.guarded_center_z_max_target_log_residual must be positive'
            )
        target_delta_log = torch.log(
            gt_valid / base_valid.clamp_min(1.0e-6)
        ).clamp(
            min=-max_target_log_residual,
            max=max_target_log_residual,
        )
        residual_target_beta = float(OmegaConf.select(
            self.cfg,
            'loss.guarded_center_z_residual_target_beta',
            default=0.01,
        ))
        residual_target_per_item = F.smooth_l1_loss(
            delta_valid,
            target_delta_log,
            beta=residual_target_beta,
            reduction='none',
        )
        loss_residual_target = balanced(residual_target_per_item)

        signed_error = candidate_valid - gt_valid
        group_bias_terms = []
        for category_id in torch.unique(categories_valid):
            category_mask = categories_valid == category_id
            source_terms = []
            for source_id in torch.unique(sources_valid[category_mask]):
                group_mask = category_mask & (sources_valid == source_id)
                if torch.any(group_mask):
                    source_terms.append(signed_error[group_mask].mean().abs())
            if source_terms:
                group_bias_terms.append(torch.stack(source_terms).mean())
        loss_group_bias = (
            torch.stack(group_bias_terms).mean() if group_bias_terms else zero.float()
        )

        candidate_gain = base_abs.detach() - candidate_abs.detach()
        if weights['confidence'] > 0.0:
            confidence_margin = float(OmegaConf.select(
                self.cfg,
                'loss.guarded_center_z_confidence_margin_m',
                default=0.005,
            ))
            confidence_target_mode = str(OmegaConf.select(
                self.cfg,
                'loss.guarded_center_z_confidence_target_mode',
                default='hard',
            )).lower()
            if confidence_target_mode == 'hard':
                confidence_target = (candidate_gain > confidence_margin).float()
            elif confidence_target_mode == 'soft_gain':
                confidence_temperature = max(float(OmegaConf.select(
                    self.cfg,
                    'loss.guarded_center_z_confidence_temperature_m',
                    default=0.002,
                )), 1.0e-6)
                confidence_target = torch.sigmoid(
                    (candidate_gain - confidence_margin) / confidence_temperature
                )
            else:
                raise ValueError(
                    'loss.guarded_center_z_confidence_target_mode must be '
                    f"'hard' or 'soft_gain', got {confidence_target_mode!r}"
                )
        else:
            # This is only a candidate-quality diagnostic; no gate objective is
            # constructed when its configured weight is zero.
            confidence_target = (candidate_gain > 0.0).float()
        confidence_start_epoch = int(OmegaConf.select(
            self.cfg,
            'loss.guarded_center_z_confidence_start_epoch',
            default=2,
        ))
        if (
            weights['confidence'] > 0.0
            and int(getattr(self, 'epoch', 0)) >= confidence_start_epoch
        ):
            confidence_per_item = F.binary_cross_entropy_with_logits(
                logits_valid,
                confidence_target,
                reduction='none',
            )
            if bool(OmegaConf.select(
                self.cfg,
                'loss.guarded_center_z_confidence_balance',
                default=False,
            )):
                positive_mass = confidence_target.detach().mean().clamp(
                    min=0.05,
                    max=0.95,
                )
                confidence_weights = (
                    confidence_target.detach() * (0.5 / positive_mass)
                    + (1.0 - confidence_target.detach())
                    * (0.5 / (1.0 - positive_mass))
                )
                max_confidence_weight = float(OmegaConf.select(
                    self.cfg,
                    'loss.guarded_center_z_confidence_max_weight',
                    default=5.0,
                ))
                confidence_per_item = confidence_per_item * confidence_weights.clamp(
                    max=max_confidence_weight
                )
            loss_confidence = balanced(confidence_per_item)
            confidence_weight = weights['confidence']
        else:
            loss_confidence = zero.float()
            confidence_weight = 0.0

        weighted = (
            weights['metric'] * loss_metric
            + weights['log'] * loss_log
            + weights['residual_target'] * loss_residual_target
            + weights['improvement'] * loss_improvement
            + confidence_weight * loss_confidence
            + weights['group_bias'] * loss_group_bias
            + weights['delta'] * loss_delta
        )

        applied = ~fallback_valid
        applied_precision = (
            confidence_target[applied].mean() if torch.any(applied) else zero.float()
        )
        return weighted.to(dtype=zero.dtype), {
            'loss/guarded_center_z_metric': float(loss_metric),
            'loss/guarded_center_z_log': float(loss_log),
            'loss/guarded_center_z_residual_target': float(
                loss_residual_target
            ),
            'loss/guarded_center_z_improvement': float(loss_improvement),
            'loss/guarded_center_z_confidence': float(loss_confidence),
            'loss/guarded_center_z_group_bias': float(loss_group_bias),
            'loss/guarded_center_z_delta_regularize': float(loss_delta),
            'loss/guarded_center_z_valid_ratio': float(valid.float().mean()),
            'translation/guarded_center_z_candidate_mae_cm': float(
                candidate_abs.mean() * 100.0
            ),
            'translation/guarded_center_z_final_mae_cm': float(
                final_abs.mean() * 100.0
            ),
            'translation/guarded_center_z_base_mae_cm': float(
                base_abs.mean() * 100.0
            ),
            'translation/guarded_center_z_candidate_signed_cm': float(
                signed_error.mean() * 100.0
            ),
            'translation/guarded_center_z_final_signed_cm': float(
                (final_valid - gt_valid).mean() * 100.0
            ),
            'translation/guarded_center_z_candidate_improved_ratio': float(
                (candidate_abs < base_abs).float().mean()
            ),
            'translation/guarded_center_z_applied_ratio': float(applied.float().mean()),
            'translation/guarded_center_z_applied_precision': float(applied_precision),
            'translation/guarded_center_z_regression_ratio': float(
                (final_abs > base_abs).float().mean()
            ),
            'translation/guarded_center_z_confidence_mean': float(
                torch.sigmoid(logits_valid).mean()
            ),
            'translation/guarded_center_z_confidence_target_mean': float(
                confidence_target.mean()
            ),
            'translation/guarded_center_z_confidence_positive_ratio': float(
                (confidence_target > 0.5).float().mean()
            ),
            'translation/guarded_center_z_candidate_gain_cm': float(
                candidate_gain.mean() * 100.0
            ),
            'translation/guarded_center_z_residual_target_mae': float(
                (delta_valid - target_delta_log).abs().mean()
            ),
            'translation/guarded_center_z_fallback_ratio': float(
                fallback_valid.float().mean()
            ),
        }

    def _compute_metric_depth_v5_loss(self, end_points, real_data):
        """Train-only in-domain metric calibration with class/source balance."""
        weight_metric = float(
            OmegaConf.select(
                self.cfg, "loss.weight_metric_depth_v5_metric_z", default=0.0
            )
        )
        weight_log_offset = float(
            OmegaConf.select(
                self.cfg, "loss.weight_metric_depth_v5_log_offset", default=0.0
            )
        )
        weight_improvement = float(
            OmegaConf.select(
                self.cfg, "loss.weight_metric_depth_v5_improvement", default=0.0
            )
        )
        weight_bias = float(
            OmegaConf.select(
                self.cfg, "loss.weight_metric_depth_v5_group_bias", default=0.0
            )
        )
        weight_bottle = float(
            OmegaConf.select(
                self.cfg, "loss.weight_metric_depth_v5_bottle", default=0.0
            )
        )
        weight_delta = float(
            OmegaConf.select(
                self.cfg,
                "loss.weight_metric_depth_v5_delta_regularize",
                default=0.0,
            )
        )
        zero = end_points['pred_size'].new_zeros(())
        empty = {
            'loss/metric_depth_v5_metric_z': 0.0,
            'loss/metric_depth_v5_log_offset': 0.0,
            'loss/metric_depth_v5_improvement': 0.0,
            'loss/metric_depth_v5_group_bias': 0.0,
            'loss/metric_depth_v5_bottle': 0.0,
            'loss/metric_depth_v5_delta_regularize': 0.0,
            'loss/metric_depth_v5_valid_ratio': 0.0,
            'translation/metric_depth_v5_mae_cm': 0.0,
            'translation/metric_depth_v5_base_mae_cm': 0.0,
            'translation/metric_depth_v5_surface_mae_cm': 0.0,
            'translation/metric_depth_v5_balanced_mae_cm': 0.0,
            'translation/metric_depth_v5_improved_ratio': 0.0,
            'translation/metric_depth_v5_signed_z_cm': 0.0,
            'translation/metric_depth_v5_bottle_mae_cm': 0.0,
            'translation/metric_depth_v5_bottle_base_mae_cm': 0.0,
            'translation/metric_depth_v5_bottle_signed_z_cm': 0.0,
        }
        if max(
            weight_metric,
            weight_log_offset,
            weight_improvement,
            weight_bias,
            weight_bottle,
            weight_delta,
        ) <= 0.0:
            return zero, empty

        required = {
            'pred_metric_depth_v5_depth',
            'pred_metric_depth_v5_base_depth',
            'pred_metric_depth_v5_surface_depth',
            'pred_metric_depth_v5_delta_log_depth',
        }
        if not required.issubset(end_points):
            if end_points.get('translation_prediction_mode') == 'metric_depth_v5_center_z':
                missing = sorted(required.difference(end_points))
                raise KeyError(f"V5 metric-depth loss missing outputs: {missing}")
            return zero, empty

        pred_z = end_points['pred_metric_depth_v5_depth'].view(-1)
        base_z = end_points['pred_metric_depth_v5_base_depth'].to(
            device=pred_z.device, dtype=pred_z.dtype
        ).view(-1)
        surface_z = end_points['pred_metric_depth_v5_surface_depth'].to(
            device=pred_z.device, dtype=pred_z.dtype
        ).view(-1)
        delta_log_z = end_points['pred_metric_depth_v5_delta_log_depth'].to(
            device=pred_z.device, dtype=pred_z.dtype
        ).view(-1)
        gt_z = real_data['translation_label'][:, 2].to(
            device=pred_z.device, dtype=pred_z.dtype
        ).view(-1)
        categories = real_data.get('category_label')
        if categories is None:
            categories = torch.zeros_like(pred_z, dtype=torch.long)
        else:
            categories = categories.to(device=pred_z.device).long().view(-1)
        sources = real_data.get('source_id')
        if sources is None:
            sources = torch.zeros_like(pred_z, dtype=torch.long)
        else:
            sources = sources.to(device=pred_z.device).long().view(-1)

        valid = (
            torch.isfinite(pred_z)
            & torch.isfinite(base_z)
            & torch.isfinite(surface_z)
            & torch.isfinite(delta_log_z)
            & torch.isfinite(gt_z)
            & (pred_z > 1.0e-6)
            & (base_z > 1.0e-6)
            & (surface_z > 1.0e-6)
            & (gt_z > 1.0e-6)
        )
        if not torch.any(valid):
            return zero, empty

        pred_valid = pred_z[valid]
        base_valid = base_z[valid]
        surface_valid = surface_z[valid]
        gt_valid = gt_z[valid]
        delta_valid = delta_log_z[valid]
        categories_valid = categories[valid]
        sources_valid = sources[valid]
        beta = float(
            OmegaConf.select(self.cfg, "loss.metric_depth_v5_beta", default=0.02)
        )

        metric_per_item = F.smooth_l1_loss(
            pred_valid, gt_valid, beta=beta, reduction='none'
        )
        pred_log_offset = torch.log(
            pred_valid / surface_valid.clamp_min(1.0e-6)
        )
        gt_log_offset = torch.log(
            gt_valid / surface_valid.clamp_min(1.0e-6)
        )
        log_offset_per_item = F.smooth_l1_loss(
            pred_log_offset,
            gt_log_offset,
            beta=beta,
            reduction='none',
        )
        pred_abs = (pred_valid - gt_valid).abs()
        base_abs = (base_valid - gt_valid).abs()
        margin = float(
            OmegaConf.select(
                self.cfg, "loss.metric_depth_v5_improvement_margin", default=0.0
            )
        )
        improvement_per_item = F.relu(pred_abs - base_abs + margin)

        loss_metric = self._category_source_balanced_mean(
            metric_per_item, categories_valid, sources_valid
        )
        loss_log_offset = self._category_source_balanced_mean(
            log_offset_per_item, categories_valid, sources_valid
        )
        loss_improvement = self._category_source_balanced_mean(
            improvement_per_item, categories_valid, sources_valid
        )
        group_bias_terms = []
        signed_error = pred_valid - gt_valid
        for category_id in torch.unique(categories_valid):
            category_mask = categories_valid == category_id
            source_bias_terms = []
            for source_id in torch.unique(sources_valid[category_mask]):
                group_mask = category_mask & (sources_valid == source_id)
                if torch.any(group_mask):
                    source_bias_terms.append(signed_error[group_mask].mean().abs())
            if source_bias_terms:
                group_bias_terms.append(torch.stack(source_bias_terms).mean())
        loss_bias = (
            torch.stack(group_bias_terms).mean() if group_bias_terms else zero
        )
        loss_delta = self._category_source_balanced_mean(
            delta_valid.abs(), categories_valid, sources_valid
        )

        bottle_class_id = int(
            OmegaConf.select(
                self.cfg, "loss.metric_depth_v5_bottle_class_id", default=0
            )
        )
        bottle_mask = categories_valid == bottle_class_id
        if torch.any(bottle_mask):
            bottle_per_item = (
                metric_per_item[bottle_mask] + log_offset_per_item[bottle_mask]
            )
            loss_bottle = self._source_balanced_mean(
                bottle_per_item, sources_valid[bottle_mask]
            )
            bottle_mae = pred_abs[bottle_mask].mean()
            bottle_base_mae = base_abs[bottle_mask].mean()
            bottle_signed = signed_error[bottle_mask].mean()
        else:
            loss_bottle = zero
            bottle_mae = zero
            bottle_base_mae = zero
            bottle_signed = zero

        loss = (
            weight_metric * loss_metric
            + weight_log_offset * loss_log_offset
            + weight_improvement * loss_improvement
            + weight_bias * loss_bias
            + weight_bottle * loss_bottle
            + weight_delta * loss_delta
        )
        balanced_mae = self._category_source_balanced_mean(
            pred_abs, categories_valid, sources_valid
        )
        return loss, {
            'loss/metric_depth_v5_metric_z': float(loss_metric),
            'loss/metric_depth_v5_log_offset': float(loss_log_offset),
            'loss/metric_depth_v5_improvement': float(loss_improvement),
            'loss/metric_depth_v5_group_bias': float(loss_bias),
            'loss/metric_depth_v5_bottle': float(loss_bottle),
            'loss/metric_depth_v5_delta_regularize': float(loss_delta),
            'loss/metric_depth_v5_valid_ratio': float(valid.float().mean()),
            'translation/metric_depth_v5_mae_cm': float(pred_abs.mean() * 100.0),
            'translation/metric_depth_v5_base_mae_cm': float(base_abs.mean() * 100.0),
            'translation/metric_depth_v5_surface_mae_cm': float(
                (surface_valid - gt_valid).abs().mean() * 100.0
            ),
            'translation/metric_depth_v5_balanced_mae_cm': float(
                balanced_mae * 100.0
            ),
            'translation/metric_depth_v5_improved_ratio': float(
                (pred_abs < base_abs).float().mean()
            ),
            'translation/metric_depth_v5_signed_z_cm': float(
                signed_error.mean() * 100.0
            ),
            'translation/metric_depth_v5_bottle_mae_cm': float(
                bottle_mae * 100.0
            ),
            'translation/metric_depth_v5_bottle_base_mae_cm': float(
                bottle_base_mae * 100.0
            ),
            'translation/metric_depth_v5_bottle_signed_z_cm': float(
                bottle_signed * 100.0
            ),
        }

    def _compute_final_translation_loss(self, end_points, real_data):
        weight_final = float(OmegaConf.select(self.cfg, "loss.weight_final_translation", default=0.0))
        weight_residual = float(OmegaConf.select(self.cfg, "loss.weight_residual_regularize", default=0.0))
        zero = end_points['pred_size'].new_zeros(())
        empty = {
            'loss/final_translation': 0.0,
            'loss/final_translation_xy': 0.0,
            'loss/final_translation_z': 0.0,
            'loss/final_translation_log_z': 0.0,
            'loss/final_translation_size_normalized': 0.0,
            'loss/residual_regularize': 0.0,
            'loss/final_translation_valid_ratio': 0.0,
        }
        if weight_final <= 0.0 and weight_residual <= 0.0:
            return zero, empty
        if (
            end_points.get('translation_prediction_mode')
            in {
                'deterministic_center_depth',
                'ray_depth_v1',
                'ray_depth_v2',
                'ray_depth_v3',
                'ray_depth_v4_center_z',
                'metric_depth_v5_center_z',
                'e2e_projective_uvz',
            }
        ):
            final_translation = end_points.get('pred_translation')
            if final_translation is None:
                return zero, empty
            gt_translation = real_data['translation_label'].to(
                device=final_translation.device,
                dtype=final_translation.dtype,
            )
            valid = (
                torch.isfinite(final_translation).all(dim=1)
                & torch.isfinite(gt_translation).all(dim=1)
                & (final_translation[:, 2] > 1.0e-6)
                & (gt_translation[:, 2] > 1.0e-6)
            )
            if not torch.any(valid):
                return zero, empty

            beta = float(
                OmegaConf.select(
                    self.cfg,
                    "loss.final_translation_beta",
                    default=0.02,
                )
            )
            loss_xy = F.smooth_l1_loss(
                final_translation[valid, :2],
                gt_translation[valid, :2],
                beta=beta,
            )
            loss_z = F.smooth_l1_loss(
                final_translation[valid, 2],
                gt_translation[valid, 2],
                beta=beta,
            )
            loss_log_z = F.smooth_l1_loss(
                torch.log(final_translation[valid, 2].clamp_min(1.0e-6)),
                torch.log(gt_translation[valid, 2].clamp_min(1.0e-6)),
                beta=beta,
            )
            loss_size_normalized = zero
            weight_size_normalized = float(
                OmegaConf.select(
                    self.cfg,
                    "loss.weight_translation_size_normalized",
                    default=0.0,
                )
            )
            gt_size = real_data.get('size_label')
            if weight_size_normalized > 0.0 and gt_size is not None:
                gt_size = gt_size.to(
                    device=final_translation.device,
                    dtype=final_translation.dtype,
                )
                object_scale = torch.linalg.vector_norm(
                    gt_size[valid],
                    dim=1,
                    keepdim=True,
                ).clamp_min(0.05)
                normalized_error = (
                    final_translation[valid] - gt_translation[valid]
                ) / object_scale
                loss_size_normalized = F.smooth_l1_loss(
                    normalized_error,
                    torch.zeros_like(normalized_error),
                    beta=float(
                        OmegaConf.select(
                            self.cfg,
                            "loss.translation_size_normalized_beta",
                            default=0.1,
                        )
                    ),
                )
            weight_xy = float(
                OmegaConf.select(
                    self.cfg,
                    "loss.weight_deterministic_translation_xy",
                    default=1.0,
                )
            )
            weight_z = float(
                OmegaConf.select(
                    self.cfg,
                    "loss.weight_deterministic_translation_z",
                    default=5.0,
                )
            )
            weight_log_z = float(
                OmegaConf.select(
                    self.cfg,
                    "loss.weight_deterministic_translation_log_z",
                    default=1.0,
                )
            )
            loss_final = (
                weight_xy * loss_xy
                + weight_z * loss_z
                + weight_log_z * loss_log_z
                + weight_size_normalized * loss_size_normalized
            )
            coarse = end_points.get('coarse_translation')
            if coarse is None:
                loss_residual = zero
            else:
                coarse = coarse.to(
                    device=final_translation.device,
                    dtype=final_translation.dtype,
                )
                loss_residual = (
                    final_translation[valid] - coarse[valid]
                ).norm(dim=1).mean()
            loss = weight_final * loss_final + weight_residual * loss_residual
            return loss, {
                'loss/final_translation': float(loss_final),
                'loss/final_translation_xy': float(loss_xy),
                'loss/final_translation_z': float(loss_z),
                'loss/final_translation_log_z': float(loss_log_z),
                'loss/final_translation_size_normalized': float(
                    loss_size_normalized
                ),
                'loss/residual_regularize': float(loss_residual),
                'loss/final_translation_valid_ratio': float(valid.float().mean()),
            }

        required = ['noisy_translation', 'pred_translation', 'diffusion_steps', 'coarse_translation']
        if any(key not in end_points for key in required):
            return zero, empty

        steps = end_points['diffusion_steps'].long()
        x0_raw = self._predict_x0_from_noise(
            end_points['noisy_translation'],
            end_points['pred_translation'],
            steps,
        )
        target_mode = end_points.get(
            'translation_target_mode',
            OmegaConf.select(self.cfg, "diffusion.translation_target", default="absolute"),
        )
        if target_mode in {"point_center_residual", "gated_center_residual"}:
            residual = x0_raw
        else:
            residual_scale = end_points.get('translation_residual_scale_values')
            if residual_scale is None:
                residual_scale = torch.ones_like(x0_raw[:, :1])
            residual = x0_raw * residual_scale.to(device=x0_raw.device, dtype=x0_raw.dtype)
            residual_bound = float(OmegaConf.select(self.cfg, "diffusion.translation_residual_bound", default=0.0))
            if residual_bound > 0.0:
                bound = residual.new_tensor(residual_bound)
                residual = bound * torch.tanh(residual / bound.clamp_min(1.0e-6))
        final_translation = end_points['coarse_translation'].to(dtype=residual.dtype) + residual
        gt_translation = real_data['translation_label'].to(device=final_translation.device, dtype=final_translation.dtype)
        valid = torch.isfinite(final_translation).all(dim=1) & torch.isfinite(gt_translation).all(dim=1)
        if not torch.any(valid):
            return zero, empty

        beta = float(OmegaConf.select(self.cfg, "loss.final_translation_beta", default=0.02))
        loss_final = F.smooth_l1_loss(
            final_translation[valid],
            gt_translation[valid],
            beta=beta,
        )
        loss_residual = residual[valid].norm(dim=1).mean()
        loss = weight_final * loss_final + weight_residual * loss_residual
        return loss, {
            'loss/final_translation': float(loss_final),
            'loss/final_translation_xy': 0.0,
            'loss/final_translation_z': 0.0,
            'loss/final_translation_log_z': 0.0,
            'loss/final_translation_size_normalized': 0.0,
            'loss/residual_regularize': float(loss_residual),
            'loss/final_translation_valid_ratio': float(valid.float().mean()),
        }

    def _compute_center_anchor_losses(self, end_points, real_data):
        weight_anchor = float(OmegaConf.select(self.cfg, "loss.weight_center_anchor_ray", default=0.0))
        weight_final = float(
            OmegaConf.select(self.cfg, "loss.weight_final_translation_reprojection", default=0.0)
        )
        weight_corner = float(
            OmegaConf.select(
                self.cfg,
                "loss.weight_translation_corner_reprojection",
                default=0.0,
            )
        )
        zero = end_points['pred_size'].new_zeros(())
        empty = {
            'loss/center_anchor_ray': 0.0,
            'loss/final_translation_reprojection': 0.0,
            'loss/translation_corner_reprojection': 0.0,
            'loss/center_anchor_valid_ratio': 0.0,
            'loss/final_translation_reprojection_valid_ratio': 0.0,
            'loss/translation_corner_reprojection_valid_ratio': 0.0,
            'anchor/point_weight': 0.0,
            'anchor/bbox_weight': 0.0,
            'anchor/mask_weight': 0.0,
        }
        if weight_anchor <= 0.0 and weight_final <= 0.0 and weight_corner <= 0.0:
            return zero, empty
        required = ['coarse_translation', 'anchor_cam_k', 'anchor_bbox_wh']
        if any(key not in end_points for key in required):
            return zero, empty

        coarse = end_points['coarse_translation']
        cam_k = end_points['anchor_cam_k'].to(device=coarse.device, dtype=coarse.dtype)
        bbox_wh = end_points['anchor_bbox_wh'].to(device=coarse.device, dtype=coarse.dtype).clamp_min(1.0)
        gt = real_data['translation_label'].to(device=coarse.device, dtype=coarse.dtype)
        gt_valid = torch.isfinite(gt).all(dim=1) & (gt[:, 2] > 1.0e-6)
        beta = float(OmegaConf.select(self.cfg, "loss.center_anchor_reprojection_beta", default=0.05))

        anchor_valid = gt_valid & torch.isfinite(coarse).all(dim=1) & (coarse[:, 2] > 1.0e-6)
        if torch.any(anchor_valid):
            coarse_uv = project_translation_to_image(coarse[anchor_valid], cam_k[anchor_valid])
            gt_uv = project_translation_to_image(gt[anchor_valid], cam_k[anchor_valid])
            anchor_error = (coarse_uv - gt_uv) / bbox_wh[anchor_valid]
            loss_anchor = F.smooth_l1_loss(
                anchor_error,
                torch.zeros_like(anchor_error),
                beta=beta,
            )
        else:
            loss_anchor = zero

        loss_final_reprojection = zero
        final_valid_ratio = zero
        diffusion_keys = ['noisy_translation', 'pred_translation', 'diffusion_steps']
        deterministic_mode = end_points.get('translation_prediction_mode') in {
            'deterministic_center_depth',
            'ray_depth_v1',
            'ray_depth_v2',
            'ray_depth_v3',
            'ray_depth_v4_center_z',
            'metric_depth_v5_center_z',
        }
        if (
            weight_final > 0.0
            and deterministic_mode
            and end_points.get('pred_translation') is not None
        ):
            final_translation = end_points['pred_translation']
            final_valid = (
                gt_valid
                & torch.isfinite(final_translation).all(dim=1)
                & (final_translation[:, 2] > 1.0e-6)
            )
            final_valid_ratio = final_valid.float().mean()
            if torch.any(final_valid):
                final_uv = project_translation_to_image(
                    final_translation[final_valid],
                    cam_k[final_valid],
                )
                gt_uv = project_translation_to_image(
                    gt[final_valid],
                    cam_k[final_valid],
                )
                final_error = (final_uv - gt_uv) / bbox_wh[final_valid]
                loss_final_reprojection = F.smooth_l1_loss(
                    final_error,
                    torch.zeros_like(final_error),
                    beta=beta,
                )
        elif weight_final > 0.0 and all(key in end_points for key in diffusion_keys):
            x0_raw = self._predict_x0_from_noise(
                end_points['noisy_translation'],
                end_points['pred_translation'],
                end_points['diffusion_steps'].long(),
            )
            target_mode = end_points.get(
                'translation_target_mode',
                OmegaConf.select(self.cfg, "diffusion.translation_target", default="absolute"),
            )
            if target_mode in {"point_center_residual", "gated_center_residual"}:
                residual = x0_raw
            else:
                residual_scale = end_points.get('translation_residual_scale_values')
                if residual_scale is None:
                    residual_scale = torch.ones_like(x0_raw[:, :1])
                residual = x0_raw * residual_scale.to(device=x0_raw.device, dtype=x0_raw.dtype)
                residual_bound = float(
                    OmegaConf.select(self.cfg, "diffusion.translation_residual_bound", default=0.0)
                )
                if residual_bound > 0.0:
                    bound = residual.new_tensor(residual_bound)
                    residual = bound * torch.tanh(residual / bound.clamp_min(1.0e-6))
            final_translation = coarse.to(dtype=residual.dtype) + residual
            final_valid = gt_valid & torch.isfinite(final_translation).all(dim=1) & (final_translation[:, 2] > 1.0e-6)
            final_valid_ratio = final_valid.float().mean()
            if torch.any(final_valid):
                final_uv = project_translation_to_image(final_translation[final_valid], cam_k[final_valid])
                gt_uv = project_translation_to_image(gt[final_valid], cam_k[final_valid])
                final_error = (final_uv - gt_uv) / bbox_wh[final_valid]
                loss_final_reprojection = F.smooth_l1_loss(
                    final_error,
                    torch.zeros_like(final_error),
                    beta=beta,
                )

        loss_corner_reprojection = zero
        corner_valid_ratio = zero
        if (
            weight_corner > 0.0
            and deterministic_mode
            and end_points.get('pred_translation') is not None
            and 'rotation_label' in real_data
            and 'size_label' in real_data
        ):
            final_translation = end_points['pred_translation']
            gt_rotation = real_data['rotation_label'].to(
                device=final_translation.device,
                dtype=final_translation.dtype,
            )
            gt_size = real_data['size_label'].to(
                device=final_translation.device,
                dtype=final_translation.dtype,
            )
            geometry_valid = (
                gt_valid
                & torch.isfinite(final_translation).all(dim=1)
                & (final_translation[:, 2] > 1.0e-6)
                & torch.isfinite(gt_rotation).flatten(1).all(dim=1)
                & torch.isfinite(gt_size).all(dim=1)
                & (gt_size > 1.0e-6).all(dim=1)
            )
            canonical_corners = canonical_bbox_axis_keypoints(
                device=final_translation.device,
                dtype=final_translation.dtype,
            )[:8]
            metric_corners = metric_keypoints_from_size(
                gt_size,
                canonical_corners,
            )
            pred_corners = transform_keypoints(
                metric_corners,
                gt_rotation,
                final_translation,
            )
            gt_corners = transform_keypoints(
                metric_corners,
                gt_rotation,
                gt,
            )
            geometry_valid = (
                geometry_valid
                & torch.isfinite(pred_corners).flatten(1).all(dim=1)
                & torch.isfinite(gt_corners).flatten(1).all(dim=1)
                & (pred_corners[:, :, 2] > 1.0e-6).all(dim=1)
                & (gt_corners[:, :, 2] > 1.0e-6).all(dim=1)
            )
            corner_valid_ratio = geometry_valid.float().mean()
            if torch.any(geometry_valid):
                selected_pred = pred_corners[geometry_valid]
                selected_gt = gt_corners[geometry_valid]
                selected_cam_k = cam_k[geometry_valid]
                fx, fy, cx, cy = selected_cam_k.unbind(dim=-1)

                def project_corners(points):
                    depth = points[:, :, 2].clamp_min(1.0e-6)
                    u = fx.unsqueeze(1) * points[:, :, 0] / depth + cx.unsqueeze(1)
                    v = fy.unsqueeze(1) * points[:, :, 1] / depth + cy.unsqueeze(1)
                    return torch.stack([u, v], dim=-1)

                pred_corner_uv = project_corners(selected_pred)
                gt_corner_uv = project_corners(selected_gt)
                corner_error = (
                    pred_corner_uv - gt_corner_uv
                ) / bbox_wh[geometry_valid].unsqueeze(1)
                loss_corner_reprojection = F.smooth_l1_loss(
                    corner_error,
                    torch.zeros_like(corner_error),
                    beta=float(
                        OmegaConf.select(
                            self.cfg,
                            "loss.translation_corner_reprojection_beta",
                            default=0.05,
                        )
                    ),
                )

        weights = end_points.get('anchor_gate_weights')
        if weights is not None:
            mean_weights = weights.detach().mean(dim=0)
            point_weight, bbox_weight, mask_weight = [float(value) for value in mean_weights]
        else:
            point_weight = bbox_weight = mask_weight = 0.0
        loss = (
            weight_anchor * loss_anchor
            + weight_final * loss_final_reprojection
            + weight_corner * loss_corner_reprojection
        )
        return loss, {
            'loss/center_anchor_ray': float(loss_anchor),
            'loss/final_translation_reprojection': float(loss_final_reprojection),
            'loss/translation_corner_reprojection': float(
                loss_corner_reprojection
            ),
            'loss/center_anchor_valid_ratio': float(anchor_valid.float().mean()),
            'loss/final_translation_reprojection_valid_ratio': float(final_valid_ratio),
            'loss/translation_corner_reprojection_valid_ratio': float(
                corner_valid_ratio
            ),
            'anchor/point_weight': point_weight,
            'anchor/bbox_weight': bbox_weight,
            'anchor/mask_weight': mask_weight,
        }

    def _compute_canonical_keypoint_loss(self, end_points, real_data):
        weight = float(OmegaConf.select(self.cfg, "loss.weight_canonical_keypoint", default=0.0))
        weight_rot_geo = float(OmegaConf.select(self.cfg, "loss.weight_canonical_keypoint_rot_geo", default=0.0))
        pred_keypoints = end_points.get('pred_keypoints_cam')
        zero = end_points['pred_size'].new_zeros(())
        if weight <= 0.0 or pred_keypoints is None:
            return zero, {
                'loss/canonical_keypoint': 0.0,
                'loss/canonical_keypoint_rot_geo': 0.0,
                'loss/canonical_keypoint_residual': 0.0,
                'loss/canonical_keypoint_valid_ratio': 0.0,
            }

        gt_size = real_data['size_label']
        gt_rot = real_data['rotation_label']
        gt_trans = real_data['translation_label']
        canonical = canonical_bbox_axis_keypoints(gt_size.device, gt_size.dtype)
        metric_keypoints = metric_keypoints_from_size(gt_size, canonical)
        gt_keypoints = transform_keypoints(metric_keypoints, gt_rot, gt_trans)

        finite = (
            torch.isfinite(pred_keypoints.reshape(pred_keypoints.shape[0], -1)).all(dim=1)
            & torch.isfinite(gt_size).all(dim=1)
            & torch.isfinite(gt_rot.reshape(gt_rot.shape[0], -1)).all(dim=1)
            & torch.isfinite(gt_trans).all(dim=1)
        )
        if not torch.any(finite):
            return zero, {
                'loss/canonical_keypoint': 0.0,
                'loss/canonical_keypoint_rot_geo': 0.0,
                'loss/canonical_keypoint_residual': 0.0,
                'loss/canonical_keypoint_valid_ratio': 0.0,
            }

        if bool(OmegaConf.select(self.cfg, "loss.canonical_keypoint_symmetry_aware", default=True)):
            gt_keypoints = self._select_symmetry_keypoint_target(
                pred_keypoints,
                metric_keypoints,
                gt_rot,
                gt_trans,
                real_data.get('category_label'),
            )

        beta = float(OmegaConf.select(self.cfg, "loss.canonical_keypoint_beta", default=0.02))
        loss_keypoint = F.smooth_l1_loss(
            pred_keypoints[finite],
            gt_keypoints[finite],
            beta=beta,
        )

        loss_rot_geo = zero
        residual_mean = 0.0
        if weight_rot_geo > 0.0:
            with autocast(False):
                pred_rot, _, residual = solve_rotation_from_keypoints(
                    metric_keypoints[finite].float(),
                    pred_keypoints[finite].float(),
                )
                gt_rot_valid = gt_rot[finite].float()
                categories = real_data.get('category_label')
                categories = categories[finite] if categories is not None else None
                if bool(OmegaConf.select(self.cfg, "loss.canonical_keypoint_symmetry_aware", default=True)):
                    loss_rot_geo = self._symmetry_aware_geodesic_loss(
                        pred_rot, gt_rot_valid, categories).to(dtype=zero.dtype)
                else:
                    rel_rot = torch.matmul(pred_rot.transpose(1, 2), gt_rot_valid)
                    trace = rel_rot[:, 0, 0] + rel_rot[:, 1, 1] + rel_rot[:, 2, 2]
                    cos_theta = ((trace - 1.0) * 0.5).clamp(min=-1.0 + 1.0e-4, max=1.0 - 1.0e-4)
                    loss_rot_geo = torch.acos(cos_theta).mean().to(dtype=zero.dtype)
                residual_mean = float(residual.mean())

        loss = loss_keypoint + weight_rot_geo * loss_rot_geo
        return loss * weight, {
            'loss/canonical_keypoint': float(loss_keypoint),
            'loss/canonical_keypoint_rot_geo': float(loss_rot_geo),
            'loss/canonical_keypoint_residual': residual_mean,
            'loss/canonical_keypoint_valid_ratio': float(finite.float().mean()),
        }

    def _select_symmetry_keypoint_target(self, pred_keypoints, metric_keypoints, gt_rot_mat, gt_translation, category_label):
        batch_size = pred_keypoints.shape[0]
        target = transform_keypoints(metric_keypoints, gt_rot_mat, gt_translation)
        sym_mask = self._symmetric_mask(category_label, batch_size, pred_keypoints.device)
        if not torch.any(sym_mask):
            return target

        with torch.no_grad():
            y_rots = self._y_rotation_candidates(pred_keypoints.device, pred_keypoints.dtype)
            gt_sym = torch.matmul(gt_rot_mat[sym_mask].unsqueeze(1), y_rots.unsqueeze(0))
            metric_sym = metric_keypoints[sym_mask]
            candidate = torch.matmul(
                metric_sym.unsqueeze(1),
                gt_sym.transpose(2, 3),
            ) + gt_translation[sym_mask].view(-1, 1, 1, 3)
            dist = torch.linalg.norm(
                candidate - pred_keypoints[sym_mask].unsqueeze(1),
                dim=-1,
            ).mean(dim=-1)
            best_idx = dist.argmin(dim=1)
            selected = candidate[torch.arange(candidate.shape[0], device=candidate.device), best_idx]

        target = target.clone()
        target[sym_mask] = selected
        return target

    def _compute_nocs_correspondence_loss(self, end_points, real_data):
        weight = float(OmegaConf.select(self.cfg, "loss.weight_nocs_correspondence", default=0.0))
        weight_rot_geo = float(OmegaConf.select(self.cfg, "loss.weight_nocs_rot_geo", default=0.0))
        weight_confidence = float(OmegaConf.select(self.cfg, "loss.weight_nocs_confidence", default=0.0))
        pred_nocs = end_points.get('pred_nocs')
        confidence_logits = end_points.get('pred_nocs_confidence_logits')
        zero = end_points['pred_size'].new_zeros(())
        empty = {
            'loss/nocs_correspondence': 0.0,
            'loss/nocs_rot_geo': 0.0,
            'loss/nocs_confidence': 0.0,
            'loss/nocs_fit_residual': 0.0,
            'loss/nocs_valid_ratio': 0.0,
        }
        if weight <= 0.0 or pred_nocs is None:
            return zero, empty

        pts = real_data.get('pts_metric', real_data['pts'])
        gt_size = real_data['size_label']
        gt_rot = real_data['rotation_label']
        gt_trans = real_data['translation_label']
        gt_nocs = inverse_transform_points(pts, gt_rot, gt_trans, gt_size)

        valid = (
            torch.isfinite(pred_nocs).all(dim=-1)
            & torch.isfinite(gt_nocs).all(dim=-1)
            & torch.isfinite(pts).all(dim=-1)
            & (pts[:, :, 2] > 0.0)
        )
        if 'pts_metric_valid' in real_data:
            valid = valid & real_data['pts_metric_valid'].to(device=pts.device).bool()
        finite_sample = (
            valid.any(dim=1)
            & torch.isfinite(gt_size).all(dim=1)
            & torch.isfinite(gt_rot.reshape(gt_rot.shape[0], -1)).all(dim=1)
            & torch.isfinite(gt_trans).all(dim=1)
        )
        if not torch.any(finite_sample):
            return zero, empty

        if bool(OmegaConf.select(self.cfg, "loss.nocs_symmetry_aware", default=True)):
            gt_nocs = self._select_symmetry_nocs_target(
                pred_nocs,
                pts,
                gt_size,
                gt_rot,
                gt_trans,
                real_data.get('category_label'),
            )

        beta = float(OmegaConf.select(self.cfg, "loss.nocs_beta", default=0.02))
        point_valid = valid & finite_sample.view(-1, 1)
        loss_nocs = F.smooth_l1_loss(
            pred_nocs[point_valid],
            gt_nocs[point_valid],
            beta=beta,
        )

        loss_confidence = zero
        if weight_confidence > 0.0 and confidence_logits is not None:
            target_confidence = point_valid.to(dtype=confidence_logits.dtype)
            loss_confidence = F.binary_cross_entropy_with_logits(confidence_logits, target_confidence)

        loss_rot_geo = zero
        residual_mean = 0.0
        if weight_rot_geo > 0.0:
            with autocast(False):
                confidence = end_points.get('pred_nocs_confidence')
                if confidence is None:
                    confidence = point_valid.to(dtype=pred_nocs.dtype)
                confidence = confidence * point_valid.to(dtype=confidence.dtype)
                metric_points = pred_nocs.float() * gt_size.float().unsqueeze(1).clamp_min(1.0e-4)
                pred_rot, _, residual, _ = solve_weighted_rotation_from_points(
                    metric_points[finite_sample],
                    pts[finite_sample].float(),
                    confidence[finite_sample].float(),
                )
                gt_rot_valid = gt_rot[finite_sample].float()
                categories = real_data.get('category_label')
                categories = categories[finite_sample] if categories is not None else None
                if bool(OmegaConf.select(self.cfg, "loss.nocs_symmetry_aware", default=True)):
                    loss_rot_geo = self._symmetry_aware_geodesic_loss(
                        pred_rot, gt_rot_valid, categories).to(dtype=zero.dtype)
                else:
                    rel_rot = torch.matmul(pred_rot.transpose(1, 2), gt_rot_valid)
                    trace = rel_rot[:, 0, 0] + rel_rot[:, 1, 1] + rel_rot[:, 2, 2]
                    cos_theta = ((trace - 1.0) * 0.5).clamp(min=-1.0 + 1.0e-4, max=1.0 - 1.0e-4)
                    loss_rot_geo = torch.acos(cos_theta).mean().to(dtype=zero.dtype)
                residual_mean = float(residual.mean())

        loss = loss_nocs + weight_rot_geo * loss_rot_geo + weight_confidence * loss_confidence
        return loss * weight, {
            'loss/nocs_correspondence': float(loss_nocs),
            'loss/nocs_rot_geo': float(loss_rot_geo),
            'loss/nocs_confidence': float(loss_confidence),
            'loss/nocs_fit_residual': residual_mean,
            'loss/nocs_valid_ratio': float(point_valid.float().mean()),
        }

    def _select_symmetry_nocs_target(self, pred_nocs, pts, gt_size, gt_rot_mat, gt_translation, category_label):
        batch_size = pred_nocs.shape[0]
        target = inverse_transform_points(pts, gt_rot_mat, gt_translation, gt_size)
        sym_mask = self._symmetric_mask(category_label, batch_size, pred_nocs.device)
        if not torch.any(sym_mask):
            return target

        with torch.no_grad():
            y_rots = self._y_rotation_candidates(pred_nocs.device, pred_nocs.dtype)
            gt_sym = torch.matmul(gt_rot_mat[sym_mask].unsqueeze(1), y_rots.unsqueeze(0))
            pts_sym = pts[sym_mask]
            metric_sym = torch.matmul(
                pts_sym.unsqueeze(1) - gt_translation[sym_mask].view(-1, 1, 1, 3),
                gt_sym,
            )
            candidate = metric_sym / gt_size[sym_mask].view(-1, 1, 1, 3).clamp_min(1.0e-6)
            dist = torch.linalg.norm(
                candidate - pred_nocs[sym_mask].unsqueeze(1),
                dim=-1,
            ).mean(dim=-1)
            best_idx = dist.argmin(dim=1)
            selected = candidate[torch.arange(candidate.shape[0], device=candidate.device), best_idx]

        target = target.clone()
        target[sym_mask] = selected
        return target

    def _compute_x0_pose_loss(self, end_points, real_data):
        weight_x0 = float(OmegaConf.select(self.cfg, "loss.weight_x0", default=0.0))
        weight_x0_rot_geo = float(OmegaConf.select(self.cfg, "loss.weight_x0_rot_geo", default=0.0))
        zero = end_points['pred_size'].new_zeros(())
        if weight_x0 <= 0.0:
            return zero, {
                'loss/x0': 0.0,
                'loss/x0_size': 0.0,
                'loss/x0_translation': 0.0,
                'loss/x0_rotation': 0.0,
                'loss/x0_rotation_geo': 0.0,
                'loss/x0_mask_ratio': 0.0,
                'loss/x0_rotation_geo_valid_ratio': 0.0,
            }

        deterministic_translation = (
            end_points.get('translation_prediction_mode')
            == 'deterministic_center_depth'
        )
        required_keys = [
            'noisy_size', 'noisy_rotation',
            'pred_size', 'pred_rotation',
            'diffusion_steps', 'gt_rotation_9d',
        ]
        if not deterministic_translation:
            required_keys.extend(
                ['noisy_translation', 'pred_translation']
            )
        if any(key not in end_points for key in required_keys):
            raise KeyError("x0 loss requires model training outputs: {}".format(required_keys))

        steps = end_points['diffusion_steps'].long()
        t_max = int(OmegaConf.select(self.cfg, "loss.x0_loss_t_max", default=300))
        if t_max > 0:
            mask = steps < t_max
        else:
            mask = torch.ones_like(steps, dtype=torch.bool)

        if not torch.any(mask):
            return zero, {
                'loss/x0': 0.0,
                'loss/x0_size': 0.0,
                'loss/x0_translation': 0.0,
                'loss/x0_rotation': 0.0,
                'loss/x0_rotation_geo': 0.0,
                'loss/x0_mask_ratio': 0.0,
                'loss/x0_rotation_geo_valid_ratio': 0.0,
            }

        x0_size = self._predict_x0_from_noise(end_points['noisy_size'], end_points['pred_size'], steps)
        x0_translation = None
        if not deterministic_translation:
            x0_translation = self._predict_x0_from_noise(
                end_points['noisy_translation'], end_points['pred_translation'], steps)
        x0_rotation = self._predict_x0_from_noise(
            end_points['noisy_rotation'], end_points['pred_rotation'], steps)

        valid_size = mask & torch.isfinite(x0_size).all(dim=1) & torch.isfinite(real_data['size_label']).all(dim=1)
        if torch.any(valid_size):
            loss_x0_size = F.mse_loss(x0_size[valid_size], real_data['size_label'][valid_size])
        else:
            loss_x0_size = zero

        loss_x0_translation = zero
        if not deterministic_translation:
            translation_target = end_points.get('translation_target', real_data['translation_label'])
            translation_clamp_abs = float(OmegaConf.select(self.cfg, "loss.x0_translation_clamp_abs", default=0.0))
            if translation_clamp_abs > 0.0:
                x0_translation = x0_translation.clamp(min=-translation_clamp_abs, max=translation_clamp_abs)
                translation_target = translation_target.clamp(min=-translation_clamp_abs, max=translation_clamp_abs)
            x0_translation_beta = float(OmegaConf.select(self.cfg, "loss.x0_translation_beta", default=0.1))

            valid_translation = mask & torch.isfinite(x0_translation).all(dim=1) & torch.isfinite(translation_target).all(dim=1)
            if torch.any(valid_translation):
                loss_x0_translation = F.smooth_l1_loss(
                    x0_translation[valid_translation],
                    translation_target[valid_translation],
                    beta=x0_translation_beta,
                )

        valid_rotation = mask & torch.isfinite(x0_rotation).all(dim=1) & torch.isfinite(end_points['gt_rotation_9d']).all(dim=1)
        if torch.any(valid_rotation):
            rotation_target_9d = end_points['gt_rotation_9d']
            if bool(OmegaConf.select(self.cfg, "loss.x0_rot_symmetry_aware", default=False)):
                rotation_target_9d = self._select_symmetry_rotation_9d_target(
                    x0_rotation,
                    real_data['rotation_label'],
                    real_data.get('category_label'),
                )
            loss_x0_rotation = F.mse_loss(x0_rotation[valid_rotation], rotation_target_9d[valid_rotation])
        else:
            loss_x0_rotation = zero

        loss_x0_rotation_geo = zero
        loss_x0_rotation_geo_valid_ratio = 0.0
        if weight_x0_rot_geo > 0.0:
            finite_rot_label = torch.isfinite(real_data['rotation_label'].reshape(real_data['rotation_label'].shape[0], -1)).all(dim=1)
            valid_geo = valid_rotation & finite_rot_label
            if torch.any(valid_geo):
                with autocast(False):
                # with autocast(device_type=self.device.split(':')[0], enabled=False):
                    pred_rot_mat = nine_d_to_rotation_matrix(x0_rotation[valid_geo].float())
                    gt_rot_mat = real_data['rotation_label'][valid_geo].float()
                    if bool(OmegaConf.select(self.cfg, "loss.x0_rot_symmetry_aware", default=False)):
                        categories = real_data.get('category_label')
                        categories = categories[valid_geo] if categories is not None else None
                        loss_x0_rotation_geo = self._symmetry_aware_geodesic_loss(
                            pred_rot_mat, gt_rot_mat, categories).to(dtype=zero.dtype)
                    else:
                        rel_rot = torch.matmul(pred_rot_mat.transpose(1, 2), gt_rot_mat)
                        trace = rel_rot[:, 0, 0] + rel_rot[:, 1, 1] + rel_rot[:, 2, 2]
                        cos_theta = ((trace - 1.0) * 0.5).clamp(min=-1.0 + 1.0e-4, max=1.0 - 1.0e-4)
                        loss_x0_rotation_geo = torch.acos(cos_theta).mean().to(dtype=zero.dtype)
            loss_x0_rotation_geo_valid_ratio = float(valid_geo.float().mean())
        loss_x0 = loss_x0_size + loss_x0_translation + loss_x0_rotation + weight_x0_rot_geo * loss_x0_rotation_geo

        return loss_x0 * weight_x0, {
            'loss/x0': float(loss_x0),
            'loss/x0_size': float(loss_x0_size),
            'loss/x0_translation': float(loss_x0_translation),
            'loss/x0_rotation': float(loss_x0_rotation),
            'loss/x0_rotation_geo': float(loss_x0_rotation_geo),
            'loss/x0_mask_ratio': float(mask.float().mean()),
            'loss/x0_rotation_geo_valid_ratio': loss_x0_rotation_geo_valid_ratio,
        }

    def _symmetry_class_ids(self):
        return {
            int(class_id)
            for class_id in OmegaConf.select(
                self.cfg,
                "loss.x0_rot_symmetry_classes",
                default=[0, 1, 3],
            )
        }

    def _category_vector(self, category_label, batch_size, device):
        if category_label is None:
            return torch.full((batch_size,), -1, dtype=torch.long, device=device)
        return category_label.to(device=device).long().view(-1)

    def _y_rotation_candidates(self, device, dtype):
        steps = int(OmegaConf.select(self.cfg, "loss.x0_rot_symmetry_steps", default=36))
        steps = max(1, steps)
        angles = torch.arange(steps, device=device, dtype=dtype) * (2.0 * math.pi / float(steps))
        cos_theta = torch.cos(angles)
        sin_theta = torch.sin(angles)
        rots = torch.zeros((steps, 3, 3), device=device, dtype=dtype)
        rots[:, 0, 0] = cos_theta
        rots[:, 0, 2] = sin_theta
        rots[:, 1, 1] = 1.0
        rots[:, 2, 0] = -sin_theta
        rots[:, 2, 2] = cos_theta
        return rots

    def _symmetric_mask(self, category_label, batch_size, device):
        categories = self._category_vector(category_label, batch_size, device)
        sym_mask = torch.zeros((batch_size,), dtype=torch.bool, device=device)
        for class_id in self._symmetry_class_ids():
            sym_mask |= categories == class_id
        return sym_mask

    def _asymmetric_direct_rotation_mask(self, category_label, batch_size, device):
        categories = self._category_vector(category_label, batch_size, device)
        default_classes = [2, 4, 5]
        asym_classes = {
            int(class_id)
            for class_id in OmegaConf.select(
                self.cfg,
                "loss.direct_rotation_asym_classes",
                default=default_classes,
            )
        }
        mask = torch.zeros((batch_size,), dtype=torch.bool, device=device)
        for class_id in asym_classes:
            mask |= categories == class_id
        return mask

    def _symmetry_aware_geodesic_losses(self, pred_rot_mat, gt_rot_mat, category_label):
        batch_size = pred_rot_mat.shape[0]
        sym_mask = self._symmetric_mask(category_label, batch_size, pred_rot_mat.device)
        losses = pred_rot_mat.new_empty((batch_size,))

        nonsym_mask = ~sym_mask
        if torch.any(nonsym_mask):
            rel_rot = torch.matmul(pred_rot_mat[nonsym_mask].transpose(1, 2), gt_rot_mat[nonsym_mask])
            trace = rel_rot[:, 0, 0] + rel_rot[:, 1, 1] + rel_rot[:, 2, 2]
            cos_theta = ((trace - 1.0) * 0.5).clamp(min=-1.0 + 1.0e-4, max=1.0 - 1.0e-4)
            losses[nonsym_mask] = torch.acos(cos_theta)

        if torch.any(sym_mask):
            y_rots = self._y_rotation_candidates(pred_rot_mat.device, pred_rot_mat.dtype)
            gt_sym = torch.matmul(gt_rot_mat[sym_mask].unsqueeze(1), y_rots.unsqueeze(0))
            rel_rot = torch.matmul(pred_rot_mat[sym_mask].transpose(1, 2).unsqueeze(1), gt_sym)
            trace = rel_rot[..., 0, 0] + rel_rot[..., 1, 1] + rel_rot[..., 2, 2]
            cos_theta = ((trace - 1.0) * 0.5).clamp(min=-1.0 + 1.0e-4, max=1.0 - 1.0e-4)
            losses[sym_mask] = torch.acos(cos_theta).min(dim=1).values

        return losses

    def _symmetry_aware_geodesic_loss(self, pred_rot_mat, gt_rot_mat, category_label):
        losses = self._symmetry_aware_geodesic_losses(pred_rot_mat, gt_rot_mat, category_label)
        return losses.mean()

    def _select_symmetry_rotation_6d_target(self, pred_rotation_6d, gt_rot_mat, category_label):
        batch_size = pred_rotation_6d.shape[0]
        target_rot = gt_rot_mat
        sym_mask = self._symmetric_mask(category_label, batch_size, pred_rotation_6d.device)
        if not torch.any(sym_mask):
            return rotation_matrix_to_6d(target_rot)

        with torch.no_grad():
            pred_rot_mat = six_d_to_rotation_matrix(pred_rotation_6d[sym_mask].float())
            y_rots = self._y_rotation_candidates(pred_rotation_6d.device, torch.float32)
            gt_sym = torch.matmul(gt_rot_mat[sym_mask].float().unsqueeze(1), y_rots.unsqueeze(0))
            rel_rot = torch.matmul(pred_rot_mat.transpose(1, 2).unsqueeze(1), gt_sym)
            trace = rel_rot[..., 0, 0] + rel_rot[..., 1, 1] + rel_rot[..., 2, 2]
            best_idx = trace.argmax(dim=1)
            selected = gt_sym[torch.arange(gt_sym.shape[0], device=gt_sym.device), best_idx]

        target_rot = target_rot.clone()
        target_rot[sym_mask] = selected.to(dtype=target_rot.dtype)
        return rotation_matrix_to_6d(target_rot)

    def _select_symmetry_rotation_9d_target(self, pred_rotation_9d, gt_rot_mat, category_label):
        batch_size = pred_rotation_9d.shape[0]
        target_rot = gt_rot_mat
        sym_mask = self._symmetric_mask(category_label, batch_size, pred_rotation_9d.device)
        if not torch.any(sym_mask):
            return rotation_matrix_to_9d(target_rot)

        with torch.no_grad():
            pred_rot_mat = nine_d_to_rotation_matrix(pred_rotation_9d[sym_mask].float())
            y_rots = self._y_rotation_candidates(pred_rotation_9d.device, torch.float32)
            gt_sym = torch.matmul(gt_rot_mat[sym_mask].float().unsqueeze(1), y_rots.unsqueeze(0))
            rel_rot = torch.matmul(pred_rot_mat.transpose(1, 2).unsqueeze(1), gt_sym)
            trace = rel_rot[..., 0, 0] + rel_rot[..., 1, 1] + rel_rot[..., 2, 2]
            best_idx = trace.argmax(dim=1)
            selected = gt_sym[torch.arange(gt_sym.shape[0], device=gt_sym.device), best_idx]

        target_rot = target_rot.clone()
        target_rot[sym_mask] = selected.to(dtype=target_rot.dtype)
        return rotation_matrix_to_9d(target_rot)

    def _predict_x0_from_noise(self, noisy_sample, pred_noise, steps):
        model = self.model.module if hasattr(self.model, "module") else self.model
        alphas_cumprod = model.noise_scheduler.alphas_cumprod.to(
            device=noisy_sample.device, dtype=noisy_sample.dtype)
        alpha_t = alphas_cumprod[steps].view(-1, *([1] * (noisy_sample.ndim - 1)))
        alpha_t = alpha_t.clamp(min=1.0e-8, max=1.0)
        sigma_t = (1.0 - alpha_t).clamp(min=0.0).sqrt()
        return (noisy_sample - sigma_t * pred_noise) / alpha_t.sqrt()

    def _grads_are_finite(self):
        for param in self.model.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                return False
        return True

    def _params_are_finite(self):
        with torch.no_grad():
            for param in self.model.parameters():
                if not torch.isfinite(param).all():
                    return False
        return True

    def _mark_progress(self, tag):
        self._last_progress_ts = time.time()
        self._last_progress_tag = tag

    def _start_hang_watchdog(self):
        if not self.watchdog_enabled:
            self.logger.warning("[HangWatchdog] disabled")
            return
        try:
            self._hang_dump_file = open(self._hang_dump_path, "a", encoding="utf-8", buffering=1)
            faulthandler.enable(file=self._hang_dump_file, all_threads=True)
        except Exception as exc:
            self.logger.error("[HangWatchdog] failed to initialize faulthandler: %s", str(exc))
            self.watchdog_enabled = False
            return

        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="train-hang-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()
        self.logger.warning(
            "[HangWatchdog] enabled timeout=%ss check_interval=%ss dump_path=%s",
            self.watchdog_timeout_sec,
            self.watchdog_check_interval_sec,
            self._hang_dump_path,
        )

    def _watchdog_loop(self):
        while not self._watchdog_stop_event.wait(self.watchdog_check_interval_sec):
            now = time.time()
            stall_sec = now - self._last_progress_ts
            if stall_sec < self.watchdog_timeout_sec:
                continue
            if now - self._watchdog_last_dump_ts < self.watchdog_dump_cooldown_sec:
                continue
            self._watchdog_last_dump_ts = now
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
            header = (
                f"\n[{ts}] [HangWatchdog] no progress for {stall_sec:.1f}s; "
                f"last_progress={self._last_progress_tag}\n"
            )
            self.logger.error(header.strip())
            try:
                if self._hang_dump_file is not None:
                    self._hang_dump_file.write(header)
                    self._hang_dump_file.flush()
                    faulthandler.dump_traceback(file=self._hang_dump_file, all_threads=True)
                    self._hang_dump_file.write("\n")
                    self._hang_dump_file.flush()
            except Exception as exc:
                self.logger.error("[HangWatchdog] dump failed: %s", str(exc))

    def _stop_hang_watchdog(self):
        self._watchdog_stop_event.set()
        if self._watchdog_thread is not None and self._watchdog_thread.is_alive():
            self._watchdog_thread.join(timeout=2.0)
        try:
            faulthandler.disable()
        except Exception:
            pass
        if self._hang_dump_file is not None:
            try:
                self._hang_dump_file.close()
            except Exception:
                pass

    def dynamic_load_config(self):
        if os.path.isfile(self.cfg_path):
            # Preserve the same recursive extends semantics used by train.py.
            cfg = load_config(self.cfg_path)
            max_lr = float(OmegaConf.select(
                cfg, "cyclic_lr.max_lr", default=self.max_lr
            ))
            base_lr = float(OmegaConf.select(
                cfg, "cyclic_lr.base_lr", default=self.base_lr
            ))
            manual_save_pth = list(OmegaConf.select(
                cfg, "train.manual_save_pth", default=self.manual_save_pth
            ))
            if self.manual_save_pth != manual_save_pth:
                self.manual_save_pth = manual_save_pth
                self.logger.info(">>>>> New Save Point Loaded:{}!!\n".format(self.manual_save_pth))
            if (not math.isclose(self.base_lr, base_lr, rel_tol=1e-9, abs_tol=1e-12) or
                    not math.isclose(self.max_lr, max_lr, rel_tol=1e-9, abs_tol=1e-12)):
                self.max_lr = max_lr
                self.base_lr = base_lr
                self.lr_scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.base_lr,
                                                                max_lr=self.max_lr,
                                                                step_size_up=self.num_mini_batch * 2,
                                                                mode='triangular', cycle_momentum=False,
                                                                last_epoch=self.iter - 1)
                self.logger.info(">>>>> New Lr Loaded: max lr:{}, base lr:{}!!\n".format(self.max_lr, self.base_lr))
        else:
            self.logger.warning(">>>>> Cannot open config file:{}!!\n".format(self.cfg_path))

    def get_logger_info(self, prefix, dict_info):
        info = prefix
        for key, value in self._select_console_metrics(dict_info).items():
            if 'T_' in key:
                info = info + '{}: {:.3f}\t'.format(key, value)
            else:
                info = info + '{}: {:.5f}\t'.format(key, value)

        return info

    def _select_console_metrics(self, dict_info):
        """Filter console and training_logger metrics when a config opts in."""
        configured = OmegaConf.select(
            self.cfg,
            "logging.train_metrics",
            default=None,
        )
        if configured is None:
            return dict(dict_info)

        configured = OmegaConf.to_container(configured, resolve=True)
        if isinstance(configured, list):
            configured = {str(key): str(key) for key in configured}
        if not isinstance(configured, dict):
            raise TypeError("logging.train_metrics must be a mapping or list")

        selected = {}
        for source_name, display_name in configured.items():
            source_name = str(source_name)
            if source_name not in dict_info or display_name in (None, ""):
                continue
            selected[str(display_name)] = dict_info[source_name]
        return selected

    def _select_tensorboard_metrics(self, dict_info, mode):
        """Apply an optional source-tag to TensorBoard-tag allowlist.

        Loss dictionaries intentionally remain complete for console logging,
        epoch averaging, and checkpoint selection.  Only TensorBoard output is
        filtered, so inactive experimental branches do not create permanent
        zero-valued curves in a new run.
        """
        configured = OmegaConf.select(
            self.cfg,
            f"tensorboard.{mode}_metrics",
            default=None,
        )
        if configured is None:
            return dict(dict_info)

        configured = OmegaConf.to_container(configured, resolve=True)
        if isinstance(configured, list):
            configured = {str(key): str(key) for key in configured}
        if not isinstance(configured, dict):
            raise TypeError(
                f"tensorboard.{mode}_metrics must be a mapping or list"
            )

        selected = {}
        for source_name, tensorboard_name in configured.items():
            source_name = str(source_name)
            if source_name not in dict_info or tensorboard_name in (None, ""):
                continue
            tensorboard_name = str(tensorboard_name)
            if tensorboard_name in selected:
                raise ValueError(
                    "duplicate TensorBoard metric tag configured: "
                    f"{tensorboard_name}"
                )
            selected[tensorboard_name] = dict_info[source_name]
        return selected

    def write_summary(self, dict_info, mode):
        summary_metrics = self._select_tensorboard_metrics(dict_info, mode)
        if not summary_metrics:
            return
        keys = list(summary_metrics.keys())
        values = list(summary_metrics.values())
        if mode == "train":
            self.tb_writer.update_scalar(
                list_name=keys, list_value=values, index_counter=0, prefix="")
        elif mode in {"eval", "val"}:
            self.tb_writer.update_scalar(
                list_name=keys,
                list_value=values,
                index_counter=1,
                prefix="eval_" if mode == "eval" else "val/",
            )
        else:
            raise ValueError(f"Unsupported TensorBoard summary mode: {mode}")

    def data_perpare(self, data):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].cuda()

        data_out = {
            'rgb': data['rgb'],
            'pts': data['pts'],
            'latent_code': data['latent_code'],
            'category_name': data['category_name'],
            'instance_name': data['instance_name'],
            'choose': data['choose'],
            'category_label': data['category_label'],
            'sym_info': data['sym_info'],
            'gt_R': rotation_matrix_to_6d(data['rotation_label']),
            'gt_t': data['translation_label'],
            'gt_s': data['size_label'],
            'qo': data['qo'],
        }

        return data_out


class TestingSolver:
    """
    A class for testing models with batched data processing.

    This class handles the complete test pipeline including data extraction,
    model inference, result processing, and output saving.
    """

    def __init__(self, model, save_path, dataloader, cfg, logger=None):
        """
        Initialize the ModelTester.

        Args:
            model: The neural network model to test
            save_path: Directory path to save test results
        """
        self.model = model
        self.save_path = save_path
        self.bbox_img_dir = os.path.join(save_path, "bbox_img")
        self.dataloader = dataloader
        self.cfg = cfg
        self.logger = logger or logging.getLogger(__name__)
        self.image_size = cfg.test.img_size
        self.draw_bbox_during_test = bool(getattr(cfg, "draw_bbox_during_test", False))

        # Create output directories
        os.makedirs(save_path, exist_ok=True)
        if self.draw_bbox_during_test:
            os.makedirs(self.bbox_img_dir, exist_ok=True)

        # Set model to evaluation mode
        if self.model is not None:
            self.model.eval()

    def test(self):
        """
        Execute the test procedure on the entire dataset.

        Args:
            dataloader: DataLoader providing batched test data

        Returns:
            None, results are saved to disk
        """
        total_batches = len(self.dataloader)

        with tqdm(total=total_batches, desc="Testing Progress") as progress_bar:
            for batch_idx, batch_data in enumerate(self.dataloader):
                self._process_batch(batch_data, batch_idx)
                progress_bar.update(1)

    def denoise_sanity(self):
        total_batches = len(self.dataloader)
        stats = {
            'matched': 0,
            'noise_size_l1': [],
            'noise_translation_l1': [],
            'noise_rotation_l1': [],
            'x0_size_rel': [],
            'x0_translation_cm': [],
            'x0_rotation_deg': [],
            'timestep': [],
        }

        was_training = self.model.training
        self.model.train()
        self._set_batchnorm_eval()
        with tqdm(total=total_batches, desc="Denoise Sanity") as progress_bar:
            for _, batch_data in enumerate(self.dataloader):
                self._accumulate_denoise_sanity(batch_data, stats)
                progress_bar.update(1)
        if not was_training:
            self.model.eval()

        self._log_denoise_sanity(stats)

    def _process_batch(self, batch_data, batch_idx):
        """
        Process a single batch of data.

        Args:
            batch_data: Dictionary containing the current batch data
            batch_idx: Index of the current batch
            dataloader: Reference to the DataLoader for metadata access
        """
        if getattr(self.cfg, "sanity_gt", False):
            result = self._process_gt_sanity(batch_data)
        else:
            result = self._process_with_model(batch_data)
        result_index = batch_idx + int(getattr(self.cfg, 'result_index_offset', 0))
        filepath = os.path.join(self.save_path, f"results_{result_index:06d}.pkl")

        with open(filepath, 'wb') as file:
            cPickle.dump(result, file)
        if self.draw_bbox_during_test:
            self._draw_box_to_image(result, self.bbox_img_dir, result_index)

    def _extract_sample(self, batch_data, sample_idx):
        """
        Extract a single sample from batched data.

        Args:
            batch_data: Full batch data dictionary
            sample_idx: Index of sample to extract within batch

        Returns:
            Dictionary containing single sample data
        """
        sample = {}

        for key, tensor in batch_data.items():
            # Extract the specific sample from batch dimension
            sample_tensor = tensor[sample_idx]

            # Add dimension if tensor is 1D
            if sample_tensor.dim() == 1:
                sample_tensor = sample_tensor.unsqueeze(-1)

            sample[key] = sample_tensor

        return sample

    def _build_base_result(self, sample):
        """
        Construct the base result dictionary from sample data.

        Args:
            sample: Single sample data dictionary

        Returns:
            Base result dictionary with ground truth and predictions
        """
        return {
            'gt_class_ids': sample['gt_class_ids'].numpy().flatten(),
            'gt_bboxes': sample['gt_bboxes'].numpy(),
            'gt_RTs': sample['gt_RTs'].numpy(),
            'gt_scales': sample['gt_scales'].numpy(),
            'gt_handle_visibility': sample['gt_handle_visibility'].numpy().flatten(),

            'pred_class_ids': sample['pred_class_ids'].numpy().flatten(),
            'pred_bboxes': sample['pred_bboxes'].numpy(),
            'pred_scores': sample['pred_scores'].numpy().flatten(),
            'ori_img': sample['ori_img']
        }

    def _process_with_model(self, batch_data):
        """
        Process sample through the neural network model.

        Args:
            batch_data: Input batch data

        Returns:
            Result dictionary with model predictions
        """
        batch_data = {
            key: value.squeeze(0) if torch.is_tensor(value) and value.size(0) == 1 else value
            for key, value in batch_data.items()
        }
        ori_img = batch_data.pop('ori_img', None)

        if self._model_requires_octree():
            batch_points = init_batch_points(batch_data)
            batch_data['batch_octree'] = build_batch_octree(
                batch_points,
                self.cfg.octree.depth,
                self.cfg.octree.full_depth,
            )
        batch_data = batch_to_device(batch_data, self.cfg.device)

        with torch.no_grad():
            end_points = self.model(batch_data)

        # Extract and process model predictions
        pred_translation = end_points['pred_translation']
        pred_size = end_points['pred_size']
        pred_rotation = end_points['pred_rotation']
        pred_rt_scale = torch.ones((pred_size.size(0), 1), device=pred_size.device, dtype=pred_size.dtype)
        pred_eval_scales = pred_size
        if getattr(self.cfg, "scale_mode", "raw") == "norm":
            pred_rt_scale = torch.norm(pred_size, dim=1, keepdim=True).clamp_min(1.0e-6)
            pred_eval_scales = pred_size / pred_rt_scale

        if getattr(self.cfg, "debug_translation", False):
            gt_translation = batch_data['gt_RTs'][:, :3, 3]
            self.logger.warning(
                "translation debug - pred min/max/mean: {} / {} / {}; gt min/max/mean: {} / {} / {}".format(
                    pred_translation.min(dim=0).values.detach().cpu().numpy(),
                    pred_translation.max(dim=0).values.detach().cpu().numpy(),
                    pred_translation.mean(dim=0).detach().cpu().numpy(),
                    gt_translation.min(dim=0).values.detach().cpu().numpy(),
                    gt_translation.max(dim=0).values.detach().cpu().numpy(),
                    gt_translation.mean(dim=0).detach().cpu().numpy(),
                )
            )

        # Construct 4x4 transformation matrices (RTs)
        num_instances = pred_rotation.size(0)
        pred_RTs = torch.eye(4, device=pred_rotation.device).unsqueeze(0).repeat(num_instances, 1, 1)
        pred_RTs[:, :3, 3] = pred_translation
        pred_RTs[:, :3, :3] = pred_rotation * pred_rt_scale.unsqueeze(-1)

        result_class_ids = end_points.get("pred_joint_class_ids")
        if result_class_ids is None:
            result_class_ids = batch_data['pred_class_ids']
        result_scores = batch_data['pred_scores'].reshape(-1)
        joint_class_score = end_points.get("pred_joint_class_score")
        joint_score_mode = str(
            OmegaConf.select(
                self.cfg, "joint_geometry.score_mode", default="preserve"
            )
        ).lower()
        if joint_class_score is not None:
            joint_class_score = joint_class_score.to(result_scores).reshape(-1)
            if joint_score_mode == "multiply":
                result_scores = result_scores * joint_class_score
            elif joint_score_mode == "joint":
                result_scores = joint_class_score
            elif joint_score_mode != "preserve":
                raise ValueError(
                    "joint_geometry.score_mode must be preserve, multiply, or joint"
                )

        result = {
            'gt_class_ids': batch_data['gt_class_ids'].detach().cpu().numpy().reshape(-1).astype(np.int32),
            'gt_bboxes': batch_data['gt_bboxes'].detach().cpu().numpy().reshape(-1, 4),
            'gt_RTs': batch_data['gt_RTs'].detach().cpu().numpy().reshape(-1, 4, 4),
            'gt_scales': batch_data['gt_scales'].detach().cpu().numpy().reshape(-1, 3),
            'gt_handle_visibility': batch_data['gt_handle_visibility'].detach().cpu().numpy().reshape(-1),
            'pred_class_ids': result_class_ids.detach().cpu().numpy().reshape(-1).astype(np.int32),
            'pred_bboxes': batch_data['pred_bboxes'].detach().cpu().numpy().reshape(-1, 4),
            'pred_scores': result_scores.detach().cpu().numpy().reshape(-1),
            'pred_RTs': pred_RTs.detach().cpu().numpy(),
            'pred_scales': pred_eval_scales.detach().cpu().numpy().reshape(-1, 3),
            # Preserve the raw pose components needed for no-training R/S
            # counterfactual evaluation.  pred_RTs/pred_scales remain the
            # canonical evaluator inputs, so existing consumers are unchanged.
            'pred_size_raw': pred_size.detach().cpu().numpy().reshape(-1, 3),
            'pred_output_rotation': pred_rotation.detach().cpu().numpy().reshape(-1, 3, 3),
            'direct_rs_result_schema': 'direct_rs_v1',
        }
        if 'pred_diffusion_rotation' in end_points:
            result['pred_diffusion_rotation'] = (
                end_points['pred_diffusion_rotation']
                .detach().cpu().numpy().reshape(-1, 3, 3)
            )
        elif end_points.get('output_pose_mode') is None:
            result['pred_diffusion_rotation'] = result['pred_output_rotation'].copy()
        if 'output_pose_mode' in end_points:
            result['output_pose_mode'] = str(end_points['output_pose_mode'])
        optional_direct_rs_fields = {
            'pred_diffusion_size': ('pred_diffusion_size_raw', (-1, 3)),
            'pred_metric_size': ('pred_metric_size_raw', (-1, 3)),
            'pred_direct_rotation': ('pred_direct_rotation', (-1, 3, 3)),
            'pred_direct_rotation_6d': ('pred_direct_rotation_6d', (-1, 6)),
            'pred_structured_size': ('pred_structured_size_raw', (-1, 3)),
            'pred_structured_rotation': ('pred_structured_rotation', (-1, 3, 3)),
            'pred_structured_rotation_6d': (
                'pred_structured_rotation_6d',
                (-1, 6),
            ),
            'pred_rotation_v6': ('pred_rotation_v6', (-1, 3, 3)),
            'pred_rotation_v6_base': ('pred_rotation_v6_base', (-1, 3, 3)),
            'pred_rotation_v6_hypotheses': (
                'pred_rotation_v6_hypotheses',
                (-1, int(OmegaConf.select(
                    self.cfg,
                    'rotation_v6.num_hypotheses',
                    default=4,
                )), 3, 3),
            ),
            'pred_rotation_v6_axis_angle': (
                'pred_rotation_v6_axis_angle',
                (-1, int(OmegaConf.select(
                    self.cfg,
                    'rotation_v6.num_hypotheses',
                    default=4,
                )), 3),
            ),
            'pred_rotation_v6_logits': (
                'pred_rotation_v6_logits',
                (-1, int(OmegaConf.select(
                    self.cfg,
                    'rotation_v6.num_hypotheses',
                    default=4,
                ))),
            ),
            'pred_rotation_v6_selected_index': (
                'pred_rotation_v6_selected_index',
                (-1,),
            ),
            'pred_rotation_v9': ('pred_rotation_v9', (-1, 3, 3)),
            'pred_rotation_v9_translation': (
                'pred_rotation_v9_translation',
                (-1, 3),
            ),
            'pred_rotation_v9_residual': (
                'pred_rotation_v9_residual',
                (-1,),
            ),
            'pred_rotation_v9_valid_ratio': (
                'pred_rotation_v9_valid_ratio',
                (-1,),
            ),
            'pred_rotation_v9_confidence_mean': (
                'pred_rotation_v9_confidence_mean',
                (-1,),
            ),
            'pred_rotation_v9_solved_valid': (
                'pred_rotation_v9_solved_valid',
                (-1,),
            ),
            'pred_absolute_center_translation': (
                'pred_absolute_center_translation',
                (-1, 3),
            ),
            'pred_absolute_center_translation_raw': (
                'pred_absolute_center_translation_raw',
                (-1, 3),
            ),
            'pred_absolute_center_base_translation': (
                'pred_absolute_center_base_translation',
                (-1, 3),
            ),
            'pred_absolute_center_surface_center': (
                'pred_absolute_center_surface_center',
                (-1, 3),
            ),
            'pred_absolute_center_canonical_offset': (
                'pred_absolute_center_canonical_offset',
                (-1, 3),
            ),
            'pred_absolute_center_point_entropy': (
                'pred_absolute_center_point_entropy',
                (-1,),
            ),
            'pred_absolute_center_query_entropy': (
                'pred_absolute_center_query_entropy',
                (-1,),
            ),
            'pred_absolute_center_fallback_mask': (
                'pred_absolute_center_fallback_mask',
                (-1,),
            ),
            'pred_translation_base': (
                'pred_translation_base',
                (-1, 3),
            ),
            'pred_metric_center_z_surface': (
                'pred_metric_center_z_surface',
                (-1,),
            ),
            'pred_metric_center_z_q25': (
                'pred_metric_center_z_q25',
                (-1,),
            ),
            'pred_metric_center_z_q75': (
                'pred_metric_center_z_q75',
                (-1,),
            ),
            'pred_metric_center_z_valid_ratio': (
                'pred_metric_center_z_valid_ratio',
                (-1,),
            ),
            'pred_metric_center_z_mask_count': (
                'pred_metric_center_z_mask_count',
                (-1,),
            ),
            'pred_metric_center_z_finite_count': (
                'pred_metric_center_z_finite_count',
                (-1,),
            ),
            'pred_metric_center_z_valid_count': (
                'pred_metric_center_z_valid_count',
                (-1,),
            ),
            'pred_metric_center_z_depth_min': (
                'pred_metric_center_z_depth_min',
                (-1,),
            ),
            'pred_metric_center_z_depth_max': (
                'pred_metric_center_z_depth_max',
                (-1,),
            ),
            'pred_metric_center_z_valid': (
                'pred_metric_center_z_valid',
                (-1,),
            ),
            'pred_metric_center_z_enabled': (
                'pred_metric_center_z_enabled',
                (-1,),
            ),
            'pred_metric_center_z_route': (
                'pred_metric_center_z_route',
                (-1,),
            ),
            'pred_metric_center_z_delta_log': (
                'pred_metric_center_z_delta_log',
                (-1,),
            ),
            'pred_class_logits': ('pred_joint_class_logits', (-1, 6)),
            'pred_class_probabilities': (
                'pred_joint_class_probabilities',
                (-1, 6),
            ),
            'pred_detector_category': (
                'pred_detector_category_zero_indexed',
                (-1,),
            ),
            'pred_joint_category': (
                'pred_joint_category_zero_indexed',
                (-1,),
            ),
            'pred_joint_class_score': ('pred_joint_class_score', (-1,)),
            'pred_joint_geometry_gate': ('pred_joint_geometry_gate', (-1,)),
            'pred_joint_geometry_valid': ('pred_joint_geometry_valid', (-1,)),
            'pred_joint_translation_delta': (
                'pred_joint_translation_delta',
                (-1, 3),
            ),
            'pred_joint_log_size_delta': (
                'pred_joint_log_size_delta',
                (-1, 3),
            ),
            'pred_joint_quality_logit': ('pred_joint_quality_logit', (-1,)),
            'pred_rotation_base': ('pred_rotation_base', (-1, 3, 3)),
            'pred_guarded_center_z_translation': (
                'pred_guarded_center_z_translation',
                (-1, 3),
            ),
            'pred_guarded_center_z_candidate_translation': (
                'pred_guarded_center_z_candidate_translation',
                (-1, 3),
            ),
            'pred_guarded_center_z_base_translation': (
                'pred_guarded_center_z_base_translation',
                (-1, 3),
            ),
            'pred_guarded_center_z_depth': (
                'pred_guarded_center_z_depth',
                (-1,),
            ),
            'pred_guarded_center_z_candidate_depth': (
                'pred_guarded_center_z_candidate_depth',
                (-1,),
            ),
            'pred_guarded_center_z_base_depth': (
                'pred_guarded_center_z_base_depth',
                (-1,),
            ),
            'pred_guarded_center_z_surface_depth': (
                'pred_guarded_center_z_surface_depth',
                (-1,),
            ),
            'pred_guarded_center_z_delta_log_depth': (
                'pred_guarded_center_z_delta_log_depth',
                (-1,),
            ),
            'pred_guarded_center_z_raw_delta_log_depth': (
                'pred_guarded_center_z_raw_delta_log_depth',
                (-1,),
            ),
            'pred_guarded_center_z_improve_logit': (
                'pred_guarded_center_z_improve_logit',
                (-1,),
            ),
            'pred_guarded_center_z_confidence': (
                'pred_guarded_center_z_confidence',
                (-1,),
            ),
            'pred_guarded_center_z_fallback_mask': (
                'pred_guarded_center_z_fallback_mask',
                (-1,),
            ),
            'pred_guarded_center_z_invalid_mask': (
                'pred_guarded_center_z_invalid_mask',
                (-1,),
            ),
            'pred_guarded_center_z_low_confidence_mask': (
                'pred_guarded_center_z_low_confidence_mask',
                (-1,),
            ),
            'pred_guarded_center_z_excessive_residual_mask': (
                'pred_guarded_center_z_excessive_residual_mask',
                (-1,),
            ),
        }
        for source_key, (result_key, shape) in optional_direct_rs_fields.items():
            if source_key in end_points:
                result[result_key] = (
                    end_points[source_key].detach().cpu().numpy().reshape(*shape)
                )
        if joint_class_score is not None:
            result['joint_geometry_score_mode'] = joint_score_mode
        if 'translation_prediction_mode' in end_points:
            result['translation_prediction_mode'] = str(
                end_points['translation_prediction_mode']
            )
        if 'joint_geometry_rotation_source' in end_points:
            result['joint_geometry_rotation_source'] = str(
                end_points['joint_geometry_rotation_source']
            )
        if 'pred_keypoint_rotation' in end_points:
            keypoint_rotation = end_points['pred_keypoint_rotation']
            keypoint_translation = end_points['pred_keypoint_translation']
            keypoint_RTs = torch.eye(4, device=keypoint_rotation.device).unsqueeze(0).repeat(num_instances, 1, 1)
            keypoint_RTs[:, :3, 3] = keypoint_translation
            keypoint_RTs[:, :3, :3] = keypoint_rotation * pred_rt_scale.unsqueeze(-1)
            result['pred_keypoint_RTs'] = keypoint_RTs.detach().cpu().numpy()
            result['pred_keypoint_rotation'] = keypoint_rotation.detach().cpu().numpy()
            result['pred_keypoint_translation'] = keypoint_translation.detach().cpu().numpy().reshape(-1, 3)
            result['pred_keypoint_residual'] = end_points['pred_keypoint_residual'].detach().cpu().numpy().reshape(-1)
            result['pred_keypoints_cam'] = end_points['pred_keypoints_cam'].detach().cpu().numpy()
        if 'pred_nocs_rotation' in end_points:
            nocs_rotation = end_points['pred_nocs_rotation']
            nocs_translation = end_points['pred_nocs_translation']
            nocs_RTs = torch.eye(4, device=nocs_rotation.device).unsqueeze(0).repeat(num_instances, 1, 1)
            nocs_RTs[:, :3, 3] = nocs_translation
            nocs_RTs[:, :3, :3] = nocs_rotation * pred_rt_scale.unsqueeze(-1)
            result['pred_nocs_RTs'] = nocs_RTs.detach().cpu().numpy()
            result['pred_nocs_rotation'] = nocs_rotation.detach().cpu().numpy()
            result['pred_nocs_translation'] = nocs_translation.detach().cpu().numpy().reshape(-1, 3)
            result['pred_nocs_residual'] = end_points['pred_nocs_residual'].detach().cpu().numpy().reshape(-1)
            result['pred_nocs_valid_ratio'] = end_points['pred_nocs_valid_ratio'].detach().cpu().numpy().reshape(-1)
        if 'coarse_translation' in end_points:
            result['pred_coarse_translation'] = end_points['coarse_translation'].detach().cpu().numpy().reshape(-1, 3)
            result['pred_translation_xyz'] = pred_translation.detach().cpu().numpy().reshape(-1, 3)
            result['translation_target_mode'] = getattr(self.cfg.diffusion, "translation_target", "absolute")
            result['translation_prediction_mode'] = end_points.get(
                'translation_prediction_mode',
                getattr(
                    self.cfg.diffusion,
                    "translation_target",
                    "absolute",
                ),
            )
        if 'pred_translation_residual' in end_points:
            result['pred_translation_residual'] = end_points['pred_translation_residual'].detach().cpu().numpy().reshape(-1, 3)
        if 'pred_translation_residual_raw' in end_points:
            result['pred_translation_residual_raw'] = end_points['pred_translation_residual_raw'].detach().cpu().numpy().reshape(-1, 3)
        if 'pred_translation_depth' in end_points:
            result['pred_translation_depth'] = end_points['pred_translation_depth'].detach().cpu().numpy().reshape(-1)
        if 'pred_translation_base_depth' in end_points:
            result['pred_translation_base_depth'] = end_points[
                'pred_translation_base_depth'
            ].detach().cpu().numpy().reshape(-1)
        if 'pred_translation_center_uv' in end_points:
            result['pred_translation_center_uv'] = end_points['pred_translation_center_uv'].detach().cpu().numpy().reshape(-1, 2)
        if 'pred_translation_center_offset_normalized' in end_points:
            result['pred_translation_center_offset_normalized'] = end_points[
                'pred_translation_center_offset_normalized'
            ].detach().cpu().numpy().reshape(-1, 2)
        if 'pred_translation_log_depth_residual' in end_points:
            result['pred_translation_log_depth_residual'] = end_points[
                'pred_translation_log_depth_residual'
            ].detach().cpu().numpy().reshape(-1)
        if 'pred_translation_fallback_weight' in end_points:
            result['pred_translation_fallback_weight'] = end_points[
                'pred_translation_fallback_weight'
            ].detach().cpu().numpy().reshape(-1)
        if 'pred_center_z_v4_base_translation' in end_points:
            result['pred_center_z_v4_base_translation'] = end_points[
                'pred_center_z_v4_base_translation'
            ].detach().cpu().numpy().reshape(-1, 3)
        for key in (
            'pred_center_z_v4_depth',
            'pred_center_z_v4_base_depth',
            'pred_center_z_v4_surface_depth',
            'pred_center_z_v4_delta_log_depth',
            'pred_center_z_v4_raw_delta_log_depth',
        ):
            if key in end_points:
                result[key] = end_points[key].detach().cpu().numpy().reshape(-1)
        if 'pred_translation_vote_dispersion' in end_points:
            result['pred_translation_vote_dispersion'] = end_points[
                'pred_translation_vote_dispersion'
            ].detach().cpu().numpy().reshape(-1, 6)
        if 'pred_translation_point_prediction' in end_points:
            result['pred_translation_point_prediction'] = end_points[
                'pred_translation_point_prediction'
            ].detach().cpu().numpy().reshape(-1, 3)
        if 'pred_translation_point_center_uv' in end_points:
            result['pred_translation_point_center_uv'] = end_points[
                'pred_translation_point_center_uv'
            ].detach().cpu().numpy().reshape(-1, 2)
        if 'pred_translation_point_depth_prediction' in end_points:
            result['pred_translation_point_depth_prediction'] = end_points[
                'pred_translation_point_depth_prediction'
            ].detach().cpu().numpy().reshape(-1)
        if 'pred_translation_vote_confidence' in end_points:
            result['pred_translation_vote_confidence_mean'] = end_points[
                'pred_translation_vote_confidence'
            ].mean(dim=1).detach().cpu().numpy().reshape(-1, 2)
        center_anchor_result_fields = {
            'point_anchor': 'pred_point_anchor',
            'anchor_gate_weights': 'anchor_gate_weights',
            'point_center_uv': 'point_center_uv',
            'bbox_center_uv': 'bbox_center_uv',
            'mask_center_uv': 'mask_center_uv',
            'fused_center_uv': 'fused_center_uv',
        }
        for source_key, result_key in center_anchor_result_fields.items():
            if source_key in end_points:
                result[result_key] = end_points[source_key].detach().cpu().numpy()
        if ori_img is not None:
            result['ori_img'] = ori_img[0] if isinstance(ori_img, list) else ori_img

        return result

    @staticmethod
    def _squeeze_single_image_batch(batch_data):
        return {
            key: value.squeeze(0) if torch.is_tensor(value) and value.size(0) == 1 else value
            for key, value in batch_data.items()
        }

    def _process_gt_sanity(self, batch_data):
        batch_data = self._squeeze_single_image_batch(batch_data)
        ori_img = batch_data.get('ori_img')

        gt_class_ids = batch_data['gt_class_ids'].detach().cpu().numpy().reshape(-1).astype(np.int32)
        gt_bboxes = batch_data['gt_bboxes'].detach().cpu().numpy().reshape(-1, 4)
        gt_RTs = batch_data['gt_RTs'].detach().cpu().numpy().reshape(-1, 4, 4)
        gt_scales = batch_data['gt_scales'].detach().cpu().numpy().reshape(-1, 3)
        gt_handle_visibility = batch_data['gt_handle_visibility'].detach().cpu().numpy().reshape(-1)

        result = {
            'gt_class_ids': gt_class_ids,
            'gt_bboxes': gt_bboxes,
            'gt_RTs': gt_RTs,
            'gt_scales': gt_scales,
            'gt_handle_visibility': gt_handle_visibility,
            'pred_class_ids': gt_class_ids.copy(),
            'pred_bboxes': gt_bboxes.copy(),
            'pred_scores': np.ones((gt_class_ids.shape[0],), dtype=np.float32),
            'pred_RTs': gt_RTs.copy(),
            'pred_scales': gt_scales.copy(),
        }
        if ori_img is not None:
            result['ori_img'] = ori_img[0] if isinstance(ori_img, list) else ori_img
        return result

    @staticmethod
    def _bbox_iou_yxyx(box_a, box_b):
        y1 = max(float(box_a[0]), float(box_b[0]))
        x1 = max(float(box_a[1]), float(box_b[1]))
        y2 = min(float(box_a[2]), float(box_b[2]))
        x2 = min(float(box_a[3]), float(box_b[3]))
        inter = max(0.0, y2 - y1) * max(0.0, x2 - x1)
        area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
        area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    @classmethod
    def _match_pred_to_gt(cls, batch_data):
        pred_class_ids = batch_data['pred_class_ids'].detach().cpu().numpy().reshape(-1)
        pred_bboxes = batch_data['pred_bboxes'].detach().cpu().numpy().reshape(-1, 4)
        gt_class_ids = batch_data['gt_class_ids'].detach().cpu().numpy().reshape(-1)
        gt_bboxes = batch_data['gt_bboxes'].detach().cpu().numpy().reshape(-1, 4)

        pred_indices = []
        gt_indices = []
        used_pred = set()
        for gt_idx, gt_class_id in enumerate(gt_class_ids):
            best_pred_idx = -1
            best_iou = -1.0
            for pred_idx, pred_class_id in enumerate(pred_class_ids):
                if pred_idx in used_pred or pred_class_id != gt_class_id:
                    continue
                iou = cls._bbox_iou_yxyx(gt_bboxes[gt_idx], pred_bboxes[pred_idx])
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = pred_idx
            if best_pred_idx >= 0:
                used_pred.add(best_pred_idx)
                pred_indices.append(best_pred_idx)
                gt_indices.append(gt_idx)
        return pred_indices, gt_indices

    @staticmethod
    def _split_srt_torch(gt_rts):
        rotation_scale = gt_rts[:, :3, :3]
        translation = gt_rts[:, :3, 3]
        scale = torch.linalg.det(rotation_scale).abs().clamp_min(1.0e-12).pow(1.0 / 3.0)
        rotation = rotation_scale / scale.view(-1, 1, 1)
        return rotation, translation

    @staticmethod
    def _estimate_x0(noisy, pred_noise, diffusion_steps, scheduler):
        alphas_cumprod = scheduler.alphas_cumprod.to(noisy.device)[diffusion_steps].view(-1, 1)
        return (noisy - torch.sqrt(1.0 - alphas_cumprod) * pred_noise) / torch.sqrt(alphas_cumprod)

    @staticmethod
    def _rotation_error_deg_torch(pred_rotation, gt_rotation):
        rel = torch.matmul(pred_rotation, gt_rotation.transpose(1, 2))
        cos = ((rel[:, 0, 0] + rel[:, 1, 1] + rel[:, 2, 2]) - 1.0) * 0.5
        cos = cos.clamp(-1.0, 1.0)
        return torch.rad2deg(torch.acos(cos))

    def _accumulate_denoise_sanity(self, batch_data, stats):
        batch_data = self._squeeze_single_image_batch(batch_data)
        batch_data.pop('ori_img', None)

        pred_indices, gt_indices = self._match_pred_to_gt(batch_data)
        if not pred_indices:
            return

        pred_indices_t = torch.as_tensor(pred_indices, dtype=torch.long)
        gt_indices_t = torch.as_tensor(gt_indices, dtype=torch.long)
        sanity_batch = {
            'rgb': batch_data['rgb'][pred_indices_t],
            'depth': batch_data['depth'][pred_indices_t],
            'pts': batch_data['pts'][pred_indices_t],
            'choose': batch_data['choose'][pred_indices_t],
            'rgb_points': batch_data['rgb_points'][pred_indices_t],
            'category_label': batch_data['category_label'][pred_indices_t],
        }
        for key in (
            'metric_depth',
            'pts_local',
            'pts_metric',
            'pts_metric_valid',
            'bbox',
            'pred_bboxes',
            'cam_k',
            'image_hw',
        ):
            if key in batch_data:
                sanity_batch[key] = batch_data[key][pred_indices_t]

        gt_rts = batch_data['gt_RTs'][gt_indices_t].float()
        gt_rotation, gt_translation = self._split_srt_torch(gt_rts)
        sanity_batch['rotation_label'] = gt_rotation
        sanity_batch['translation_label'] = gt_translation
        sanity_batch['size_label'] = batch_data['gt_scales'][gt_indices_t].float()

        if self._model_requires_octree():
            batch_points = init_batch_points(sanity_batch)
            sanity_batch['batch_octree'] = build_batch_octree(
                batch_points,
                self.cfg.octree.depth,
                self.cfg.octree.full_depth,
            )
        sanity_batch = batch_to_device(sanity_batch, self.cfg.device)

        with torch.no_grad():
            end_points = self.model(sanity_batch)

        diffusion_steps = end_points['diffusion_steps']
        x0_size = self._estimate_x0(
            end_points['noisy_size'],
            end_points['pred_size'],
            diffusion_steps,
            self._model_ref().noise_scheduler,
        )
        x0_rotation_9d = self._estimate_x0(
            end_points['noisy_rotation'],
            end_points['pred_rotation'],
            diffusion_steps,
            self._model_ref().noise_scheduler,
        )
        x0_rotation = nine_d_to_rotation_matrix(x0_rotation_9d)

        gt_size = sanity_batch['size_label']
        gt_translation = sanity_batch['translation_label']
        gt_rotation = sanity_batch['rotation_label']
        deterministic_translation = (
            end_points.get('translation_prediction_mode')
            == 'deterministic_center_depth'
        )
        if deterministic_translation:
            x0_translation_for_error = end_points['pred_translation']
        else:
            x0_translation = self._estimate_x0(
                end_points['noisy_translation'],
                end_points['pred_translation'],
                diffusion_steps,
                self._model_ref().noise_scheduler,
            )
            target_mode = end_points.get(
                'translation_target_mode',
                OmegaConf.select(self.cfg, "diffusion.translation_target", default="absolute"),
            )
            if target_mode in {"point_center_residual", "gated_center_residual"} and 'coarse_translation' in end_points:
                x0_translation_for_error = end_points['coarse_translation'].to(dtype=x0_translation.dtype) + x0_translation
            elif target_mode in {
                    "bbox_scale_residual",
                    "metric_coarse_residual",
                    "scaled_point_residual",
            } and 'coarse_translation' in end_points:
                residual_scale = end_points.get('translation_residual_scale_values')
                if residual_scale is None:
                    residual_scale = torch.ones_like(x0_translation[:, :1])
                residual = x0_translation * residual_scale.to(device=x0_translation.device, dtype=x0_translation.dtype)
                residual_bound = float(OmegaConf.select(self.cfg, "diffusion.translation_residual_bound", default=0.0))
                if residual_bound > 0.0:
                    bound = residual.new_tensor(residual_bound)
                    residual = bound * torch.tanh(residual / bound.clamp_min(1.0e-6))
                x0_translation_for_error = end_points['coarse_translation'].to(dtype=residual.dtype) + residual
            else:
                x0_translation_for_error = x0_translation

        stats['matched'] += int(gt_size.size(0))
        stats['noise_size_l1'].extend(torch.mean(torch.abs(end_points['pred_size'] - end_points['delta_size']), dim=1).detach().cpu().tolist())
        if not deterministic_translation:
            stats['noise_translation_l1'].extend(torch.mean(torch.abs(end_points['pred_translation'] - end_points['delta_translation']), dim=1).detach().cpu().tolist())
        stats['noise_rotation_l1'].extend(torch.mean(torch.abs(end_points['pred_rotation'] - end_points['delta_rotation']), dim=1).detach().cpu().tolist())
        stats['x0_size_rel'].extend(
            (torch.linalg.norm(x0_size - gt_size, dim=1) / torch.linalg.norm(gt_size, dim=1).clamp_min(1.0e-12)).detach().cpu().tolist()
        )
        stats['x0_translation_cm'].extend(
            (torch.linalg.norm(x0_translation_for_error - gt_translation, dim=1) * 100.0).detach().cpu().tolist()
        )
        stats['x0_rotation_deg'].extend(self._rotation_error_deg_torch(x0_rotation, gt_rotation).detach().cpu().tolist())
        stats['timestep'].extend(diffusion_steps.detach().cpu().tolist())

    @staticmethod
    def _log_stat(logger, name, values):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            logger.warning("{}: no values".format(name))
            return
        logger.warning(
            "{}: mean={:.4f}, median={:.4f}, p75={:.4f}, p90={:.4f}, min={:.4f}, max={:.4f}".format(
                name,
                float(np.mean(values)),
                float(np.median(values)),
                float(np.percentile(values, 75)),
                float(np.percentile(values, 90)),
                float(np.min(values)),
                float(np.max(values)),
            )
        )

    def _log_denoise_sanity(self, stats):
        self.logger.warning("####### One-step Denoise Sanity ###################")
        self.logger.warning("matched instances: {}".format(stats['matched']))
        self._log_stat(self.logger, "noise size L1", stats['noise_size_l1'])
        self._log_stat(self.logger, "noise translation L1", stats['noise_translation_l1'])
        self._log_stat(self.logger, "noise rotation6d L1", stats['noise_rotation_l1'])
        self._log_stat(self.logger, "x0 size relative L2", stats['x0_size_rel'])
        self._log_stat(self.logger, "x0 translation error (cm)", stats['x0_translation_cm'])
        self._log_stat(self.logger, "x0 rotation error (degree)", stats['x0_rotation_deg'])
        self._log_denoise_sanity_by_timestep(stats)

    def _model_ref(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _model_requires_octree(self):
        return bool(getattr(self._model_ref(), "requires_octree", True))

    def _set_batchnorm_eval(self):
        for module in self.model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()

    def _log_denoise_sanity_by_timestep(self, stats):
        timesteps = np.asarray(stats['timestep'], dtype=np.int64)
        if timesteps.size == 0:
            return

        max_step = int(getattr(self.cfg.diffusion, "train_steps", int(timesteps.max()) + 1))
        bins = [
            (0, 100),
            (100, 300),
            (300, 600),
            (600, 1000),
            (1000, max_step),
        ]
        value_keys = [
            ("noise translation L1", "noise_translation_l1"),
            ("noise rotation6d L1", "noise_rotation_l1"),
            ("x0 translation error (cm)", "x0_translation_cm"),
            ("x0 rotation error (degree)", "x0_rotation_deg"),
            ("x0 size relative L2", "x0_size_rel"),
        ]

        self.logger.warning("####### Denoise Sanity by Timestep ###################")
        for lo, hi in bins:
            if lo >= max_step:
                continue
            hi = min(hi, max_step)
            mask = (timesteps >= lo) & (timesteps < hi)
            self.logger.warning("----- timestep [{}, {}) n={} -----".format(lo, hi, int(mask.sum())))
            if not np.any(mask):
                continue
            for label, key in value_keys:
                values = np.asarray(stats[key], dtype=np.float64)[mask]
                self._log_stat(self.logger, label, values)

    def _add_default_predictions(self, result, sample):
        """
        Add default/placeholder predictions when model is not used.

        Args:
            result: Current result dictionary
            sample: Sample data for dimension reference

        Returns:
            Result dictionary with default prediction values
        """
        num_instances = sample['pred_class_ids'].numpy().shape[0]

        # Create default identity transformation matrices
        default_RTs = np.zeros((num_instances, 4, 4))
        default_RTs[:, :3, :3] = np.eye(3)

        # Create default unit scales
        default_scales = np.ones((num_instances, 3))

        # Add defaults to result
        result['pred_RTs'] = default_RTs
        result['pred_scales'] = default_scales

        return result

    def _draw_box_to_image(
            self, result, out_dir, img_id,
            draw_2d=True, draw_3d=True,
            draw_gt=True, draw_pred=True,
            suffix=None,
            match_pred_to_gt=True):
        intrinsics = np.array([[591.0125, 0, 322.525], [0, 590.16775, 244.11084], [0, 0, 1]])
        if 'ori_img' not in result:
            return

        img = np.asarray(result['ori_img']).copy()
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        gt_class_ids = result['gt_class_ids']
        gt_bboxes = result['gt_bboxes']
        gt_RTs = result['gt_RTs']
        gt_scales = result['gt_scales']
        pred_class_ids = result['pred_class_ids']
        pred_bboxes = result['pred_bboxes']
        pred_RTs = result['pred_RTs']
        pred_scales = result['pred_scales']
        pred_scores = result.get('pred_scores', np.ones((len(pred_class_ids),), dtype=np.float32))

        if match_pred_to_gt and draw_pred:
            pred_class_ids, pred_bboxes, pred_RTs, pred_scales, pred_scores = TestingSolver._match_pred_to_gt_for_vis(
                gt_class_ids, gt_RTs, gt_scales,
                result.get('gt_handle_visibility', np.ones_like(gt_class_ids)),
                pred_bboxes, pred_class_ids, pred_scores, pred_RTs, pred_scales)

        if draw_2d:
            if draw_gt:
                for bbox in gt_bboxes.astype(np.int32):
                    y1, x1, y2, x2 = bbox
                    cv2.rectangle(img, (x1, y1), (x2, y2), (220, 40, 40), 2)
            if draw_pred:
                for bbox in pred_bboxes.astype(np.int32):
                    y1, x1, y2, x2 = bbox
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 180, 0), 2)

        if draw_3d:
            if draw_gt:
                for i in range(gt_RTs.shape[0]):
                    bbox_3d = get_3d_bbox(gt_scales[i, :], 0)
                    transformed_bbox_3d = transform_coordinates_3d(bbox_3d, gt_RTs[i, :, :])
                    projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
                    img = draw_bboxes(img, projected_bbox, (255, 0, 0))

            if draw_pred:
                for i in range(pred_RTs.shape[0]):
                    bbox_3d = get_3d_bbox(pred_scales[i, :], 0)
                    transformed_bbox_3d = transform_coordinates_3d(bbox_3d, pred_RTs[i, :, :])
                    projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
                    img = draw_bboxes(img, projected_bbox, (0, 255, 0))

        os.makedirs(out_dir, exist_ok=True)
        suffix = "" if suffix is None else f"_{suffix}"
        out_path = os.path.join(out_dir, f"real_{img_id:06d}_bbox{suffix}.png")
        cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return out_path

    @staticmethod
    def _match_pred_to_gt_for_vis(
            gt_class_ids, gt_RTs, gt_scales, gt_handle_visibility,
            pred_bboxes, pred_class_ids, pred_scores, pred_RTs, pred_scales):
        synset_names = ['BG', 'bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']

        if len(gt_class_ids) == 0 or len(pred_class_ids) == 0:
            return pred_class_ids, pred_bboxes, pred_RTs, pred_scales, pred_scores

        gt_match, pred_order = compute_3d_matches_for_each_gt(
            gt_class_ids.astype(np.int32),
            np.asarray(gt_RTs),
            np.asarray(gt_scales),
            np.asarray(gt_handle_visibility),
            synset_names,
            np.asarray(pred_bboxes),
            np.asarray(pred_class_ids).astype(np.int32),
            np.asarray(pred_scores),
            np.asarray(pred_RTs),
            np.asarray(pred_scales),
        )

        valid_gt = gt_match >= 0
        if not np.any(valid_gt):
            return (
                np.zeros((0,), dtype=np.asarray(pred_class_ids).dtype),
                np.zeros((0, 4), dtype=np.asarray(pred_bboxes).dtype),
                np.zeros((0, 4, 4), dtype=np.asarray(pred_RTs).dtype),
                np.zeros((0, 3), dtype=np.asarray(pred_scales).dtype),
                np.zeros((0,), dtype=np.asarray(pred_scores).dtype),
            )

        pred_order = np.asarray(pred_order, dtype=np.int64)
        matched_pred_indices = pred_order[gt_match[valid_gt].astype(np.int64)]
        return (
            pred_class_ids[matched_pred_indices],
            pred_bboxes[matched_pred_indices],
            pred_RTs[matched_pred_indices],
            pred_scales[matched_pred_indices],
            pred_scores[matched_pred_indices],
        )
