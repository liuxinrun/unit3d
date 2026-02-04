# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Sequence
from mmdet3d.evaluation import panoptic_seg_eval, seg_eval
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
import copy
from .indoor_eval import indoor_eval
from mmdet3d.registry import METRICS
from mmdet3d.structures import get_box_type
from pathlib import Path
import torch
import numpy as np
from .instance_seg_eval import instance_seg_eval
from os import path as osp
import os
import multiprocessing as mp
import pandas as pd
import matplotlib.pyplot as plt
@METRICS.register_module()
class IndoorMetric_(BaseMetric):
    """Indoor scene evaluation metric.

    Args:
        iou_thr (float or List[float]): List of iou threshold when calculate
            the metric. Defaults to [0.25, 0.5].
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
    """

    def __init__(self,
                 datasets,
                 datasets_classes,
                 metric_meta,
                 attn_vis_dir=None,
                 seg_vis_output_dir=None,
                 only_det=False,
                 det_vis_dir: str = None,
                 iou_thr: List[float] = [0.25, 0.5],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super(IndoorMetric_, self).__init__(
            prefix=prefix, collect_device=collect_device)
        self.iou_thr = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        self.datasets = datasets
        self.datasets_classes = datasets_classes
        self.det_vis_dir=det_vis_dir
        self.only_det=only_det
        self.metric_meta=metric_meta
        self.seg_vis_output_dir=seg_vis_output_dir
        self.attn_vis_dir=attn_vis_dir
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            
            pred_3d = data_sample['pred_instances_3d']
            pred_mask_3d=data_sample['pred_pts_seg']
            pred_3d['dataset'] = self.get_dataset(data_sample['lidar_path'])
            eval_ann_info = data_sample['eval_ann_info']
            cpu_pred_3d = dict()
            for k, v in pred_3d.items():
                if hasattr(v, 'to'):
                    cpu_pred_3d[k] = v.to('cpu')
                else:
                    cpu_pred_3d[k] = v
            for k, v in pred_mask_3d.items():
                if hasattr(v, 'to'):
                    cpu_pred_3d[k] = v.to('cpu')
                else:
                    cpu_pred_3d[k] = v
            
            self.results.append((eval_ann_info, cpu_pred_3d))####
    def map_inst_markup(self,
                        pts_semantic_mask,
                        pts_instance_mask,
                        valid_class_ids,
                        num_stuff_cls):
        """Map gt instance and semantic classes back from panoptic annotations.

        Args:
            pts_semantic_mask (np.array): of shape (n_raw_points,)
            pts_instance_mask (np.array): of shape (n_raw_points.)
            valid_class_ids (Tuple): of len n_instance_classes
            num_stuff_cls (int): number of stuff classes
        
        Returns:
            Tuple:
                np.array: pts_semantic_mask of shape (n_raw_points,)
                np.array: pts_instance_mask of shape (n_raw_points,)
        """
        pts_instance_mask -= num_stuff_cls
        pts_instance_mask[pts_instance_mask < 0] = -1#
        
        if self.datasets[0]=='s3dis':
            pts_semantic_mask -= (num_stuff_cls-1)
        else:
            pts_semantic_mask -= num_stuff_cls
        pts_semantic_mask[pts_instance_mask == -1] = -1

        mapping = np.array(list(valid_class_ids) + [-1])
        pts_semantic_mask = mapping[pts_semantic_mask]
        
        return pts_semantic_mask, pts_instance_mask
    def map_inst_markup12(self,
                        pts_semantic_mask,
                        pts_instance_mask,
                        valid_class_ids,
                        num_stuff_cls):
        """Map gt instance and semantic classes back from panoptic annotations.

        Args:
            pts_semantic_mask (np.array): of shape (n_raw_points,)
            pts_instance_mask (np.array): of shape (n_raw_points.)
            valid_class_ids (Tuple): of len n_instance_classes
            num_stuff_cls (int): number of stuff classes
        
        Returns:
            Tuple:
                np.array: pts_semantic_mask of shape (n_raw_points,)
                np.array: pts_instance_mask of shape (n_raw_points,)
        """
        pts_instance_mask -= num_stuff_cls
        pts_instance_mask[pts_instance_mask < 0] = -1#
        
        
        pts_semantic_mask -= num_stuff_cls
        pts_semantic_mask[pts_instance_mask == -1] = -1

        mapping = np.array(list(valid_class_ids) + [-1])
        pts_semantic_mask = mapping[pts_semantic_mask]
        
        return pts_semantic_mask, pts_instance_mask
    def compute_seg_metrics(self,logger,results):
        vaild_idxs=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)
        gt_semantic_masks_inst_task = []
        gt_instance_masks_inst_task = []
        pred_instance_masks_inst_task = []
        pred_instance_labels = []
        pred_instance_scores = []
        num_stuff_cls=2
        
        for eval_ann, single_pred_results in results:###
            
            sem_mask, inst_mask = self.map_inst_markup(
                eval_ann['pts_semantic_mask'].copy(), 
                eval_ann['pts_instance_mask'].copy(), 
                vaild_idxs[num_stuff_cls:],
                num_stuff_cls)
            
            gt_semantic_masks_inst_task.append(sem_mask)
            gt_instance_masks_inst_task.append(inst_mask)  
 

            pred_instance_masks_inst_task.append(
                torch.tensor(single_pred_results['pts_instance_mask'][0]))
            
            pred_instance_labels.append(
                torch.tensor(single_pred_results['mask_instance_labels']))
            pred_instance_scores.append(
                torch.tensor(single_pred_results['mask_instance_scores']))

        ret_inst = instance_seg_eval(
                gt_semantic_masks_inst_task,
                gt_instance_masks_inst_task,
                pred_instance_masks_inst_task,
                pred_instance_labels,
                pred_instance_scores,
                valid_class_ids=vaild_idxs[num_stuff_cls:],
                class_labels=self.dataset_meta['classes'],
                logger=logger)
        
        
        return ret_inst

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        ann_infos = [[] for _ in self.datasets]
        pred_results = [[] for _ in self.datasets]
        if not self.only_det:
            seg_inst_result=self.compute_seg_metrics(logger,results,self.seg_vis_output_dir)
        for eval_ann, sinlge_pred_results in results:
            idx = self.datasets.index(sinlge_pred_results['dataset'])
            ann_infos[idx].append(eval_ann)
            pred_results[idx].append(sinlge_pred_results)
            
        
        # some checkpoints may not record the key "box_type_3d"
        box_type_3d, box_mode_3d = get_box_type(
            self.dataset_meta.get('box_type_3d', 'depth'))

        ret_dict = {}
        for i in range(len(self.datasets)):
            ret_dict[self.datasets[i]] = indoor_eval(
                                                ann_infos[i],
                                                pred_results[i],
                                                self.iou_thr,
                                                self.datasets_classes[i],
                                                logger=logger,
                                                box_mode_3d=box_mode_3d)
        if not self.only_det:
            for k, v in seg_inst_result.items():  
                ret_dict[k] = v



        return ret_dict

    def get_dataset(self, lidar_path):
        for dataset in self.datasets:
            if dataset in lidar_path.split('/'):
                return dataset


@METRICS.register_module()
class IndoorMetric_sempan(IndoorMetric_):
    """Indoor scene evaluation metric.

    Args:
        iou_thr (float or List[float]): List of iou threshold when calculate
            the metric. Defaults to [0.25, 0.5].
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix will
            be used instead. Defaults to None.
    """

    def __init__(self,
                 datasets,
                 datasets_classes,
                 stuff_class_inds, 
                 thing_class_inds, 
                 min_num_points, 
                 id_offset,
                 sem_mapping,
                 inst_mapping,
                 metric_meta,
                 seg_vis_output_dir=None,
                 attn_vis_dir=None,
                 only_det=False,
                 submission_prefix_instance=None,
                 submission_prefix_semantic=None,
                 logger_keys=[('miou',),
                              ('all_ap', 'all_ap_50%', 'all_ap_25%'), 
                              ('pq',)],
                 det_vis_dir: str = None,
                 iou_thr: List[float] = [0.25, 0.5],
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super(IndoorMetric_sempan, self).__init__(
            datasets=datasets,datasets_classes=datasets_classes,metric_meta=metric_meta,
            prefix=prefix,only_det=only_det,seg_vis_output_dir=seg_vis_output_dir, attn_vis_dir=attn_vis_dir,
            det_vis_dir=det_vis_dir, collect_device=collect_device)
        self.iou_thr = [iou_thr] if isinstance(iou_thr, float) else iou_thr
        self.datasets = datasets
        self.datasets_classes = datasets_classes
        self.det_vis_dir = det_vis_dir
        self.thing_class_inds=thing_class_inds
        self.stuff_class_inds = stuff_class_inds
        self.min_num_points = min_num_points
        self.id_offset = id_offset
        self.sem_mapping = np.array(sem_mapping)
        self.inst_mapping = np.array(inst_mapping)
        self.metric_meta=metric_meta
        self.submission_prefix_instance = submission_prefix_instance
        self.submission_prefix_semantic = submission_prefix_semantic
        self.seg_vis_output_dir = seg_vis_output_dir
        self.logger_keys = logger_keys
        
    

    def compute_seg_metrics(self,logger,results,vis_output_dir=None):
        
        vaild_idxs=self.sem_mapping
        gt_semantic_masks_inst_task = []
        gt_instance_masks_inst_task = []
        pred_instance_masks_inst_task = []
        pred_instance_labels = []
        pred_instance_scores = []
        
        gt_semantic_masks_sem_task = []
        pred_semantic_masks_sem_task = []

        gt_masks_pan = []
        pred_masks_pan = []
        
        num_stuff_cls=len(self.stuff_class_inds)

        classes = self.metric_meta['classes']
        for eval_ann, single_pred_results in results:###
            if self.metric_meta['dataset_name'] == 'S3DIS':
                sem_mask, inst_mask = self.map_inst_markup(
                eval_ann['pts_semantic_mask'].copy(), 
                eval_ann['pts_instance_mask'].copy(), 
                self.thing_class_inds,
                num_stuff_cls) 
            
            else:
                sem_mask, inst_mask = self.map_inst_markup(
                eval_ann['pts_semantic_mask'].copy(), 
                eval_ann['pts_instance_mask'].copy(), 
                vaild_idxs[num_stuff_cls:],
                num_stuff_cls)
            

            gt_semantic_masks_inst_task.append(sem_mask)
            gt_instance_masks_inst_task.append(inst_mask)  
            
           
            gt_masks_pan.append(eval_ann)
            gt_semantic_masks_sem_task.append(eval_ann['pts_semantic_mask'])  
            pred_semantic_masks_sem_task.append(
                single_pred_results['pts_semantic_mask'][0])        
            pred_masks_pan.append({
                'pts_instance_mask': \
                    single_pred_results['pts_instance_mask'][1],
                'pts_semantic_mask': \
                    single_pred_results['pts_semantic_mask'][1]
            })
            pred_instance_masks_inst_task.append(
                torch.tensor(single_pred_results['pts_instance_mask'][0]))
            
            pred_instance_labels.append(
                torch.tensor(single_pred_results['mask_instance_labels']))
            pred_instance_scores.append(
                torch.tensor(single_pred_results['mask_instance_scores']))

        stuff_classes = [classes[i] for i in self.stuff_class_inds]
        thing_classes = [classes[i] for i in self.thing_class_inds]
        if self.metric_meta['dataset_name'] == 'S3DIS':
            # :-1 for unlabeled
            ret_inst = instance_seg_eval(
                gt_semantic_masks_inst_task,
                gt_instance_masks_inst_task,
                pred_instance_masks_inst_task,
                pred_instance_labels,
                pred_instance_scores,
                valid_class_ids=self.thing_class_inds,
                class_labels=thing_classes,
                logger=logger)
        else:
            ret_inst = instance_seg_eval(
                gt_semantic_masks_inst_task,
                gt_instance_masks_inst_task,
                pred_instance_masks_inst_task,
                pred_instance_labels,
                pred_instance_scores,
                valid_class_ids=vaild_idxs[num_stuff_cls:],
                class_labels=self.dataset_meta['classes'],
                logger=logger)
        
        label2cat = self.metric_meta['label2cat']
        ignore_index = self.metric_meta['ignore_index']
        ret_pan = panoptic_seg_eval(
            gt_masks_pan, pred_masks_pan, classes, thing_classes,
            stuff_classes, self.min_num_points, self.id_offset,
            label2cat, ignore_index, logger)
        ret_sem = seg_eval(
            gt_semantic_masks_sem_task,
            pred_semantic_masks_sem_task,
            label2cat,
            ignore_index[0],
            logger=logger)
        metrics = dict()
        for ret, keys in zip((ret_sem, ret_inst, ret_pan), self.logger_keys):
            for key in keys:
                metrics[key] = ret[key]
        return metrics
    

