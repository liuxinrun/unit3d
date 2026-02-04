from typing import Dict, List, Optional

import torch
import numpy as np

from mmengine.logging import MMLogger

from mmdet3d.evaluation.metrics import SegMetric
from mmdet3d.registry import METRICS
from .instance_seg_eval import instance_seg_eval
from mmdet3d.evaluation import panoptic_seg_eval
from mmdet3d.evaluation import seg_eval

import multiprocessing as mp
from os import path as osp
import os

@METRICS.register_module()
class UnifiedPanopticSegInstMetric(SegMetric):
    # the order of classes must be [stuff classes, thing classes, unlabeled]
    # id_offset is usually equal to 2**16 for panoptic_seg_eval and is used to
    # separate inst and sem labels for each point
    def __init__(self,
                 thing_class_inds: List[int],
                 stuff_class_inds: List[int],
                 min_num_points: int,
                 id_offset: int,
                 sem_mapping: List[int],   
                 inst_mapping: List[int],   
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 pklfile_prefix: str = None,
                 metric_meta: Optional[Dict] = None,
                 submission_prefix_semantic: str = None,
                 submission_prefix_instance: str = None,
                 **kwargs):
        self.thing_class_inds = thing_class_inds
        self.stuff_class_inds = stuff_class_inds
        self.min_num_points = min_num_points
        self.id_offset = id_offset
        self.metric_meta = metric_meta
        self.sem_mapping = np.array(sem_mapping)
        self.inst_mapping = np.array(inst_mapping)
        self.submission_prefix_semantic = submission_prefix_semantic
        self.submission_prefix_instance = submission_prefix_instance

        super(UnifiedPanopticSegInstMetric, self).__init__(
            pklfile_prefix=pklfile_prefix,
            prefix=prefix,
            collect_device=collect_device,
            **kwargs)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        self.valid_class_ids = self.dataset_meta['seg_valid_class_ids']
        label2cat = self.metric_meta['label2cat']
        ignore_index = self.metric_meta['ignore_index']
        classes = self.metric_meta['classes']
        thing_classes = [classes[i] for i in self.thing_class_inds]
        stuff_classes = [classes[i] for i in self.stuff_class_inds]
        num_stuff_cls = len(stuff_classes)

        if self.submission_prefix_instance is not None:
            self.format_results_instance(results)
        if self.submission_prefix_semantic is not None:
            self.format_results_semantic(results)
        if self.submission_prefix_semantic is not None or self.submission_prefix_instance is not None:
            return {} 
        gt_semantic_masks_inst_task = []
        gt_instance_masks_inst_task = []
        pred_instance_masks_inst_task = []
        pred_instance_labels = []
        pred_instance_scores = []

        gt_semantic_masks_sem_task = []
        pred_semantic_masks_sem_task = []

        gt_masks_pan = []
        pred_masks_pan = []

        for eval_ann, single_pred_results in results:
            if self.metric_meta['dataset_name'] == 'S3DIS':
                pan_gt = {}
                pan_gt['pts_semantic_mask'] = eval_ann['pts_semantic_mask']
                pan_gt['pts_instance_mask'] = \
                                    eval_ann['pts_instance_mask'].copy()

                for stuff_cls in self.stuff_class_inds:
                    pan_gt['pts_instance_mask'][\
                        pan_gt['pts_semantic_mask'] == stuff_cls] = \
                                    np.max(pan_gt['pts_instance_mask']) + 1

                pan_gt['pts_instance_mask'] = np.unique(
                                                pan_gt['pts_instance_mask'],
                                                return_inverse=True)[1]
                gt_masks_pan.append(pan_gt)
            else:
                gt_masks_pan.append(eval_ann)
            pred_masks_pan.append({'pts_instance_mask' : single_pred_results['pts_instance_mask'][1], 
                                   'pts_semantic_mask' : single_pred_results['pts_semantic_mask'][1]})

            gt_semantic_masks_sem_task.append(eval_ann['pts_semantic_mask'])            
            pred_semantic_masks_sem_task.append(
                single_pred_results['pts_semantic_mask'][0])
            if self.metric_meta['dataset_name'] == 'S3DIS':
                gt_semantic_masks_inst_task.append(eval_ann['pts_semantic_mask'])
                gt_instance_masks_inst_task.append(eval_ann['pts_instance_mask'])  
            else:
                sem_mask, inst_mask = self.map_inst_markup(eval_ann['pts_semantic_mask'].copy(), 
                                          eval_ann['pts_instance_mask'].copy(), 
                                          self.valid_class_ids[num_stuff_cls:],
                                          num_stuff_cls)
            

                gt_semantic_masks_inst_task.append(sem_mask)
                gt_instance_masks_inst_task.append(inst_mask)           
            pred_instance_masks_inst_task.append(
                torch.tensor(single_pred_results['pts_instance_mask'][0]))
            pred_instance_labels.append(torch.tensor(single_pred_results['instance_labels']))
            pred_instance_scores.append(torch.tensor(single_pred_results['instance_scores']))

        ret_pan = panoptic_seg_eval(gt_masks_pan, pred_masks_pan, classes,
                                     thing_classes, stuff_classes,
                                     self.min_num_points, self.id_offset,
                                     label2cat, ignore_index, logger)

        ret_sem = seg_eval(
            gt_semantic_masks_sem_task,
            pred_semantic_masks_sem_task,
            label2cat,
            ignore_index[0],
            logger=logger)
        if self.metric_meta['dataset_name'] == 'S3DIS':
            # :-1 for unlabeled
            ret_inst = instance_seg_eval(
                gt_semantic_masks_inst_task,
                gt_instance_masks_inst_task,
                pred_instance_masks_inst_task,
                pred_instance_labels,
                pred_instance_scores,
                valid_class_ids=self.valid_class_ids,
                class_labels=classes[thing_classes],
                logger=logger)
        else:
            ret_inst = instance_seg_eval(
                gt_semantic_masks_inst_task,
                gt_instance_masks_inst_task,
                pred_instance_masks_inst_task,
                pred_instance_labels,
                pred_instance_scores,
                valid_class_ids=self.valid_class_ids[num_stuff_cls:],
                class_labels=classes[num_stuff_cls:-1], # -1 because the last class is 'unlabeled', which is only needed for panoptic and semantic evaluation
                logger=logger)

        return {"pan_metric" : ret_pan, "sem_metric" : ret_sem, "inst_metric" : ret_inst}

    def map_inst_markup(self, pts_semantic_mask, pts_instance_mask, valid_class_ids, num_stuff_cls):
        pts_instance_mask -= num_stuff_cls
        pts_instance_mask[pts_instance_mask < 0] = -1
        pts_semantic_mask -= num_stuff_cls
        pts_semantic_mask[pts_instance_mask == -1] = -1

        mapping = np.array(list(valid_class_ids) + [-1])
        pts_semantic_mask = mapping[pts_semantic_mask]
        
        return pts_semantic_mask, pts_instance_mask

    def format_results_semantic(self, results):
        submission_prefix = self.submission_prefix_semantic
        os.makedirs(submission_prefix)

        for eval_ann, single_pred_results in results:
            scan_idx = eval_ann['lidar_idx']
            pred_sem_mask = single_pred_results['pts_semantic_mask'][0].astype(np.int)
            pred_label = self.sem_mapping[pred_sem_mask]
            
            curr_file = f'{submission_prefix}/{scan_idx}.txt'
            np.savetxt(curr_file, pred_label, fmt='%d')

    def format_results_instance(self, results):
        submission_prefix = self.submission_prefix_instance
        os.makedirs(submission_prefix)
        
        scans_idxs = []
        pred_results = [] 

        for eval_ann, single_pred_results in results:           
            scans_idxs.append(eval_ann['lidar_idx'])
            pred_results.append((single_pred_results['pts_instance_mask'][0],
                                single_pred_results['instance_labels'],
                                single_pred_results['instance_scores']))
        
        save_pred_instances(submission_prefix, scans_idxs, pred_results, self.inst_mapping)    

def save_single_instance(root, scan_id, insts, mapping):
    f = open(osp.join(root, f'{scan_id}.txt'), 'w')
    os.makedirs(osp.join(root, 'predicted_masks'), exist_ok=True)
    for i, (mask, label, score) in enumerate(zip(insts[0], insts[1], insts[2])):
        f.write(f'predicted_masks/{scan_id}_{i:03d}.txt {mapping[label]} {score:.4f}\n')
        mask_path = osp.join(root, 'predicted_masks', f'{scan_id}_{i:03d}.txt')
        np.savetxt(mask_path, mask, fmt='%d')
    f.close()

def save_pred_instances(root, scan_ids, pred_insts, mapping):
    os.makedirs(root, exist_ok=True)
    roots = [root] * len(scan_ids)
    mappings = [mapping] * len(scan_ids)
    pool = mp.Pool()
    pool.starmap(save_single_instance, zip(roots, scan_ids, pred_insts, mappings))
    pool.close()
    pool.join()