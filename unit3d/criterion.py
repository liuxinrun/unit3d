import torch
from typing import Optional
from mmdet3d.structures import AxisAlignedBboxOverlaps3D
from torch import Tensor
import torch.nn.functional as F
import math
from .structures import InstanceData_
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.structures import DepthInstance3DBoxes
import copy
from torch_scatter import scatter_mean, scatter_max, scatter_min

def batch_sigmoid_bce_loss(inputs, targets):
    """Sigmoid BCE loss.

    Args:
        inputs: of shape (n_queries, n_points).
        targets: of shape (n_gts, n_points).
    
    Returns:
        Tensor: Loss of shape (n_queries, n_gts).
    """
    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction='none')
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction='none')

    pos_loss = torch.einsum('nc,mc->nm', pos, targets)
    neg_loss = torch.einsum('nc,mc->nm', neg, (1 - targets))
    
    return (pos_loss + neg_loss) / inputs.shape[1]


def batch_dice_loss(inputs, targets):
    """Dice loss.

    Args:
        inputs: of shape (n_queries, n_points).
        targets: of shape (n_gts, n_points).
    
    Returns:
        Tensor: Loss of shape (n_queries, n_gts).
    """
    inputs = inputs.sigmoid()
    numerator = 2 * torch.einsum('nc,mc->nm', inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def get_iou(inputs, targets):
    """IoU for to equal shape masks.

    Args:
        inputs (Tensor): of shape (n_gts, n_points).
        targets (Tensor): of shape (n_gts, n_points).
    
    Returns:
        Tensor: IoU of shape (n_gts,).
    """
    inputs = inputs.sigmoid()
    binarized_inputs = (inputs >= 0.5).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


def dice_loss(inputs, targets):
    """Compute the DICE loss, similar to generalized IOU for masks.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
            The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs.
            Stores the binary classification label for each element in inputs
            (0 for the negative class and 1 for the positive class).
    
    Returns:
        Tensor: loss value.
    """
    inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()

@MODELS.register_module()
class SegDetCriterion():
    """Universal 3D detection criterion.

    Args:
        matcher (Callable): A class or function for matching queries with 
            ground truth (GT) instances.
        loss_weight (List[float]): A list of two weights corresponding to 
            classification loss and regression loss, respectively.
        iter_matcher (bool): Flag indicating whether to use a separate 
            matcher for each encoder layer.
        bbox_loss_simple (dict): Configuration for bounding box loss 
            w/o angle.
        bbox_loss_rotated (dict): Configuration for bounding box loss 
            with angle.
        datasets (List[str]): A list of dataset names for each scene 
            in batch.
        datasets_weights (List[float]): A list of loss weights corresponding 
            to each dataset in `datasets`.
        topk (List[int]): A list of integer indicating the number 
            of top predictions to consider for each gt bbox 
            when computing losses.
    """

    def __init__(self, seg_matcher, det_matcher,seg_loss_weight,det_loss_weight, non_object_weight,
                 fix_dice_loss_weight, iter_matcher, fix_mean_loss,
                 bbox_loss_simple, bbox_loss_rotated,num_semantic_classes,
                 datasets, datasets_weights, topk):
        self.bbox_loss_simple = MODELS.build(bbox_loss_simple)
        self.bbox_loss_rotated = MODELS.build(bbox_loss_rotated)
        self.seg_matcher = TASK_UTILS.build(seg_matcher)
        self.det_matcher = TASK_UTILS.build(det_matcher)
        self.num_semantic_classes=num_semantic_classes
        self.non_object_weight = non_object_weight
        self.seg_loss_weight = seg_loss_weight
        self.det_loss_weight = det_loss_weight
        self.fix_dice_loss_weight = fix_dice_loss_weight
        self.iter_matcher = iter_matcher
        self.fix_mean_loss = fix_mean_loss
        self.datasets = datasets 
        self.datasets_weights = datasets_weights
        self.topk = topk
        class_weight = [1] * 18 + [non_object_weight]
        self.class_weight=class_weight
        self.expert_number = 8
    
    def seg_det_get_layer_loss(self, aux_outputs_seg,aux_outputs_det, insts_seg,insts_det,
                                datasets_names, indices=None):
        """Per-layer auxiliary loss.

        Args:
            aux_outputs (Dict): A dictionary containing auxiliary outputs, with
                the following keys:
                - 'cls_preds' (List[Tensor]): A list of tensors of shape 
                (n_queries, n_classes + 1) for each sample in the batch, 
                representing predicted class scores.
                - 'bboxes' (List[Tensor]): A list of tensors of shape
                (n_queries, 7) for each sample in the batch, representing 
                the predicted bounding box coordinates.
            
            insts (List[InstanceData_]): A list of ground truth instances for 
                each sample in the batch, where each instance contains:
                - labels_3d (Tensor): Shape (n_gts_i,), containing the labels 
                for the ground truth bounding boxes.
                - bboxes_3d (DepthInstance3DBoxes): ground truth bounding boxes.

            datasets_names (List[str]): A list of dataset names corresponding 
                to each sample in the batch.

            indices (Optional[List[Tuple[Tensor]]]): Indices for matching 
                predicted bboxes with ground truth bboxes. If None, 
                these will be computed internally.

        Returns:
            Tensor: loss value.

        """
        
        cls_preds_seg = aux_outputs_seg['cls_preds']
        pred_scores = aux_outputs_seg['scores']
        pred_masks = aux_outputs_seg['masks']
        indices_seg=None
        indices_det=None
        if indices_seg is None:
            indices_seg = []
            for i in range(len(insts_seg)):
                pred_instances = InstanceData_(
                    scores=cls_preds_seg[i],
                    masks=pred_masks[i])
                gt_instances = InstanceData_(
                    labels=insts_seg[i].labels_3d,
                    masks=insts_seg[i].sp_masks)
                if insts_seg[i].get('query_masks') is not None:
                    gt_instances.query_masks = insts_seg[i].query_masks
                indices_seg.append(self.seg_matcher(pred_instances, gt_instances))

        cls_losses_seg = []
        for cls_pred, inst, (idx_q, idx_gt) in zip(cls_preds_seg, insts_seg, indices_seg):
            n_classes = cls_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), n_classes, dtype=torch.long)
            cls_target[idx_q] = inst.labels_3d[idx_gt]
            cls_losses_seg.append(F.cross_entropy(
                cls_pred, cls_target, cls_pred.new_tensor(self.class_weight)))
        cls_loss_seg = torch.mean(torch.stack(cls_losses_seg))

        # 3 other losses
        score_losses, mask_bce_losses, mask_dice_losses = [], [], []
        for mask, score, inst, (idx_q, idx_gt) in zip(pred_masks, pred_scores,
                                                      insts_seg, indices_seg):
            if len(inst.labels_3d) == 0:
                continue

            pred_mask = mask[idx_q]
            tgt_mask = inst.sp_masks[idx_gt]
            mask_bce_losses.append(F.binary_cross_entropy_with_logits(
            pred_mask, tgt_mask.float()))
            mask_dice_losses.append(dice_loss(pred_mask, tgt_mask.float()))
            
            # check if skip objectness loss
            if score is None:
                continue

            pred_score = score[idx_q]
            with torch.no_grad():
                tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)

            filter_id, _ = torch.where(tgt_score > 0.5)
            if filter_id.numel():
                tgt_score = tgt_score[filter_id]
                pred_score = pred_score[filter_id]
                score_losses.append(F.mse_loss(pred_score, tgt_score))

        if len(score_losses):
            score_loss = torch.stack(score_losses).sum() / len(pred_masks)
        else:
            score_loss = torch.tensor(0)

        if len(mask_bce_losses):
            mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
            mask_dice_loss = torch.stack(mask_dice_losses).sum() / len(pred_masks)

            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss / len(pred_masks) * 4
            
            if self.fix_mean_loss:
                mask_bce_loss  = mask_bce_loss * len(pred_masks) \
                    / len(mask_bce_losses)
                mask_dice_loss  = mask_dice_loss * len(pred_masks) \
                    / len(mask_dice_losses)
        else:
            mask_bce_loss = torch.tensor(0)
            mask_dice_loss = torch.tensor(0)
        


        cls_preds_det = aux_outputs_det['cls_preds']
        pred_bboxes = aux_outputs_det['bboxes']
        pred_routers = aux_outputs_det.get('routers', None)
        if indices_det is None:
            indices_det = []
            for i in range(len(insts_det)):
                idx = self.datasets.index(datasets_names[i])
                pred_instances = InstanceData_(
                    scores=cls_preds_det[i],
                    bboxes=pred_bboxes[i])
                gt_instances = InstanceData_(
                    labels=insts_det[i].labels_3d,
                    query_masks=insts_det[i].query_masks,
                    bboxes=torch.cat((insts_det[i].bboxes_3d.gravity_center, 
                                      insts_det[i].bboxes_3d.tensor[:, 3:] if \
                                      insts_det[i].bboxes_3d.with_yaw else \
                                      insts_det[i].bboxes_3d.tensor[:, 3:6]),
                                      dim=1))
                indices_det.append(self.det_matcher(pred_instances, gt_instances))

        cls_losses_det = []
        for dataset_name, cls_pred, inst, (idx_q, idx_gt) in \
                zip(datasets_names, cls_preds_det, insts_det, indices_det):
            num_classes = cls_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), num_classes, dtype=torch.long)
            cls_target[idx_q] = inst.labels_3d[idx_gt]
            
            idx = self.datasets.index(dataset_name)
            weight = self.datasets_weights[idx]

            cls_losses_det.append(weight * F.cross_entropy(
                cls_pred, cls_target, cls_pred.new_tensor([1] * num_classes + \
                                                          [self.non_object_weight])))
        cls_loss_det = torch.mean(torch.stack(cls_losses_det))

        bbox_losses = []
        for dataset_name, bbox, inst, (idx_q, idx_gt) in zip(datasets_names,
                                                pred_bboxes, insts_det, indices_det):
            if len(inst.labels_3d) == 0:
                continue
            pred_bbox = bbox[idx_q]
            tgt_bbox = inst.bboxes_3d[idx_gt]
            tgt_bbox = torch.cat((tgt_bbox.gravity_center, 
                                  tgt_bbox.tensor[:, 3:] if \
                                  tgt_bbox.with_yaw else \
                                  tgt_bbox.tensor[:, 3:6]),
                                  dim=1)     

            idx = self.datasets.index(dataset_name)
            weight = self.datasets_weights[idx]

            if tgt_bbox.shape[1] == 7: # rotated case
                bbox_losses.append(weight * self.bbox_loss_rotated(
                                        _bbox_to_loss(pred_bbox), 
                                        _bbox_to_loss(tgt_bbox)).mean())
            else:
                bbox_losses.append(weight * self.bbox_loss_simple(
                                        _bbox_to_loss(pred_bbox), 
                                        _bbox_to_loss(tgt_bbox)).mean())
        if len(bbox_losses):
            bbox_loss = torch.stack(bbox_losses).mean()
        else:
            bbox_loss = torch.tensor(0)
        loss_values_seg=[self.seg_loss_weight[0] * cls_loss_seg ,
            self.seg_loss_weight[1] * mask_bce_loss ,
            self.seg_loss_weight[2] * mask_dice_loss, 
            self.seg_loss_weight[3] * score_loss
            ]
        loss_values_det=[self.det_loss_weight[0] * cls_loss_det ,
            self.det_loss_weight[1] * bbox_loss]
        
        total_loss_seg=sum(loss_values_seg)
        total_loss_det=sum(loss_values_det)
        return loss_values_seg,loss_values_det,total_loss_seg,total_loss_det

    def __call__(self, x_seg,x_det, insts_seg,insts_det,  datasets_names):
        """Loss main function.

            pred (Dict): A dictionary containing auxiliary outputs, with
                the following keys:
                - 'cls_preds' (List[Tensor]): A list of tensors of shape 
                (n_queries, n_classes + 1) for each sample in the batch, 
                representing predicted class scores.
                - 'bboxes' (List[Tensor]): A list of tensors of shape
                (n_queries, 7) for each sample in the batch, representing 
                the predicted bounding box coordinates.
            
            insts (List[InstanceData_]): A list of ground truth instances for 
                each sample in the batch, where each instance contains:
                - labels_3d (Tensor): Shape (n_gts_i,), containing the labels 
                for the ground truth bounding boxes.
                - bboxes_3d (DepthInstance3DBoxes): ground truth bounding boxes.

            datasets_names (List[str]): A list of dataset names corresponding 
                to each sample in the batch.
        
        Returns:
            Dict: with instance loss value.
        """
        loss_values_seg,loss_values_det,total_loss_seg,total_loss_det = self.seg_det_get_layer_loss( x_seg,x_det,insts_seg,insts_det, datasets_names)
        
        if 'aux_outputs' in x_seg:
            if self.iter_matcher:
                indices = None
            for i, aux_outputs_seg in enumerate(x_seg['aux_outputs']):
                aux_outputs_det=x_det['aux_outputs'][i]
                loss_values_seg_aux,loss_values_det_aux,total_loss_seg_aux,total_loss_det_aux \
                    = self.seg_det_get_layer_loss(aux_outputs_seg,aux_outputs_det,insts_seg,insts_det, datasets_names)
                for i in range(len(loss_values_seg)):
                    loss_values_seg[i] += loss_values_seg_aux[i]
                for i in range(len(loss_values_det)):
                    loss_values_det[i] += loss_values_det_aux[i]
                total_loss_det+=total_loss_det_aux
                total_loss_seg+=total_loss_seg_aux
        loss_name_seg=['cls_loss_seg', 'mask_bce_loss', 'mask_dice_loss', 'score_loss']
        loss_name_det=['cls_loss_det', 'bbox_loss', 'router_loss']
        loss_dict_seg = dict(zip(loss_name_seg, loss_values_seg))
        loss_dict_det = dict(zip(loss_name_det, loss_values_det))
        loss_dict={**loss_dict_seg, **loss_dict_det}
       
        return loss_dict

@MODELS.register_module()
class SegDetCriterionAlign(SegDetCriterion):
    """Universal 3D detection criterion.

    Args:
        matcher (Callable): A class or function for matching queries with 
            ground truth (GT) instances.
        loss_weight (List[float]): A list of two weights corresponding to 
            classification loss and regression loss, respectively.
        iter_matcher (bool): Flag indicating whether to use a separate 
            matcher for each encoder layer.
        bbox_loss_simple (dict): Configuration for bounding box loss 
            w/o angle.
        bbox_loss_rotated (dict): Configuration for bounding box loss 
            with angle.
        datasets (List[str]): A list of dataset names for each scene 
            in batch.
        datasets_weights (List[float]): A list of loss weights corresponding 
            to each dataset in `datasets`.
        topk (List[int]): A list of integer indicating the number 
            of top predictions to consider for each gt bbox 
            when computing losses.
    """

    def __init__(self, matcher,seg_loss_weight,det_loss_weight, non_object_weight,
                 fix_dice_loss_weight, iter_matcher, fix_mean_loss,
                 bbox_loss_simple, bbox_loss_rotated,num_semantic_classes,
                 datasets, datasets_weights, topk):
        super(SegDetCriterion, self).__init__()
        self.bbox_loss_simple = MODELS.build(bbox_loss_simple)
        self.bbox_loss_rotated = MODELS.build(bbox_loss_rotated)
        self.matcher = TASK_UTILS.build(matcher)
        self.num_semantic_classes=num_semantic_classes
        self.non_object_weight = non_object_weight
        self.seg_loss_weight = seg_loss_weight
        self.det_loss_weight = det_loss_weight
        self.fix_dice_loss_weight = fix_dice_loss_weight
        self.iter_matcher = iter_matcher
        self.fix_mean_loss = fix_mean_loss
        self.datasets = datasets 
        self.datasets_weights = datasets_weights
        self.topk = topk
        if self.datasets==['s3dis']:
            class_weight = [1] * (num_semantic_classes-8) + [non_object_weight]
        else:
            class_weight = [1] * (num_semantic_classes-2) + [non_object_weight]
        self.class_weight=class_weight

        
    def align_layer_loss(self, aux_outputs_seg,aux_outputs_det, insts,
                         sp_pts_masks,points_ori, datasets_names ,indices):
        
        
        cls_preds_seg = aux_outputs_seg['cls_preds']
        pred_scores = aux_outputs_seg['scores']
        pred_masks = aux_outputs_seg['masks']
        cls_preds_det = aux_outputs_det['cls_preds']
        pred_bboxes = aux_outputs_det['bboxes']
        pred_routers = aux_outputs_det.get('routers', None)
        
        if indices is None:
            indices = []
            for i in range(len(insts)):
                pred_instances = InstanceData_(
                    scores_seg=cls_preds_seg[i],
                    masks=pred_masks[i],
                    scores_det=cls_preds_det[i],
                    bboxes=pred_bboxes[i])
                gt_instances = InstanceData_(
                    labels=insts[i].labels_3d,
                    masks=insts[i].sp_masks,
                    query_masks=insts[i].query_masks,
                    bboxes=torch.cat((insts[i].bboxes_3d.gravity_center, 
                                      insts[i].bboxes_3d.tensor[:, 3:] if \
                                      insts[i].bboxes_3d.with_yaw else \
                                      insts[i].bboxes_3d.tensor[:, 3:6]),
                                      dim=1))
                indices.append(self.matcher(pred_instances, gt_instances))
                
        cls_losses_seg = []
        for cls_pred, inst, (idx_q, idx_gt) in zip(cls_preds_seg, insts, indices):
            n_classes = cls_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), n_classes, dtype=torch.long)
            cls_target[idx_q] = inst.labels_3d[idx_gt]
            cls_losses_seg.append(F.cross_entropy(
                cls_pred, cls_target, cls_pred.new_tensor(self.class_weight)))
        cls_loss_seg = torch.mean(torch.stack(cls_losses_seg))

        # 3 other losses
        score_losses, mask_bce_losses, mask_dice_losses = [], [], []
        for mask, score, inst, (idx_q, idx_gt) in zip(pred_masks, pred_scores,
                                                      insts, indices):
            if len(inst.labels_3d) == 0:
                continue

            pred_mask = mask[idx_q]
            tgt_mask = inst.sp_masks[idx_gt]
            mask_bce_losses.append(F.binary_cross_entropy_with_logits(
            pred_mask, tgt_mask.float()))
            mask_dice_losses.append(dice_loss(pred_mask, tgt_mask.float()))
            
            # check if skip objectness loss
            if score is None:
                continue

            pred_score = score[idx_q]
            with torch.no_grad():
                tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)

            filter_id, _ = torch.where(tgt_score > 0.5)
            if filter_id.numel():
                tgt_score = tgt_score[filter_id]
                pred_score = pred_score[filter_id]
                score_losses.append(F.mse_loss(pred_score, tgt_score))

        if len(score_losses):
            score_loss = torch.stack(score_losses).sum() / len(pred_masks)
        else:
            score_loss = torch.tensor(0)

        if len(mask_bce_losses):
            mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
            mask_dice_loss = torch.stack(mask_dice_losses).sum() / len(pred_masks)

            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss / len(pred_masks) * 4
            
            if self.fix_mean_loss:
                mask_bce_loss  = mask_bce_loss * len(pred_masks) \
                    / len(mask_bce_losses)
                mask_dice_loss  = mask_dice_loss * len(pred_masks) \
                    / len(mask_dice_losses)
        else:
            mask_bce_loss = torch.tensor(0)
            mask_dice_loss = torch.tensor(0)


        cls_losses_det = []
        for dataset_name, cls_pred, inst, (idx_q, idx_gt) in \
                zip(datasets_names, cls_preds_det, insts, indices):
            num_classes = cls_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), num_classes, dtype=torch.long)
            cls_target[idx_q] = inst.labels_3d[idx_gt]
            
            idx = self.datasets.index(dataset_name)
            weight = self.datasets_weights[idx]

            cls_losses_det.append(weight * F.cross_entropy(
                cls_pred, cls_target, cls_pred.new_tensor([1] * num_classes + \
                                                          [self.non_object_weight])))
        cls_loss_det = torch.mean(torch.stack(cls_losses_det))
        

        bbox_losses = []
        for dataset_name, bbox, inst, (idx_q, idx_gt) in zip(datasets_names,
                                                pred_bboxes, insts, indices):
            if len(inst.labels_3d) == 0:
                continue
            pred_bbox = bbox[idx_q]
            tgt_bbox = inst.bboxes_3d[idx_gt]
            tgt_bbox = torch.cat((tgt_bbox.gravity_center, 
                                  tgt_bbox.tensor[:, 3:] if \
                                  tgt_bbox.with_yaw else \
                                  tgt_bbox.tensor[:, 3:6]),
                                  dim=1)     

            idx = self.datasets.index(dataset_name)
            weight = self.datasets_weights[idx]

            if tgt_bbox.shape[1] == 7: # rotated case
                bbox_losses.append(weight * self.bbox_loss_rotated(
                                        _bbox_to_loss(pred_bbox), 
                                        _bbox_to_loss(tgt_bbox)).mean())
            else:
                bbox_losses.append(weight * self.bbox_loss_simple(
                                        _bbox_to_loss(pred_bbox), 
                                        _bbox_to_loss(tgt_bbox)).mean())
        if len(bbox_losses):
            bbox_loss = torch.stack(bbox_losses).mean()
        else:
            bbox_loss = torch.tensor(0)
        
        loss_values_seg=[self.seg_loss_weight[0] * cls_loss_seg ,
            self.seg_loss_weight[1] * mask_bce_loss ,
            self.seg_loss_weight[2] * mask_dice_loss, 
            self.seg_loss_weight[3] * score_loss
            ]
        loss_values_det=[self.det_loss_weight[0] * cls_loss_det ,
            self.det_loss_weight[1] * bbox_loss]

        return loss_values_seg,loss_values_det
    
    def seg_det_get_layer_loss(self, aux_outputs_seg,aux_outputs_det, insts,
                                datasets_names,indices=None,dc_inst_arr_mask=None):
        """Per-layer auxiliary loss.

        Args:
            aux_outputs (Dict): A dictionary containing auxiliary outputs, with
                the following keys:
                - 'cls_preds' (List[Tensor]): A list of tensors of shape 
                (n_queries, n_classes + 1) for each sample in the batch, 
                representing predicted class scores.
                - 'bboxes' (List[Tensor]): A list of tensors of shape
                (n_queries, 7) for each sample in the batch, representing 
                the predicted bounding box coordinates.
            
            insts (List[InstanceData_]): A list of ground truth instances for 
                each sample in the batch, where each instance contains:
                - labels_3d (Tensor): Shape (n_gts_i,), containing the labels 
                for the ground truth bounding boxes.
                - bboxes_3d (DepthInstance3DBoxes): ground truth bounding boxes.

            datasets_names (List[str]): A list of dataset names corresponding 
                to each sample in the batch.

            indices (Optional[List[Tuple[Tensor]]]): Indices for matching 
                predicted bboxes with ground truth bboxes. If None, 
                these will be computed internally.

        Returns:
            Tensor: loss value.

        """
        
        cls_preds_seg = aux_outputs_seg['cls_preds']
        pred_scores = aux_outputs_seg['scores']
        pred_masks = aux_outputs_seg['masks']
        cls_preds_det = aux_outputs_det['cls_preds']
        pred_bboxes = aux_outputs_det['bboxes']

        if indices is None:
            indices = []
            for i in range(len(insts)):
                pred_instances = InstanceData_(
                    scores_seg=cls_preds_seg[i],
                    masks=pred_masks[i],
                    scores_det=cls_preds_det[i],
                    bboxes=pred_bboxes[i])
                gt_instances = InstanceData_(
                    labels=insts[i].labels_3d,
                    masks=insts[i].sp_masks,
                    query_masks=insts[i].query_masks,
                    bboxes=torch.cat((insts[i].bboxes_3d.gravity_center, 
                                      insts[i].bboxes_3d.tensor[:, 3:] if \
                                      insts[i].bboxes_3d.with_yaw else \
                                      insts[i].bboxes_3d.tensor[:, 3:6]),
                                      dim=1))
                indices.append(self.matcher(pred_instances, gt_instances))

        cls_losses_seg = []
        for cls_pred, inst, (idx_q, idx_gt) in zip(cls_preds_seg, insts, indices):
            n_classes = cls_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), n_classes, dtype=torch.long)
            cls_target[idx_q] = inst.labels_3d[idx_gt]
            cls_losses_seg.append(F.cross_entropy(
                cls_pred, cls_target, cls_pred.new_tensor(self.class_weight)))
        cls_loss_seg = torch.mean(torch.stack(cls_losses_seg))
                
        # 3 other losses
        score_losses, mask_bce_losses, mask_dice_losses = [], [], []
        for mask, score, inst, (idx_q, idx_gt) in zip(pred_masks, pred_scores,
                                                      insts, indices):
            if len(inst.labels_3d) == 0:
                continue

            pred_mask = mask[idx_q]
            tgt_mask = inst.sp_masks[idx_gt]
            mask_bce_losses.append(F.binary_cross_entropy_with_logits(
            pred_mask, tgt_mask.float()))
            mask_dice_losses.append(dice_loss(pred_mask, tgt_mask.float()))
            
            # check if skip objectness loss
            if score is None:
                continue

            pred_score = score[idx_q]
            with torch.no_grad():
                tgt_score = get_iou(pred_mask, tgt_mask).unsqueeze(1)

            filter_id, _ = torch.where(tgt_score > 0.5)
            if filter_id.numel():
                tgt_score = tgt_score[filter_id]
                pred_score = pred_score[filter_id]
                score_losses.append(F.mse_loss(pred_score, tgt_score))

        if len(score_losses):
            score_loss = torch.stack(score_losses).sum() / len(pred_masks)
        else:
            score_loss = torch.tensor(0)

        if len(mask_bce_losses):
            mask_bce_loss = torch.stack(mask_bce_losses).sum() / len(pred_masks)
            mask_dice_loss = torch.stack(mask_dice_losses).sum() / len(pred_masks)

            if self.fix_dice_loss_weight:
                mask_dice_loss = mask_dice_loss / len(pred_masks) * 4
            
            if self.fix_mean_loss:
                mask_bce_loss  = mask_bce_loss * len(pred_masks) \
                    / len(mask_bce_losses)
                mask_dice_loss  = mask_dice_loss * len(pred_masks) \
                    / len(mask_dice_losses)
        else:
            mask_bce_loss = torch.tensor(0)
            mask_dice_loss = torch.tensor(0)
        
        cls_losses_det = []
        for dataset_name, cls_pred, inst, (idx_q, idx_gt) in \
                zip(datasets_names, cls_preds_det, insts, indices):
            num_classes = cls_pred.shape[1] - 1
            cls_target = cls_pred.new_full(
                (len(cls_pred),), num_classes, dtype=torch.long)
            cls_target[idx_q] = inst.labels_3d[idx_gt]
            
            idx = self.datasets.index(dataset_name)
            weight = self.datasets_weights[idx]

            cls_losses_det.append(weight * F.cross_entropy(
                cls_pred, cls_target, cls_pred.new_tensor([1] * num_classes + \
                                                          [self.non_object_weight])))
        cls_loss_det = torch.mean(torch.stack(cls_losses_det))

        bbox_losses = []
        for dataset_name, bbox, inst, (idx_q, idx_gt) in zip(datasets_names,
                                                pred_bboxes, insts, indices):
            if len(inst.labels_3d) == 0:
                continue
            pred_bbox = bbox[idx_q]
            tgt_bbox = inst.bboxes_3d[idx_gt]
            tgt_bbox = torch.cat((tgt_bbox.gravity_center, 
                                  tgt_bbox.tensor[:, 3:] if \
                                  tgt_bbox.with_yaw else \
                                  tgt_bbox.tensor[:, 3:6]),
                                  dim=1)     

            idx = self.datasets.index(dataset_name)
            weight = self.datasets_weights[idx]

            if tgt_bbox.shape[1] == 7: # rotated case
                bbox_losses.append(weight * self.bbox_loss_rotated(
                                        _bbox_to_loss(pred_bbox), 
                                        _bbox_to_loss(tgt_bbox)).mean())
            else:
                bbox_losses.append(weight * self.bbox_loss_simple(
                                        _bbox_to_loss(pred_bbox), 
                                        _bbox_to_loss(tgt_bbox)).mean())
        if len(bbox_losses):
            bbox_loss = torch.stack(bbox_losses).mean()
        else:
            bbox_loss = torch.tensor(0)

        loss_values_seg=[self.seg_loss_weight[0] * cls_loss_seg ,
            self.seg_loss_weight[1] * mask_bce_loss ,
            self.seg_loss_weight[2] * mask_dice_loss, 
            self.seg_loss_weight[3] * score_loss
            ]
        loss_values_det=[self.det_loss_weight[0] * cls_loss_det ,
            self.det_loss_weight[1] * bbox_loss]
        total_loss_seg=sum(loss_values_seg)
        total_loss_det=sum(loss_values_det)
        
        return loss_values_seg,loss_values_det,total_loss_seg,total_loss_det,indices
   
    def __call__(self, x_seg,x_det, insts,  datasets_names,sp_pts_masks,points):
        """Loss main function.

            pred (Dict): A dictionary containing auxiliary outputs, with
                the following keys:
                - 'cls_preds' (List[Tensor]): A list of tensors of shape 
                (n_queries, n_classes + 1) for each sample in the batch, 
                representing predicted class scores.
                - 'bboxes' (List[Tensor]): A list of tensors of shape
                (n_queries, 7) for each sample in the batch, representing 
                the predicted bounding box coordinates.
            
            insts (List[InstanceData_]): A list of ground truth instances for 
                each sample in the batch, where each instance contains:
                - labels_3d (Tensor): Shape (n_gts_i,), containing the labels 
                for the ground truth bounding boxes.
                - bboxes_3d (DepthInstance3DBoxes): ground truth bounding boxes.

            datasets_names (List[str]): A list of dataset names corresponding 
                to each sample in the batch.
        
        Returns:
            Dict: with instance loss value.
        """

        loss_name_seg=['cls_loss_seg','mask_bce_loss','mask_dice_loss', 'score_loss']
        loss_name_det=['cls_loss_det','bbox_loss']
        
        loss_values_seg,loss_values_det,total_loss_seg,total_loss_det,last_layer_indices = self.seg_det_get_layer_loss( x_seg,x_det,insts, datasets_names)
        last_layer={}
        total_loss_det = sum(loss_values_det)
        total_loss_seg = sum(loss_values_seg)
        
        lossseg = [loss.clone().detach() for loss in loss_values_seg]
        lossdet = [loss.clone().detach() for loss in loss_values_det]
        for i in range(len(loss_values_seg)):
            last_layer['last_layer'+loss_name_seg[i]]=lossseg[i]
        for i in range(len(loss_values_det)):
            last_layer['last_layer'+loss_name_det[i]]=lossdet[i]
        if 'aux_outputs' in x_seg:
            if self.iter_matcher:
                indices = None
            for i, aux_outputs_seg in enumerate(x_seg['aux_outputs']):
                aux_outputs_det=x_det['aux_outputs'][i]
                loss_values_seg_aux,loss_values_det_aux,total_loss_seg_aux,total_loss_det_aux,_ \
                                = self.seg_det_get_layer_loss(aux_outputs_seg,aux_outputs_det,insts, datasets_names)
                for i in range(len(loss_values_seg)):
                    loss_values_seg[i]+=loss_values_seg_aux[i]
                for i in range(len(loss_values_det)):
                    loss_values_det[i]+=loss_values_det_aux[i]
                total_loss_det+=sum(loss_values_det_aux)
                total_loss_seg+=sum(loss_values_seg_aux)
        
        loss_dict_seg = dict(zip(loss_name_seg, loss_values_seg))
        loss_dict_det = dict(zip(loss_name_det, loss_values_det))
        loss_dict={**loss_dict_seg, **loss_dict_det, **last_layer}
        return loss_dict

def _bbox_to_loss(bbox):
    """Transform box to the axis-aligned or rotated iou loss format.

    Args:
        bbox (Tensor): 3D box of shape (N, 6) or (N, 7).

    Returns:
        Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
    """
    # # rotated iou loss accepts (x, y, z, w, h, l, heading)
    if bbox.shape[-1] != 6:
        return bbox

    # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
    return torch.stack(
        (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
            bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
            bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
        dim=-1)

@TASK_UTILS.register_module()
class BOXMASKClassificationCost:
    """Classification cost for queries.

    Args:
        weigth (float): Weight of the cost.
    """
    def __init__(self, weight):
        self.weight = weight
    
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                must contain `scores` of shape (n_pred_bboxes, n_classes + 1),
            gt_instances (:obj:`InstanceData_`): Ground truth which must contain
                `labels` of shape (n_gt_bboxes,).

        Returns:
            Tensor: Cost of shape (n_pred_bboxes, n_gt_bboxes).
        """
        scores_seg = pred_instances.scores_seg.softmax(-1)
        scores_det = pred_instances.scores_det.softmax(-1)
        cost_seg = -scores_seg[:, gt_instances.labels]
        cost_det = -scores_det[:, gt_instances.labels]
        return (cost_seg+cost_det) * self.weight
@TASK_UTILS.register_module()
class QueryClassificationCost:
    """Classification cost for queries.

    Args:
        weigth (float): Weight of the cost.
    """
    def __init__(self, weight):
        self.weight = weight
    
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                must contain `scores` of shape (n_pred_bboxes, n_classes + 1),
            gt_instances (:obj:`InstanceData_`): Ground truth which must contain
                `labels` of shape (n_gt_bboxes,).

        Returns:
            Tensor: Cost of shape (n_pred_bboxes, n_gt_bboxes).
        """
        scores = pred_instances.scores.softmax(-1)
        cost = -scores[:, gt_instances.labels]
        return cost * self.weight
@TASK_UTILS.register_module()
class xyzmeansCost:

    def __init__(self, weight):
        self.weight = weight
    
    def __call__(self, pred_instances, gt_instances, **kwargs):

        pred_xyz_mean = pred_instances.pred_xyz_means
        gt_xyz_mean = gt_instances.xyz_means
        cost =  torch.cdist(
                pred_xyz_mean, gt_xyz_mean, p=1
            )

        return cost * self.weight

@TASK_UTILS.register_module()
class BboxCostJointTraining:
    """Regression cost for bounding boxes.

    Args:
        weigth (float): Weight of the cost.
        bbox_loss_simple (dict): Configuration for 
            bounding box loss w/o angle.
        bbox_loss_rotated (dict): Configuration for 
            bounding box loss with angle.
    """
    def __init__(self, weight, loss_simple, loss_rotated):
        self.weight = weight
        self.loss_simple = MODELS.build(loss_simple)
        self.loss_rotated = MODELS.build(loss_rotated)

    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                mast contain `bboxes` of shape (n_pred_bboxes, 6) or 
                (n_pred_bboxes, 7).
            gt_instances (:obj:`InstanceData_`): Ground truth which must 
                contain `bboxes` of shape (n_gt_bboxes, 6) or 
                (n_gt_bboxes, 7).
        
        Returns:
            Tensor: Cost of shape (n_pred_bboxes, n_gt_bboxes).
        """
        pred_bboxes = pred_instances.bboxes.\
                        unsqueeze(axis=1).repeat(1, 
                                                 gt_instances.bboxes.shape[0], 
                                                 1)
        gt_bboxes = gt_instances.bboxes.\
                        unsqueeze(axis=0).repeat(pred_bboxes.shape[0], 
                                                 1, 1)
        assert gt_instances.bboxes.shape[1] == pred_instances.bboxes.shape[1]  
        if gt_instances.bboxes.shape[1] == 7: #rotated case
            cost = self.loss_rotated(_bbox_to_loss(pred_bboxes), 
                                    _bbox_to_loss(gt_bboxes))
        else:
            cost = self.loss_simple(_bbox_to_loss(pred_bboxes), 
                                    _bbox_to_loss(gt_bboxes))
        return cost * self.weight

@TASK_UTILS.register_module()
class UniMatcher:
    """Match only queries to their including objects.

    Args:
        costs (List[Callable]): Cost functions.
    """

    def __init__(self, costs,topk):
        self.costs = []
        self.inf = 1e8
        self.topk = topk
        for cost in costs:
            self.costs.append(TASK_UTILS.build(cost))

    @torch.no_grad()
    def __call__(self, pred_instances, gt_instances,  **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                can contain `bboxes` of shape (n_pred_bboxes, 6), `scores`
                of shape (n_pred_bboxes, n_classes + 1),
            gt_instances (:obj:`InstanceData_`): Ground truth which can contain
                `labels` of shape (n_gt_bboxes,), `bboxes` of shape (n_gt_bboxes, 6),
            topk (int): Limit topk matches per bbox.

        Returns:
            Tuple:
                Tensor: Query ids of shape (n_matched,),
                Tensor: Object ids of shape (n_matched,).
        """
        labels = gt_instances.labels
        n_gts = len(labels)
        if n_gts == 0:
            return labels.new_empty((0,)), labels.new_empty((0,))
        
        cost_values = []
        for cost in self.costs:
            cost_values.append(cost(pred_instances, gt_instances))
        # of shape (n_queries, n_gts)
        cost_value = torch.stack(cost_values).sum(dim=0)
        cost_value = torch.where(
            gt_instances.query_masks.T, cost_value, self.inf)

        values = torch.topk(
            cost_value, self.topk + 1, dim=0, sorted=True,
            largest=False).values[-1:, :]
        ids = torch.argwhere(cost_value < values)
        return ids[:, 0], ids[:, 1]

@TASK_UTILS.register_module()
class SparseMatcher:
    """Match only queries to their including objects.

    Args:
        costs (List[Callable]): Cost functions.
        topk (int): Limit topk matches per query.
    """

    def __init__(self, costs, topk):
        self.topk = topk
        self.costs = []
        self.inf = 1e8
        for cost in costs:
            self.costs.append(TASK_UTILS.build(cost))

    @torch.no_grad()
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                can contain `masks` of shape (n_queries, n_points), `scores`
                of shape (n_queries, n_classes + 1),
            gt_instances (:obj:`InstanceData_`): Ground truth which can contain
                `labels` of shape (n_gts,), `masks` of shape (n_gts, n_points),
                `query_masks` of shape (n_gts, n_queries).

        Returns:
            Tuple:
                Tensor: Query ids of shape (n_matched,),
                Tensor: Object ids of shape (n_matched,).
        """
        labels = gt_instances.labels
        n_gts = len(labels)
        if n_gts == 0:
            return labels.new_empty((0,)), labels.new_empty((0,))
        
        cost_values = []
        for cost in self.costs:
            cost_values.append(cost(pred_instances, gt_instances))
        # of shape (n_queries, n_gts)
        cost_value = torch.stack(cost_values).sum(dim=0)
        cost_value = torch.where(
            gt_instances.query_masks.T, cost_value, self.inf)

        values = torch.topk(
            cost_value, self.topk + 1, dim=0, sorted=True,
            largest=False).values[-1:, :]
        ids = torch.argwhere(cost_value < values)
        return ids[:, 0], ids[:, 1]



def ensure_tensor(x, name):
    if not torch.is_tensor(x):
        raise TypeError(f"{name} must be a torch.Tensor")
    return x

@TASK_UTILS.register_module()
class CenterRegDistance:
    def __init__(self, weight):
        self.weight = weight
    def __call__(self, pred_instances, gt_instances, **kwargs):
        gt_centers = gt_instances.bboxes[:,:3]
        pre_centers=pred_instances.center_preds
        pre_sizes=pred_instances.size_preds
        pred_regs=pred_instances.center_regs
        n_pred = pre_centers.shape[0]
        n_gt = gt_centers.shape[0]
        pc = pre_centers.unsqueeze(1).expand(n_pred, n_gt, 3)
        ps = pre_sizes.unsqueeze(1).expand(n_pred, n_gt, 3)
        gc = gt_centers.unsqueeze(0).expand(n_pred, n_gt, 3)
        gt_center_reg = (gc - pc) / (ps + 1e-5)
        pr = pred_regs.unsqueeze(1).expand(n_pred, n_gt, 3)
        dist = torch.abs(pr - gt_center_reg).sum(dim=-1)
        return dist * self.weight
    
@TASK_UTILS.register_module()
class SizeRegDistance:
    
    def __init__(self, weight):
        self.weight = weight
    def __call__(self, pred_instances, gt_instances, **kwargs):
        gt_sizes=gt_instances.bboxes[:,3:]
        pre_sizes=pred_instances.size_preds
        pred_regs=pred_instances.size_regs
        n_pred = pre_sizes.shape[0]
        n_gt = gt_sizes.shape[0]
        ps = pre_sizes.unsqueeze(1).expand(n_pred, n_gt, 3)
        gs = gt_sizes.unsqueeze(0).expand(n_pred, n_gt, 3)
        gt_size_reg = torch.log((gs + 1e-5) / (ps + 1e-5))
        pr = pred_regs.unsqueeze(1).expand(n_pred, n_gt, 3)
        dist = torch.abs(pr - gt_size_reg).sum(dim=-1)
        return dist * self.weight
    


@TASK_UTILS.register_module()
class MaskBCECost:
    """Sigmoid BCE cost for masks.

    Args:
        weigth (float): Weight of the cost.
    """
    def __init__(self, weight):
        self.weight = weight
    
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                mast contain `masks` of shape (n_queries, n_points).
            gt_instances (:obj:`InstanceData_`): Ground truth which must contain
                `labels` of shape (n_gts,), `masks` of shape (n_gts, n_points).
        
        Returns:
            Tensor: Cost of shape (n_queries, n_gts).
        """
        cost = batch_sigmoid_bce_loss(
            pred_instances.masks, gt_instances.masks.float())
        return cost * self.weight


@TASK_UTILS.register_module()
class MaskDiceCost:
    """Dice cost for masks.

    Args:
        weigth (float): Weight of the cost.
    """
    def __init__(self, weight):
        self.weight = weight
    
    def __call__(self, pred_instances, gt_instances, **kwargs):
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData_`): Predicted instances which
                mast contain `masks` of shape (n_queries, n_points).
            gt_instances (:obj:`InstanceData_`): Ground truth which must contain
                `masks` of shape (n_gts, n_points).
        
        Returns:
            Tensor: Cost of shape (n_queries, n_gts).
        """
        cost = batch_dice_loss(
            pred_instances.masks, gt_instances.masks.float())
        return cost * self.weight