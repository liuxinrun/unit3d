from mmdet3d.registry import MODELS
from .structures import InstanceData_


@MODELS.register_module()
class ScanNetUnifiedCriterion:
    """Simply call semantic and instance criterions.

    Args:
        num_semantic_classes (int): Number of semantic classes.
        sem_criterion (ConfigDict): Class for semantic loss calculation.
        inst_criterion (ConfigDict): Class for instance loss calculation.
    """

    def __init__(self, num_semantic_classes, sem_criterion, inst_criterion,use_ray_loss=False):
        self.num_semantic_classes = num_semantic_classes
        self.sem_criterion = MODELS.build(sem_criterion)
        self.inst_criterion = MODELS.build(inst_criterion)
        self.use_ray_loss=use_ray_loss
    def __call__(self, x_seg,x_det, insts,  datasets_names,sp_pts_masks,points):
        """Calculate loss.

        Args:
            pred (Dict):
                List `cls_preds` of shape len batch_size, each of shape
                    (n_queries, n_classes + 1)
                List `scores` of len batch_size each of shape (n_queries, 1)
                List `masks` of len batch_size each of shape
                    (n_queries, n_points)
                Dict `aux_preds` with list of cls_preds, scores, and masks
                List `sem_preds` of len batch_size each of shape
                    (n_queries, n_classes + 1).
            insts (list): Ground truth of len batch_size,
                each InstanceData_ with
                    `sp_masks` of shape (n_gts_i + n_classes + 1, n_points_i)
                    `labels_3d` of shape (n_gts_i + n_classes + 1,)
                    `query_masks` of shape
                        (n_gts_i + n_classes + 1, n_queries_i).

        Returns:
            Dict: with semantic and instance loss values.
        """
        sem_gts = []
        inst_gts = []
        n = self.num_semantic_classes
        for i in range(len(x_seg['masks'])):
            sem_gt = InstanceData_()
            if insts[i].get('query_masks') is not None:
                sem_gt.sp_masks = insts[i].query_masks[-n - 1:, :]
            else:
                sem_gt.sp_masks = insts[i].sp_masks[-n - 1:, :]
            sem_gts.append(sem_gt)

            inst_gt = InstanceData_()
            inst_gt.sp_masks = insts[i].sp_masks[:-n - 1, :]
            inst_gt.labels_3d = insts[i].labels_3d[:-n - 1]
            inst_gt.bboxes_3d = insts[i].bboxes_3d
            inst_gt.ori_sp_centers = insts[i].ori_sp_centers
            inst_gt.xyz_mean = insts[i].xyz_mean
            if insts[i].get('query_masks') is not None:
                inst_gt.query_masks = insts[i].query_masks[:-n - 1,
                 :]
            inst_gts.append(inst_gt)
        
        
        loss = self.inst_criterion( x_seg,x_det, inst_gts,  datasets_names,sp_pts_masks,points)
        loss.update(self.sem_criterion(x_seg, sem_gts, alignlayer=False))
        
        return loss
    



@MODELS.register_module()
class S3disUnifiedCriterion:
    """Simply call semantic and instance criterions.

    Args:
        num_semantic_classes (int): Number of semantic classes.
        sem_criterion (ConfigDict): Class for semantic loss calculation.
        inst_criterion (ConfigDict): Class for instance loss calculation.
    """

    def __init__(self, num_semantic_classes, sem_criterion, inst_criterion,use_ray_loss=False):
        self.num_semantic_classes = num_semantic_classes
        self.sem_criterion = MODELS.build(sem_criterion)
        self.inst_criterion = MODELS.build(inst_criterion)
        self.use_ray_loss=use_ray_loss
    def __call__(self, x_seg,x_det, insts,  datasets_names,sp_pts_masks,points):
        """Calculate loss.

        Args:
            pred (Dict):
                List `cls_preds` of shape len batch_size, each of shape
                    (n_queries, n_classes + 1)
                List `scores` of len batch_size each of shape (n_queries, 1)
                List `masks` of len batch_size each of shape
                    (n_queries, n_points)
                Dict `aux_preds` with list of cls_preds, scores, and masks
                List `sem_preds` of len batch_size each of shape
                    (n_queries, n_classes + 1).
            insts (list): Ground truth of len batch_size,
                each InstanceData_ with
                    `sp_masks` of shape (n_gts_i + n_classes + 1, n_points_i)
                    `labels_3d` of shape (n_gts_i + n_classes + 1,)
                    `query_masks` of shape
                        (n_gts_i + n_classes + 1, n_queries_i).

        Returns:
            Dict: with semantic and instance loss values.
        """
        sem_gts = []
        inst_gts = []
        n = self.num_semantic_classes
        for i in range(len(x_seg['masks'])):
            sem_gt = InstanceData_()
            if insts[i].get('query_masks') is not None:
                sem_gt.sp_masks = insts[i].query_masks[-n:, :]
            else:
                sem_gt.sp_masks = insts[i].sp_masks[-n:, :]
            sem_gts.append(sem_gt)

            inst_gt = InstanceData_()
            inst_gt.sp_masks = insts[i].sp_masks[:-n, :]
            inst_gt.labels_3d = insts[i].labels_3d[:-n]
            inst_gt.bboxes_3d = insts[i].bboxes_3d
            inst_gt.ori_sp_centers = insts[i].ori_sp_centers
            
            if insts[i].get('query_masks') is not None:
                inst_gt.query_masks = insts[i].query_masks[:-n,
                 :]
            inst_gts.append(inst_gt)
        
        
        loss = self.inst_criterion( x_seg,x_det, inst_gts,  datasets_names,sp_pts_masks,points)
        loss.update(self.sem_criterion(x_seg, sem_gts, alignlayer=False))
        
        return loss
    