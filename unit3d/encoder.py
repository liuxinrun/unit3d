import torch
import torch.nn as nn
import itertools
import numpy as np
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
import torch
from torch import nn
from .PositionEmbeddingCoords import PositionEmbeddingCoordsSine
import spconv.pytorch as spconv
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
import functools
import math
from .blocks_spherical_mask import MLP, GenericMLP, ResidualBlock, UBlock, conv_with_kaiming_uniform
from .spherical_mask_model_utils import get_instance_info_dyco_ray,get_cropped_instance_label,get_batch_offsets,\
get_spp_gt_ray
from .maskboxRPE import MaskPredictorWithBoxRPE
class CrossAttentionLayer(BaseModule):
    """Cross attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, dropout, fix=False, return_attn = False):
        super().__init__()
        self.fix = fix
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        # todo: why BaseModule doesn't call it without us?
        self.init_weights()
        self.return_attn=return_attn
    def init_weights(self):
        """Init weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sources, queries, attn_masks=None):
        """Forward pass.

        Args:
            sources (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).
            queries (List[Tensor]): of len batch_size,
                each of shape(n_queries_i, d_model).
            attn_masks (List[Tensor] or None): of len batch_size,
                each of shape (n_queries, n_points).
        
        Return:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        outputs = []
        attn_weights_list=[]
        for i in range(len(sources)):
            k = v = sources[i]
            attn_mask = attn_masks[i] if attn_masks is not None else None
            if self.return_attn:
                output, attn_weights= self.attn(queries[i], k, v, attn_mask=attn_mask)
                attn_weights_list.append(attn_weights.detach().cpu())
            else:
                output, _ = self.attn(queries[i], k, v, attn_mask=attn_mask)
            if self.fix:
                output = self.dropout(output)
            output = output + queries[i]
            if self.fix:
                output = self.norm(output)
            outputs.append(output)
            
        if self.return_attn:
            return outputs, attn_weights_list 
        return outputs


class SelfAttentionLayer_pos(BaseModule):
    """Self attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, pos):
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        
        Returns:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        out = []
        for i in range(len(x)):

            z, _ = self.attn(x[i]+pos[i], x[i]+pos[i], x[i])
            z = self.dropout(z) + x[i]
            z = self.norm(z)
            out.append(z)
        return out
class SelfAttentionLayer(BaseModule):
    """Self attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        
        Returns:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        out = []
        for i in range(len(x)):

            z, _ = self.attn(x[i], x[i], x[i])
            z = self.dropout(z) + x[i]
            z = self.norm(z)
            out.append(z)
        return out

class FFN(BaseModule):
    """Feed forward network.

    Args:
        d_model (int): Model dimension.
        hidden_dim (int): Hidden dimension.
        dropout (float): Dropout rate.
        activation_fn (str): 'relu' or 'gelu'.
    """

    def __init__(self, d_model, hidden_dim, dropout, activation_fn):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU() if activation_fn == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        
        Returns:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        out = []
        for y in x:
            z = self.net(y)
            z = z + y
            z = self.norm(z)
            out.append(z)
        return out

class PredBBox(nn.Module):
    """Prediction module for bounding boxes.

    Args:
        d_model (int): Number of channels for model layers.
        n_bbox_outs (int): The number of outputs for 
            bounding box parameters.
        bbox_init_normal (bool, optional): If True, 
            initializes the linear layer weights using 
            a normal distribution. Defaults to False.
    """
    def __init__(self, d_model, n_bbox_outs, bbox_init_normal=False):
        super(PredBBox, self).__init__()
        self.linear = nn.Linear(d_model, n_bbox_outs)
        if bbox_init_normal:
            nn.init.normal_(self.linear.weight, std=.01)

    def forward(self, x):
        """Forward pass to predict bounding boxes.

        Args:
            x (Tensor): Input tensor of shape (n_bboxes, d_model).

        Returns:
            Tensor: A tensor of shape (n_bboxes, n_bbox_outs) 
                containing the predicted bounding boxes.
        """
        x = self.linear(x)
        return torch.hstack((torch.exp(x[:, :6]), 
                             x[:, 6:]))


@MODELS.register_module()
class QueryDecoder(BaseModule):
    """Query decoder.

    Args:
        num_layers (int): Number of transformer layers.
        num_instance_queries (int): Number of instance queries.
        num_semantic_queries (int): Number of semantic queries.
        num_classes (int): Number of classes.
        in_channels (int): Number of input channels.
        d_model (int): Number of channels for model layers.
        num_heads (int): Number of head in attention layer.
        hidden_dim (int): Dimension of attention layer.
        dropout (float): Dropout rate for transformer layer.
        activation_fn (str): 'relu' of 'gelu'.
        iter_pred (bool): Whether to predict iteratively.
        attn_mask (bool): Whether to use mask attention.
        pos_enc_flag (bool): Whether to use positional enconding.
    """

    def __init__(self, num_layers, num_instance_queries, num_semantic_queries,
                 num_classes, in_channels, d_model, num_heads, hidden_dim,
                 dropout, activation_fn, iter_pred, attn_mask, fix_attention,
                 objectness_flag,vis_attn=False, **kwargs):
        super().__init__()
        self.objectness_flag = objectness_flag
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.num_queries = num_instance_queries + num_semantic_queries
        if num_instance_queries + num_semantic_queries > 0:
            self.query = nn.Embedding(num_instance_queries + num_semantic_queries, d_model)
        if num_instance_queries == 0:
            self.query_proj = nn.Sequential(
                nn.Linear(in_channels, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model))
        self.cross_attn_layers = nn.ModuleList([])
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        for i in range(num_layers):
            self.cross_attn_layers.append(
                CrossAttentionLayer(
                    d_model, num_heads, dropout, fix_attention, return_attn=vis_attn))
            self.self_attn_layers.append(
                SelfAttentionLayer(d_model, num_heads, dropout))
            self.ffn_layers.append(
                FFN(d_model, hidden_dim, dropout, activation_fn))
        self.out_norm = nn.LayerNorm(d_model)
        self.out_cls = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, num_classes + 1))
        if objectness_flag:
            self.out_score = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.x_mask = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model))
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask

    def _get_queries(self, queries=None, batch_size=None):
        """Get query tensor.

        Args:
            queries (List[Tensor], optional): of len batch_size,
                each of shape (n_queries_i, in_channels).
            batch_size (int, optional): batch size.
        
        Returns:
            List[Tensor]: of len batch_size, each of shape
                (n_queries_i, d_model).
        """
        if batch_size is None:
            batch_size = len(queries)
        
        result_queries = []
        for i in range(batch_size):
            result_query = []
            if hasattr(self, 'query'):
                result_query.append(self.query.weight)
            if queries is not None:
                result_query.append(self.query_proj(queries[i]))
            result_queries.append(torch.cat(result_query))
        return result_queries

    def _forward_head(self, queries, mask_feats):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        cls_preds, pred_scores, pred_masks, attn_masks = [], [], [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])
            cls_preds.append(self.out_cls(norm_query))
            pred_score = self.out_score(norm_query) if self.objectness_flag \
                else None
            pred_scores.append(pred_score)
            pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        return cls_preds, pred_scores, pred_masks, attn_masks
    
    def forward_simple(self, x, queries):
        """Simple forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, and scores.
        """
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
        cls_preds, pred_scores, pred_masks, _ = self._forward_head(
            queries, mask_feats)
        return dict(
            cls_preds=cls_preds,
            masks=pred_masks,
            scores=pred_scores)

    def forward_iter_pred(self, x, queries):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and aux_outputs.
        """
        cls_preds, pred_scores, pred_masks = [], [], []
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        cls_pred, pred_score, pred_mask, attn_mask = self._forward_head(
            queries, mask_feats)
        cls_preds.append(cls_pred)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
            cls_pred, pred_score, pred_mask, attn_mask = self._forward_head(
                queries, mask_feats)
            cls_preds.append(cls_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        aux_outputs = [
            {'cls_preds': cls_pred, 'masks': masks, 'scores': scores}
            for cls_pred, scores, masks in zip(
                cls_preds[:-1], pred_scores[:-1], pred_masks[:-1])]
        return dict(
            cls_preds=cls_preds[-1],
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            aux_outputs=aux_outputs)

    def forward(self, x, queries=None):
        """Forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and possibly aux_outputs.
        """
        if self.iter_pred:
            return self.forward_iter_pred(x, queries)
        else:
            return self.forward_simple(x, queries)

@MODELS.register_module()
class UniDet3DEncoder(BaseModule):
    """Encoder for the UniDet3D model.

    Args:
        num_layers (int): Number of transformer layers.
        datasets_classes (List[List[str]]): List of classes for 
            each dataset.
        in_channels (int): Number of input channels.
        d_model (int): Number of channels for model layers.
        num_heads (int): Number of head in attention layer.
        hidden_dim (int): Dimension of attention layer.
        dropout (float): Dropout rate for transformer layer.
        activation_fn (str): 'relu' of 'gelu'.
        datasets (List[str]): List of dataset names.
        angles (List[Bool]): Whether to use angle prediction.
    """

    def __init__(self, num_layers, datasets_classes, in_channels, 
                 d_model, num_heads, hidden_dim, dropout, activation_fn,
                 datasets, angles, get_box_type=False, get_query=False, group=1, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.datasets = datasets
        self.angles = angles 
        self.group = group
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model))
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        for i in range(num_layers):
            self.self_attn_layers.append(
                SelfAttentionLayer(d_model, num_heads, dropout))
            self.ffn_layers.append(
                FFN(d_model, hidden_dim, dropout, activation_fn))
        self.out_norm = nn.LayerNorm(d_model)
        self.outs_cls = nn.ModuleList([])
        self.outs_cls_align = nn.ModuleList([])

        unique_cls = sorted(list(set(itertools.chain.from_iterable(
                            datasets_classes)))) + ['no_obj']
        self.outs_cls = nn.Sequential(
                                nn.Linear(d_model, d_model), nn.ReLU(),
                                nn.Linear(d_model, len(unique_cls)))
        self.datasets_cls_idxs = []
        for dataset_classes in datasets_classes:
            dataset_cls_idxs = []
            for cls in dataset_classes:
                dataset_cls_idxs.append(unique_cls.index(cls))
            self.datasets_cls_idxs.append(dataset_cls_idxs + [-1])

        self.out_bboxes = PredBBox(d_model, 8)
        self.get_query = get_query
        self.get_box_type = get_box_type
    def _forward_head_align(self, feats, sp_centers, datasets_names):
        """Prediction head forward.

        Args:
            feats (List[Tensor]): of len batch_size,
                each of shape (n_bboxes_i, d_model).
            sp_centers (List[Tensor]): of len batch_size,
                each of shape (n_bboxes_i, 3) representing 
                the spatial centers for the predicted 
                bounding boxes.
            datasets_names (List[str]): A list of dataset names 
                corresponding to each input feature tensor.

        Returns:
            Tuple[List[Tensor], List[Tensor]]:
                    - List[Tensor]: Classification predictions 
                        of length `batch_size`, with each tensor 
                        having a shape of (n_bboxes_i, n_classes + 1).
                    - List[Tensor]: Bounding box predictions of 
                        length `batch_size`, with each tensor having 
                        a shape of (n_bboxes_i, 7) (or 6 if `angles` 
                        are not applicable).
        """
        cls_preds, pred_bboxes = [], []
        for j in range(len(feats)):
            i = j % (len(feats) // self.group)
            norm_query = self.out_norm(feats[i])
            idx = self.datasets.index(datasets_names[i])
            class_idxs = norm_query.new_tensor(
                            self.datasets_cls_idxs[idx]).long()
            cls_preds.append(self.outs_cls_align(norm_query)[:, class_idxs])
            pred_bbox = self.out_bboxes_align(norm_query)
            if not self.angles[idx]:
                pred_bbox = pred_bbox[:, :6]

            pred_bbox = _bbox_pred_to_bbox(sp_centers[i], pred_bbox)
            
            pred_bboxes.append(pred_bbox)
        
        return cls_preds, pred_bboxes

    def _forward_head(self, feats, sp_centers, datasets_names):
        """Prediction head forward.

        Args:
            feats (List[Tensor]): of len batch_size,
                each of shape (n_bboxes_i, d_model).
            sp_centers (List[Tensor]): of len batch_size,
                each of shape (n_bboxes_i, 3) representing 
                the spatial centers for the predicted 
                bounding boxes.
            datasets_names (List[str]): A list of dataset names 
                corresponding to each input feature tensor.

        Returns:
            Tuple[List[Tensor], List[Tensor]]:
                    - List[Tensor]: Classification predictions 
                        of length `batch_size`, with each tensor 
                        having a shape of (n_bboxes_i, n_classes + 1).
                    - List[Tensor]: Bounding box predictions of 
                        length `batch_size`, with each tensor having 
                        a shape of (n_bboxes_i, 7) (or 6 if `angles` 
                        are not applicable).
        """
        cls_preds, pred_bboxes  = [], []
        for j in range(len(feats)):
            if self.training:
                i = j % (len(feats) // self.group)
            else:
                i = j
            norm_query = self.out_norm(feats[i])
            idx = self.datasets.index(datasets_names[i])
            class_idxs = norm_query.new_tensor(
                            self.datasets_cls_idxs[idx]).long()
            cls_preds.append(self.outs_cls(norm_query)[:, class_idxs])
            pred_bbox = self.out_bboxes(norm_query)
            if not self.angles[idx]:
                pred_bbox = pred_bbox[:, :6]

            pred_bbox = _bbox_pred_to_bbox(sp_centers[i], pred_bbox)
            
            pred_bboxes.append(pred_bbox)
        
        return cls_preds, pred_bboxes

    def forward(self, x, sp_centers, datasets_names):
        """Forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_bboxes_i, in_channels).
            sp_centers (List[Tensor]): A list of spatial centers 
                for the superpoints, with a length of `batch_size`. 
                Each tensor has a shape of (n_bboxes_i, 3).
            datasets_names (List[str]): A list of dataset names 
                corresponding to each input feature tensor.
        Returns:
            Dict: with cls_preds, pred_bboxes and aux_outputs.
        """
        cls_preds, pred_bboxes= [], []
        feats = [self.input_proj(y) for y in x]
        cls_pred, pred_bbox = \
            self._forward_head(feats, sp_centers, datasets_names)
        cls_preds.append(cls_pred)
        pred_bboxes.append(pred_bbox)

        for i in range(self.num_layers):
            feats = self.self_attn_layers[i](feats)
            feats = self.ffn_layers[i](feats)
            
            cls_pred, pred_bbox = \
                self._forward_head(feats, sp_centers, datasets_names)
            cls_preds.append(cls_pred)
            pred_bboxes.append(pred_bbox)
            
        aux_outputs = [
            dict(
                cls_preds=cls_pred,
                bboxes=bboxes,
                )
            for cls_pred, bboxes in zip(
                cls_preds[:-1], pred_bboxes[:-1])]
        if self.get_query:
            return dict(
            queries=feats,
            cls_preds=cls_preds[-1],
            bboxes=pred_bboxes[-1],
            aux_outputs=aux_outputs)
        else:
            return dict(
            cls_preds=cls_preds[-1],
            bboxes=pred_bboxes[-1],
            aux_outputs=aux_outputs)

@MODELS.register_module()
class Instance_relative_Decoder(QueryDecoder):

    def __init__(self, num_instance_classes, num_semantic_classes,num_heads,dropout,fix_attention,
                 d_model, num_semantic_linears,num_layers, pos_normlize=False,
                 use_pos_mlp=True,pos_type='fourier',get_query=False, **kwargs):
        super().__init__(
            num_classes=num_instance_classes,num_layers=num_layers,num_heads=num_heads,
              dropout=dropout, fix_attention=fix_attention,d_model=d_model, **kwargs)
        assert num_semantic_linears in [1, 2]
        if num_semantic_linears == 2:
            self.out_sem = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, num_semantic_classes + 1))
            
        else:
            self.out_sem = nn.Linear(d_model, num_semantic_classes + 1)
        self.get_query = get_query
        self.self_attn_layers = nn.ModuleList([])
        for i in range(num_layers):
            self.self_attn_layers.append(
                SelfAttentionLayer_pos(
                    d_model, num_heads, dropout))
        self.out_bboxes = PredBBox(d_model, 8)
        self.corner_mlp = nn.ModuleList([])
        for _ in range(num_layers):
            self.corner_mlp.append(nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model)))
        self.pos_type=pos_type
        self.d_model=d_model
        self.use_pos_mlp=use_pos_mlp
        if use_pos_mlp:
            self.pos_mlp = nn.ModuleList([])
            for _ in range(num_layers):
                self.pos_mlp.append(nn.Sequential(
                    nn.Linear(d_model, d_model), nn.ReLU(),
                    nn.Linear(d_model, d_model)))
        self.pos_emb=PositionEmbeddingCoordsSine(normalize=pos_normlize, pos_type=pos_type, d_pos=d_model)
        self.pos_normlize=pos_normlize

    def _forward_head(self, queries, mask_feats, sp_centers, last_flag):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_instance_classes + 1).
                List[Tensor] or None: Semantic predictions of len batch_size,
                    each of shape (n_queries_i, n_semantic_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        cls_preds, sem_preds, pred_scores, pred_masks, pred_bboxes = \
            [], [], [], [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])
            cls_preds.append(self.out_cls(norm_query))
            if last_flag:
                sem_preds.append(self.out_sem(norm_query))
            pred_score = self.out_score(norm_query) if self.objectness_flag \
                else None
            pred_scores.append(pred_score)
            pred_bbox = self.out_bboxes(norm_query)
            
            pred_bbox = pred_bbox[:, :6]
            pred_bbox = _bbox_pred_to_bbox(sp_centers[i], pred_bbox)
            pred_bboxes.append(pred_bbox)
            pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
            pred_masks.append(pred_mask)
        sem_preds = sem_preds if last_flag else None
        return cls_preds, sem_preds, pred_scores, pred_masks, pred_bboxes
    
    def forward(self, x, queries, sp_centers):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with instance scores, semantic scores, masks, scores,
                and aux_outputs.
        """
        cls_preds, sem_preds, pred_scores, pred_masks, pred_bboxes =[], [], [], [], []
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        cls_pred, sem_pred, pred_score, pred_mask, pred_bbox = \
            self._forward_head(queries, mask_feats, sp_centers, last_flag=False)
        cls_preds.append(cls_pred)
        sem_preds.append(sem_pred)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        pred_bboxes.append(pred_bbox)
        num_channels = self.d_model if self.pos_type=='sine' else None
        if self.pos_normlize:
            pbmin = torch.stack([bbox[:,:3].min(0)[0] for bbox in pred_bbox])  
            pbmax = torch.stack([bbox[:,:3].max(0)[0] for bbox in pred_bbox])  
            q_pos=[self.pos_emb(pred_bbox[k][:,:3].unsqueeze(0),num_channels=num_channels,input_range=(pbmin[k],pbmax[k])).transpose(1, 2).squeeze(0) for k in range(len(queries))]
        else:
            q_pos=[self.pos_emb(pred_bbox[k][:,:3].unsqueeze(0),num_channels=num_channels).transpose(1, 2).squeeze(0) for k in range(len(queries))]
        corner_pos_list=generate_corners_pos(pred_bbox)
        num_examples = len(corner_pos_list)  
        num_corners = 8
        pos_cor_list = [list(torch.unbind(corner_pos, dim=1)) for corner_pos in corner_pos_list] 
        pos_i_mins = [[pos_cor.min(0)[0].unsqueeze(0) for pos_cor in pos_cor_i] for pos_cor_i in pos_cor_list]  
        pos_i_maxs = [[pos_cor.max(0)[0].unsqueeze(0) for pos_cor in pos_cor_i] for pos_cor_i in pos_cor_list]
        if self.pos_normlize:
            sine_corner = [sum([self.pos_emb(pos_cor_i[j].unsqueeze(0),  
            num_channels=num_channels,  
            input_range=(pos_i_mins[k][j], pos_i_maxs[k][j])  
        ).transpose(1, 2).squeeze(0) for j in range(num_corners)]) for k, pos_cor_i in enumerate(pos_cor_list)]  
        else:
            sine_corner = [sum([self.pos_emb(pos_cor_i[j].unsqueeze(0),  
            num_channels=num_channels).transpose(1, 2).squeeze(0) for j in range(num_corners)]) for k, pos_cor_i in enumerate(pos_cor_list)]  
        
        for i in range(len(self.cross_attn_layers)):
            if self.use_pos_mlp:
                q_pos_proj=[self.pos_mlp[i](q_pos[k]) for k in range(len(queries))]
            else:
                q_pos_proj=q_pos
            queries = self.self_attn_layers[i](queries,q_pos_proj)
            queries=[queries[k]+self.corner_mlp[i](sine_corner[k]) for k in range(len(queries))]
            queries = self.cross_attn_layers[i](inst_feats, queries)
            queries = self.ffn_layers[i](queries)
            last_flag = i == len(self.cross_attn_layers) - 1
            cls_pred, sem_pred, pred_score, pred_mask, pred_bbox = \
                self._forward_head(queries, mask_feats, sp_centers, last_flag)
            cls_preds.append(cls_pred)
            sem_preds.append(sem_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)
            pred_bboxes.append(pred_bbox)
            if self.pos_normlize:
                pbmin= [pred_bbox[k][:,:3].min(0)[0].unsqueeze(0) for k in range(len(x))]
                pbmax= [pred_bbox[k][:,:3].max(0)[0].unsqueeze(0) for k in range(len(x))]
                q_pos=[self.pos_emb(pred_bbox[k][:,:3].unsqueeze(0),num_channels=num_channels,input_range=(pbmin[k],pbmax[k])).transpose(1, 2).squeeze(0) for k in range(len(queries))]
            else:
                q_pos=[self.pos_emb(pred_bbox[k][:,:3].unsqueeze(0),num_channels=num_channels).transpose(1, 2).squeeze(0) for k in range(len(queries))]
            
            corner_pos_list=generate_corners_pos(pred_bbox)
            pos_cor_list = [list(torch.unbind(corner_pos, dim=1)) for corner_pos in corner_pos_list] 
            pos_i_mins = [[pos_cor.min(0)[0].unsqueeze(0) for pos_cor in pos_cor_i] for pos_cor_i in pos_cor_list]  
            pos_i_maxs = [[pos_cor.max(0)[0].unsqueeze(0) for pos_cor in pos_cor_i] for pos_cor_i in pos_cor_list]
            
            if self.pos_normlize:
                sine_corner = [sum([self.pos_emb(pos_cor_i[j].unsqueeze(0),  
                num_channels=num_channels,  
                input_range=(pos_i_mins[k][j], pos_i_maxs[k][j])  
            ).transpose(1, 2).squeeze(0) for j in range(num_corners)]) for k, pos_cor_i in enumerate(pos_cor_list)]  
            else:
                sine_corner = [sum([self.pos_emb(pos_cor_i[j].unsqueeze(0),  
                num_channels=num_channels).transpose(1, 2).squeeze(0) for j in range(num_corners)]) for k, pos_cor_i in enumerate(pos_cor_list)]  

            
        aux_outputs_det = [
            dict(
                cls_preds=cls_pred,
                bboxes=bboxes,
                )
            for cls_pred, bboxes in zip(
                cls_preds[:-1], pred_bboxes[:-1])]
        aux_outputs_seg = [
            dict(
                cls_preds=cls_pred,
                sem_preds=sem_pred,
                masks=masks,
                scores=scores)
            for cls_pred, sem_pred, scores, masks in zip(
                cls_preds[:-1], sem_preds[:-1],
                pred_scores[:-1], pred_masks[:-1])]
        x_seg=dict(
                cls_preds=cls_preds[-1],
                masks=pred_masks[-1],
                sem_preds=sem_preds[-1],
                scores=pred_scores[-1],
                aux_outputs=aux_outputs_seg)
        x_det=dict(
            cls_preds=cls_preds[-1],
            bboxes=pred_bboxes[-1],
            aux_outputs=aux_outputs_det)

        return x_seg,x_det
    

def _bbox_pred_to_bbox(points, bbox_pred):
    """Transform predicted bbox parameters to bbox.

    Args:
        points (Tensor): Final locations of shape (N, 3)
        bbox_pred (Tensor): Predicted bbox parameters of shape (N, 6)
            or (N, 8).

    Returns:
        Tensor: Transformed 3D box of shape (N, 6) or (N, 7).
    """
    if bbox_pred.shape[0] == 0:
        return bbox_pred

    x_center = points[:, 0] + (bbox_pred[:, 1] - bbox_pred[:, 0]) / 2
    y_center = points[:, 1] + (bbox_pred[:, 3] - bbox_pred[:, 2]) / 2
    z_center = points[:, 2] + (bbox_pred[:, 5] - bbox_pred[:, 4]) / 2

    # dx_min, dx_max, dy_min, dy_max, dz_min, dz_max -> x, y, z, w, l, h
    base_bbox = torch.stack([
        x_center,
        y_center,
        z_center,
        bbox_pred[:, 0] + bbox_pred[:, 1],
        bbox_pred[:, 2] + bbox_pred[:, 3],
        bbox_pred[:, 4] + bbox_pred[:, 5],
    ], -1)

    # axis-aligned case
    if bbox_pred.shape[1] == 6:
        return base_bbox

    # rotated case: ..., sin(2a)ln(q), cos(2a)ln(q)
    scale = bbox_pred[:, 0] + bbox_pred[:, 1] + \
        bbox_pred[:, 2] + bbox_pred[:, 3]
    q = torch.exp(
        torch.sqrt(
            torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)))
    alpha = 0.5 * torch.atan2(bbox_pred[:, 6], bbox_pred[:, 7])
    return torch.stack(
        (x_center, y_center, z_center, scale / (1 + q), scale /
            (1 + q) * q, bbox_pred[:, 5] + bbox_pred[:, 4], alpha),
        dim=-1)

def corners_to_box(corners):
    min_xyz = corners.min(dim=0)[0]
    max_xyz = corners.max(dim=0)[0]
    center = (min_xyz + max_xyz) / 2
    dims = max_xyz - min_xyz
    return center, dims


def generate_corners(boxes):
    """
    Args:
        boxes: [B, N, 6] (x,y,z,w,l,h)
    Returns:
        corners: [B, N, 8, 3]
    """
    B, N, _ = boxes.shape
    half_size = boxes[..., 3:6] / 2  # [B, N, 3] (w/2, l/2, h/2)
    

    base_corners = torch.tensor([
        [-1, -1, -1], [1, -1, -1],
        [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1],
        [1, 1, 1], [-1, 1, 1]
    ], dtype=boxes.dtype, device=boxes.device)  # [8, 3]
    
    scaled_corners = base_corners * half_size.unsqueeze(2)  # [B, N, 8, 3]
    
    centers = boxes[..., :3].unsqueeze(2)  # [B, N, 1, 3]
    return scaled_corners + centers  # [B, N, 8, 3]

def generate_corners_pos(boxes_list):
    base_corners = torch.tensor([
        [-1, -1, -1], [1, -1, -1],
        [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1],
        [1, 1, 1], [-1, 1, 1]
    ], dtype=torch.float32)  # [8,3]
    corner_list=[]
    for boxes in  boxes_list:
        device = boxes.device
        half_size = boxes[:, 3:6] / 2  # [n,3]
        scaled = half_size.unsqueeze(1) * base_corners.to(device)  # [n,8,3]
        centers = boxes[:, :3].unsqueeze(1)  # [n,1,3]
        corners = scaled + centers  # [n,8,3]
        corner_list.append(corners)

    
    return corner_list
