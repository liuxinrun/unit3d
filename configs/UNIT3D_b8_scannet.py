_base_ = ['mmdet3d::_base_/default_runtime.py']
custom_imports = dict(imports=['unit3d'])
import math

classes_scannet = ['cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
                    'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'showercurtrain',
                    'toilet', 'sink', 'bathtub', 'otherfurniture']
classes_s3dis = ['table', 'chair', 'sofa', 'bookcase', 'board']

classes_multiscan = ['door', 'table',  'chair',  'cabinet',  'window',  'sofa',  'microwave',  'pillow',  
         'tv_monitor',  'curtain',  'trash_can',  'suitcase',  'sink',  'backpack',  'bed',  
         'refrigerator',  'toilet']

classes_3rscan = classes_scannet

classes_scannetpp = ['table', 'door', 'ceiling lamp', 'cabinet', 'blinds', 'curtain', 'chair', 'storage cabinet', 'office chair', 'bookshelf', 'whiteboard', 'window', 'box', 
                     'monitor', 'shelf', 'heater', 'kitchen cabinet', 'sofa', 'bed', 'trash can', 'book', 'plant', 'blanket', 'tv', 'computer tower', 'refrigerator', 'jacket', 
                     'sink', 'bag', 'picture', 'pillow', 'towel', 'suitcase', 'backpack', 'crate', 'keyboard', 'rack', 'toilet', 'printer', 'poster', 'painting', 'microwave', 'shoes', 
                     'socket', 'bottle', 'bucket', 'cushion', 'basket', 'shoe rack', 'telephone', 'file folder', 'laptop', 'plant pot', 'exhaust fan', 'cup', 'coat hanger', 'light switch', 
                     'speaker', 'table lamp', 'kettle', 'smoke detector', 'container', 'power strip', 'slippers', 'paper bag', 'mouse', 'cutting board', 'toilet paper', 'paper towel', 
                     'pot', 'clock', 'pan', 'tap', 'jar', 'soap dispenser', 'binder', 'bowl', 'tissue box', 'whiteboard eraser', 'toilet brush', 'spray bottle', 'headphones', 'stapler', 'marker']

classes_arkitscenes = ['cabinet', 'refrigerator', 'shelf', 'stove', 'bed',
                        'sink', 'washer', 'toilet', 'bathtub', 'oven',
                        'dishwasher', 'fireplace', 'stool', 'chair', 'table',
                        'tv_monitor', 'sofa']
score_thr = 0.0
# model settings
num_channels=32
voxel_size=0.02
num_instance_classes = 18
num_semantic_classes = 20
d_in=6
model = dict(
    type='UNIT3D',
    data_preprocessor=dict(type='Det3DDataPreprocessor_'),
    in_channels=9,
    num_channels=num_channels,
    num_classes=num_instance_classes,
    voxel_size=voxel_size,
    min_spatial_shape=128,
    query_thr_det=3000,
    query_thr_seg=0.5,
    bbox_by_mask=[True],
    target_by_distance=[False],
    use_superpoints=[True],
    fast_nms=[True],
    with_normals=True,
    minus_mean=True,########
    backbone=dict(
        type='SpConvUNet',
        num_planes=[num_channels * (i + 1) for i in range(5)],
        return_blocks=True),
    decoder=dict(
        type='Instance_relative_Decoder',
        num_layers=6,
        num_instance_queries=0,
        num_semantic_queries=0,
        num_instance_classes=num_instance_classes,
        num_semantic_classes=num_semantic_classes,
        num_semantic_linears=1,
        in_channels=32,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        pos_type='sine',
        pos_normlize=True,
        iter_pred=True,
        attn_mask=True,
        fix_attention=True,
        objectness_flag=False,
        get_query=True),
    datasets=['scannet'],
    criterion=dict(
        type='ScanNetUnifiedCriterion',
        num_semantic_classes=num_semantic_classes,
        sem_criterion=dict(
            type='ScanNetSemanticCriterion',
            ignore_index=num_semantic_classes,
            loss_weight=0.2),
        inst_criterion=dict(
        type='SegDetCriterionAlign',
            datasets=['scannet'],
            num_semantic_classes=num_semantic_classes,
            datasets_weights=[1],
            bbox_loss_simple=dict(
                type='UniDet3DAxisAlignedIoULoss',
                mode='mpdiou',
                reduction='none'),
            bbox_loss_rotated=dict(
                type='UniDet3DRotatedIoU3DLoss',
                mode='diou',
                reduction='none'),
            matcher=dict(
                type='UniMatcher',
                costs=[
                    dict(type='BOXMASKClassificationCost', weight=0.25),
                    dict(type='MaskBCECost', weight=1.0),
                    dict(type='MaskDiceCost', weight=1.0),
                    dict(type='BboxCostJointTraining', 
                            weight=2.0,
                            loss_simple=dict(
                                type='UniDet3DAxisAlignedIoULoss',
                                mode='mpdiou',
                                reduction='none'),
                            loss_rotated=dict(
                                type='UniDet3DRotatedIoU3DLoss',
                                mode='diou',
                                reduction='none'))],
                topk=1),
            seg_loss_weight=[0.25, 1.0, 1.0, 0.5],
            det_loss_weight=[0.25, 1.0],
            non_object_weight=0.1,
            topk=[1],
            fix_dice_loss_weight=False,
            iter_matcher=True,
            fix_mean_loss=True)),
    train_cfg=dict(topk=1),
    test_cfg=dict(
        low_sp_thr=0.18,
        inst_score_thr=0.0,
        sp_score_thr=0.4,
        pan_score_thr=0.5,
        npoint_thr=100,
        up_sp_thr=0.81,
        det_topk_insts=1000,
        seg_topk_insts=600,
        obj_normalization=True,
        score_thr=score_thr,
        stuff_classes=[0, 1],
        nms=True,
        matrix_nms_kernel='linear',
        iou_thr=[0.5]))


# scannet dataset settings

metainfo_scannet = dict(classes=classes_scannet)
data_root_scannet = 'data/scannet/'

max_class_scannet = 20
dataset_type_scannet = 'ScanNetDetDataset'
data_prefix_scannet = dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask',
    sp_pts_mask='super_points')

train_pipeline_scannet = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadAnnotations3D_',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True,
        with_sp_mask_3d=True,
        with_normals=True),
    dict(type='GlobalAlignmentWithNormals', rotation_axis=2),
    dict(type='PointSegClassMapping'),
    dict(
        type='RandomFlip3DWithNormals',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTransWithNormals',
        rot_range=[-3.14, 3.14],
        scale_ratio_range=[0.8, 1.2],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
    dict(
        type='NormalizePointsColor_',
        color_mean=[127.5, 127.5, 127.5]),
    dict(
        type='AddSuperPointAnnotations',
        num_classes=max_class_scannet,
        
        stuff_classes=[0, 1]),
    dict(
        type='ElasticTransfrom',
        gran=[6, 20],
        mag=[40, 160],
        voxel_size=voxel_size,
        p=0.5),
    dict(
        type='Pack3DDetInputs_',
        keys=[
            'points', 'gt_labels_3d', 'pts_semantic_mask', 'pts_instance_mask',
            'sp_pts_mask', 'gt_sp_masks', 'elastic_coords'
        ])
]
test_pipeline_scannet = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    
    dict(
        type='LoadAnnotations3D_',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True,
        with_sp_mask_3d=True,
        with_normals=True),
    dict(type='GlobalAlignmentWithNormals', rotation_axis=2),
    dict(type='PointSegClassMapping'),
    
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='NormalizePointsColor_',
                color_mean=[127.5, 127.5, 127.5]),
            dict(
        type='AddSuperPointAnnotations',
        num_classes=max_class_scannet,
        stuff_classes=[0, 1])]),
    dict(type='Pack3DDetInputs_', keys=['points', 'sp_pts_mask'])
]



# run settings
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset_',
        datasets=[dict(
                    type=dataset_type_scannet,
                    ann_file='scannet_infos_train.pkl',
                    data_prefix=data_prefix_scannet,
                    data_root=data_root_scannet,
                    metainfo=metainfo_scannet,
                    pipeline=train_pipeline_scannet,
                    ignore_index=max_class_scannet,
                    scene_idxs=None,
                    test_mode=False)] #+ \

                    ))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset_',
        datasets= \
                [dict(
                    type=dataset_type_scannet,
                    ann_file='scannet_infos_val.pkl',
                    data_prefix=data_prefix_scannet,
                    data_root=data_root_scannet,
                    metainfo=metainfo_scannet,
                    pipeline=test_pipeline_scannet,
                    ignore_index=max_class_scannet,
                    test_mode=True)] #+ \

                    ))

test_dataloader = val_dataloader
sem_mapping = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
inst_mapping = sem_mapping[2:]

class_names = [
    'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
    'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
    'curtain', 'refrigerator', 'showercurtrain', 'toilet', 'sink',
    'bathtub', 'otherfurniture']
class_names += ['unlabeled']
label2cat = {i: name for i, name in enumerate(class_names)}
metric_meta = dict(
    label2cat=label2cat,
    ignore_index=[num_semantic_classes],
    classes=class_names,
    dataset_name='ScanNet')
test_evaluator = dict(type='IndoorMetric_sempan', 
                      datasets=['scannet'],
                      datasets_classes=[classes_scannet],
                      stuff_class_inds=[0, 1], 
                      thing_class_inds=list(range(2, num_semantic_classes)), 
                      min_num_points=1, 
                      id_offset=2**16,
                      metric_meta=metric_meta,
                      sem_mapping=sem_mapping,
                      inst_mapping=inst_mapping)

val_evaluator = test_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001 * 2, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2))

param_scheduler = dict(type='PolyLR', begin=0, end=1024, power=0.9)

custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,           
        max_keep_ckpts=4,     
        save_last=True,       
        save_best='all_ap',  
        rule='greater'        
    )
)
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=1024,
    dynamic_intervals=[(1, 16), (1024 - 16, 1)])
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
