from mmdet3d.registry import DATASETS
from mmdet3d.datasets.s3dis_dataset import S3DISDataset
import os.path as osp
from mmengine.logging import print_log
import logging
import numpy as np
@DATASETS.register_module()
class S3DISSegDataset_(S3DISDataset):
    METAINFO = {
        'classes':
        ('ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
         'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter'),
        'palette': [[0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 255, 0],
                    [255, 0, 255], [100, 100, 255], [200, 200, 100],
                    [170, 120, 200], [255, 0, 0], [200, 100, 100],
                    [10, 200, 100], [200, 200, 200], [50, 50, 50]],
        'seg_valid_class_ids':
        tuple(range(13)),
        'seg_all_class_ids':
        tuple(range(14))  # possibly with 'stair' class
    }
from mmengine.fileio import join_path, list_from_file, load
@DATASETS.register_module()
class S3DISSegDetDataset(S3DISDataset):
    """S3DISSegDetDataset dataset.

    Args:
        partition(float): Defaults to 1, the part of 
            the dataset that will be used.
    """
    METAINFO = {
        'classes': ('ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
                    'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter'),
        'palette': [[0, 255, 0], [0, 0, 255], [0, 255, 255], [255, 255, 0],
                    [255, 0, 255], [100, 100, 255], [200, 200, 100],
                    [170, 120, 200], [255, 0, 0], [200, 100, 100],
                    [10, 200, 100], [200, 200, 200], [50, 50, 50]],
        'seg_valid_class_ids': tuple(range(13)),
        'seg_all_class_ids': tuple(range(14))
    }
    def __init__(self,
                 partition: float = 1,
                 **kwargs) -> None:
        self.partition = partition
        kwargs.pop("scene_idxs", None)  # 添加此行
        super().__init__(**kwargs)
    
    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        info['super_pts_path'] = osp.join(
            self.data_prefix.get('sp_pts_mask', ''), info['super_pts_path'])

        info = super().parse_data_info(info)

        return info
    def load_data_list(self):
        """Load annotations from an annotation file named as ``self.ann_file``"""
        annotations = load(self.ann_file)

        
        if isinstance(annotations, list):
            raw_data_list = annotations
            # 如果你在构造函数传入了 metainfo，就用它；否则就空 dict
            metainfo = getattr(self, 'metainfo', {}) or {}
        else:
            if not isinstance(annotations, dict):
                raise TypeError(
                    f'The annotations loaded from annotation file '
                    f'should be a dict, but got {type(annotations)}!'
                )
            if 'data_list' not in annotations or 'metainfo' not in annotations:
                raise ValueError(
                    'Annotation must have data_list and metainfo keys'
                )
            metainfo = annotations['metainfo']
            raw_data_list = annotations['data_list']

        # 把 metainfo 合并到 self._metainfo（保持原有逻辑不变）
        for k, v in metainfo.items():
            self._metainfo.setdefault(k, v)

        # 后面保持原有解析流程
        data_list = []
        for raw_data_info in raw_data_list:
            data_info = self.parse_data_info(raw_data_info)
            if isinstance(data_info, dict):
                data_list.append(data_info)
            elif isinstance(data_info, list):
                for item in data_info:
                    if not isinstance(item, dict):
                        raise TypeError(
                            'data_info must be list of dict, but got '
                            f'{type(item)}'
                        )
                data_list.extend(data_info)
            else:
                raise TypeError(
                    'data_info should be a dict or list of dict, '
                    f'but got {type(data_info)}'
                )
        return data_list

    def __getitem__(self, idx: int) -> dict:
        """Get the idx-th image and data information of dataset after
        ``self.pipeline``, and ``full_init`` will be called if the dataset has
        not been fully initialized.

        During training phase, if ``self.pipeline`` get ``None``,
        ``self._rand_another`` will be called until a valid image is fetched or
         the maximum limit of refetech is reached.

        Args:
            idx (int): The index of self.data_list.

        Returns:
            dict: The idx-th image and data information of dataset after
            ``self.pipeline``.
        """
        # Performing full initialization by calling `__getitem__` will consume
        # extra memory. If a dataset is not fully initialized by setting
        # `lazy_init=True` and then fed into the dataloader. Different workers
        # will simultaneously read and parse the annotation. It will cost more
        # time and memory, although this may work. Therefore, it is recommended
        # to manually call `full_init` before dataset fed into dataloader to
        # ensure all workers use shared RAM from master process.

        if not self.test_mode:
            if self.serialize_data:
                dataset_len = len(self.data_address)
            else:
                dataset_len = len(self.data_list)
            idx = np.random.randint(0, dataset_len)
        if not self._fully_initialized:
            print_log(
                'Please call `full_init()` method manually to accelerate '
                'the speed.',
                logger='current',
                level=logging.WARNING)
            self.full_init()

        if self.test_mode:
            data = self.prepare_data(idx)
            if data is None:
                raise Exception('Test time pipline should not get `None` '
                                'data_sample')
            return data

        for _ in range(self.max_refetch + 1):
            data = self.prepare_data(idx)
            # Broken images or random augmentations may cause the returned data
            # to be None
            if data is None:
                idx = self._rand_another()
                continue
            return data

    def __len__(self) -> int:
        """Get the length of filtered dataset and automatically call
        ``full_init`` if the  dataset has not been fully init.

        Returns:
            int: The length of filtered dataset.
        """

        if self.serialize_data:
            dataset_len = len(self.data_address)
        else:
            dataset_len = len(self.data_list)
        if not self.test_mode:
            return int(self.partition * dataset_len)
        else:
            return dataset_len
