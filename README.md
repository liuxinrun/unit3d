## UNIT3D: Unified Instance-relative Transformer for Indoor 3D Object Detection and Segmentation

This repository contains an implementation of UNIT3D, a unified 3D scene understanding framework that integrates object detection, semantic, instance, and panoptic segmentation tasks within a single model.

### Installation

This implementation is based on [mmdetection3d](https://github.com/open-mmlab/mmdetection3d) framework `v1.1.0`. If not using Docker, please follow [getting_started.md](https://github.com/open-mmlab/mmdetection3d/blob/22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61/docs/en/get_started.md) for the installation instructions. You can check your environment to our [requirement.txt](requirement.txt).

### Getting Started

Please see [test_train.md](https://github.com/open-mmlab/mmdetection3d/blob/22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61/docs/en/user_guides/train_test.md) for some basic usage examples.

#### Data Preprocessing

UNIT3D is trained and tested using 3 datasets: [ScanNet](data/scannet), [ScanNet200](data/scannet200), [S3DIS](data/s3dis).
Preprocessed data can be found at our [Hugging Face](https://huggingface.co/datasets/maksimko123/UniDet3D). Download each archive, unpack, and move into the corresponding directory in [data](data). Please comply with the license agreement before downloading the data.

Alternatively, you can preprocess the data by youself. 
Training data can be prepared according to the [instructions](data/scannet/README.md).

Superpoints for ScanNet and ScanNet200 are provided as a part of the original annotation. For the rest datasets, you can either download pre-computed superpoints at UniDet3D [Hugging Face](https://huggingface.co/datasets/maksimko123/UniDet3D), or compute them using [superpoint_transformer](https://github.com/drprojects/superpoint_transformer).

#### Training

To train UNIT3D, simply run the [training](tools/train.py) script:

```bash
python tools/train.py configs/UNIT3D_b8_scannet.py
```
or dist train for multi GPU to run the [training](tools/dist_train.sh) script:
```bash
bash tools/dist_train.py configs/UNIT3D_b8_scannet.py 2
```

#### Testing

To test a trained model, you can run the [testing](tools/test.py) script:

```bash
python tools/test.py configs/UNIT3D_b8_scannet.py \
    work_dirs/UNIT3D_b8_scannet.py/epoch_1024.pth
```
