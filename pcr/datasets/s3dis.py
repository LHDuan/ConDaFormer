"""
S3DIS Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import numpy as np
import pickle
import torch
import random
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence

from pcr.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose, TRANSFORMS


@DATASETS.register_module()
class S3DISDataset(Dataset):
    def __init__(self,
                 split=("Area_1", "Area_2", "Area_3", "Area_4", "Area_6"),
                 data_root='data/s3dis',
                 transform=None,
                 test_mode=False,
                 test_cfg=None,
                 npy=False,
                 loop=1):
        super(S3DISDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.loop = loop if not test_mode else 1    # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        self.npy = npy
        self.suffix = "*.pkl" if self.npy else "*.pth"
        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info("Totally {} x {} samples in {} set.".format(len(self.data_list), self.loop, split))

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, self.suffix))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, self.suffix))
        else:
            raise NotImplementedError
        return data_list

    def get_data(self, idx):
        if self.npy:
            file = open(self.data_list[idx % len(self.data_list)], 'rb')
            data = pickle.load(file)
            file.close()
        else:
            data = torch.load(self.data_list[idx % len(self.data_list)])
        coord = data["coord"]
        color = data["color"]
        if "semantic_gt" in data.keys():
            label = data["semantic_gt"].reshape([-1])
        else:
            label = np.zeros(coord.shape[0])
        data_dict = dict(coord=coord, color=color, label=label)
        return data_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)]).split(".")[0]

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        label = data_dict.pop("label")
        data_dict = self.transform(data_dict)
        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(
                aug(deepcopy(data_dict))
            )

        input_dict_list = []
        for data in data_dict_list:
            if self.test_voxelize.mode == "val":
                input_dict_list.append(self.test_voxelize(data))
            elif self.test_voxelize.mode == "test":
                data_part_list = self.test_voxelize(data)
                for data_part in data_part_list:
                    if self.test_crop:
                        data_part = self.test_crop(data_part)
                    else:
                        data_part = [data_part]
                    input_dict_list += data_part

        for i in range(len(input_dict_list)):
            input_dict_list[i] = self.post_transform(input_dict_list[i])
        return input_dict_list, label

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop
