import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset

from pointllm.data.utils import *
from pointllm.utils import *

shape_name = {
    "airplane": 0,
    "bathtub": 1,
    "bed": 2,
    "bench": 3,
    "bookshelf": 4,
    "bottle": 5,
    "bowl": 6,
    "car": 7,
    "chair": 8,
    "cone": 9,
    "cup": 10,
    "curtain": 11,
    "desk": 12,
    "door": 13,
    "dresser": 14,
    "flower_pot": 15,
    "glass_box": 16,
    "guitar": 17,
    "keyboard": 18,
    "lamp": 19,
    "laptop": 20,
    "mantel": 21,
    "monitor": 22,
    "night_stand": 23,
    "person": 24,
    "piano": 25,
    "plant": 26,
    "radio": 27,
    "range_hood": 28,
    "sink": 29,
    "sofa": 30,
    "stairs": 31,
    "stool": 32,
    "table": 33,
    "tent": 34,
    "toilet": 35,
    "tv_stand": 36,
    "vase": 37,
    "wardrobe": 38,
    "xbox": 39,
}
reversed_shape_name = {v: k for k, v in shape_name.items()}


class CustonModelNet(Dataset):
    def __init__(self, data_path, use_adv: bool = True, use_color: bool = True):
        """
        Args:
            data_args:
                split: train or test
        """
        super(CustonModelNet, self).__init__()

        self.data_path = data_path
        self.index_path = os.path.join(self.data_path, "modelnet40_index.txt")
        self.index = None
        self.mode = "adv" if use_adv else "ori"
        self.use_color = use_color

        self.list_of_points = None
        self.list_of_classnames = None

        self.load_index()
        self.load_data()

    def load_data(self):
        self.list_of_points = {"ori": [], "adv": []}
        self.list_of_classnames = {"ori": [], "adv": []}

        for data in self.index:
            ori_filename = os.path.join(self.data_path, data["ori_file"])
            adv_filename = os.path.join(self.data_path, data["adv_file"])

            ori_pts = np.load(ori_filename)
            adv_pts = np.load(adv_filename)

            self.list_of_points["ori"].append(ori_pts)
            self.list_of_points["adv"].append(adv_pts)

            self.list_of_classnames["ori"].append(data["ori_label"])
            self.list_of_classnames["adv"].append(data["adv_label"])

        self.list_of_points["ori"] = np.stack(self.list_of_points["ori"])
        self.list_of_points["adv"] = np.stack(self.list_of_points["adv"])

        self.list_of_classnames["ori"] = np.stack(self.list_of_classnames["ori"])
        self.list_of_classnames["adv"] = np.stack(self.list_of_classnames["adv"])

    def load_index(self):
        import csv

        self.index = []
        with open(self.index_path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 4:
                    entry = {
                        "ori_file": row[0].strip(),
                        "adv_file": row[1].strip(),
                        "ori_label": row[2].strip(),
                        "adv_label": row[3].strip(),
                    }
                    self.index.append(entry)

    def __len__(self):
        import pdb;pdb.set_trace()
        return len(self.list_of_classnames["ori"])

    def _get_item(self, index):
        point_set, label = (
            self.list_of_points[index][self.mode],
            self.list_of_classnames[index][self.mode],
        )

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        if self.use_height:
            self.gravity_dim = 1
            height_array = (
                point_set[:, self.gravity_dim : self.gravity_dim + 1]
                - point_set[:, self.gravity_dim : self.gravity_dim + 1].min()
            )
            point_set = np.concatenate((point_set, height_array), axis=1)

        point_set = (
            np.concatenate((point_set, np.zeros_like(point_set)), axis=-1)
            if self.use_color
            else point_set
        )

        return point_set, label.item()  # * ndarray, int

    def pc_norm(self, pc):
        """pc: NxC, return NxC"""
        xyz = pc[:, :3]
        other_feature = pc[:, 3:]

        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        m = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
        xyz = xyz / m

        pc = np.concatenate((xyz, other_feature), axis=1)
        return pc

    def __getitem__(self, index):
        point_set, label = (
            self.list_of_points[self.mode][index],
            self.list_of_classnames["ori"][index],
        )
        point_set = (
            np.concatenate((point_set, np.zeros_like(point_set)), axis=-1)
            if self.use_color
            else point_set
        )

        current_points = torch.from_numpy(point_set).float()  # * N, C tensors

        label_name = label.item()
        label = shape_name[label_name]

        data_dict = {
            "indice": index,  # * int
            "point_clouds": current_points,  # * tensor of N, C
            "labels": label,  # * int
            "label_names": label_name,  # * str
        }

        return data_dict
