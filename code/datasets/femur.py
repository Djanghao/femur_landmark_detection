import torchvision.transforms as transforms
import numpy as np
import os, random, math, torch
import torch.utils.data as data
import json
import pandas as pd
from PIL import Image
from copy import deepcopy
from .augment import cc_augment, augment_patch, to_PIL, gray_to_PIL
from utils import radial

# choose a scale: distance between point 0 and point 4
SCALE_DIS = 50

class Femur_Base(data.Dataset):
    def __init__(self, pathDataset, size=[384, 384], mode="Train", id_shot=None):
        self.size = size
        self.num_landmark = 19
        self.list = []
        self.mode = mode

        self.num_repeat = 5

        self.pth_Image = os.path.join(pathDataset, "img_gray")
        self.labels = pd.read_csv(
            os.path.join(pathDataset, "all_yx.csv"), header=None, index_col=0
        )

        # file index
        index_set = set(self.labels.index)
        files = [i[:-4] for i in sorted(os.listdir(self.pth_Image))]
        files = [i for i in files if int(i) in index_set]

        n = len(files)
        train_num = 132  # round(n*0.7)
        val_num = 0  # round(n*0.1)
        test_num = 38
        if mode == "Train" or mode == "Infer_Train":
            self.indexes = files[:train_num]
        elif mode == "validate":
            self.indexes = files[train_num:-test_num]
        elif mode == "Test":
            self.indexes = files[-test_num:]
        elif mode == 'All':
            self.indexes = files[:]
        else:
            raise Exception("Unknown phase: {phase}".format(phase=mode))
        
        self.list = [{"ID": "{}".format(i)} for i in self.indexes]

        normalize = transforms.Normalize([0], [1])
        transformList = []
        transformList.append(transforms.Resize(self.size))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        self.transform = transforms.Compose(transformList)

        self.images = {
            item["ID"]: Image.open(
                os.path.join(self.pth_Image, item["ID"] + ".jpg")
            ).convert("RGB")
            for item in self.list
        }

        self.gts = {
            item["ID"]: self._get_landmark_gt(
                item["ID"], self.images[item["ID"]].size[::-1]
            )
            for item in self.list
        }
        # comment

        self.scale_rates = {
            item["ID"]: self._get_landmark_gt(
                item["ID"], self.images[item["ID"]].size[::-1], True
            )[1]
            for item in self.list
        }

    def __len__(self):
        return len(self.list)

    def readLandmark(self, name, origin_size):
        li = list(self.labels.loc[int(name), :])
        r1, r2 = [i / j for i, j in zip(self.size, origin_size)]
        points = [
            tuple([round(li[i + 1] * r1), round(li[i] * r2)])
            for i in range(0, len(li), 2)
        ]
        points
        return points

    def compute_spacing(self, gt):
        physical_factor = SCALE_DIS / radial(gt[0], gt[4])
        return [physical_factor, physical_factor]

    def _get_landmark_gt(self, name, origin_size, get_scale_rate=False):
        points = self.readLandmark(name, origin_size)
        if not get_scale_rate:
            return points
        scale_rate = self.compute_spacing(points)
        return points, scale_rate

    def get_landmark_gt(self, name, origin_size=None, get_scale_rate=False):
        if get_scale_rate:
            return self.gts[name], self.scale_rates[name]
        else:
            return self.gts[name]

class Femur_TPL_Voting(Femur_Base):
    def __init__(
        self,
        pathDataset,
        mode,
        size=[800, 640],
        do_repeat=True,
        ssl_dir=None,
        R_ratio=0.05,
        pseudo=True,
        id_shot=None,
    ):
        super().__init__(pathDataset=pathDataset, size=size, mode=mode, id_shot=id_shot)

        self.ssl_dir = ssl_dir
        self.do_repeat = do_repeat
        self.Radius = int(max(size) * R_ratio)
        self.pseudo = pseudo

        mask = torch.zeros(2 * self.Radius, 2 * self.Radius, dtype=torch.float)
        guassian_mask = torch.zeros(2 * self.Radius, 2 * self.Radius, dtype=torch.float)
        for i in range(2 * self.Radius):
            for j in range(2 * self.Radius):
                distance = np.linalg.norm([i + 1 - self.Radius, j + 1 - self.Radius])
                if distance < self.Radius:
                    mask[i][j] = 1
                    guassian_mask[i][j] = math.exp(
                        -math.pow(distance, 2) / math.pow(self.Radius, 2)
                    )
        self.mask = mask
        self.guassian_mask = guassian_mask

        self.offset_x = torch.zeros(2 * self.Radius, 2 * self.Radius, dtype=torch.float)
        self.offset_y = torch.zeros(2 * self.Radius, 2 * self.Radius, dtype=torch.float)
        for i in range(2 * self.Radius):
            self.offset_x[:, i] = self.Radius - i
            self.offset_y[i, :] = self.Radius - i
        self.offset_x = self.offset_x * self.mask / self.Radius
        self.offset_y = self.offset_y * self.mask / self.Radius

        normalize = transforms.Normalize([0.5], [0.5])
        transformList = []
        transformList.append(transforms.Resize(self.size))
        if mode == "Train" in mode:
            transforms.ColorJitter(brightness=0.25, contrast=0.35)
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        self.transform = transforms.Compose(transformList)

        if (mode == "Train" in mode) and do_repeat:
            temp = self.list.copy()
            for _ in range(self.num_repeat):
                self.list.extend(temp)

    def __getitem__(self, index):
        item = self.list[index]

        image = self.images[item["ID"]]
        origin_size = image.size[::-1]
        image = self.transform(image)

        landmark_list = list()

        if (self.mode == "Train" in self.mode) and self.do_repeat and self.pseudo:
            landmark_list = list()
            with open(
                "{0}/pseudo_labels/{1}.json".format(self.ssl_dir, item["ID"]), "r"
            ) as f:
                landmark_dict = json.load(f)
            for key, value in landmark_dict.items():
                landmark_list.append(value)
            scale_rate = 1.0
        else:
            landmark_list, scale_rate = self.get_landmark_gt(
                item["ID"], origin_size, get_scale_rate=True
            )

        y, x = image.shape[-2], image.shape[-1]
        mask = torch.zeros((self.num_landmark, y, x), dtype=torch.float)
        offset_x = torch.zeros((self.num_landmark, y, x), dtype=torch.float)
        offset_y = torch.zeros((self.num_landmark, y, x), dtype=torch.float)

        for i, landmark in enumerate(landmark_list):
            margin_x_left = max(0, landmark[1] - self.Radius)
            margin_x_right = min(x, landmark[1] + self.Radius)
            margin_y_bottom = max(0, landmark[0] - self.Radius)
            margin_y_top = min(y, landmark[0] + self.Radius)

            mask[i][
                margin_y_bottom:margin_y_top, margin_x_left:margin_x_right
            ] = self.mask[
                0 : margin_y_top - margin_y_bottom, 0 : margin_x_right - margin_x_left
            ]
            offset_x[i][
                margin_y_bottom:margin_y_top, margin_x_left:margin_x_right
            ] = self.offset_x[
                0 : margin_y_top - margin_y_bottom, 0 : margin_x_right - margin_x_left
            ]
            offset_y[i][
                margin_y_bottom:margin_y_top, margin_x_left:margin_x_right
            ] = self.offset_y[
                0 : margin_y_top - margin_y_bottom, 0 : margin_x_right - margin_x_left
            ]

        return image, mask, offset_y, offset_x, landmark_list, item["ID"], scale_rate
