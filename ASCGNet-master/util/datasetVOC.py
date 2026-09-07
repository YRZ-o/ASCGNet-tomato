import os
import os.path
import cv2
import numpy as np

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import random
import time
from tqdm import tqdm

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

# PASCAL VOC 2012 标准颜色映射表
VOC_COLORMAP = {
    (0, 0, 0): 0,  # 背景 (background)
    (128, 0, 0): 1,  # 飞机 (aeroplane)
    (0, 128, 0): 2,  # 自行车 (bicycle)
    (128, 128, 0): 3,  # 鸟 (bird)
    (0, 0, 128): 4,  # 船 (boat)
    (128, 0, 128): 5,  # 瓶子 (bottle)
    (0, 128, 128): 6,  # 巴士 (bus)
    (128, 128, 128): 7,  # 汽车 (car)
    (64, 0, 0): 8,  # 猫 (cat)
    (192, 0, 0): 9,  # 椅子 (chair)
    (64, 128, 0): 10,  # 牛 (cow)
    (192, 128, 0): 11,  # 餐桌 (dining table)
    (64, 0, 128): 12,  # 狗 (dog)
    (192, 0, 128): 13,  # 马 (horse)
    (64, 128, 128): 14,  # 摩托车 (motorbike)
    (192, 128, 128): 15,  # 人 (person)
    (0, 64, 0): 16,  # 盆栽植物 (potted plant)
    (128, 64, 0): 17,  # 羊 (sheep)
    (0, 192, 0): 18,  # 沙发 (sofa)
    (128, 192, 0): 19,  # 火车 (train)
    (0, 64, 128): 20,  # 显示器/电视 (tv/monitor)
    (224, 224, 192): 255  # 忽略边界 (ignore/border)
}

# 全局标签缓存
LABEL_CACHE = {}
CACHE_ENABLED = True
COLOR_TOLERANCE = 5


def rgb_to_index(rgb_label):
    """
    将RGB标签图像转换为索引格式（带容差处理）

    参数:
        rgb_label: RGB格式的标签图像 (H, W, 3)

    返回:
        index_label: 索引格式的标签图像 (H, W)
    """
    h, w = rgb_label.shape[:2]
    index_label = np.zeros((h, w), dtype=np.uint8)

    # 将RGB图像转换为浮点型以便计算距离
    rgb_float = rgb_label.astype(np.float32)

    # 对颜色映射表中的每种颜色进行匹配
    for rgb_color, index in VOC_COLORMAP.items():
        target_color = np.array(rgb_color, dtype=np.float32)

        # 计算欧氏距离
        color_diff = np.sqrt(np.sum((rgb_float - target_color) ** 2, axis=2))

        # 找到距离小于容差的像素
        mask = color_diff < COLOR_TOLERANCE

        # 设置对应的索引值
        index_label[mask] = index

    return index_label


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split=0, data_root=None, data_list=None, sub_list=None):
    assert split in [0, 1, 2, 3, 10, 11, 999]
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))

    # Shaban uses these lines to remove small objects:
    # if util.change_coordinates(mask, 32.0, 0.0).sum() > 2:
    #    filtered_item.append(item)
    # which means the mask will be downsampled to 1/32 of the original size and the valid area should be larger than 2,
    # therefore the area in original size should be accordingly larger than 2 * 32 * 32
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Processing data...".format(sub_list))
    sub_class_file_list = {}
    for sub_c in sub_list:
        sub_class_file_list[sub_c] = []

    for l_idx in tqdm(range(len(list_read))):
        line = list_read[l_idx]
        line = line.strip()
        line_split = line.split(' ')
        image_name = os.path.join(data_root, line_split[0])
        label_name = os.path.join(data_root, line_split[1])
        item = (image_name, label_name)

        # 修改标签读取方式：从灰度图改为彩色图，并转换为索引格式
        if CACHE_ENABLED and label_name in LABEL_CACHE:
            label = LABEL_CACHE[label_name].copy()
        else:
            label_rgb = cv2.imread(label_name, cv2.IMREAD_COLOR)
            label_rgb = cv2.cvtColor(label_rgb, cv2.COLOR_BGR2RGB)
            label = rgb_to_index(label_rgb)  # 转换为索引格式
            if CACHE_ENABLED:
                LABEL_CACHE[label_name] = label.copy()

        label_class = np.unique(label).tolist()

        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)

        new_label_class = []
        for c in label_class:
            if c in sub_list:
                tmp_label = np.zeros_like(label)
                target_pix = np.where(label == c)
                tmp_label[target_pix[0], target_pix[1]] = 1
                if tmp_label.sum() >= 2 * 32 * 32:
                    new_label_class.append(c)

        label_class = new_label_class

        if len(label_class) > 0:
            image_label_list.append(item)
            for c in label_class:
                if c in sub_list:
                    sub_class_file_list[c].append(item)

    print("Checking image&label pair {} list done! ".format(split))
    return image_label_list, sub_class_file_list


class SemData(Dataset):
    def __init__(self, split=3, shot=1, data_root=None, data_list=None, transform=None, mode='train', use_coco=False,
                 use_split_coco=False):
        assert mode in ['train', 'val', 'test']

        self.mode = mode
        self.split = split
        self.shot = shot
        self.data_root = data_root

        if not use_coco:
            self.class_list = list(range(1, 21))  # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
            if self.split == 3:
                self.sub_list = list(range(1, 16))  # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
                self.sub_val_list = list(range(16, 21))  # [16,17,18,19,20]
            elif self.split == 2:
                self.sub_list = list(range(1, 11)) + list(range(16, 21))  # [1,2,3,4,5,6,7,8,9,10,16,17,18,19,20]
                self.sub_val_list = list(range(11, 16))  # [11,12,13,14,15]
            elif self.split == 1:
                self.sub_list = list(range(1, 6)) + list(range(11, 21))  # [1,2,3,4,5,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = list(range(6, 11))  # [6,7,8,9,10]
            elif self.split == 0:
                self.sub_list = list(range(6, 21))  # [6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                self.sub_val_list = list(range(1, 6))  # [1,2,3,4,5]

        else:
            if use_split_coco:
                print('INFO: using SPLIT COCO')
                self.class_list = list(range(1, 81))
                if self.split == 3:
                    self.sub_val_list = list(range(4, 81, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))
                elif self.split == 2:
                    self.sub_val_list = list(range(3, 80, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))
                elif self.split == 1:
                    self.sub_val_list = list(range(2, 79, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))
                elif self.split == 0:
                    self.sub_val_list = list(range(1, 78, 4))
                    self.sub_list = list(set(self.class_list) - set(self.sub_val_list))
            else:
                print('INFO: using COCO')
                self.class_list = list(range(1, 81))
                if self.split == 3:
                    self.sub_list = list(range(1, 61))
                    self.sub_val_list = list(range(61, 81))
                elif self.split == 2:
                    self.sub_list = list(range(1, 41)) + list(range(61, 81))
                    self.sub_val_list = list(range(41, 61))
                elif self.split == 1:
                    self.sub_list = list(range(1, 21)) + list(range(41, 81))
                    self.sub_val_list = list(range(21, 41))
                elif self.split == 0:
                    self.sub_list = list(range(21, 81))
                    self.sub_val_list = list(range(1, 21))

        print('sub_list: ', self.sub_list)
        print('sub_val_list: ', self.sub_val_list)

        if self.mode == 'train':
            self.data_list, self.sub_class_file_list = make_dataset(split, data_root, data_list, self.sub_list)
            assert len(self.sub_class_file_list.keys()) == len(self.sub_list)
        elif self.mode == 'val':
            self.data_list, self.sub_class_file_list = make_dataset(split, data_root, data_list, self.sub_val_list)
            assert len(self.sub_class_file_list.keys()) == len(self.sub_val_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        label_class = []
        image_path, label_path = self.data_list[index]

        # 加载图像
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.float32(image)

        # 修改标签加载方式：从灰度图改为彩色图，并转换为索引格式
        if CACHE_ENABLED and label_path in LABEL_CACHE:
            label = LABEL_CACHE[label_path].copy()
        else:
            label_rgb = cv2.imread(label_path, cv2.IMREAD_COLOR)
            label_rgb = cv2.cvtColor(label_rgb, cv2.COLOR_BGR2RGB)
            label = rgb_to_index(label_rgb)  # 转换为索引格式
            if CACHE_ENABLED:
                LABEL_CACHE[label_path] = label.copy()

        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Query Image & label shape mismatch: " + image_path + " " + label_path + "\n"))

        label_class = np.unique(label).tolist()
        if 0 in label_class:
            label_class.remove(0)
        if 255 in label_class:
            label_class.remove(255)
        new_label_class = []
        for c in label_class:
            if c in self.sub_val_list:
                if self.mode == 'val' or self.mode == 'test':
                    new_label_class.append(c)
            if c in self.sub_list:
                if self.mode == 'train':
                    new_label_class.append(c)
        label_class = new_label_class
        assert len(label_class) > 0

        class_chosen = label_class[random.randint(1, len(label_class)) - 1]
        class_chosen = class_chosen
        target_pix = np.where(label == class_chosen)
        ignore_pix = np.where(label == 255)
        label[:, :] = 0
        if target_pix[0].shape[0] > 0:
            label[target_pix[0], target_pix[1]] = 1
        label[ignore_pix[0], ignore_pix[1]] = 255

        file_class_chosen = self.sub_class_file_list[class_chosen]
        num_file = len(file_class_chosen)

        support_image_path_list = []
        support_label_path_list = []
        support_idx_list = []
        for k in range(self.shot):
            support_idx = random.randint(1, num_file) - 1
            support_image_path = image_path
            support_label_path = label_path
            while ((
                           support_image_path == image_path and support_label_path == label_path) or support_idx in support_idx_list):
                support_idx = random.randint(1, num_file) - 1
                support_image_path, support_label_path = file_class_chosen[support_idx]
            support_idx_list.append(support_idx)
            support_image_path_list.append(support_image_path)
            support_label_path_list.append(support_label_path)

        support_image_list = []
        support_label_list = []
        subcls_list = []
        for k in range(self.shot):
            if self.mode == 'train':
                subcls_list.append(self.sub_list.index(class_chosen))
            else:
                subcls_list.append(self.sub_val_list.index(class_chosen))
            support_image_path = support_image_path_list[k]
            support_label_path = support_label_path_list[k]
            support_image = cv2.imread(support_image_path, cv2.IMREAD_COLOR)
            support_image = cv2.cvtColor(support_image, cv2.COLOR_BGR2RGB)
            support_image = np.float32(support_image)

            # 修改支持集标签加载方式：同样需要转换为索引格式
            if CACHE_ENABLED and support_label_path in LABEL_CACHE:
                support_label = LABEL_CACHE[support_label_path].copy()
            else:
                support_label_rgb = cv2.imread(support_label_path, cv2.IMREAD_COLOR)
                support_label_rgb = cv2.cvtColor(support_label_rgb, cv2.COLOR_BGR2RGB)
                support_label = rgb_to_index(support_label_rgb)  # 转换为索引格式
                if CACHE_ENABLED:
                    LABEL_CACHE[support_label_path] = support_label.copy()

            target_pix = np.where(support_label == class_chosen)
            ignore_pix = np.where(support_label == 255)
            support_label[:, :] = 0
            support_label[target_pix[0], target_pix[1]] = 1
            support_label[ignore_pix[0], ignore_pix[1]] = 255
            if support_image.shape[0] != support_label.shape[0] or support_image.shape[1] != support_label.shape[1]:
                raise (RuntimeError(
                    "Support Image & label shape mismatch: " + support_image_path + " " + support_label_path + "\n"))
            support_image_list.append(support_image)
            support_label_list.append(support_label)
        assert len(support_label_list) == self.shot and len(support_image_list) == self.shot

        raw_label = label.copy()
        if self.transform is not None:
            image, label = self.transform(image, label)
            for k in range(self.shot):
                support_image_list[k], support_label_list[k] = self.transform(support_image_list[k],
                                                                              support_label_list[k])

        s_xs = support_image_list
        s_ys = support_label_list
        s_x = s_xs[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_x = torch.cat([s_xs[i].unsqueeze(0), s_x], 0)
        s_y = s_ys[0].unsqueeze(0)
        for i in range(1, self.shot):
            s_y = torch.cat([s_ys[i].unsqueeze(0), s_y], 0)

        if self.mode == 'train':
            return image, label, s_x, s_y, subcls_list
        else:
            return image, label, s_x, s_y, subcls_list, raw_label