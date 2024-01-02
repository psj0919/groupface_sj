import pickle
import sys, os

import cv2
import torch
import numpy as np
import torch.utils.data.dataset
from tqdm import tqdm
#
import random
from utilss.augmentation import ArgumentationSchedule
import torch.nn.functional as F


def makefile(train_path, cache_file=" "):
    IDs = []
    IDsLabels = {}
    file_paths = []
    file_IDs = []
    file_labels = []

    if os.path.exists(cache_file) is False:
        for dir in tqdm(os.listdir(train_path)):
            if os.path.isdir(os.path.join(train_path, dir)) is False:
                raise ("DIR Error")
            IDs.append(dir)

            label_idx = 0
            for ID in IDs:
                IDsLabels[ID] = label_idx
                label_idx += 1

        for dir in tqdm(os.listdir(train_path)):
            if os.path.isdir(os.path.join(train_path, dir)) is False:
                raise ("DIR Error")
            for file in os.listdir(os.path.join(train_path, dir)):
                if os.path.splitext(file)[1] in [".jpg", ".bmp", ".png"]:
                    file_paths.append(os.path.join(train_path, dir, file))
                    file_IDs.append(dir)
                    file_labels.append(IDsLabels[dir])
        print("data set loaded from scratch len: {}".format(len(file_paths)))
        sys.stdout.flush()
        with open(cache_file, "wb") as f:
            pickle.dump([file_paths,
                         file_IDs,
                         file_labels,
                         IDs,
                         IDsLabels], f)
    else:
        print("start loading from cache")
        sys.stdout.flush()
        with open(cache_file, "rb") as f:
            file_paths, file_IDs, file_labels, IDs, IDsLabels = pickle.load(f)
        print("data set loaded from cache len: {}".format(len(file_paths)))
        sys.stdout.flush()

if __name__=='__main__':
    train_path = "/storage/sjpark/VGGFace2/vgg_train_6s_hs_SW_960"
    cache_file = "/storage/sjpark/VGGFace2/cache/vgg_train_6s_hs_SW_960.pikcle"
    img_size = 224
    makefile(train_path= train_path,cache_file=cache_file)