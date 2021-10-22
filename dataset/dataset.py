import os
import cv2
import paddle
import numpy as np

from paddle.vision.transforms import functional as F
from paddle.io import Dataset

class MyDataset(Dataset):
    
    def __init__(self, img_dir):

        self.img_path_list = []
        self.label_path_list = []
        for d in os.listdir(img_dir):
            img_path = img_dir + d +'/ori.jpg'
            label_path = img_dir + d +'/res.jpg'
            self.img_path_list.append(img_path)
            self.label_path_list.append(label_path)

    def __len__(self):
        return len(self.img_path_list)

    def process_img(self, img, label):

        img = cv2.resize(img, (320, 320))
        label = cv2.resize(label, (320, 320))
        img = img.transpose((2,0,1))
        label = label.transpose((2,0,1))

        label_diff = np.float32(label) - np.float32(img)
        img = img / 255.0
        label_diff = label_diff / 2.55
        img = paddle.to_tensor(img, dtype='float32')
        label_diff = paddle.to_tensor(label_diff, dtype='float32')

        return img, label_diff

    def __getitem__(self, idx):
        
        img = cv2.imread(self.img_path_list[idx])
        label = cv2.imread(self.label_path_list[idx])
        img, label_diff = self.process_img(img, label)
        return {
            'img': img,
            'label': label_diff
        }
        