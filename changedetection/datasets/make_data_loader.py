import argparse
import os

import imageio
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import MambaCD.changedetection.datasets.imutils as imutils


def img_loader(path):
    img = np.array(imageio.imread(path), np.float32)
    return img


def one_hot_encoding(image, num_classes=8):
    # Create a one hot encoded tensor
    one_hot = np.eye(num_classes)[image.astype(np.uint8)]

    # Move the channel axis to the front
    # one_hot = np.moveaxis(one_hot, -1, 0)

    return one_hot



class ChangeDetectionDatset(Dataset):
    def __init__(self, dataset_name,dataset_path, data_list, crop_size, max_iters=None, type='train', data_loader=img_loader):
        # print(dataset_path)
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.data_list = data_list
        self.loader = data_loader
        self.type = type
        self.data_pro_type = self.type

        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
            self.data_list = self.data_list[0:max_iters]
        self.crop_size = crop_size

    def __transforms(self, aug, pre_img, post_img, label):
        # print(label.shape)
        if aug:
            if self.dataset_name == "DSIFN-CD":
                label = Image.fromarray(label)
                label = label.resize((pre_img.shape[0], pre_img.shape[1]))
                label = np.array(label)
            pre_img, post_img, label = imutils.random_crop_new(pre_img, post_img, label, self.crop_size)
            pre_img, post_img, label = imutils.random_fliplr(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_flipud(pre_img, post_img, label)
            pre_img, post_img, label = imutils.random_rot(pre_img, post_img, label)
    
        pre_img = imutils.normalize_img(pre_img)  # imagenet normalization
        pre_img = np.transpose(pre_img, (2, 0, 1))

        post_img = imutils.normalize_img(post_img)  # imagenet normalization
        post_img = np.transpose(post_img, (2, 0, 1))

        return pre_img, post_img, label

    def __getitem__(self, index):
        if self.dataset_name=='LEVIR-CD' or self.dataset_name == 'LEVIR-CD+':
            pre_path = os.path.join(self.dataset_path, 'A', self.data_list[index])
            post_path = os.path.join(self.dataset_path,'B', self.data_list[index])
            label_path = os.path.join(self.dataset_path, 'label', self.data_list[index])
        if self.dataset_name=='SYSU':
            pre_path = os.path.join(self.dataset_path, 'time1', self.data_list[index])
            post_path = os.path.join(self.dataset_path,'time2', self.data_list[index])
            label_path = os.path.join(self.dataset_path, 'label', self.data_list[index])
        if self.dataset_name == 'DSIFN-CD':
            pre_path = os.path.join(self.dataset_path, 't1', self.data_list[index])
            post_path = os.path.join(self.dataset_path,'t2', self.data_list[index])
            if self.type=='train':
                label_path = os.path.join(self.dataset_path, 'm/m', self.data_list[index].split('.')[0]+'.png')
            else:
                label_path = os.path.join(self.dataset_path, 'mask', self.data_list[index].split('.')[0]+'.tif')
        if self.dataset_name=='WHU-CD':
            pre_path = os.path.join(self.dataset_path, 'A', self.data_list[index])
            post_path = os.path.join(self.dataset_path,'B', self.data_list[index])
            # if self.type=='train':
            #     label_path = os.path.join(self.dataset_path, 'label/OUT_0_1', self.data_list[index].split('.')[0]+'.png')
            # else:
            label_path = os.path.join(self.dataset_path, 'label', self.data_list[index])
        pre_img = self.loader(pre_path)
        post_img = self.loader(post_path)
        label = self.loader(label_path)
        if self.dataset_name != "DSIFN-CD":
            label = label / 255
        else:
            if self.type == 'train':
                label = label / 255

        if 'train'  in self.data_pro_type:
            pre_img, post_img, label = self.__transforms(True, pre_img, post_img, label)
        else:
            pre_img, post_img, label = self.__transforms(False, pre_img, post_img, label)
            label = np.asarray(label)

        data_idx = self.data_list[index]
        return pre_img, post_img, label, data_idx

    def __len__(self):
        return len(self.data_list)

def make_data_loader(args, **kwargs):  # **kwargs could be omitted
    if 'SYSU' in args.dataset or 'LEVIR-CD+' in args.dataset or 'WHU-CD' in args.dataset or 'LEVIR-CD' in args.dataset or 'DSIFN-CD' in args.dataset or 'SYSU' in args.dataset:
        dataset = ChangeDetectionDatset(args.dataset,args.train_dataset_path, args.train_data_name_list, args.crop_size, args.max_iters, args.type)
        # train_sampler = DistributedSampler(dataset, shuffle=True)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs, num_workers=16,
                                 drop_last=False)
        return data_loader
    
    else:
        raise NotImplementedError