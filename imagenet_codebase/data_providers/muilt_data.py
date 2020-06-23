# Once for All: Train One Network and Specialize it for Efficient Deployment
# Han Cai, Chuang Gan, Tianzhe Wang, Zhekai Zhang, Song Han
# International Conference on Learning Representations (ICLR), 2020.

import warnings
import os
import math
import numpy as np

import torch.utils.data
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets

from .datasets import FileListLabeledDataset,GivenSizeSampler
from .transforms import RandomResizedCrop, Compose, Resize, CenterCrop, ToTensor, \
    Normalize, RandomHorizontalFlip, ColorJitter, Lighting
import numpy as np

from imagenet_codebase.data_providers.base_provider import DataProvider, MyRandomResizedCrop, MyDistributedSampler


class ImagenetDataProvider(DataProvider):
    DEFAULT_PATH = ['/home/testuser/data2/yangdecheng/data/TR-NMA-0417/CX_20200417',\
                    '/home/testuser/data2/yangdecheng/data/TR-NMA-0417/TK_20200417',\
                    '/home/testuser/data2/yangdecheng/data/TR-NMA-0417/ZR_20200417',\
                    '/home/testuser/data2/yangdecheng/data/TR-NMA-0417/TX_20200417',\
                    '/home/testuser/data2/yangdecheng/data/TR-NMA-0224/WM_20200224']
    
    def __init__(self, save_path=None, train_batch_size=256, test_batch_size=512, valid_size=None, n_worker=32,
                 resize_scale=0.08, distort_color=None, image_size=224,
                 num_replicas=None, rank=None):
        
        warnings.filterwarnings('ignore')
        self._save_path = save_path
        self.num_tasks = len(self.train_path)
        
        self.image_size = image_size  # int or list of int
        self.distort_color = distort_color
        self.resize_scale = resize_scale
        self.n_worker = n_worker

        self._valid_transform_dict = {}
        if not isinstance(self.image_size, int):
            assert isinstance(self.image_size, list)
            from imagenet_codebase.data_providers.my_data_loader import MyDataLoader
            self.image_size.sort()  # e.g., 160 -> 224
            MyRandomResizedCrop.IMAGE_SIZE_LIST = self.image_size.copy()
            MyRandomResizedCrop.ACTIVE_SIZE = max(self.image_size)

            for img_size in self.image_size:
                self._valid_transform_dict[img_size] = self.build_valid_transform(img_size)
            self.active_img_size = max(self.image_size)
            self.valid_transforms = self._valid_transform_dict[self.active_img_size]
            train_loader_class = MyDataLoader  # randomly sample image size for each batch of training image
        else:
            self.active_img_size = self.image_size
            self.valid_transforms = self.build_valid_transform()
            train_loader_class = torch.utils.data.DataLoader

        self.train_transforms = self.build_train_transform()
        train_dataset = self.train_dataset(self.train_transforms)
        
        if valid_size is not None:
            if not isinstance(valid_size, int):
                pass
                # assert isinstance(valid_size, float) and 0 < valid_size < 1
                # valid_size = [len(train_dataset[i].samples) * valid_size[i] for i in range(self.num_tasks)]
            
            valid_dataset = self.train_dataset(self.valid_transforms)
            # train_indexes, valid_indexes = self.random_sample_valid_set(len(train_dataset.samples), valid_size)
            
            train_longest_size = max([int(np.ceil(len(td) / float(bs))) for td, bs in zip(train_dataset, train_batch_size)])
            self.train_longest_size = train_longest_size
            self.train_batch_size = train_batch_size
            valid_longest_size = max([int(np.ceil(len(td) / float(bs))) for td, bs in zip(valid_dataset, valid_size)])
            train_sampler = [GivenSizeSampler(td, total_size=train_longest_size * bs, rand_seed=0) for td, bs in zip(train_dataset, train_batch_size)]
            valid_sampler = [GivenSizeSampler(td, total_size=valid_longest_size * bs, rand_seed=0) for td, bs in zip(valid_dataset, valid_size)]

            # if num_replicas is not None:
            #     train_sampler = MyDistributedSampler(train_dataset, num_replicas, rank, np.array(train_indexes))
            #     valid_sampler = MyDistributedSampler(valid_dataset, num_replicas, rank, np.array(valid_indexes))
            # else:
            #     train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indexes)
            #     valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_indexes)

            self.train = [train_loader_class(
                train_dataset[i], batch_size=train_batch_size[i], sampler=train_sampler[i],
                num_workers=n_worker, pin_memory=False,
            ) for i in range(self.num_tasks)]
            self.valid = [torch.utils.data.DataLoader(
                valid_dataset[i], batch_size=valid_size[i], sampler=valid_sampler[i],
                num_workers=n_worker, pin_memory=False,
            ) for i in range(self.num_tasks)]
        else:
            if num_replicas is not None:
                train_sampler = [GivenSizeSampler(td, total_size=train_longest_size * bs, rand_seed=0) for td, bs in zip(train_dataset, train_batch_size)]
                self.train = train_loader_class(
                    train_dataset, batch_size=train_batch_size, sampler=train_sampler,
                    num_workers=n_worker, pin_memory=False
                )
            else:
                self.train = train_loader_class(
                    train_dataset, batch_size=train_batch_size, shuffle=False,
                    num_workers=n_worker, pin_memory=False,
                )
            self.valid = None
        
        test_dataset = self.test_dataset(self.valid_transforms)
        if num_replicas is not None:
            test_sampler = [GivenSizeSampler(td, total_size=valid_longest_size * bs, rand_seed=0) for td, bs in zip(valid_dataset, valid_size)]
            self.test = [torch.utils.data.DataLoader(
            test_dataset[i], batch_size=test_batch_size[i], sampler=test_sampler[i], shuffle=False, num_workers=n_worker, pin_memory=False,
        ) for i in range(self.num_tasks)]
        else:
            self.test = [torch.utils.data.DataLoader(
            test_dataset[i], batch_size=test_batch_size[i], sampler=test_sampler[i], shuffle=False, num_workers=n_worker, pin_memory=False,
        ) for i in range(self.num_tasks)]
        
        if self.valid is None:
            self.valid = self.test
    
    @staticmethod
    def name():
        return 'muilt_data'
    
    @property
    def data_shape(self):
        return 3, self.active_img_size, self.active_img_size  # C, H, W
    
    @property
    def n_classes(self):
        return [5,3,2,2,7]
    
    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = self.DEFAULT_PATH
        return self._save_path
    
    @property
    def data_url(self):
        raise ValueError('unable to download %s' % self.name())
    
    def train_dataset(self, _transforms):
        dataset = [FileListLabeledDataset(self.train_list[i], self.train_path[i], self.train_transforms) for i in range(self.num_tasks)]
        return dataset
    
    def test_dataset(self, _transforms):
        dataset = [FileListLabeledDataset(self.valid_list[i], self.valid_path[i], self.valid_transforms) for i in range(self.num_tasks)]
        return dataset
    
    @property
    def train_path(self):
        return self.save_path

    @property
    def train_list(self):
        return ['/home/testuser/data2/yangdecheng/data/TR-NMA-0417/CX_20200417/txt/cx_val.txt',\
                '/home/testuser/data2/yangdecheng/data/TR-NMA-0417/TK_20200417/txt/tk_val.txt',\
                '/home/testuser/data2/yangdecheng/data/TR-NMA-0417/ZR_20200417/txt/zr_val.txt',\
                '/home/testuser/data2/yangdecheng/data/TR-NMA-0417/TX_20200417/txt/tx_val.txt',\
                '/home/testuser/data2/yangdecheng/data/TR-NMA-0224/WM_20200224/txt/wm_val.txt'] 

    @property
    def valid_path(self):
        return self.save_path
    
    @property
    def valid_list(self):
        return ['/home/testuser/data2/yangdecheng/data/TR-NMA-0417/CX_20200417/txt/cx_val.txt',\
                '/home/testuser/data2/yangdecheng/data/TR-NMA-0417/TK_20200417/txt/tk_val.txt',\
                '/home/testuser/data2/yangdecheng/data/TR-NMA-0417/ZR_20200417/txt/zr_val.txt',\
                '/home/testuser/data2/yangdecheng/data/TR-NMA-0417/TX_20200417/txt/tx_val.txt',\
                '/home/testuser/data2/yangdecheng/data/TR-NMA-0224/WM_20200224/txt/wm_val.txt'] 

    @property
    def normalize(self):
        return Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])
    
    def build_train_transform(self, image_size=None, print_log=True):
        if image_size is None:
            image_size = self.image_size
        if print_log:
            print('Color jitter: %s, resize_scale: %s, img_size: %s' %
                  (self.distort_color, self.resize_scale, image_size))

        if self.distort_color == 'torch':
            color_transform = ColorJitter(brightness=[0.5,1.5], contrast=[0.5,1.5], saturation=[0.5,1.5], hue= 0) #brightness=32. / 255., saturation=0.5
        elif self.distort_color == 'tf':
            color_transform = transforms.ColorJitter(brightness=32. / 255., saturation=0.5)
        else:
            color_transform = None
        
        if isinstance(image_size, list):
            resize_transform_class = MyRandomResizedCrop
            print('Use MyRandomResizedCrop: %s, \t %s' % MyRandomResizedCrop.get_candidate_image_size(),
                  'sync=%s, continuous=%s' % (MyRandomResizedCrop.SYNC_DISTRIBUTED, MyRandomResizedCrop.CONTINUOUS))
        else:
            resize_transform_class = RandomResizedCrop

        train_transforms = [
            resize_transform_class(image_size, scale=(self.resize_scale, 1.0)),
            RandomHorizontalFlip(),
        ]
        if color_transform is not None:
            train_transforms.append(color_transform)
        train_transforms += [
            ToTensor(),
            self.normalize,
        ]

        train_transforms = Compose(train_transforms)
        return train_transforms

    def build_valid_transform(self, image_size=None):
        if image_size is None:
            image_size = self.active_img_size
        return Compose([
            Resize([128,128]),
            CenterCrop(image_size),
            ToTensor(),
            self.normalize,
        ])

    def assign_active_img_size(self, new_img_size):
        self.active_img_size = new_img_size
        if self.active_img_size not in self._valid_transform_dict:
            self._valid_transform_dict[self.active_img_size] = self.build_valid_transform()
        # change the transform of the valid and test set
        # self.valid.dataset.transform = self._valid_transform_dict[self.active_img_size]
        # self.test.dataset.transform = self._valid_transform_dict[self.active_img_size]
    
    def build_sub_train_loader(self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None):
        # used for resetting running statistics
        if self.__dict__.get('sub_train_%d' % self.active_img_size, None) is None:
            if num_worker is None:
                num_worker = self.n_worker
            
            # n_samples = len(self.train.dataset.samples)
            # g = torch.Generator()
            # g.manual_seed(DataProvider.SUB_SEED)
            # rand_indexes = torch.randperm(n_samples, generator=g).tolist()
            
            new_train_dataset = self.train_dataset(
                self.build_train_transform(image_size=self.active_img_size, print_log=False))
            # chosen_indexes = rand_indexes[:n_images]
            if num_replicas is not None:
                sub_sampler = [GivenSizeSampler(td, total_size=self.train_longest_size * bs, rand_seed=0) for td, bs in zip(new_train_dataset, self.train_batch_size)]
            else:
                sub_sampler = [GivenSizeSampler(td, total_size=self.train_longest_size * bs, rand_seed=0) for td, bs in zip(new_train_dataset, self.train_batch_size)]
            sub_data_loader = [torch.utils.data.DataLoader(
                new_train_dataset[i], batch_size=self.train_batch_size[i], sampler=sub_sampler[i],
                num_workers=num_worker, pin_memory=False,
            ) for i in range(self.num_tasks)]
            self.__dict__['sub_train_%d' % self.active_img_size] = sub_data_loader
            # for images, labels in sub_data_loader:
            #     self.__dict__['sub_train_%d' % self.active_img_size].append((images, labels))
        return self.__dict__['sub_train_%d' % self.active_img_size]
