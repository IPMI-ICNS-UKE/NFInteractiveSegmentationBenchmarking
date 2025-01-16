import os
from os import path

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

from dataset.range_transform import im_normalization, im_mean
from dataset.reseed import reseed


class NeurofibromaDataset(Dataset):
    def __init__(self, im_root, gt_root, max_jump, subset=None):
        self.im_root = im_root
        self.gt_root = gt_root
        self.max_jump = max_jump

        self.videos = []
        self.frames = {}
        
        vid_list = sorted(os.listdir(self.im_root))
        # Pre-filtering
        for vid in vid_list:
            if subset is not None:
                if vid not in subset:
                    continue
            frames_all = sorted(os.listdir(os.path.join(self.im_root, vid)))
            frames = []
            for file in frames_all:
                if file.endswith('.jpg') or file.endswith('.png') or \
                    file.endswith('.PNG') or file.endswith('.JPG'):
                    frames.append(file)

            if len(frames) < 3:
                continue
            self.frames[vid] = frames
            self.videos.append(vid)

        print('%d out of %d videos accepted in %s.' % (len(self.videos), len(vid_list), im_root))
        
        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        

        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=10, shear=5, interpolation=InterpolationMode.BICUBIC, fill=im_mean),
        ])

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=10, shear=5, interpolation=InterpolationMode.NEAREST, fill=0),
        ])
        
        self.all_im_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((384, 384), scale=(0.8,1.00), interpolation=InterpolationMode.BICUBIC)
        ])

        self.all_gt_dual_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((384, 384), scale=(0.8,1.00), interpolation=InterpolationMode.NEAREST)
        ])
        
        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def __getitem__(self, idx):
        video = self.videos[idx]
        info = {}
        info['name'] = video

        vid_im_path = path.join(self.im_root, video)
        vid_gt_path = path.join(self.gt_root, video)
        frames = self.frames[video]

        trials = 0
        while trials < 5:
            info['frames'] = [] # Appended with actual frames

            # Don't want to bias towards beginning/end
            this_max_jump = min(len(frames), self.max_jump)
            start_idx = np.random.randint(len(frames)-this_max_jump+1)
            f1_idx = start_idx + np.random.randint(this_max_jump+1) + 1
            f1_idx = min(f1_idx, len(frames)-this_max_jump, len(frames)-1)

            f2_idx = f1_idx + np.random.randint(this_max_jump+1) + 1
            f2_idx = min(f2_idx, len(frames)-this_max_jump//2, len(frames)-1)

            frames_idx = [start_idx, f1_idx, f2_idx]
            if np.random.rand() < 0.5:
                # Reverse time
                frames_idx = frames_idx[::-1]

            sequence_seed = np.random.randint(2147483647)
            images = []
            masks = []
            target_object = None
            for f_idx in frames_idx:
                jpg_name = frames[f_idx][:-4] + '.jpg'
                png_name = frames[f_idx][:-4] + '.png'
                info['frames'].append(jpg_name)

                reseed(sequence_seed)
                this_im = Image.open(path.join(vid_im_path, jpg_name)).convert('RGB')
                this_im = self.all_im_dual_transform(this_im)
                reseed(sequence_seed)
                this_gt = Image.open(path.join(vid_gt_path, png_name)).convert('P')
                this_gt = self.all_gt_dual_transform(this_gt)

                pairwise_seed = np.random.randint(2147483647)
                reseed(pairwise_seed)
                this_im = self.pair_im_dual_transform(this_im)
                reseed(pairwise_seed)
                this_gt = self.pair_gt_dual_transform(this_gt)

                this_im = self.final_im_transform(this_im)
                this_gt = np.array(this_gt)

                images.append(this_im)
                masks.append(this_gt)

            images = torch.stack(images, 0)

            labels = np.unique(masks[0])
            # Remove background
            labels = labels[labels!=0]

            if len(labels) == 0:
                target_object = -1 # all black if no objects
                has_second_object = False
                trials += 1
            else:
                target_object = np.random.choice(labels)
                has_second_object = (len(labels) > 1)
                if has_second_object:
                    labels = labels[labels!=target_object]
                    second_object = np.random.choice(labels)
                break

        masks = np.stack(masks, 0)
        tar_masks = (masks==target_object).astype(np.float32)[:,np.newaxis,:,:]
        if has_second_object:
            sec_masks = (masks==second_object).astype(np.float32)[:,np.newaxis,:,:]
            selector = torch.FloatTensor([1, 1])
        else:
            sec_masks = np.zeros_like(tar_masks)
            selector = torch.FloatTensor([1, 0])

        cls_gt = np.zeros((3, 384, 384), dtype=np.int)
        cls_gt[tar_masks[:,0] > 0.5] = 1
        cls_gt[sec_masks[:,0] > 0.5] = 2

        data = {
            'rgb': images,
            'gt': tar_masks,
            'cls_gt': cls_gt,
            'sec_gt': sec_masks,
            'selector': selector,
            'info': info,
        }

        return data

    def __len__(self):
        return len(self.videos)
