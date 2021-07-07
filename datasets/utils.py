"""
Created by Hamid Eghbal-zadeh at 05.02.21
Johannes Kepler University of Linz
"""
from random import shuffle

from torchvision import transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from torchvision.datasets.folder import ImageFolder

from pathlib import Path
import os
import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms



def set_path_info():
    dirname = os.path.dirname(__file__)
    path = Path(dirname)
    parent = str(path.parent)
    db_path = os.path.join(os.path.join(parent, 'datasets'), "db.path")

    path_info = np.loadtxt(db_path, dtype=str, delimiter="\n")

    PATH_DICT = {}
    for l in path_info:
        db_name, path = l.split(':')
        PATH_DICT[db_name] = path
    return PATH_DICT


def get_database_stats(loader):
    mean = 0.
    std = 0.
    mini = 0.
    maxi = 0.
    cnt = 0
    for data in loader:
        try:
            images, _ = data
            batch_samples = images.size(0)  # batch size (the last batch can have smaller size!)
            cnt += batch_samples
            images = images.view(batch_samples, images.size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)
            mini = images.min() if images.min() < mini else mini
            maxi = images.max() if images.max() > maxi else maxi
        except:
            print('ok!')
    mean /= cnt
    std /= cnt
    print('mean: {} std: {} min: {} max: {}'.format(mean, std, mini, maxi))

    return mean, std


def rescale_linear(array, old_min, old_max, new_min, new_max):
    array = np.moveaxis(array.cpu().detach().numpy(), 1, 3)
    """Rescale an arrary linearly."""
    m = (new_max - new_min) / (old_max - old_min)
    b = new_min - m * old_min
    return m * array + b


def Class2FolderTargetTransform(class2folder):
    def getkey(key):
        folder = int(class2folder[key])
        return folder

    return getkey


def get_dataset_loaders(ds_name, crop_size, img_size, batch_size, n_workers, aug, dec_out_nonlin, seq_len=0,
                        device=None):
    PATH_DICT = set_path_info()
    trn_transform, val_transform = get_transform(ds_name, crop_size, img_size, aug, dec_out_nonlin)



    valid_size = 0.2
    np.random.seed(123)
    dataset = ImageFolder(root=PATH_DICT[ds_name],
                          transform=trn_transform
                          )

    # fixing the mixmatch between folder names and assigned labels.
    class2folder = {}
    for folder, label in dataset.class_to_idx.items():
        class2folder[label] = folder

    target_transform = Class2FolderTargetTransform(class2folder)
    dataset.target_transform = target_transform

    num_ds = len(dataset)
    indices = list(range(num_ds))
    num_val_imgs = int(np.floor(valid_size * num_ds))
    num_train_imgs = num_ds - num_val_imgs

    np.random.shuffle(indices)
    train_idx, valid_idx = indices[num_val_imgs:], indices[:num_val_imgs]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    trn_loader = DataLoader(dataset,
                            sampler=train_sampler,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=True,
                            num_workers=n_workers)

    val_loader = DataLoader(dataset, sampler=valid_sampler,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=True,
                            num_workers=n_workers)
    return trn_loader, val_loader, num_train_imgs, num_val_imgs


def get_dataset_loaders_vae(ds_name, crop_size, img_size, batch_size, n_workers, aug, dec_out_nonlin, seq_len=0,
                            device=None):
    PATH_DICT = set_path_info()
    trn_transform, val_transform = get_transform(ds_name, crop_size, img_size, aug, dec_out_nonlin)

    valid_size = 0.2
    np.random.seed(123)
    dataset = ImageFolder(root=PATH_DICT[ds_name],
                          transform=trn_transform
                          )
    num_ds = len(dataset)
    # fixing the mixmatch between folder names and assigned labels.
    class2folder = {}
    for folder, label in dataset.class_to_idx.items():
        class2folder[label] = folder

    target_transform = Class2FolderTargetTransform(class2folder)
    dataset.target_transform = target_transform

    trn_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True,
                            num_workers=n_workers)

    return trn_loader, num_ds


def Context2ColorTargetTransform(context2color, class2folder):
    def getkey(key):
        folder = int(class2folder[key])
        color_ix = context2color[folder]
        return color_ix

    return getkey


class GaussianBlur(object):
    """blur a single image on CPU"""

    def __init__(self, kernel_size):
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(3, 3, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=3)
        self.blur_v = nn.Conv2d(3, 3, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=3)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):

        img = img.unsqueeze(0)
        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(3, 1)

        self.blur_h.weight.data.copy_(x.view(3, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(3, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        # img = self.tensor_to_pil(img)

        return img


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)





def get_disentangled_loaders(ds_name, crop_size, img_size, batch_size, n_workers, aug, dec_out_nonlin, alt_img_count=1,
                             device=None):
    from datasets.disentangled_dataset import DisentangledFolder
    valid_size = 0.2
    np.random.seed(123)
    PATH_DICT = set_path_info()
    trn_transform, val_transform = get_transform(ds_name, crop_size, img_size, aug, dec_out_nonlin)
    dataset = DisentangledFolder(root=PATH_DICT[ds_name],
                                 transform=trn_transform,
                                 alt_img_count=alt_img_count
                                 )

    num_ds = len(dataset)
    indices = list(range(num_ds))
    num_train_imgs = num_ds



    train_sampler = SequentialSampler(indices)

    trn_loader = DataLoader(dataset, sampler=train_sampler,
                            batch_size=batch_size,
                            shuffle=False,
                            drop_last=True,
                            num_workers=n_workers)


    return trn_loader, num_train_imgs


def get_transform(ds_name, crop_size, img_size, aug, dec_out_nonlin):
    if dec_out_nonlin == 'tanh':
        new_max = 1
        new_min = -1



        old_min = 0.
        old_max = 1.

        m = (new_max - new_min) / (old_max - old_min)
        b = new_min - m * old_min

        SetRange = transforms.Lambda(
            lambda X: m * X + b)
    trans = []

    if 'hf' in aug:
        trans.append(transforms.RandomHorizontalFlip())
    if 'vf' in aug:
        trans.append(transforms.RandomVerticalFlip())
    # if 'rr' in aug:
    p = None
    for a in aug:
        if 'rr' in a:
            p = int(a[2:])
    if p is not None:
        trans.append(transforms.RandomRotation(p))
    p_lst = None
    for a in aug:
        if 'cj' in a:
            p_lst = []
            for p in a[2:].split('_')[1:]:
                p_lst.append(float(p))
    if p_lst is not None:
        trans.append(transforms.ColorJitter(p_lst[0], p_lst[1], p_lst[2]))
    # if 'rc' in aug:
    p = None
    for a in aug:
        if 'rc' in a:
            p = int(a[2:])
    if p is not None:
        trans.append(transforms.RandomCrop(img_size, p))
    trans.append(transforms.ToTensor())
    if dec_out_nonlin == 'tanh':
        trans.append(SetRange)

    for a in aug:
        if 'gb' in a:
            trans.append(GaussianBlur(kernel_size=int(0.1 * img_size)))

    trn_transform = transforms.Compose(trans)
    trans = []

    trans.append(transforms.ToTensor())
    if dec_out_nonlin == 'tanh':
        trans.append(SetRange)

    val_transform = transforms.Compose(trans)

    return trn_transform, val_transform


def seq_shuffle(sequence, seq_len):
    data_len = len(sequence)
    assert data_len % seq_len == 0
    order_list = list(range(0, data_len, seq_len))
    shuffle(order_list)

    shuffled_seq = []
    for oix in order_list:
        start_ix = oix
        stop_ix = oix + seq_len
        shuffled_seq.extend(sequence[start_ix:stop_ix])
    return shuffled_seq


if __name__ == "__main__":
    sequence = list(range(20))
    shuffled_sequence = seq_shuffle(sequence, 4)
    print(shuffled_sequence)
