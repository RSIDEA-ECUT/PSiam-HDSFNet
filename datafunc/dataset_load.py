import torch.utils.data as data
import glob
import os

import numpy as np
import rasterio
from tqdm import tqdm


def normalize_cau(imgs):
    d = len(imgs)
    for i in range(d):
        if imgs[i, :, :].max() == imgs[i, :, :].min():
            norm = imgs[i, :, :] / imgs[i, :, :].max()
            imgs[i, :, :] = norm
        else:
            imgs[i, :, :] = (imgs[i, :, :] - imgs[i, :, :].min()) / (
                    imgs[i, :, :].max() - imgs[i, :, :].min())

    return imgs


def load_cau1(path):
    bands_selected = [1]
    with rasterio.open(path) as data:
        cas1 = data.read(bands_selected)
    cas1 = cas1.astype(np.float32)
    cas1 = normalize_cau(cas1)
    return cas1


def load_cau2(path):
    bands_selected = [1, 2, 3, 4]
    with rasterio.open(path) as data:
        cas2 = data.read(bands_selected)
    cas2 = cas2.astype(np.float32)
    cas2 = normalize_cau(cas2)
    return cas2


def load_lc(path):
    with rasterio.open(path) as data:
        lc = data.read(1)
    lc = lc.astype(np.float32)
    lc[lc == 1.] = 1.
    lc[lc == 0.] = 0.
    return lc


def load_sample(sample, unlabeled=False):
    img = load_cau2(sample["opt"])
    img = np.concatenate((img, load_cau1(sample["vv"])), axis=0)
    if unlabeled:
        return {'image': img, 'id': sample["id"]}
    else:
        lc = load_lc(sample["lc"])
        return {'image': img, 'label': lc, 'id': sample["id"]}


class dataset_load(data.Dataset):
    def __init__(self,
                 path,
                 subset='train',
                 unlabeled=True,
                 train_index=None):
        super(dataset_load, self).__init__()

        assert subset in ['train', 'test']
        self.unlabeled = unlabeled
        self.train_index = train_index
        self.n_inputs = 4
        self.n_classes = 2

        assert os.path.exists(path)

        if subset == 'train':
            train_list = []
            for trainfolder in ['train']:
                train_list += [os.path.join(trainfolder, x) for x in
                               os.listdir(os.path.join(path, trainfolder))]
            train_list = [x for x in train_list if "opt" in x]
            sample_dirs = train_list
        else:
            test_list = []
            for testfolder in ['test']:
                test_list += [os.path.join(testfolder, x) for x in
                              os.listdir(os.path.join(path, testfolder))]
            test_list = [x for x in test_list if "opt" in x]
            sample_dirs = test_list

        self.samples = []
        for folder in sample_dirs:
            opt_locations = glob.glob(os.path.join(path, f"{folder}/*.png"), recursive=True)
            for opt_loc in tqdm(opt_locations, desc="[Load]"):
                vv_loc = opt_loc.replace("opt", "vv")
                lc_loc = opt_loc.replace("opt", "flood_vv")
                self.samples.append(
                    {"lc": lc_loc, "vv": vv_loc, "opt": opt_loc, "id": os.path.basename(lc_loc)})

            if self.train_index:
                Tindex = np.load(self.train_index)
                self.samples = [self.samples[i] for i in Tindex]

        print("loaded", len(self.samples),
              "samples from the CHU Flood subset", subset)

    def __getitem__(self, index):
        sample = self.samples[index]
        data_sample = load_sample(sample, unlabeled=self.unlabeled)
        return data_sample

    def __len__(self):
        return len(self.samples)
