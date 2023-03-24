import os
import glob
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# NOTE: do not remove the imports below
from torchnet.dataset import TensorDataset, ResampleDataset
from abstract_getters import AbstractGetters
from networks import Encoder, Decoder, ImgClassifier


class Getters(AbstractGetters):
    LABEL_DICT = {
        0: ["digit_class", 10],  # label_index: label_name, num_classes
    }

    def get_encs_decs(self, flags, liks):
        M = flags.num_modalities
        return [Encoder(flags.style_dim, flags.class_dim) for m in range(M)], \
               [Decoder(flags.style_dim, flags.class_dim, liks[m], flags.llik_scale_param[m]) for m in range(M)]

    def get_img_to_digit_clfs(self, flags, path=""):
        M = flags.num_modalities
        img_to_digit_clfs = [ImgClassifier() for _ in range(M)]
        for m in range(M):
            if flags.cuda:
                img_to_digit_clfs[m].cuda()
            img_to_digit_clfs[m].load_state_dict(torch.load(os.path.join(path, 'pretrained_img_to_digit_clf')))
        return img_to_digit_clfs

    def get_data_loaders(self, batch_size, num_modalities, num_workers, shuffle=True, device="cuda",
                         random_noise=False, data_dir=None):
        exp_dir = os.path.split(os.path.abspath(__file__))[0]
        if not os.path.exists(os.path.join(exp_dir, "data/MMMNIST", "train-mmmnist-1-idx.pt")):
            make_mmmnist(M=num_modalities)
        # get transformed indices
        global ts
        ts = []
        global ss
        ss = []
        for m in range(num_modalities):
            mth_train_path = os.path.join(exp_dir, "data/MMMNIST", "train-mmmnist-" + str(m + 1) + "-idx.pt")
            mth_test_path = os.path.join(exp_dir, "data/MMMNIST", "test-mmmnist-" + str(m + 1) + "-idx.pt")
            if not os.path.exists(mth_train_path) or not os.path.exists(mth_test_path):
                make_mmmnist(M=num_modalities)
            ts.append(torch.load(mth_train_path))
            ss.append(torch.load(mth_test_path))
        M = num_modalities
        # load base datasets
        global t_data_loaders
        t_data_loaders = []
        global s_data_loaders
        s_data_loaders = []

        if not random_noise:
            tx = transforms.ToTensor()
        else:
            tx = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x + torch.randn_like(x))])

        for m in range(M):
            kwargs = {"num_workers": num_workers, "pin_memory": True} if device == "cuda" else {}
            t_m = DataLoader(datasets.MNIST(os.path.join(exp_dir, "data/MMMNIST"), train=True, download=True,
                                            transform=tx), batch_size=batch_size, shuffle=shuffle, **kwargs)
            s_m = DataLoader(datasets.MNIST(os.path.join(exp_dir, "data/MMMNIST"), train=False, download=True,
                                            transform=tx), batch_size=batch_size, shuffle=shuffle, **kwargs)
            t_data_loaders.append(t_m)
            s_data_loaders.append(s_m)

        cmd = "train_mmmnist = TensorDataset(["
        for m in range(M):
            cmd = cmd + "ResampleDataset(t_data_loaders[" + str(m) + " ].dataset, lambda d, i: ts[" + \
                  str(m) + "][i], size=len(ts[" + str(m) + "])), "
        cmd = cmd + "])"
        ldict = {'t_data_loaders': t_data_loaders, 'ts': ts}
        exec(cmd, globals(), ldict)
        train_mmmnist = ldict["train_mmmnist"]

        cmd = "test_mmmnist = TensorDataset(["
        for m in range(M):
            cmd = cmd + "ResampleDataset(s_data_loaders[" + str(m) + "].dataset, lambda d, i: ss[" + \
                  str(m) + "][i], size=len(ss[" + str(m) + "])), "
        cmd = cmd + "])"
        ldict = {'s_data_loaders': s_data_loaders, 'ts': ss}
        exec(cmd, globals(), ldict)
        test_mmmnist = ldict["test_mmmnist"]

        kwargs = {"num_workers": num_workers, "pin_memory": True} if device == "cuda" else {}
        train = DataLoader(train_mmmnist, batch_size=batch_size, shuffle=shuffle, drop_last=True, **kwargs)
        test = DataLoader(test_mmmnist, batch_size=batch_size, shuffle=shuffle, drop_last=True, **kwargs)

        return train, test


def rand_match_on_idx(ls, lis, max_d=10000, dm=10, M=3):
    """
    ls: list of sorted labels, for each modality
    lis: indices of sorted labels in original lists, for each modality
    """
    _lis = []
    for l in ls[0].unique():    # assuming both have same idxs
        l_idxs = []
        n = max_d + 1
        for m in range(M):
            l_idxs.append(lis[m][ls[m] == l])
            n = min(n, l_idxs[m].size(0))
        n = min(n, max_d)
        for m in range(M):
            l_idxs[m] = l_idxs[m][:n]
            _lis.append([])
        for _ in range(dm):
            for m in range(M):
                _lis[m].append(l_idxs[m][torch.randperm(n)])
    for m in range(M):
        _lis[m] = torch.cat(_lis[m])
    return _lis


def make_mmmnist(M, datapath="", dm=1, max_d=10000, random_seed=42):

    # Remove old files
    mmmnist_path = os.path.join(datapath, "data/MMMNIST")
    fileList = glob.glob(os.path.join(mmmnist_path, "*.pt"))
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)

    # set random seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)

    # get the datasets
    train_sets = []
    test_sets = []
    train_ls = []
    train_lis = []
    test_ls = []
    test_lis = []
    tx = transforms.ToTensor()
    for m in range(M):
        train_mnist_m = datasets.MNIST(mmmnist_path, train=True, download=True, transform=tx)
        train_sets.append(train_mnist_m)
        test_mnist_m = datasets.MNIST(mmmnist_path, train=False, download=True, transform=tx)
        test_sets.append(test_mnist_m)
        train_l_m, train_li_m = train_mnist_m.targets.sort()
        test_l_m, test_li_m = test_mnist_m.targets.sort()
        train_ls.append(train_l_m)
        train_lis.append(train_li_m)
        test_ls.append(test_l_m)
        test_lis.append(test_li_m)

    # Perform index matching here
    train_idxs = rand_match_on_idx(train_ls, train_lis, dm=dm, max_d=max_d, M=M)
    print("len train idx:", *[len(train_idxs[m]) for m in range(M)])
    test_idxs = rand_match_on_idx(test_ls, test_lis, dm=dm, max_d=max_d, M=M)
    print("len test idx:", *[len(test_idxs[m]) for m in range(M)])
    for m in range(M):
        torch.save(train_idxs[m], os.path.join(mmmnist_path, "train-mmmnist-" + str(m + 1) + "-idx.pt"))
        torch.save(test_idxs[m], os.path.join(mmmnist_path, "test-mmmnist-" + str(m + 1) + "-idx.pt"))
