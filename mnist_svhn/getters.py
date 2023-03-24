import os
import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchnet.dataset import TensorDataset, ResampleDataset
from abstract_getters import AbstractGetters
from networks import MNISTEncoder, SVHNEncoder, MNISTDecoder, SVHNDecoder, MNISTImgClassifier, SVHNImgClassifier


class Getters(AbstractGetters):
    LABEL_DICT = {
        0: ["digit_class", 10],  # label_index: label_name, num_classes
    }

    def get_encs_decs(self, flags, liks):
        return [MNISTEncoder(flags.style_dim, flags.class_dim), SVHNEncoder(flags.style_dim, flags.class_dim)], \
               [MNISTDecoder(flags.style_dim, flags.class_dim, liks[0], flags.llik_scale_param[0]),
                SVHNDecoder(flags.style_dim, flags.class_dim, liks[1], flags.llik_scale_param[1])]

    def get_img_to_digit_clfs(self, flags, path=""):
        M = flags.num_modalities
        img_to_digit_clfs = [MNISTImgClassifier(), SVHNImgClassifier()]
        if flags.cuda:
            for m in range(M):
                img_to_digit_clfs[m].cuda()
        img_to_digit_clfs[0].load_state_dict(torch.load(os.path.join(path, 'pretrained_img_to_digit_clf')))
        img_to_digit_clfs[1].load_state_dict(torch.load(os.path.join(path, 'pretrained_img_to_digit_clf_SVHN')))
        return img_to_digit_clfs

    def get_data_loaders(self, batch_size, num_modalities, num_workers, shuffle=True, device="cuda",
                         random_noise=False, data_dir=None):
        exp_dir = os.path.split(os.path.abspath(__file__))[0]
        if not (os.path.exists(os.path.join(exp_dir, "data/MNIST_SVHN", "train-ms-mnist-idx.pt"))
                and os.path.exists(os.path.join(exp_dir, "data/MNIST_SVHN", "train-ms-svhn-idx.pt"))
                and os.path.exists(os.path.join(exp_dir, "data/MNIST_SVHN", "test-ms-mnist-idx.pt"))
                and os.path.exists(os.path.join(exp_dir, "data/MNIST_SVHN", "test-ms-svhn-idx.pt"))):
            raise RuntimeError("Generate transformed indices first (using make_mnist_svhn.py)")
        # get transformed indices
        t_mnist = torch.load(os.path.join(exp_dir, "data/MNIST_SVHN", "train-ms-mnist-idx.pt"))
        t_svhn = torch.load(os.path.join(exp_dir, "data/MNIST_SVHN", "train-ms-svhn-idx.pt"))
        s_mnist = torch.load(os.path.join(exp_dir, "data/MNIST_SVHN", "test-ms-mnist-idx.pt"))
        s_svhn = torch.load(os.path.join(exp_dir, "data/MNIST_SVHN", "test-ms-svhn-idx.pt"))

        # load base datasets
        kwargs = {"num_workers": num_workers, "pin_memory": True} if device == "cuda" else {}
        tx = transforms.ToTensor()
        t1 = DataLoader(datasets.MNIST(os.path.join(exp_dir, "data/MNIST_SVHN"), train=True, download=True,
                                       transform=tx), batch_size=batch_size, shuffle=shuffle, **kwargs)
        s1 = DataLoader(datasets.MNIST(os.path.join(exp_dir, "data/MNIST_SVHN"), train=False, download=True,
                                       transform=tx), batch_size=batch_size, shuffle=shuffle, **kwargs)
        kwargs = {"num_workers": 1, "pin_memory": True} if device == "cuda" else {}
        tx = transforms.ToTensor()
        t2 = DataLoader(datasets.SVHN(os.path.join(exp_dir, "data/MNIST_SVHN"), split='train', download=True,
                                      transform=tx), batch_size=batch_size, shuffle=shuffle, **kwargs)
        s2 = DataLoader(datasets.SVHN(os.path.join(exp_dir, "data/MNIST_SVHN"), split='test', download=True,
                                      transform=tx), batch_size=batch_size, shuffle=shuffle, **kwargs)

        train_mnist_svhn = TensorDataset([ResampleDataset(t1.dataset, lambda d, i: t_mnist[i], size=len(t_mnist)),
                                          ResampleDataset(t2.dataset, lambda d, i: t_svhn[i], size=len(t_svhn))])
        test_mnist_svhn = TensorDataset([ResampleDataset(s1.dataset, lambda d, i: s_mnist[i], size=len(s_mnist)),
                                         ResampleDataset(s2.dataset, lambda d, i: s_svhn[i], size=len(s_svhn))])

        kwargs = {"num_workers": num_workers, "pin_memory": True} if device == "cuda" else {}
        train = DataLoader(train_mnist_svhn, batch_size=batch_size, shuffle=shuffle, drop_last=True, **kwargs)
        test = DataLoader(test_mnist_svhn, batch_size=batch_size, shuffle=shuffle, drop_last=True, **kwargs)

        return train, test
