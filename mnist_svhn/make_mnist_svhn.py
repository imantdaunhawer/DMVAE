# Code adapted from Shi et al.
# https://github.com/iffsid/mmvae
import torch
import numpy as np
import random

from torchvision import datasets, transforms

RANDOM_SEED = 42
MAX_D = 10000   # maximum number of datapoints per class
DM = 20         # data multiplier: random permutations to match; adopted from Shi et al. (2019)


def rand_match_on_idx(l1, idx1, l2, idx2, max_d=10000, dm=10):
    """
    l*: sorted labels
    idx*: indices of sorted labels in original list
    """
    _idx1, _idx2 = [], []
    for l in l1.unique():  # assuming both have same idxs
        l_idx1, l_idx2 = idx1[l1 == l], idx2[l2 == l]
        n = min(l_idx1.size(0), l_idx2.size(0), max_d)
        l_idx1, l_idx2 = l_idx1[:n], l_idx2[:n]
        for _ in range(dm):
            _idx1.append(l_idx1[torch.randperm(n)])
            _idx2.append(l_idx2[torch.randperm(n)])
    return torch.cat(_idx1), torch.cat(_idx2)


if __name__ == '__main__':
    # set random seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # get the individual datasets
    tx = transforms.ToTensor()
    train_mnist = datasets.MNIST("data/MNIST_SVHN", train=True, download=True, transform=tx)
    test_mnist = datasets.MNIST("data/MNIST_SVHN", train=False, download=True, transform=tx)
    train_svhn = datasets.SVHN("data/MNIST_SVHN", split="train", download=True, transform=tx)
    test_svhn = datasets.SVHN("data/MNIST_SVHN", split="test", download=True, transform=tx)
    # svhn labels need extra work
    train_svhn.labels = torch.LongTensor(train_svhn.labels.squeeze().astype(int)) % 10
    test_svhn.labels = torch.LongTensor(test_svhn.labels.squeeze().astype(int)) % 10

    mnist_l, mnist_li = train_mnist.targets.sort()
    svhn_l, svhn_li = train_svhn.labels.sort()
    idx1, idx2 = rand_match_on_idx(mnist_l, mnist_li, svhn_l, svhn_li, max_d=MAX_D, dm=DM)
    print("len train idx:", len(idx1), len(idx2))
    torch.save(idx1, "data/MNIST_SVHN/train-ms-mnist-idx.pt")
    torch.save(idx2, "data/MNIST_SVHN/train-ms-svhn-idx.pt")

    mnist_l, mnist_li = test_mnist.targets.sort()
    svhn_l, svhn_li = test_svhn.labels.sort()
    idx1, idx2 = rand_match_on_idx(mnist_l, mnist_li, svhn_l, svhn_li, max_d=MAX_D, dm=DM)
    print("len test idx:", len(idx1), len(idx2))
    torch.save(idx1, "data/MNIST_SVHN/test-ms-mnist-idx.pt")
    torch.save(idx2, "data/MNIST_SVHN/test-ms-svhn-idx.pt")
