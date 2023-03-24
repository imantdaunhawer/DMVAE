import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable
from collections import OrderedDict
from utils import reparameterize


svhn_dataSize = torch.Size([3, 32, 32])
svhn_imgChans = svhn_dataSize[0]
svhn_fBase = 32


class MNISTEncoder(nn.Module):
    """
    Encoder for MNIST images, adapted from Shi'19:
    https://github.com/iffsid/mmvae
    """
    def __init__(self, style_dim, class_dim):
        super(MNISTEncoder, self).__init__()

        self.shared = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU())

        self.style_mu = nn.Linear(400, style_dim)
        self.style_logvar = nn.Linear(400, style_dim)

        self.class_mu = nn.Linear(400, class_dim)
        self.class_logvar = nn.Linear(400, class_dim)

    def forward(self, x):
        h = self.shared(x.view(-1, 784))
        return self.style_mu(h).squeeze(), self.style_logvar(h).squeeze(), self.class_mu(h).squeeze(), \
               self.class_logvar(h).squeeze()


class MNISTDecoder(nn.Module):
    """
    Decoder for MNIST images, adapted from Shi'19:
    https://github.com/iffsid/mmvae
    """
    def __init__(self, style_dim, class_dim, likelihood, scale):
        super(MNISTDecoder, self).__init__()
        self.fc1 = nn.Linear(style_dim + class_dim, 400)
        self.fc2 = nn.Linear(400, 784)
        self.relu = nn.ReLU()
        self.likelihood = likelihood
        self.scale = scale

    def forward(self, z_style, z_class, plot=False):
        z = torch.cat((z_style, z_class), dim=1)
        h = self.relu(self.fc1(z))
        h = self.fc2(h).view(-1, 1, 28, 28)
        h = torch.sigmoid(h)
        if plot is True:
            return h
        return [h, self.scale * torch.ones_like(h).cuda()]

    def compute_likelihood(self, ss_list, targets, norm_value=1.):
        px_z = self.likelihood(*ss_list).log_prob(targets).sum()
        return px_z / norm_value

    def compute_loglik(self, image_batch, style_mu, style_logvar, class_mu, class_logvar, reparam=True, freeze_content=False, reparam_c=False):
        style_latent_embeddings = reparameterize(training=reparam, mu=style_mu, logvar=style_logvar)
        if not reparam_c:
            class_latent_embeddings = reparameterize(training=False, mu=class_mu, logvar=class_logvar)
        else:
            class_latent_embeddings = reparameterize(training=reparam, mu=class_mu, logvar=class_logvar)
        if freeze_content:
            ss_list = self.forward(style_latent_embeddings, class_latent_embeddings.data)  # a list of sufficient statistics
        else:
            ss_list = self.forward(style_latent_embeddings, class_latent_embeddings)       # a list of sufficient statistics
        loglik = self.compute_likelihood(ss_list, Variable(image_batch), norm_value=len(image_batch) + 1e-8)
        return loglik


class SVHNEncoder(nn.Module):
    """
    Encoder for SVHN images, adapted from Shi'19:
    https://github.com/iffsid/mmvae
    """
    def __init__(self, style_dim, class_dim):
        super(SVHNEncoder, self).__init__()
        self.style_dim = style_dim
        self.class_dim = class_dim
        self.enc = nn.Sequential(
            # input size: 3 x 32 x 32
            nn.Conv2d(svhn_imgChans, svhn_fBase, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 16 x 16
            nn.Conv2d(svhn_fBase, svhn_fBase * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 8
            nn.Conv2d(svhn_fBase * 2, svhn_fBase * 4, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 4 x 4
        )
        if style_dim > 0:
            self.style_mu = nn.Conv2d(svhn_fBase * 4, style_dim, 4, 1, 0, bias=True)
            self.style_logvar = nn.Conv2d(svhn_fBase * 4, style_dim, 4, 1, 0, bias=True)
        self.class_mu = nn.Conv2d(svhn_fBase * 4, class_dim, 4, 1, 0, bias=True)
        self.class_logvar = nn.Conv2d(svhn_fBase * 4, class_dim, 4, 1, 0, bias=True)

    def forward(self, x):
        h = self.enc(x)
        if self.style_dim == 0:
            bs = h.shape[0]
            if bs == 1:
                dummy = torch.zeros(0).cuda()
            else:
                dummy = torch.zeros(h.shape[0], 0).cuda()
            return dummy, dummy, self.class_mu(h).squeeze(), self.class_logvar(h).squeeze()
        else:
            return self.style_mu(h).squeeze(), self.style_logvar(h).squeeze(), self.class_mu(h).squeeze(), \
                self.class_logvar(h).squeeze()


class SVHNDecoder(nn.Module):
    """
    Decoder for SVHN images, adapted from Shi'19:
    https://github.com/iffsid/mmvae
    """
    def __init__(self, style_dim, class_dim, likelihood, scale):
        super(SVHNDecoder, self).__init__()
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(style_dim + class_dim, svhn_fBase * 4, 4, 1, 0, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 4 x 4
            nn.ConvTranspose2d(svhn_fBase * 4, svhn_fBase * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 8
            nn.ConvTranspose2d(svhn_fBase * 2, svhn_fBase, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 16 x 16
            nn.ConvTranspose2d(svhn_fBase, svhn_imgChans, 4, 2, 1, bias=True),
            # Output size: 3 x 32 x 32
        )
        self.likelihood = likelihood
        self.scale = scale

    def forward(self, z_style, z_class, plot=False):
        z = torch.cat((z_style, z_class), dim=1)
        z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        out = self.dec(z.view(-1, *z.size()[-3:]))
        if plot is True:
            return out
        return [out, self.scale*torch.ones_like(out).cuda()]  # NOTE: consider learning scale param, too

    def compute_likelihood(self, ss_list, targets, norm_value=1.):
        px_z = self.likelihood(*ss_list).log_prob(targets).sum()
        return px_z / norm_value

    def compute_loglik(self, image_batch, style_mu, style_logvar, class_mu, class_logvar, reparam=True, freeze_content=False, reparam_c=False):
        style_latent_embeddings = reparameterize(training=reparam, mu=style_mu, logvar=style_logvar)
        if not reparam_c:
            class_latent_embeddings = reparameterize(training=False, mu=class_mu, logvar=class_logvar)
        else:
            class_latent_embeddings = reparameterize(training=reparam, mu=class_mu, logvar=class_logvar)
        if freeze_content:
            ss_list = self.forward(style_latent_embeddings, class_latent_embeddings.data)  # a list of sufficient statistics
        else:
            ss_list = self.forward(style_latent_embeddings, class_latent_embeddings)       # a list of sufficient statistics
        loglik = self.compute_likelihood(ss_list, Variable(image_batch), norm_value=len(image_batch) + 1e-8)
        return loglik


class SVHNImgClassifier(nn.Module):
    """
    SVHN image-to-digit classifier.
    """
    def __init__(self, out_dims=10):
        super(SVHNImgClassifier, self).__init__()

        self.enc = nn.Sequential(
            # input size: 3 x 32 x 32
            nn.Conv2d(svhn_imgChans, svhn_fBase, 4, 2, 1, bias=False),
            nn.BatchNorm2d(svhn_fBase),
            nn.ReLU(True),
            nn.Dropout2d(p=0.5),
            # size: (fBase) x 16 x 16
            nn.Conv2d(svhn_fBase, svhn_fBase * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(svhn_fBase * 2),
            nn.ReLU(True),
            nn.Dropout2d(p=0.5),
            # size: (fBase * 2) x 8 x 8
            nn.Conv2d(svhn_fBase * 2, svhn_fBase * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(svhn_fBase * 4),
            nn.ReLU(True),
            nn.Dropout2d(p=0.5),
            # size: (fBase * 4) x 4 x 4
        )
        self.hidden = nn.Conv2d(svhn_fBase * 4, 256, 4, 1, 0, bias=True)
        self.linear = nn.Linear(in_features=256, out_features=out_dims, bias=True)

    def forward(self, x):
        h = self.hidden(self.enc(x))
        dp = nn.Dropout(p=0.5)
        res = self.linear(dp(h.squeeze()).squeeze())
        return res


class ClassifierCollection(object):
    """
    Structure with classifiers for latent space representations.
    """
    def __init__(self, class_dim, style_dim, num_classes, label_ix, label_name, num_modalities=2, cuda=True):
        self.class_dim = class_dim
        self.style_dim = style_dim
        self.num_classes = num_classes
        self.label_ix = label_ix
        self.label_name = label_name
        self.cuda = cuda
        self.num_modalities = num_modalities
        self.classifier_tuples = []

    def initialize_models(self, lr=0.001, betas=(0.9, 0.999), linear=False):
        for m in range(self.num_modalities):
            if linear is True:
                C = LinearClassifier
            else:
                C = Classifier
            if self.class_dim > 0:
                clf_content = C(z_dim=self.class_dim, num_classes=self.num_classes)
                opt_content = optim.Adam(clf_content.parameters(), lr=lr, betas=betas)
                clf_content.cuda()
            else:
                clf_content = None
                opt_content = None
            if self.style_dim > 0:
                clf_style = C(z_dim=self.style_dim, num_classes=self.num_classes)
                opt_style = optim.Adam(clf_style.parameters(), lr=lr, betas=betas)
                clf_style.cuda()
            else:
                clf_style = None
                opt_style = None
            if self.class_dim > 0 and self.style_dim > 0:
                clf_concat = C(z_dim=self.class_dim+self.style_dim, num_classes=self.num_classes)
                opt_concat = optim.Adam(clf_concat.parameters(), lr=lr, betas=betas)
                clf_concat.cuda()
            else:
                clf_concat = None
                opt_concat = None

            classifier_tuple = ((clf_content, opt_content, torch.nn.CrossEntropyLoss()),
                                (clf_style, opt_style, torch.nn.CrossEntropyLoss()),
                                (clf_concat, opt_concat, torch.nn.CrossEntropyLoss()))
            self.classifier_tuples.append(classifier_tuple)

    def get_model(self, modality):
        if self.classifier_tuples:
            return self.classifier_tuples[modality]
        else:
            raise AttributeError("Model is not initialized.")

    def classify(self, modality, labels, class_mu, class_logvar, style_mu, style_logvar, train=False, reparam=True):
        """
        Run the classifiers and return the predictions for
        class, style, and concatenated embeddings.

        If train=True then train the classifiers.
        """
        # setup
        for clf, opt, _ in self.classifier_tuples[modality]:
            if clf is not None:
                opt.zero_grad()
                if train:
                    clf.train()
                else:
                    clf.eval()

        class_latent_embeddings_clf = None
        style_latent_embeddings_clf = None
        concat_latent_embeddings_clf = None
        # reparametrize embeddings (if training)
        if class_mu is not None and class_logvar is not None:
            class_latent_embeddings_clf = reparameterize(training=reparam, mu=class_mu, logvar=class_logvar)
        if style_mu is not None and style_logvar is not None:
            style_latent_embeddings_clf = reparameterize(training=reparam, mu=style_mu, logvar=style_logvar)
        if class_mu is not None and class_logvar is not None and style_mu is not None and style_logvar is not None:
            concat_latent_embeddings_clf = \
                torch.cat((class_latent_embeddings_clf, style_latent_embeddings_clf), dim=1)

        # compute classifier predictions
        latent_embeddings = [class_latent_embeddings_clf, style_latent_embeddings_clf, concat_latent_embeddings_clf]
        predictions = []
        for i, (clf, opt, lossf) in enumerate(self.classifier_tuples[modality]):

            if clf is not None:
                out = clf(latent_embeddings[i].data)
                _, pred = torch.max(out, 1)
                if train is True:
                    loss = lossf(out, labels)
                    loss.backward()
                    opt.step()
            else:
                pred = None
            predictions.append(pred)
        return predictions


class Classifier(nn.Module):
    """
    Generic classifier.
    """
    def __init__(self, z_dim, num_classes):
        super(Classifier, self).__init__()

        self.fc_model = nn.Sequential(OrderedDict([
            ('fc_1', nn.Linear(in_features=z_dim, out_features=256, bias=True)),
            ('fc_1_bn', nn.BatchNorm1d(num_features=256)),
            ('relu_1', nn.ReLU(inplace=True)),
            ('fc_2', nn.Linear(in_features=256, out_features=num_classes, bias=True))
        ]))

    def forward(self, z):
        x = self.fc_model(z)
        return x


class LinearClassifier(nn.Module):
    """
    Generic linear classifier.
    """
    def __init__(self, z_dim, num_classes):
        super(LinearClassifier, self).__init__()
        self.fc_linear = nn.Linear(z_dim, num_classes)
        nn.init.normal_(self.fc_linear.weight, std=0.01)
        nn.init.constant_(self.fc_linear.bias, 0)

    def forward(self, z):
        x = self.fc_linear(z)
        return x


class TCDiscriminator(nn.Module):
    """
    Discriminator network for estimating total correlation among a set of RVs.
    Adapted from https://github.com/1Konny/FactorVAE
    """
    def __init__(self, s_dim, c_dim, hidden_size):
        super(TCDiscriminator, self).__init__()
        assert (hidden_size % 2) == 0
        self.s_dim = s_dim
        self.c_dim = c_dim

        self.score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_size, 2),
        )

        self.map_s = nn.Sequential(
            nn.Linear(s_dim, hidden_size//2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_size//2, hidden_size//2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_size//2, hidden_size//2),
            nn.LeakyReLU(0.2, True),
        )

        self.map_c = nn.Sequential(
            nn.Linear(c_dim, hidden_size//2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_size//2, hidden_size//2),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_size//2, hidden_size//2),
            nn.LeakyReLU(0.2, True),
        )

    def forward(self, s, c):
        s = self.map_s(s)
        c = self.map_c(c)
        z = torch.cat([s, c], dim=1)
        return self.score(z).squeeze()


class MNISTImgClassifier(nn.Module):
    """
    MNIST image-to-digit classifier.
    """
    def __init__(self):
        super(MNISTImgClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)


class InfomaxNonlinearProjectionHead(nn.Module):
    def __init__(self, ndim):
        super(InfomaxNonlinearProjectionHead, self).__init__()
        self.ndim = ndim
        self.net = nn.Sequential(
            nn.Linear(ndim, ndim*2),
            nn.ReLU(),
            nn.Linear(ndim*2, ndim))

    def forward(self, x):
        return self.net(x)


class InfomaxLinearProjectionHead(nn.Module):
    def __init__(self, ndim):
        super(InfomaxLinearProjectionHead, self).__init__()

    def forward(self, x):
        return x
