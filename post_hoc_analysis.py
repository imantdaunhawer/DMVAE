import argparse
import os
# import sys
import glob

import utils
import torch
# import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
# from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision

from sklearn import mixture

import numpy as np

from argparse import Namespace

from itertools import cycle

import matplotlib
import matplotlib.pyplot as plt
from utils import LIKELIHOOD_DICT

from imp import load_source
from utils_tensorboard import (
    write_conditional_generation_to_tensorboard,
    write_unconditional_generation_to_tensorboard,
    write_conditional_fid_to_tensorboard,
    write_unconditional_fid_to_tensorboard)
from tensorboardX import SummaryWriter

import json
import uuid

# hack for plotting on remote server
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
matplotlib.use('agg')

SEED = 42

# random seed
# see https://pytorch.org/docs/stable/notes/randomness.html
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(SEED)
np.random.seed(SEED)

parser = argparse.ArgumentParser(description='Analysing results')
parser.add_argument('--experiment-dir', type=str, default=".",
                    metavar='N', help='path to experiments (tensorboard logs)')
parser.add_argument('--networks-dir', type=str, default=".", metavar="N",
                    help='directory with the relevant networks.py file')
parser.add_argument('--scaling-factors', type=float, nargs="+", default=None,
                    help='list of scaling factors for joint likelihood')
parser.add_argument('--reparam_c', action='store_true', default=None,
        help='if true, reparameterize content for conditional generation. Default: based on flags.json')
parser.add_argument('--content-density-estimation', type=str, default=None,
                    choices=["gmm1", "gmm10", "gmm100"],
                    help='fit a density estimator for on the content representations')
parser.add_argument('--latent-classification', action='store_true', default=False,
                    help='disables CUDA use')
parser.add_argument('--num-fid-samples', type=int, default=10000,
                    help='number of test and generates samples for FID computation')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA use')
cmds = parser.parse_args()

# Import relevant neural network modules
networks = load_source("networks", os.path.join(cmds.networks_dir, "networks.py"))

# Load arguments
with open(os.path.join(cmds.experiment_dir, "flags.json"), 'r') as fh:
    args_dict = json.load(fh)
flags = Namespace(**args_dict)

print("\nCMDS:")
print(cmds)
print("\nFLAGS:")
print(flags)
print()

# Import data loaders
getters = load_source("getters", os.path.join(cmds.networks_dir, "getters.py"))
gtrs = getters.Getters()
train_loader, test_loader = gtrs.get_data_loaders(batch_size=args_dict["batch_size"],
                                                         num_modalities=flags.num_modalities,
                                                         num_workers=flags.num_workers)
img_to_digit_clfs = gtrs.get_img_to_digit_clfs(flags, path=cmds.networks_dir)
reparam_c = True if cmds.reparam_c else flags.reparam_c_for_decoders

# Set up likelihoods
likelihoods = []
for l in args_dict["likelihood_str"].split("-"):
    likelihoods.append(LIKELIHOOD_DICT[l])
if len(likelihoods) == 1:  # assume similar likelihoods if only one was provided
    tmp = likelihoods[0]
    likelihoods = [tmp for _ in range(flags.num_modalities)]
assert len(likelihoods) == flags.num_modalities

# Instantiate encoders and decoders
encs, decs = gtrs.get_encs_decs(flags, likelihoods)

# Load trained models
for e in range(len(encs)):
    model = encs[e]
    model.load_state_dict(torch.load(os.path.join(cmds.experiment_dir, "checkpoints/encoder_"+str(e))))
    if args_dict["cuda"]:
        model.cuda()
    model.eval()
for d in range(len(decs)):
    model = decs[d]
    model.load_state_dict(torch.load(os.path.join(cmds.experiment_dir, "checkpoints/decoder_"+str(d))))
    if args_dict["cuda"]:
        model.cuda()
    model.eval()


def content_density_estimation(density, encoders, data, flags, num_samples=None):

    M = flags.num_modalities
    bs = flags.batch_size
    loader = cycle(data)
    representations = []  # log-likelihoods over the dataset
    if num_samples is None:
        num_iterations = len(data)
    else:
        num_iterations = num_samples // bs

    # collect representations
    for iteration in range(num_iterations):

        # load a mini-batch
        mm_batch = next(loader)

        # do the inference step and store latents and targets
        mm_class_mu = Variable(torch.empty(M, bs, flags.class_dim)).cuda()
        mm_style_mu = Variable(torch.empty(M, bs, flags.style_dim)).cuda()
        mm_class_logvar = Variable(torch.empty(M, bs, flags.class_dim)).cuda()
        mm_style_logvar = Variable(torch.empty(M, bs, flags.style_dim)).cuda()
        for m in range(M):
            encoder = encoders[m]
            image_batch = mm_batch[m][0]
            if flags.cuda:
                image_batch = image_batch.cuda()
            if flags.noisy_inputs:
                image_batch = image_batch + torch.randn_like(image_batch)
            style_mu, style_logvar, class_mu, class_logvar = encoder(Variable(image_batch))
            mm_class_mu[m] = class_mu
            mm_style_mu[m] = style_mu
            mm_class_logvar[m] = class_logvar
            mm_style_logvar[m] = style_logvar

        # compute the product
        poe_mu, poe_logvar = utils.poe(mm_class_mu, mm_class_logvar, prior_expert=flags.prior_expert)
        representations.append(poe_mu.detach().cpu().numpy())

    # fit density estimator
    X = np.array(representations).reshape(-1, flags.class_dim)
    density.fit(X)
    return density


def prepare_fid_computation(dataloader, flags, num_test_samples=None, digit=None):

    # create directories
    hash_value = str(uuid.uuid4())
    gen_path = "/tmp/%s/generated_images" % hash_value
    test_paths = ["/tmp/%s/test_images/m%d" % (hash_value, m) for m in range(flags.num_modalities)]
    for p in [gen_path, *test_paths]:
        if not os.path.exists(p):
            os.makedirs(p)
            print("Created new directory:", p)

    # save test data for each modality
    for m, p in enumerate(test_paths):
        if list(glob.glob("%s/precomputed_fid_statistics.npz" % p)):
            print("Found pre-computed FID-statistics")
            continue
        else:
            print("For FID computation, save test images for m=%d to %s" % (m, p))
            cnt = 0
            for batch in dataloader:
                for i in range(flags.batch_size):
                    image = batch[m][0][i, :, :, :]
                    label = batch[m][1][i]
                    if digit is not None and label.item() != digit:
                        continue
                    torchvision.utils.save_image(image, "{}/{}_{}.png".format(p, cnt, int(label)))
                    cnt += 1
                if num_test_samples is not None and cnt > num_test_samples:
                    break  # NOTE: take at most x test images (e.g., if the evaluation of FID scores takes too long)
        if cnt < num_test_samples:
            print("[WARN] only %d/%d images have been saved" % (cnt, num_test_samples))

    return gen_path, test_paths


def train_classifier(clf, train_loader, test_loader, optimizer, mods_in, encoders, flags, num_epochs=1000,
        early_stopping=np.inf, verbose=True, noisy_inputs=False, num_samples_per_epoch=None):

    # prep
    assert len(mods_in) == len(encoders)
    stop_counter = 0
    i = 0
    acc_prev = 0
    if num_samples_per_epoch is None:
        num_iterations_train = len(train_loader)
        num_iterations_test = len(test_loader)
    else:
        num_iterations_train = num_samples_per_epoch // flags.batch_size
        num_iterations_test = num_samples_per_epoch // flags.batch_size
    train_cycle = cycle(train_loader)
    test_cycle = cycle(test_loader)
    # NOTE: consider decreasing batch size, which is currently based on {train/test}_loader

    # run till stopping criterion
    while i < num_epochs and stop_counter < early_stopping:

        # training epoch
        for j in range(num_iterations_train):

            # do the inference step and store latents and targets
            mm_class_mu = Variable(torch.zeros(flags.num_modalities, flags.batch_size, flags.class_dim)).cuda()
            mm_style_mu = Variable(torch.zeros(flags.num_modalities, flags.batch_size, flags.style_dim)).cuda()
            mm_class_logvar = Variable(torch.zeros(flags.num_modalities, flags.batch_size, flags.class_dim)).cuda()
            mm_style_logvar = Variable(torch.zeros(flags.num_modalities, flags.batch_size, flags.style_dim)).cuda()
            labels_list = []
            mm_batch = next(train_cycle)
            for m, encoder in zip(mods_in, encoders):
                image_batch = mm_batch[m][0]
                if flags.cuda:
                    image_batch = image_batch.cuda()
                if flags.noisy_inputs:
                    image_batch = image_batch + torch.randn_like(image_batch)
                style_mu, style_logvar, class_mu, class_logvar = encoder(Variable(image_batch))
                mm_class_mu[m] = class_mu
                mm_style_mu[m] = style_mu
                mm_class_logvar[m] = class_logvar
                mm_style_logvar[m] = style_logvar
                labels_batch_m = mm_batch[m][1].cuda()
                labels_list.append(labels_batch_m)

            # compute the product
            poe_mu, poe_logvar = utils.poe(mm_class_mu, mm_class_logvar, prior_expert=flags.prior_expert)

            # classify embeddings
            out = clf(poe_mu)
            _, y_hat = torch.max(out, 1)
            loss = F.cross_entropy(out, labels_list[0])
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # test epoch
        accuracies = []
        for j in range(num_iterations_test):

            # do the inference step and store latents and targets
            mm_class_mu = Variable(torch.zeros(flags.num_modalities, flags.batch_size, flags.class_dim)).cuda()
            mm_style_mu = Variable(torch.zeros(flags.num_modalities, flags.batch_size, flags.style_dim)).cuda()
            mm_class_logvar = Variable(torch.zeros(flags.num_modalities, flags.batch_size, flags.class_dim)).cuda()
            mm_style_logvar = Variable(torch.zeros(flags.num_modalities, flags.batch_size, flags.style_dim)).cuda()
            labels_list = []
            mm_batch = next(test_cycle)
            for m, encoder in zip(mods_in, encoders):
                image_batch = mm_batch[m][0]
                if flags.cuda:
                    image_batch = image_batch.cuda()
                if flags.noisy_inputs:
                    image_batch = image_batch + torch.randn_like(image_batch)
                style_mu, style_logvar, class_mu, class_logvar = encoder(Variable(image_batch))
                mm_class_mu[m] = class_mu
                mm_style_mu[m] = style_mu
                mm_class_logvar[m] = class_logvar
                mm_style_logvar[m] = style_logvar
                labels_batch_m = mm_batch[m][1].cuda()
                labels_list.append(labels_batch_m)

            # compute the product
            poe_mu, poe_logvar = utils.poe(mm_class_mu, mm_class_logvar, prior_expert=flags.prior_expert)

            # classify embeddings
            out = clf(poe_mu)
            _, y_hat = torch.max(out, 1)
            acc = 100 * (y_hat == labels_list[0]).detach().cpu().numpy().mean()
            accuracies.append(acc)

        # handle early stopping
        acc = np.mean(accuracies)
        if acc < acc_prev:
            stop_counter += 1
        else:
            stop_counter = 0

        # meta update
        acc_prev = acc
        i += 1
        if verbose:
            print("Epoch %d, Accuracy=%.2f" % (i, acc))

    return clf


def evaluate_classifier(clf, dataloader, mods_in, encoders, flags, noisy_inputs=False):

    accuracies = []
    data_cycle = cycle(dataloader)

    for j in range(len(dataloader.dataset) // flags.batch_size):

        # do the inference step and store latents and targets
        mm_class_mu = Variable(torch.zeros(flags.num_modalities, flags.batch_size, flags.class_dim)).cuda()
        mm_style_mu = Variable(torch.zeros(flags.num_modalities, flags.batch_size, flags.style_dim)).cuda()
        mm_class_logvar = Variable(torch.zeros(flags.num_modalities, flags.batch_size, flags.class_dim)).cuda()
        mm_style_logvar = Variable(torch.zeros(flags.num_modalities, flags.batch_size, flags.style_dim)).cuda()
        labels_list = []
        mm_batch = next(data_cycle)
        for m, encoder in zip(mods_in, encoders):
            image_batch = mm_batch[m][0]
            if flags.cuda:
                image_batch = image_batch.cuda()
            if flags.noisy_inputs:
                image_batch = image_batch + torch.randn_like(image_batch)
            style_mu, style_logvar, class_mu, class_logvar = encoder(Variable(image_batch))
            mm_class_mu[m] = class_mu
            mm_style_mu[m] = style_mu
            mm_class_logvar[m] = class_logvar
            mm_style_logvar[m] = style_logvar
            labels_batch_m = mm_batch[m][1].cuda()
            labels_list.append(labels_batch_m)

        # compute the product
        poe_mu, poe_logvar = utils.poe(mm_class_mu, mm_class_logvar, prior_expert=flags.prior_expert)

        # classify embeddings
        out = clf(poe_mu)
        _, y_hat = torch.max(out, 1)
        # loss = F.cross_entropy(out, labels_list[0])
        acc = 100 * (y_hat == labels_list[0]).detach().cpu().numpy().mean()
        accuracies.append(acc)
    return np.mean(accuracies)


def save_reconstructions(encoders, decoders, samples, prior_expert, padding=-0.01):
    num_samples = samples[0].size()[1]

    for m in range(len(decoders)):
        encoder = encoders[m]
        decoder = decoders[m]
        fig = plt.figure(figsize=(9, 2))
        axs = fig.subplots(2, num_samples)
        fig.subplots_adjust(wspace=padding, hspace=padding*15)  # NOTE: hack to reduce whitespace between rows

        for i in range(num_samples):
            z = encoder(samples[m][:, i, :, :].unsqueeze(0))
            # reconstruction
            z_poe = utils.poe(z[2].unsqueeze(0).unsqueeze(0), z[3].unsqueeze(0).unsqueeze(0), prior_expert=prior_expert)
            out = decoder(z[0].unsqueeze(0), z_poe[0], plot=True)
            tmp = out.squeeze()
            tmp = torch.clamp(tmp, 0, 1)
            if len(tmp.size()) == 3:
                tmp = tmp.transpose(0, 2).transpose(0, 1)
                axs[1, i].imshow(tmp.detach().cpu().numpy())
            else:
                axs[1, i].imshow(tmp.detach().cpu().numpy(), cmap="gray")
            axs[1, i].set_axis_off()
            # ground truth
            tmp = samples[m][:, i, :, :].squeeze()
            if len(tmp.size()) == 3:
                axs[0, i].imshow(tmp.transpose(0, 2).transpose(0, 1).detach().cpu().numpy())
            else:
                tmp = (tmp - torch.min(tmp)) / (torch.max(tmp) - torch.min(tmp))
                axs[0, i].imshow(tmp.detach().cpu().numpy(), cmap="gray")
            axs[0, i].set_axis_off()

        plt.savefig(os.path.join(cmds.experiment_dir, "recon_final_M" + str(m) + ".png"), pad_inches=0, bbox_inches="tight",
                    edgecolor="black", facecolor="black")
        plt.close(fig)


def save_conditional(m_from, m_to, content_encoder, style_encoder, decoder, xsample, prior_expert, reparam_c=True,
                     num_conditional_samples=10, padding=-0.01):
    num_rows = num_conditional_samples + 1
    num_cols = xsample.shape[1]

    fig = plt.figure(figsize=(num_rows - 2, num_cols))
    axs = fig.subplots(num_rows, num_cols)
    # determine latent dimensions from a sample
    z = style_encoder(xsample[:, 0, :, :].unsqueeze(0))
    style_shape = z[0].unsqueeze(0).shape
    del z

    # ground truth in the 0th row and column
    for i in range(xsample.shape[1]):
        tmp = xsample[:, i, :, :].squeeze()
        if len(tmp.size()) == 3:
            tmp = tmp.transpose(0, 2).transpose(0, 1)
            axs[0, i].imshow(tmp.detach().cpu().numpy())
        else:
            tmp = (tmp - torch.min(tmp)) / (torch.max(tmp) - torch.min(tmp))
            axs[0, i].imshow(tmp.detach().cpu().numpy(), cmap="gray")

    # unconditional samples
    # vertical: content is fixed, style is variable
    for i in range(num_conditional_samples):
        z_random = Variable(torch.randn(style_shape).cuda(), requires_grad=False)
        for j in range(xsample.shape[1]):
            z = content_encoder(xsample[:, j, :, :].unsqueeze(0))
            poe_mu, poe_logvar = utils.poe(z[2].unsqueeze(0).unsqueeze(0), z[3].unsqueeze(0).unsqueeze(0),
                                     prior_expert=prior_expert)
            if reparam_c:  # Wu's model needs to sample from c
                poe_mu = utils.reparameterize(training=True, mu=poe_mu, logvar=poe_logvar)
            out = decoder(z_random, poe_mu, plot=True)
            tmp = out.squeeze()
            tmp = torch.clamp(tmp, 0, 1)
            if len(tmp.size()) == 3:
                tmp = tmp.transpose(0, 2).transpose(0, 1)
                axs[i + 1, j].imshow(tmp.detach().cpu().numpy())
            else:
                axs[i + 1, j].imshow(tmp.detach().cpu().numpy(), cmap="gray")

    # plot config
    fig.subplots_adjust(wspace=padding, hspace=padding)
    for i in range(num_conditional_samples + 1):
        for j in range(xsample.shape[1]):
            axs[i, j].set_axis_off()
    # save image
    plt.savefig(os.path.join(cmds.experiment_dir, "conditionals_final_M" + str(m_to) + "|M" + str(m_from) + ".png"), pad_inches=0,
                bbox_inches="tight", edgecolor="black", facecolor="black")
    plt.close(fig)


def save_loo_conditional(m_from, m_to, content_encoders, style_encoders, decoder, xsamples, prior_expert, reparam_c=True,
                     num_conditional_samples=10, padding=-0.01):
    num_rows = num_conditional_samples + len(content_encoders)
    num_cols = xsamples[0].shape[1]

    fig = plt.figure(figsize=(num_cols, num_rows))
    axs = fig.subplots(num_rows, num_cols)

    # we condition on the 1st few rows
    for m in range(len(m_from)):
        for i in range(xsamples[m].shape[1]):
            tmp = xsamples[m][:, i, :, :].squeeze()
            if len(tmp.size()) == 3:
                tmp = tmp.transpose(0, 2).transpose(0, 1)
                axs[m, i].imshow(tmp.detach().cpu().numpy())
            else:
                tmp = (tmp - torch.min(tmp)) / (torch.max(tmp) - torch.min(tmp))
                axs[m, i].imshow(tmp.detach().cpu().numpy(), cmap="gray")

    # conditional samples
    # vertical: content is fixed, style is variable
    for i in range(num_conditional_samples):
        for j in range(xsamples[0].shape[1]):
            class_mu_mm = Variable(torch.empty(len(m_from), 1, flags.class_dim)).cuda()
            class_logvar_mm = Variable(torch.empty(len(m_from), 1, flags.class_dim)).cuda()
            for k, (m, encoder) in enumerate(zip(m_from, content_encoders)):
                image_batch = xsamples[k][:, j, :, :]
                # compute embeddings
                _, _, class_mu, class_logvar = encoder(Variable(image_batch))
                class_mu_mm[k] = class_mu
                class_logvar_mm[k] = class_logvar

            # compute the product
            poe_mu, poe_logvar = utils.poe(class_mu_mm, class_logvar_mm, prior_expert=prior_expert)

            # compute outputs
            if reparam_c:
                poe_mu = utils.reparameterize(training=True, mu=poe_mu, logvar=poe_logvar)
            random_style = Variable(torch.randn((1, flags.style_dim)).cuda(), requires_grad=False)
            out = decoder(random_style, poe_mu, plot=True)
            tmp = out.squeeze()
            tmp = torch.clamp(tmp, 0, 1)
            if len(tmp.size()) == 3:
                tmp = tmp.transpose(0, 2).transpose(0, 1)
                axs[i + len(m_from), j].imshow(tmp.detach().cpu().numpy())
            else:
                axs[i + len(m_from), j].imshow(tmp.detach().cpu().numpy(), cmap="gray")

    # plot config
    fig.subplots_adjust(wspace=padding, hspace=padding)
    for i in range(num_conditional_samples + len(content_encoders)):
        for j in range(xsamples[0].shape[1]):
            axs[i, j].set_axis_off()
    # save image
    plt.savefig(os.path.join(cmds.experiment_dir, "conditionals_final_M" + str(m_to) + "|M" + str(m_from) + ".png"), pad_inches=0,
                bbox_inches="tight", edgecolor="black", facecolor="black")
    plt.close(fig)


def save_unconditionals(flags, content_density=None, num_unconditional_sample_rows=10, num_unconditional_sample_cols=10, padding=-0.01):
    for m in range(len(encs)):
        fig = plt.figure(figsize=(num_unconditional_sample_rows, num_unconditional_sample_cols))
        axs = fig.subplots(num_unconditional_sample_rows, num_unconditional_sample_cols)
        fig.subplots_adjust(wspace=padding, hspace=padding * 15)  # NOTE: hack to reduce whitespace between rows
        for i in range(num_unconditional_sample_rows):
            for j in range(num_unconditional_sample_cols):
                s = Variable(torch.randn([1, flags.style_dim]).cuda(), requires_grad=False)
                if content_density:
                    c = torch.Tensor(content_density.sample()[0]).cuda()
                else:
                    c = Variable(torch.randn([1, flags.class_dim]).cuda(), requires_grad=False)
                out = decs[m](s, c, plot=True)
                tmp = out.squeeze()
                tmp = torch.clamp(tmp, 0, 1)
                if len(tmp.size()) == 3:
                    tmp = tmp.transpose(0, 2).transpose(0, 1)
                    axs[i, j].imshow(tmp.detach().cpu().numpy())
                else:
                    axs[i, j].imshow(tmp.detach().cpu().numpy(), cmap="gray")
                axs[i, j].set_axis_off()
        plt.savefig(os.path.join(cmds.experiment_dir, "unconditionals_final_M" + str(m) + ".png"), pad_inches=0,
                    bbox_inches="tight", edgecolor="black", facecolor="black")
        plt.close(fig)


def calculate_joint_llik(encoders, decoders, get_data_loaders, flags, num_samples=1000, num_imp_samples=10,
                         scaling_factors=None):
    assert len(encoders) == len(decoders)
    M = flags.num_modalities
    bs = int(np.ceil(flags.batch_size / num_imp_samples))  # smaller batch size due to importance samples
    device = "cuda" if flags.cuda else "cpu"
    train, test = get_data_loaders(batch_size=bs, num_modalities=flags.num_modalities, num_workers=flags.num_workers,
                                   device=device)
    loader = cycle(test)
    lls_dataset = []  # log-likelihoods over the dataset
    for iteration in range(num_samples // bs):
        # load a mini-batch
        mm_batch = next(loader)

        # do the inference step and store latents and targets
        mm_class_mu = Variable(torch.empty(M, bs, flags.class_dim)).cuda()
        mm_style_mu = Variable(torch.empty(M, bs, flags.style_dim)).cuda()
        mm_class_logvar = Variable(torch.empty(M, bs, flags.class_dim)).cuda()
        mm_style_logvar = Variable(torch.empty(M, bs, flags.style_dim)).cuda()
        targets = []
        for m in range(M):
            encoder = encoders[m]
            image_batch = mm_batch[m][0]
            if flags.cuda:
                image_batch = image_batch.cuda()
            if flags.noisy_inputs:
                image_batch_out = torch.clone(image_batch)
                image_batch = image_batch + torch.randn_like(image_batch)
            style_mu, style_logvar, class_mu, class_logvar = encoder(Variable(image_batch))
            mm_class_mu[m] = class_mu
            mm_style_mu[m] = style_mu
            mm_class_logvar[m] = class_logvar
            mm_style_logvar[m] = style_logvar
            # NOTE: image_batch will be repeated k times in likelihood call
            if flags.noisy_inputs:
                targets.append(image_batch_out.view(bs, -1))
            else:
                targets.append(image_batch.view(bs, -1))

        # compute the product and repeat it k times
        poe_mu, poe_logvar = utils.poe(mm_class_mu, mm_class_logvar, prior_expert=flags.prior_expert)
        poe_mu_k = poe_mu.unsqueeze(1).repeat(1, num_imp_samples, 1)  # dims: BS K D
        poe_logvar_k = poe_logvar.unsqueeze(1).repeat(1, num_imp_samples, 1)

        # repeat the styles k times
        mm_style_mu_k = mm_style_mu.unsqueeze(2).repeat(1, 1, num_imp_samples, 1)  # dims: M BS K D
        mm_style_logvar_k = mm_style_logvar.unsqueeze(2).repeat(1, 1, num_imp_samples, 1)

        # reparameterize repeated style and contents
        c_k = utils.reparameterize(training=True, mu=poe_mu_k, logvar=poe_logvar_k)
        s_k = utils.reparameterize(training=True, mu=mm_style_mu_k, logvar=mm_style_logvar_k)

        # concat content and styles mus to a single representation
        tmp = mm_style_mu_k.unsqueeze(-1).transpose(0, -1).contiguous().view(bs, num_imp_samples, -1)
        mu_cat_k = torch.cat((poe_mu_k, tmp), dim=-1)  # dims: BS K (Dc + Ds * M)

        # concat content and styles logvars to a single representation
        tmp = mm_style_logvar_k.unsqueeze(-1).transpose(0, -1).contiguous().view(bs, num_imp_samples, -1)
        logvar_cat_k = torch.cat((poe_logvar_k, tmp), dim=-1)  # dims: BS K (Dc + Ds * M)

        # concat reparametrized content and styles to a single representation
        tmp = s_k.unsqueeze(-1).transpose(0, -1).contiguous().view(bs, num_imp_samples, -1)
        z_k = torch.cat((c_k, tmp), dim=-1)  # dims: BS K (Dc + Ds * M)

        # compute and save likelihoods (i.e., outputs)
        ss_lists = []
        lik_distributions = []
        for m in range(M):
            decoder = decoders[m]
            sm_k = s_k[m]  # reparametrized style representation for the m-th modality

            # compute the outputs (likelihoods) for x_m
            ss_list = decoder(sm_k.view(bs * num_imp_samples, -1), c_k.view(bs * num_imp_samples, -1))

            # reshape all sstats to (bs*k)*-1
            ss_list = [ss.view(bs * num_imp_samples, -1) for ss in ss_list]

            # collect sstats
            ss_lists.append(ss_list)

            # collect likelihood distributions
            lik_distributions.append(decoder.likelihood)

        # compute joint log-likelihood for the current batch
        ll_batch = utils.log_joint_estimate(ss_lists, targets, z_k, mu_cat_k, logvar_cat_k, lik_distributions,
                                      scaling_factors)
        lls_dataset.append(ll_batch.item())
    return np.mean(lls_dataset)


def calculate_marginal_llik(m, encoder, decoder, get_data_loaders, flags, num_samples=1000, num_imp_samples=10):
    elbo_array = []
    bs = max(2, int(np.ceil(flags.batch_size / num_imp_samples)))  # NOTE: use minimal batch size of two, s.t. we don't need to unsqueeze dimensions
    device = "cuda" if flags.cuda else "cpu"
    train, test = get_data_loaders(batch_size=bs, device=device, num_modalities=flags.num_modalities,
                                   num_workers=flags.num_workers)
    loader = cycle(test)
    for iteration in range(num_samples // bs):
        # load a mini-batch
        batch = next(loader)
        image_batch = batch[m][0]
        if flags.cuda:
            image_batch = image_batch.cuda()
        if flags.noisy_inputs:
            image_batch_out = torch.clone(image_batch)
            image_batch = image_batch + torch.randn_like(image_batch)
        # compute embeddings
        style_mu, style_logvar, class_mu, class_logvar = encoder(Variable(image_batch))

        # compute the product and repeat it k times
        poe_mu, poe_logvar = utils.poe(class_mu.unsqueeze(0), class_logvar.unsqueeze(0), prior_expert=flags.prior_expert)
        poe_mu_k = poe_mu.unsqueeze(1).repeat(1, num_imp_samples, 1)
        poe_logvar_k = poe_logvar.unsqueeze(1).repeat(1, num_imp_samples, 1)

        # repeat the style k times
        style_mu_k = style_mu.unsqueeze(1).repeat(1, num_imp_samples, 1)
        style_logvar_k = style_logvar.unsqueeze(1).repeat(1, num_imp_samples, 1)

        # reparameterize repeated style and content
        c = utils.reparameterize(training=True, mu=poe_mu_k, logvar=poe_logvar_k)
        sm = utils.reparameterize(training=True, mu=style_mu_k, logvar=style_logvar_k)

        # concat reparametrized content and style mus to a single representation
        mu_cat_k = torch.cat((poe_mu_k, style_mu_k), dim=-1)

        # concat reparametrized content and style logvars to a single representation
        logvar_cat_k = torch.cat((poe_logvar_k, style_logvar_k), dim=-1)

        # concat reparametrized content and styles to a single representation
        z_k = torch.cat((c, sm), dim=-1)

        # compute outputs (sufficient stats)
        ss_list = decoder(sm.view(bs*num_imp_samples, -1), c.view(bs*num_imp_samples, -1))

        # reshape all sstats to (bs*k)*-1
        ss_list = [ss.view(bs * num_imp_samples, -1) for ss in ss_list]

        # reshape image batch to (bs)*-1
        # NOTE: the image batch will be expanded to K samples in the log-likelihood call
        if flags.noisy_inputs:
            target = image_batch_out.view(bs, -1)
        else:
            target = image_batch.view(bs, -1)

        # compute log-likelihood
        ll = utils.log_marginal_estimate(ss_list, target, z_k, mu_cat_k, logvar_cat_k, decoder.likelihood)
        elbo_array.append(ll.item())

    return np.mean(elbo_array)


def calculate_conditional_llik(ms_in, m_out, encoders, decoder, get_data_loaders, flags, num_samples=1000,
                               num_imp_samples=10):
    assert type(ms_in) == list
    assert type(m_out) == int
    M = flags.num_modalities
    bs = int(np.ceil(flags.batch_size / num_imp_samples))  # smaller batch size due to importance samples
    device = "cuda" if flags.cuda else "cpu"
    train, test = get_data_loaders(batch_size=bs, device=device, num_modalities=flags.num_modalities,
                                   num_workers=flags.num_workers)
    loader = cycle(test)
    lls_dataset = []  # log-likelihoods over the dataset
    for iteration in range(num_samples // bs):
        # load a mini-batch
        mm_batch = next(loader)

        # do the inference step and store latents and targets
        mm_class_mu = Variable(torch.zeros(M, bs, flags.class_dim)).cuda()
        mm_style_mu = Variable(torch.zeros(M, bs, flags.style_dim)).cuda()
        mm_class_logvar = Variable(torch.zeros(M, bs, flags.class_dim)).cuda()
        mm_style_logvar = Variable(torch.zeros(M, bs, flags.style_dim)).cuda()
        target = mm_batch[m_out][0].cuda().view(bs, -1)
        for m in ms_in:
            encoder = encoders[m]
            if flags.noisy_inputs:
                style_mu, style_logvar, class_mu, class_logvar = encoder(Variable(mm_batch[m][0].cuda() + torch.randn_like(mm_batch[m][0].cuda())))
            else:
                style_mu, style_logvar, class_mu, class_logvar = encoder(Variable(mm_batch[m][0].cuda()))
            mm_class_mu[m] = class_mu
            mm_style_mu[m] = style_mu
            mm_class_logvar[m] = class_logvar
            mm_style_logvar[m] = style_logvar

        # compute the product and repeat it k times
        poe_mu, poe_logvar = utils.poe(mm_class_mu, mm_class_logvar, prior_expert=flags.prior_expert)
        poe_mu_k = poe_mu.unsqueeze(1).repeat(1, num_imp_samples, 1)  # dims: BS K D
        poe_logvar_k = poe_logvar.unsqueeze(1).repeat(1, num_imp_samples, 1)

        # repeat the styles k times
        mm_style_mu_k = mm_style_mu.unsqueeze(2).repeat(1, 1, num_imp_samples, 1)  # dims: M BS K D
        mm_style_logvar_k = mm_style_logvar.unsqueeze(2).repeat(1, 1, num_imp_samples, 1)

        # reparameterize repeated style and contents
        c_k = utils.reparameterize(training=True, mu=poe_mu_k, logvar=poe_logvar_k)

        # draw random styles
        # except, when m_out is also part of the input, take its style
        s_k = torch.randn_like(mm_style_mu_k)
        if m_out in ms_in:
            tmp = utils.reparameterize(training=True, mu=mm_style_mu_k, logvar=mm_style_logvar_k)
            s_k[m_out] = tmp[m_out]

        # concat content and styles mus to a single representation
        mu_cat_k = poe_mu_k

        # concat content and styles logvars to a single representation
        logvar_cat_k = poe_logvar_k

        # concat reparametrized content and styles to a single representation
        z_k = c_k

        # compute the outputs (likelihoods) for x_m
        sm_k = s_k[m_out]  # reparametrized/random style representation for the output modality
        ss_list = decoder(sm_k.view(bs * num_imp_samples, -1), c_k.view(bs * num_imp_samples, -1))

        # reshape all sstats to (bs*k)*-1
        ss_list = [ss.view(bs * num_imp_samples, -1) for ss in ss_list]

        # compute joint log-likelihood for the current batch
        ll_batch = utils.log_marginal_estimate(ss_list, target, z_k, mu_cat_k, logvar_cat_k, decoder.likelihood)
        lls_dataset.append(ll_batch.item())
    return np.mean(lls_dataset)


if __name__ == '__main__':
    print("-" * 80)
    print("Running post hoc analysis")
    print("-"*80)
    print("Experiment directory: " + cmds.experiment_dir)
    print("Save directory: " + cmds.experiment_dir)
    print("Networks directory: " + cmds.networks_dir)
    print("-" * 80)
    M = len(encs)         # number of modalities
    switched = False
    if flags.noisy_inputs:
        flags.noisy_inputs = False
        switched = True
    samples = utils.get_10_mm_digit_samples(test_loader, flags)
    if switched:
        flags.noisy_inputs = True
    if flags.noisy_inputs:
        print("Saving noisy images")
        for m in range(M):
            fig = plt.figure(figsize=(10, 2))
            axs = fig.subplots(2, 10)
            fig.subplots_adjust(wspace=-0.01, hspace=-0.01 * 15)  # NOTE: hack to reduce whitespace between rows
            # pure images
            for i in range(samples[m].shape[1]):
                tmp = samples[m][:, i, :, :].squeeze()
                if len(tmp.size()) == 3:
                    tmp = tmp.transpose(0, 2).transpose(0, 1)
                    axs[0, i].imshow(tmp.detach().cpu().numpy())
                else:
                    tmp = (tmp - torch.min(tmp)) / (torch.max(tmp) - torch.min(tmp))
                    axs[0, i].imshow(tmp.detach().cpu().numpy(), cmap="gray")
                    axs[0, i].set_axis_off()
            samples[m] = samples[m] + torch.randn_like(samples[m])
            # noisy images
            for i in range(samples[m].shape[1]):
                tmp = samples[m][:, i, :, :].squeeze()
                if len(tmp.size()) == 3:
                    tmp = tmp.transpose(0, 2).transpose(0, 1)
                    axs[1, i].imshow(tmp.detach().cpu().numpy())
                else:
                    tmp = (tmp - torch.min(tmp)) / (torch.max(tmp) - torch.min(tmp))
                    axs[1, i].imshow(tmp.detach().cpu().numpy(), cmap="gray")
                    axs[1, i].set_axis_off()
            plt.savefig(os.path.join(cmds.experiment_dir, "noisy_vs_pure_M" + str(m) + ".png"), pad_inches=0, bbox_inches="tight",
                        edgecolor="black", facecolor="black")
            plt.close(fig)
        print("-" * 80)

    # fit density estimator on content representations
    content_density = None
    if cmds.content_density_estimation is not None:
        print("Density Estimation...")
        if cmds.content_density_estimation == "gmm1":
            content_density = mixture.GaussianMixture(1, covariance_type="diag", max_iter=2000, verbose=2)
        elif cmds.content_density_estimation == "gmm10":
            content_density = mixture.GaussianMixture(10, covariance_type="full", max_iter=2000, verbose=2)
        elif cmds.content_density_estimation == "gmm100":
            content_density = mixture.GaussianMixture(100, covariance_type="full", max_iter=2000, verbose=2)
        else:
            raise ValueError("Invalid density option")
        content_density = content_density_estimation(content_density, encs, train_loader, flags, num_samples=5000)  # TODO: make 5K a hyperparameter
    print("-" * 80)

    # compute coherence metrics
    print("Computing coherences...")
    writer = SummaryWriter("/tmp")
    joint_coherence = np.mean(list(write_unconditional_generation_to_tensorboard(img_to_digit_clfs, decs, test_loader, writer, epoch=0, flags=flags, num_gen_samples=None, content_density=content_density)))
    print("Joint coherence: %.3f" % joint_coherence)
    # leave-one-out mappings
    for m in range(flags.num_modalities):
        conds = list(range(flags.num_modalities))
        conds.remove(m)
        conditional_coherence = np.mean(list(write_conditional_generation_to_tensorboard(conds, img_to_digit_clfs[m], encs[:m] + encs[m+1:], decs[m],
                                                                                        test_loader, writer, epoch=0, flags=flags, num_gen_samples=None,
                                                                                        reparam_c=reparam_c)))
        print("rest->M%d coherence: %.3f" % (m, conditional_coherence))
    # coherences for pairwise mappings m0 -> m1
    if flags.num_modalities > 2:
        for m_from in range(flags.num_modalities):
            for m_to in range(flags.num_modalities):
                if m_from != m_to:
                    conditional_coherence = np.mean(list(write_conditional_generation_to_tensorboard([m_from], img_to_digit_clfs[m_to], [encs[m_from]], decs[m_to],
                                                test_loader, writer, epoch=0, flags=flags, num_gen_samples=None, reparam_c=reparam_c)))
                    print("M%d->M%d coherence: %.3f" % (m_from, m_to, conditional_coherence))
    print("-" * 80)

    # compute classification performance
    if cmds.latent_classification:
        print("Latent Classification...")
        # classify c_m (i.e., each modality separately)
        for m in range(flags.num_modalities):
            clf_linear = networks.LinearClassifier(z_dim=flags.class_dim, num_classes=10).cuda()
            optimizer = optim.Adam(clf_linear.parameters(), lr=0.001)
            clf_linear = train_classifier(clf_linear, train_loader, test_loader, optimizer, mods_in=[m], encoders=[encs[m]], flags=flags,
                                          early_stopping=3, noisy_inputs=flags.noisy_inputs, num_samples_per_epoch=10000)
            accuracy = evaluate_classifier(clf_linear, test_loader, mods_in=[m], encoders=[encs[m]], flags=flags, noisy_inputs=flags.noisy_inputs)
            print("M%d Latent Linear Classification Accuracy: %.3f" % (m, accuracy))
        # classify joint representation c
        clf_linear = networks.LinearClassifier(z_dim=flags.class_dim, num_classes=10).cuda()
        optimizer = optim.Adam(clf_linear.parameters(), lr=0.001)
        clf_linear = train_classifier(clf_linear, train_loader, test_loader, optimizer, mods_in=list(range(flags.num_modalities)), encoders=encs, flags=flags,
                early_stopping=3, noisy_inputs=flags.noisy_inputs, num_samples_per_epoch=10000)
        accuracy = evaluate_classifier(clf_linear, test_loader, mods_in=list(range(flags.num_modalities)), encoders=encs, flags=flags, noisy_inputs=flags.noisy_inputs)
        print("MM Latent Linear Classification Accuracy: %.3f" % (accuracy))

    # compute "digit-wise" FID scores
    for digit in range(10):
        print("Preparing digit-wise FIDs...")
        gen_path_digitwise, test_paths_digitwise = prepare_fid_computation(test_loader, flags, num_test_samples=cmds.num_fid_samples, digit=digit)
        print("Computing digit-wise FIDs...")
        # leave-one-out FIDs
        for m in range(flags.num_modalities):
            conds = list(range(flags.num_modalities))
            conds.remove(m)
            fid = write_conditional_fid_to_tensorboard(m_in=conds, m_out=m, encoders=encs[:m] + encs[m+1:],
                                                 decoder=decs[m], mm_data=test_loader, writer=writer, epoch=0,
                                                 flags=flags, gen_path=gen_path_digitwise, test_path=test_paths_digitwise[m],
                                                 num_gen_samples=cmds.num_fid_samples, reparam_c=reparam_c, digit=digit)
            print("rest->M%d digit(%d)-wise FID: %.3f" % (m, digit, fid))
        # NOTE: uncomment below block for pairwise digit-wise FIDs (computation can take a while given many modalities)
        # # FID for pairwise mappings m0 -> m1
        # if flags.num_modalities > 2:
        #     for m_from in range(flags.num_modalities):
        #         for m_to in range(flags.num_modalities):
        #             if m_from != m_to:
        #                 fid = write_conditional_fid_to_tensorboard([m_from], m_to, [encs[m_from]], decs[m_to], test_loader, writer, epoch=0, flags=flags,
        #                                                      gen_path=gen_path_digitwise, test_path=test_paths_digitwise[m_to], num_gen_samples=cmds.num_fid_samples, reparam_c=reparam_c)
        #                 print("M%d->M%d digit(%d)-wise FID: %.3f" % (m_from, m_to, digit, fid))
        print("-" * 80)

    # compute FID scores
    print("Preparing FIDs...")
    gen_path, test_paths = prepare_fid_computation(test_loader, flags, num_test_samples=cmds.num_fid_samples)
    print("Computing FIDs...")
    # unconditional FIDs
    for m in range(flags.num_modalities):
        fid = write_unconditional_fid_to_tensorboard(m, decs[m], test_loader, writer, epoch=0, flags=flags, gen_path=gen_path,
                                                     test_path=test_paths[m], num_gen_samples=cmds.num_fid_samples, content_density=content_density)
        print("M%d FID : %.3f" % (m, fid))
    # leave-one-out FIDs
    for m in range(flags.num_modalities):
        conds = list(range(flags.num_modalities))
        conds.remove(m)
        fid = write_conditional_fid_to_tensorboard(m_in=conds, m_out=m, encoders=encs[:m] + encs[m+1:],
                                             decoder=decs[m], mm_data=test_loader, writer=writer, epoch=0,
                                             flags=flags, gen_path=gen_path, test_path=test_paths[m],
                                             num_gen_samples=cmds.num_fid_samples, reparam_c=reparam_c)
        print("rest->M%d FID: %.3f" % (m, fid))
    # NOTE: uncomment below block for pairwise FIDs (computation can take a while given many modalities)
    # # FID for pairwise mappings m0 -> m1
    # if flags.num_modalities > 2:
    #     for m_from in range(flags.num_modalities):
    #         for m_to in range(flags.num_modalities):
    #             if m_from != m_to:
    #                 fid = write_conditional_fid_to_tensorboard([m_from], m_to, [encs[m_from]], decs[m_to], test_loader, writer, epoch=0, flags=flags,
    #                                                      gen_path=gen_path, test_path=test_paths[m_to], num_gen_samples=cmds.num_fid_samples, reparam_c=reparam_c)
    #                 print("M%d->M%d FID: %.3f" % (m_from, m_to, fid))
    print("-" * 80)

    print("Saving reconstructions")
    save_reconstructions(encs, decs, samples, flags.prior_expert)
    print("-" * 80)
    print("Saving pairwise conditionals")
    for m_from in range(M):
        for m_to in range(M):
            if not m_from == m_to:
                save_conditional(m_from, m_to, encs[m_from], encs[m_from], decs[m_to], samples[m_from],
                                 flags.prior_expert, reparam_c=reparam_c)
    print("-" * 80)
    if flags.num_modalities > 2:
        conds = [_ for _ in range(M)]
        print("Saving leave-one-out conditionals")
        for m_to in range(M):
            save_loo_conditional(conds[:m_to]+conds[m_to+1:], m_to, encs[:m_to]+encs[m_to+1:],
                                 encs[:m_to]+encs[m_to+1:], decs[m_to], samples[:m_to]+samples[m_to+1:],
                                 flags.prior_expert, reparam_c=reparam_c)
        print("-" * 80)
    print("Saving unconditionals")
    save_unconditionals(flags, content_density=content_density)
    print("-" * 80)
    print("Computing log likelihooods:")
    print("Joint: " + str(calculate_joint_llik(encs, decs, gtrs.get_data_loaders, flags,
                                               num_samples=flags.batch_size, num_imp_samples=flags.num_imp_samples,
                                               scaling_factors=cmds.scaling_factors)))
    m_llik = 0
    for m in range(flags.num_modalities):
        m_llik += calculate_marginal_llik(m, encs[m], decs[m], get_data_loaders=gtrs.get_data_loaders,
                                          flags=flags, num_samples=flags.batch_size,
                                          num_imp_samples=flags.num_imp_samples)
    m_llik /= flags.num_modalities
    print("Average marginal:" + str(m_llik))
    loo_llik = 0
    for m in range(flags.num_modalities):
        conds = list(range(flags.num_modalities))
        conds.remove(m)
        loo_llik += calculate_conditional_llik(conds, m, encs, decs[m], get_data_loaders=gtrs.get_data_loaders,
                                               flags=flags, num_samples=flags.batch_size,
                                               num_imp_samples=flags.num_imp_samples)
    loo_llik /= flags.num_modalities
    print("Average LOO:" + str(loo_llik))
    c_p_llik = 0
    for m_from in range(flags.num_modalities):
        for m_to in range(flags.num_modalities):
            if not m_from == m_to:
                c_p_llik += calculate_conditional_llik([m_from], m_to, encs, decs[m_to],
                                                       get_data_loaders=gtrs.get_data_loaders,
                                                       flags=flags, num_samples=flags.batch_size,
                                                       num_imp_samples=flags.num_imp_samples)
    c_p_llik /= flags.num_modalities * (flags.num_modalities - 1)
    print("Average pairwise conditional:" + str(c_p_llik))
    print("-" * 80)
