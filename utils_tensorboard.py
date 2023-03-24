import torch
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import torchvision
from sklearn.manifold import TSNE

from utils import fig2data, poe, reparameterize, log_marginal_estimate, log_joint_estimate
from torch.autograd import Variable
from itertools import cycle
from fid_score import calculate_fid_given_paths


def write_reconstructions_to_tensorboard(encoders, decoders, samples, epoch, writer, prior_expert, padding=-0.01):
    """
    Plots reconstructions of the given multimodal samples to TensorBoard.

    Args:
        encoders: A list of encoders for all modalities.
        decoders: A list of decoders for all modalities.
        samples: A list with multimodal samples with dimensions [M][n_channels x BS x height x width] to be
        reconstructed.
        epoch: Number of the current training epoch.
        writer: TensorBoard SummaryWriter.
        prior_expert: A flag identifying whether to use a prior expert in the PoE.
        padding: Amount of space reserved between plots, by default: no space.

    Returns:
        None.
    """
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
            z_poe = poe(z[2].unsqueeze(0).unsqueeze(0), z[3].unsqueeze(0).unsqueeze(0), prior_expert=prior_expert)
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
                axs[0, i].imshow(tmp.detach().cpu().numpy(), cmap="gray")
            axs[0, i].set_axis_off()
        writer.add_image('Reconstruction/M%d' % m, fig2data(fig), epoch, dataformats="HWC")
        plt.clf(); plt.close("all")


def write_unconditional_sampling_figure_to_tensorboard(decoders, content_dims, style_dims, epoch, writer, nrows=5,
        ncols=5, padding=-0.01):
    """
    Plots unconditionally generated samples to TensorBoard.

    Args:
        decoders: A list of decoders for all modalities.
        content_dims: Number of dimensions in the shared latent space.
        style_dims: Number of dimensions in the modality-specific latent space.
        epoch: Number of the current training epoch.
        writer: TensorBoard SummaryWriter.
        nrows: Number of rows when plotting generated samples.
        ncols: Number of columns when plotting generated samples. By default, 25 samples are generated and plotted in
        5 rows and 5 columns, for each modality.
        padding: Amount of space reserved between plots, by default: no space.

    Returns:
        None.
    """
    for m in range(len(decoders)):
        decoder = decoders[m]
        fig = plt.figure(figsize=(7, 7))
        axs = fig.subplots(nrows, ncols)
        fig.subplots_adjust(wspace=padding, hspace=padding)

        for i in range(nrows):
            for j in range(ncols):
                s = Variable(torch.randn([1, style_dims]).cuda(), requires_grad=False)
                c = Variable(torch.randn([1, content_dims]).cuda(), requires_grad=False)
                out = decoder(s, c, plot=True)
                tmp = out.squeeze()
                tmp = torch.clamp(tmp, 0, 1)
                if len(tmp.size()) == 3:
                    tmp = tmp.transpose(0, 2).transpose(0, 1)
                    axs[i, j].imshow(tmp.detach().cpu().numpy())
                else:
                    axs[i, j].imshow(tmp.detach().cpu().numpy(), cmap="gray")
                axs[i, j].set_axis_off()
        writer.add_image('Random_Samples/M%d' % m, fig2data(fig), epoch, dataformats="HWC")
        plt.clf(); plt.close("all")


def write_intramodality_swapping_to_tensorboard(encoder, decoder, ysample, xsample, epoch, writer, prior_expert,
                                                num_prior_samples=0, figure_name="Swapping/intramodality",
                                                reparam_c=True, colorgrid=False, padding=-0.01):
    """
    Writes within-modality content and style swapping plots to TensorBoard. The first column/row shows given samples
    from the respective modality; the remaining images are the outputs of the decoder. Modality-specific factors are
    taken from the respective sample in the first column and remain fixed along the x-axis. Vice versa, shared factors
    are taken from the respective sample in the first row and remain fixed along the y-axis.

    Args:
        encoder: Encoder for the modality, for which the swapping needs to be performed.
        decoder: Decoder for the modality, for which the swapping needs to be performed.
        ysample: Samples from which the content is taken.
        xsample: Samples from which the style is taken.
        epoch: Number of the current training epoch.
        writer: TensorBoard SummaryWriter.
        prior_expert: A flag identifying whether to use a prior expert in the PoE.
        num_prior_samples: Number of samples with randomly sampled style and content.
        figure_name: Figure name.
        reparam_c: Flag identifying whether to use the reparameterization trick.
        colorgrid: Flag identifying whether space between plots needs to be colored.
        padding: Amount of space reserved between plots, by default: no space.

    Returns:
        None.
    """
    offset = num_prior_samples + 1  # plus ground truth
    num_rows = ysample.shape[1] + offset
    num_cols = xsample.shape[1] + offset

    fig = plt.figure(figsize=(10, 10))
    axs = fig.subplots(num_rows, num_cols)
    # determine number of latent dimensions from a sample
    z = encoder(xsample[:, 0, :, :].unsqueeze(0))
    content_shape = z[2].unsqueeze(0).shape
    style_shape = z[0].unsqueeze(0).shape
    del z

    # given samples in the 0th row and column
    for i in range(xsample.shape[1]):
        tmp = xsample[:, i, :, :].squeeze()
        if len(tmp.size()) == 3:
            tmp = tmp.transpose(0, 2).transpose(0, 1)
            axs[0, i + offset].imshow(tmp.detach().cpu().numpy())
        else:
            axs[0, i+offset].imshow(tmp.detach().cpu().numpy(), cmap="gray")
    for i in range(ysample.shape[1]):
        tmp = ysample[:, i, :, :].squeeze()
        if len(tmp.size()) == 3:
            tmp = tmp.transpose(0, 2).transpose(0, 1)
            axs[i + offset, 0].imshow(tmp.detach().cpu().numpy())
        else:
            axs[i + offset, 0].imshow(tmp.detach().cpu().numpy(), cmap="gray")

    # horizontal: style is fixed, content is variable
    for j in range(num_prior_samples):
        z_random = Variable(torch.randn(content_shape).cuda(), requires_grad=False)
        for i in range(ysample.shape[1]):
            z = encoder(ysample[:, i, :, :].unsqueeze(0))
            out = decoder(z[0].unsqueeze(0), z_random, plot=True)
            tmp = out.squeeze()
            tmp = torch.clamp(tmp, 0, 1)
            if len(tmp.size()) == 3:
                tmp = tmp.transpose(0, 2).transpose(0, 1)
                axs[i+offset, j+1].imshow(tmp.detach().cpu().numpy())
            else:
                axs[i + offset, j + 1].imshow(tmp.detach().cpu().numpy(), cmap="gray")
    # vertical: content is fixed, style is variable
    for i in range(num_prior_samples):
        z_random = Variable(torch.randn(style_shape).cuda(), requires_grad=False)
        for j in range(xsample.shape[1]):
            z = encoder(xsample[:, j, :, :].unsqueeze(0))
            poe_mu, poe_logvar = poe(z[2].unsqueeze(0).unsqueeze(0), z[3].unsqueeze(0).unsqueeze(0),
                                     prior_expert=prior_expert)
            if reparam_c:
                poe_mu = reparameterize(training=True, mu=poe_mu, logvar=poe_logvar)
            out = decoder(z_random, poe_mu, plot=True)
            tmp = out.squeeze()
            tmp = torch.clamp(tmp, 0, 1)
            if len(tmp.size()) == 3:
                tmp = tmp.transpose(0, 2).transpose(0, 1)
                axs[i+1, j+offset].imshow(tmp.detach().cpu().numpy())
            else:
                axs[i + 1, j + offset].imshow(tmp.detach().cpu().numpy(), cmap="gray")

    # style transfer in the rest of the matrix
    for i in range(ysample.shape[1]):
        for j in range(xsample.shape[1]):
            zi = encoder(ysample[:, i, :, :].unsqueeze(0))
            zj = encoder(xsample[:, j, :, :].unsqueeze(0))
            poe_mu, poe_logvar = poe(zj[2].unsqueeze(0).unsqueeze(0), zj[3].unsqueeze(0).unsqueeze(0), prior_expert=prior_expert)
            if reparam_c:
                poe_mu = reparameterize(training=True, mu=poe_mu, logvar=poe_logvar)
            out = decoder(zi[0].unsqueeze(0), poe_mu, plot=True)
            tmp = out.squeeze()
            tmp = torch.clamp(tmp, 0, 1)
            if len(tmp.size()) == 3:
                tmp = tmp.transpose(0, 2).transpose(0, 1)
                axs[i+offset, j+offset].imshow(tmp.detach().cpu().numpy())
            else:
                axs[i + offset, j + offset].imshow(tmp.detach().cpu().numpy(), cmap="gray")

    # plot config
    fig.subplots_adjust(wspace=padding, hspace=padding)
    for i in range(ysample.shape[1] + num_prior_samples + 1):
        for j in range(xsample.shape[1] + num_prior_samples + 1):
            axs[i, j].set_axis_off()

    if colorgrid:
        # add frame around style transfer plots
        outergs = gridspec.GridSpec(num_rows, num_cols)
        outerax = fig.add_subplot(outergs[offset:num_rows, offset:num_cols])
        rectangle = patches.Rectangle([0, 0], width=1, height=1, transform=fig.transFigure,
                                      facecolor="crimson", zorder=-1, alpha=0.5, edgecolor=None)
        outerax.set_zorder(-1)
        outerax.add_artist(rectangle)
        outerax.tick_params(axis='both', which='both', bottom=0, left=0, labelbottom=0, labelleft=0)
        outerax.set_position(np.array(outerax.get_position().bounds) + np.array([-1, -1, 2, 2]) * 0.0025)
        outerax.set_axis_off()

        # add frame around unconditional xsamples
        outergs = gridspec.GridSpec(num_rows, num_cols)
        outerax = fig.add_subplot(outergs[1:offset, offset:num_cols])
        rectangle = patches.Rectangle([0, 0], width=1, height=1, transform=fig.transFigure,
                                      facecolor="blue", zorder=-1, alpha=0.5, edgecolor=None)
        outerax.set_zorder(-1)
        outerax.add_artist(rectangle)
        outerax.tick_params(axis='both', which='both', bottom=0, left=0, labelbottom=0, labelleft=0)
        outerax.set_position(np.array(outerax.get_position().bounds) + np.array([-1, -1, 2, 2]) * 0.0025)
        outerax.set_axis_off()
        # add frame around unconditional ysamples
        outergs = gridspec.GridSpec(num_rows, num_cols)
        outerax = fig.add_subplot(outergs[offset:num_rows, 1:offset])
        rectangle = patches.Rectangle([0, 0], width=1, height=1, transform=fig.transFigure,
                                      facecolor="blue", zorder=-1, alpha=0.5, edgecolor=None)
        outerax.set_zorder(-1)
        outerax.add_artist(rectangle)
        outerax.tick_params(axis='both', which='both', bottom=0, left=0, labelbottom=0, labelleft=0)
        outerax.set_position(np.array(outerax.get_position().bounds) + np.array([-1, -1, 2, 2]) * 0.0025)
        outerax.set_axis_off()

    # save image
    writer.add_image(figure_name, fig2data(fig), epoch, dataformats="HWC")
    plt.clf(); plt.close("all")


def write_crossmodality_swapping_to_tensorboard(content_encoder, style_encoder, style_decoder, ysample, xsample, epoch,
                                                writer, prior_expert, num_prior_samples=0,
                                                figure_name="Swapping/cross-modality", reparam_c=True, colorgrid=False,
                                                padding=-0.01):
    """
    Writes cross-modality content and style swapping plots to TensorBoard. The first column/row shows given samples
    from two modalities; the remaining images are the outputs of the decoder. Modality-specific factors are taken from
    the samples of the modality in the first column and remain fixed along the x-axis. Vice versa, shared factors are
    taken from the samples of another modality in the first row and remain fixed along the y-axis.

    Args:
        content_encoder: Encoder for the modality that provides shared factors.
        style_encoder: Encoder for the modality that provides modality-specific factors.
        style_decoder: Decoder for the modality that provides modality-specific factors
        ysample: Samples from which the content is taken.
        xsample: Samples from which the style is taken.
        epoch: Number of the current training epoch.
        writer: TensorBoard SummaryWriter.
        prior_expert: Flag identifying whether to use a prior expert in the PoE.
        num_prior_samples: Number of samples with randomly sampled style and content.
        figure_name: Figure name.
        reparam_c: Flag identifying whether to use the reparameterization trick.
        colorgrid: Flag identifying whether space between plots needs to be colored.
        padding: Amount of space reserved between plots, by default: no space.

    Returns:
        None.
    """
    offset = num_prior_samples + 1  # plus ground truth
    num_rows = ysample.shape[1] + offset
    num_cols = xsample.shape[1] + 1  # no unconditional samples from fixed style

    fig = plt.figure(figsize=(10, 10))
    axs = fig.subplots(num_rows, num_cols)
    # determine latent dimensions from a sample
    z = style_encoder(ysample[:, 0, :, :].unsqueeze(0))
    style_shape = z[0].unsqueeze(0).shape
    del z

    # ground truth in the 0th row and column
    for i in range(xsample.shape[1]):
        tmp = xsample[:, i, :, :].squeeze()
        if len(tmp.size()) == 3:
            tmp = tmp.transpose(0, 2).transpose(0, 1)
            axs[0, i+1].imshow(tmp.detach().cpu().numpy())
        else:
            axs[0, i + 1].imshow(tmp.detach().cpu().numpy(), cmap="gray")
    for i in range(ysample.shape[1]):
        tmp = ysample[:, i, :, :].squeeze()
        if len(tmp.size()) == 3:
            tmp = tmp.transpose(0, 2).transpose(0, 1)
            axs[i+offset, 0].imshow(tmp.detach().cpu().numpy())
        else:
            axs[i + offset, 0].imshow(tmp.detach().cpu().numpy(), cmap="gray")

    # unconditional samples
    # vertical: content is fixed, style is variable
    for i in range(num_prior_samples):
        z_random = Variable(torch.randn(style_shape).cuda(), requires_grad=False)
        for j in range(xsample.shape[1]):
            z = content_encoder(xsample[:, j, :, :].unsqueeze(0))
            poe_mu, poe_logvar = poe(z[2].unsqueeze(0).unsqueeze(0), z[3].unsqueeze(0).unsqueeze(0), prior_expert=prior_expert)
            if reparam_c:
                poe_mu = reparameterize(training=True, mu=poe_mu, logvar=poe_logvar)
            out = style_decoder(z_random, poe_mu, plot=True)
            tmp = out.squeeze()
            tmp = torch.clamp(tmp, 0, 1)
            if len(tmp.size()) == 3:
                tmp = tmp.transpose(0, 2).transpose(0, 1)
                axs[i+1, j+1].imshow(tmp.detach().cpu().numpy())
            else:
                axs[i + 1, j + 1].imshow(tmp.detach().cpu().numpy(), cmap="gray")

    # style transfer in the rest of the matrix
    for i in range(ysample.shape[1]):
        for j in range(xsample.shape[1]):
            zi = style_encoder(ysample[:, i, :, :].unsqueeze(0))
            zj = content_encoder(xsample[:, j, :, :].unsqueeze(0))
            poe_mu, poe_logvar = poe(zj[2].unsqueeze(0).unsqueeze(0), zj[3].unsqueeze(0).unsqueeze(0), prior_expert=prior_expert)
            if reparam_c:
                poe_mu = reparameterize(training=True, mu=poe_mu, logvar=poe_logvar)
            out = style_decoder(zi[0].unsqueeze(0), poe_mu, plot=True)
            tmp = out.squeeze()
            tmp = torch.clamp(tmp, 0, 1)
            if len(tmp.size()) == 3:
                tmp = tmp.transpose(0, 2).transpose(0, 1)
                axs[i+offset, j+1].imshow(tmp.detach().cpu().numpy())
            else:
                axs[i + offset, j + 1].imshow(tmp.detach().cpu().numpy(), cmap="gray")

    # plot config
    fig.subplots_adjust(wspace=padding, hspace=padding)
    for i in range(ysample.shape[1] + num_prior_samples + 1):
        for j in range(xsample.shape[1] + 1):
            axs[i, j].set_axis_off()

    # add frame around style transfer plots
    if colorgrid:
        outergs = gridspec.GridSpec(num_rows, num_cols)
        outerax = fig.add_subplot(outergs[offset:num_rows, 1:num_cols])
        rectangle = patches.Rectangle([0, 0], width=1, height=1, transform=fig.transFigure,
                                      facecolor="crimson", zorder=-1, alpha=0.5, edgecolor=None)
        outerax.set_zorder(-1)
        outerax.add_artist(rectangle)
        outerax.tick_params(axis='both', which='both', bottom=0, left=0, labelbottom=0, labelleft=0)
        outerax.set_position(np.array(outerax.get_position().bounds) + np.array([-1, -1, 2, 2]) * 0.0025)
        outerax.set_axis_off()

        # add frame around unconditional samples
        outergs = gridspec.GridSpec(num_rows, num_cols)
        outerax = fig.add_subplot(outergs[1:offset, 1:num_cols])
        rectangle = patches.Rectangle([0, 0], width=1, height=1, transform=fig.transFigure,
                                      facecolor="blue", zorder=-1, alpha=0.5, edgecolor=None)
        outerax.set_zorder(-1)
        outerax.add_artist(rectangle)
        outerax.tick_params(axis='both', which='both', bottom=0, left=0, labelbottom=0, labelleft=0)
        outerax.set_position(np.array(outerax.get_position().bounds) + np.array([-1, -1, 2, 2]) * 0.0025)
        outerax.set_axis_off()

    # save image
    writer.add_image(figure_name, fig2data(fig), epoch, dataformats="HWC")
    plt.clf(); plt.close("all")


def write_conditional_generation_samples_to_tensorboard(content_encoder, style_encoder, decoder, xsample, epoch,
                                                        writer, prior_expert, num_samples=0,
                                                        figure_name="Conditional Generation", reparam_c=True,
                                                        padding=-0.01):
    """
    Writes a plot with samples from one modality generated conditionally on another modality to TensorBoard.

    Args:
        content_encoder: Encoder for the modality that we condition on.
        style_encoder: Encoder for the modality that is generated.
        decoder: Decoder for the modality that is generated
        xsample: Samples from the modality that we condition on.
        epoch: Number of the current training epoch.
        writer: TensorBoard SummaryWriter.
        prior_expert: A flag identifying whether to use a prior expert in the PoE.
        num_samples: Number of conditionally generated samples.
        figure_name:  Figure name.
        reparam_c: Flag identifying whether to use the reparameterization trick.
        padding: Amount of space reserved between plots, by default: no space.

    Returns:
        None.
    """
    offset = num_samples + 1  # plus ground truth
    num_rows = offset
    num_cols = xsample.shape[1] + 1  # no unconditional samples from fixed style

    fig = plt.figure(figsize=(10, num_samples))
    axs = fig.subplots(num_rows, num_cols)
    # determine number of latent dimensions from a sample
    z = style_encoder(xsample[:, 0, :, :].unsqueeze(0))
    style_shape = z[0].unsqueeze(0).shape
    del z

    # ground truth in the 0th row
    for i in range(xsample.shape[1]):
        tmp = xsample[:, i, :, :].squeeze()
        if len(tmp.size()) == 3:
            tmp = tmp.transpose(0, 2).transpose(0, 1)
            axs[0, i+1].imshow(tmp.detach().cpu().numpy())
        else:
            axs[0, i + 1].imshow(tmp.detach().cpu().numpy(), cmap="gray")

    # conditional samples
    for i in range(num_samples):
        z_random = Variable(torch.randn(style_shape).cuda(), requires_grad=False)
        for j in range(xsample.shape[1]):
            z = content_encoder(xsample[:, j, :, :].unsqueeze(0))
            poe_mu, poe_logvar = poe(z[2].unsqueeze(0).unsqueeze(0), z[3].unsqueeze(0).unsqueeze(0),
                                     prior_expert=prior_expert)
            if reparam_c:
                poe_mu = reparameterize(training=True, mu=poe_mu, logvar=poe_logvar)
            out = decoder(z_random, poe_mu, plot=True)
            tmp = out.squeeze()
            tmp = torch.clamp(tmp, 0, 1)
            if len(tmp.size()) == 3:
                tmp = tmp.transpose(0, 2).transpose(0, 1)
                axs[i+1, j+1].imshow(tmp.detach().cpu().numpy())
            else:
                axs[i + 1, j + 1].imshow(tmp.detach().cpu().numpy(), cmap="gray")

    # plot config
    fig.subplots_adjust(wspace=padding, hspace=padding)
    for i in range(num_samples + 1):
        for j in range(xsample.shape[1] + 1):
            axs[i, j].set_axis_off()

    # save image
    writer.add_image(figure_name, fig2data(fig), epoch, dataformats="HWC")
    plt.clf(); plt.close("all")


def write_marginal_loglikelihood_to_tensorboard(m, encoder, decoder, get_data_loaders, writer, epoch, flags, label="",
                                                num_samples=1000, num_imp_samples=10):
    """
    Writes marginal log-likelihood (on the test set) for the given modality to TensorBoard.

    Args:
        m: Index of the modality.
        encoder: Encoder for the modality.
        decoder: Decoder for the modality.
        get_data_loaders: Method that returns data loaders.
        writer: TensorBoard SummaryWriter.
        epoch: Number of the current training epoch.
        flags: argparse.Namespace with input arguments.
        label: Plot label.
        num_samples: Number of samples to use in the evaluation of the log-likelihood.
        num_imp_samples: Number of importance samples.

    Returns:
        Test set marginal log-likelihood for the given modality.
    """
    elbo_array = []
    bs = max(2, int(np.ceil(flags.batch_size / num_imp_samples)))
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
        poe_mu, poe_logvar = poe(class_mu.unsqueeze(0), class_logvar.unsqueeze(0), prior_expert=flags.prior_expert)
        poe_mu_k = poe_mu.unsqueeze(1).repeat(1, num_imp_samples, 1)
        poe_logvar_k = poe_logvar.unsqueeze(1).repeat(1, num_imp_samples, 1)

        # repeat the style k times
        style_mu_k = style_mu.unsqueeze(1).repeat(1, num_imp_samples, 1)
        style_logvar_k = style_logvar.unsqueeze(1).repeat(1, num_imp_samples, 1)

        # reparameterize repeated style and content
        c = reparameterize(training=True, mu=poe_mu_k, logvar=poe_logvar_k)
        sm = reparameterize(training=True, mu=style_mu_k, logvar=style_logvar_k)

        # concat reparameterized content and style mus to a single representation
        mu_cat_k = torch.cat((poe_mu_k, style_mu_k), dim=-1)

        # concat reparameterized content and style logvars to a single representation
        logvar_cat_k = torch.cat((poe_logvar_k, style_logvar_k), dim=-1)

        # concat reparameterized content and styles to a single representation
        z_k = torch.cat((c, sm), dim=-1)

        # compute outputs (sufficient stats)
        ss_list = decoder(sm.view(bs*num_imp_samples, -1), c.view(bs*num_imp_samples, -1))

        # reshape all sstats to (bs*k)*-1
        ss_list = [ss.view(bs * num_imp_samples, -1) for ss in ss_list]

        # reshape image batch to (bs)*-1;
        # NOTE: the image batch will be expanded to K samples in the log-likelihood call
        if flags.noisy_inputs:
            target = image_batch_out.view(bs, -1)
        else:
            target = image_batch.view(bs, -1)

        # compute log-likelihood
        ll = log_marginal_estimate(ss_list, target, z_k, mu_cat_k, logvar_cat_k, decoder.likelihood)
        elbo_array.append(ll.item())

    writer.add_scalars("test/MM/LogLik_unweighted", {
        "%s (IS)" % label: np.mean(elbo_array),
    }, epoch)
    return np.mean(elbo_array)


def write_joint_loglikelihood_to_tensorboard(encoders, decoders, get_data_loaders, writer, epoch, flags, label="",
                                             num_samples=1000, num_imp_samples=10, scaling_factors=None):
    """
    Writes joint log-likelihood (on the test set) to TensorBoard.

    Args:
        encoders: List of encoders for all modalities.
        decoders: List of decoders for all modalities.
        get_data_loaders: Method that returns data loaders.
        writer: TensorBoard SummaryWriter.
        epoch: Number of the current training epoch.
        flags: argparse.Namespace with input arguments.
        label: Plot label.
        num_samples: Number of samples to use in the evaluation of the log-likelihood.
        num_imp_samples: Number of importance samples.
        scaling_factors: Scaling factors for individual modalities. By default, all modalities are weighted equally.

    Returns:
        Test set joint log-likelihood.
    """
    assert len(encoders) == len(decoders)
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
            # NOTE: the image batch will be expanded to K samples in the log-likelihood call
            if flags.noisy_inputs:
                targets.append(image_batch_out.view(bs, -1))
            else:
                targets.append(image_batch.view(bs, -1))

        # compute the product and repeat it k times
        poe_mu, poe_logvar = poe(mm_class_mu, mm_class_logvar, prior_expert=flags.prior_expert)
        poe_mu_k = poe_mu.unsqueeze(1).repeat(1, num_imp_samples, 1)  # dims: BS K D
        poe_logvar_k = poe_logvar.unsqueeze(1).repeat(1, num_imp_samples, 1)

        # repeat the styles k times
        mm_style_mu_k = mm_style_mu.unsqueeze(2).repeat(1, 1, num_imp_samples, 1)  # dims: M BS K D
        mm_style_logvar_k = mm_style_logvar.unsqueeze(2).repeat(1, 1, num_imp_samples, 1)

        # reparameterize repeated style and contents
        c_k = reparameterize(training=True, mu=poe_mu_k, logvar=poe_logvar_k)
        s_k = reparameterize(training=True, mu=mm_style_mu_k, logvar=mm_style_logvar_k)

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
            ss_list = decoder(sm_k.view(bs*num_imp_samples, -1), c_k.view(bs*num_imp_samples, -1))

            # reshape all sstats to (bs*k)*-1
            ss_list = [ss.view(bs * num_imp_samples, -1) for ss in ss_list]

            # collect sstats
            ss_lists.append(ss_list)

            # collect likelihood distributions
            lik_distributions.append(decoder.likelihood)

        # compute joint log-likelihood for the current batch
        ll_batch = log_joint_estimate(ss_lists, targets, z_k, mu_cat_k, logvar_cat_k, lik_distributions, scaling_factors)
        lls_dataset.append(ll_batch.item())

    writer.add_scalars("test/MM/LogLik_unweighted", {
        "%s (IS)" % label: np.mean(lls_dataset),
    }, epoch)
    return np.mean(lls_dataset)


def write_conditional_loglikelihood_to_tensorboard(ms_in, m_out, encoders, decoder, get_data_loaders, writer, epoch,
                                                   flags, label="", num_samples=1000, num_imp_samples=10):
    """
    Writes conditional log-likelihood (on the test set) to TensorBoard.

    Args:
        ms_in: Indices of modalities to condition on.
        m_out: Index of the modality the likelihood of which needs to be calculated.
        encoders: List of encoders for modalities that we condition on.
        decoder: Decoder for the modality the likelihood of which needs to be calculated.
        get_data_loaders: Method that returns data loaders.
        writer: TensorBoard SummaryWriter.
        epoch: Number of the current training epoch.
        flags: argparse.Namespace with input arguments.
        label: Plot label.
        num_samples: Number of samples to use in the evaluation of the log-likelihood.
        num_imp_samples: Number of importance samples.

    Returns:
        Test set conditional log-likelihood.
    """
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
                style_mu, style_logvar, class_mu, class_logvar = \
                    encoder(Variable(mm_batch[m][0].cuda() + torch.randn_like(mm_batch[m][0].cuda())))
            else:
                style_mu, style_logvar, class_mu, class_logvar = encoder(Variable(mm_batch[m][0].cuda()))
            mm_class_mu[m] = class_mu
            mm_style_mu[m] = style_mu
            mm_class_logvar[m] = class_logvar
            mm_style_logvar[m] = style_logvar

        # compute the product and repeat it k times
        poe_mu, poe_logvar = poe(mm_class_mu, mm_class_logvar, prior_expert=flags.prior_expert)
        poe_mu_k = poe_mu.unsqueeze(1).repeat(1, num_imp_samples, 1)  # dims: BS K D
        poe_logvar_k = poe_logvar.unsqueeze(1).repeat(1, num_imp_samples, 1)

        # repeat the styles k times
        mm_style_mu_k = mm_style_mu.unsqueeze(2).repeat(1, 1, num_imp_samples, 1)  # dims: M BS K D
        mm_style_logvar_k = mm_style_logvar.unsqueeze(2).repeat(1, 1, num_imp_samples, 1)

        # reparameterize repeated style and contents
        c_k = reparameterize(training=True, mu=poe_mu_k, logvar=poe_logvar_k)

        # draw random styles
        # except, when m_out is also part of the input, take its style
        s_k = torch.randn_like(mm_style_mu_k)
        if m_out in ms_in:
            tmp = reparameterize(training=True, mu=mm_style_mu_k, logvar=mm_style_logvar_k)
            s_k[m_out] = tmp[m_out]

        # concat content and styles mus to a single representation
        mu_cat_k = poe_mu_k

        # concat content and styles logvars to a single representation
        logvar_cat_k = poe_logvar_k

        # concat reparametrized content and styles to a single representation
        z_k = c_k

        # compute the outputs (likelihoods) for x_m
        sm_k = s_k[m_out]  # reparameterized/random style representation for the output modality
        ss_list = decoder(sm_k.view(bs*num_imp_samples, -1), c_k.view(bs*num_imp_samples, -1))

        # reshape all sstats to (bs*k)*-1
        ss_list = [ss.view(bs * num_imp_samples, -1) for ss in ss_list]

        # compute joint log-likelihood for the current batch
        ll_batch = log_marginal_estimate(ss_list, target, z_k, mu_cat_k, logvar_cat_k, decoder.likelihood)
        lls_dataset.append(ll_batch.item())

    writer.add_scalars("test/MM/LogLik_unweighted", {
        "%s (IS)" % label: np.mean(lls_dataset),
    }, epoch)
    return np.mean(lls_dataset)


def write_conditional_generation_to_tensorboard(m_in, classifier, encoders, decoder, data, writer, epoch, flags,
        label="", num_gen_samples=None, label_ix=None, reparam_c=True, label_suffix=""):
    """
    Writes conditional generative coherence results to TensorBoard.

    Args:
        m_in: Indices of modalities to condition on.
        classifier: Classifier for the modality that needs to be generated.
        encoders: List of encoders for modalities that we condition on.
        decoder: Decoder for the modality that needs to be generated.
        data: DataLoader for the test set.
        writer: TensorBoard SummaryWriter.
        epoch: Number of the current training epoch.
        flags: argparse.Namespace with input arguments.
        label: Plot label.
        num_gen_samples: Number of samples to generate.
        label_ix: Index of the modality labels to use in the evaluation.
        reparam_c: Flag identifying whether to use the reparameterization trick.

    Returns:
        None.
    """
    assert len(m_in) == len(encoders)
    assert len(m_in) > 0

    if num_gen_samples is not None:
        n_batches = num_gen_samples // flags.batch_size
    else:
        n_batches = len(data)
    loader = cycle(data)
    for iteration in range(n_batches):
        # load a mini-batch
        batch = next(loader)
        image_batches = []
        labels_batches = []
        for m in range(flags.num_modalities):
            image_batch_m = batch[m][0]
            labels_batch_m = batch[m][1]
            if flags.noisy_inputs:
                image_batch_m = image_batch_m + torch.randn_like(image_batch_m)
            if flags.cuda:
                image_batch_m = image_batch_m.cuda()
                labels_batch_m = labels_batch_m.cuda()
            image_batches.append(image_batch_m)
            labels_batches.append(labels_batch_m)

        class_mu_mm = Variable(torch.empty(len(m_in), flags.batch_size, flags.class_dim)).cuda()
        class_logvar_mm = Variable(torch.empty(len(m_in), flags.batch_size, flags.class_dim)).cuda()
        for j, (m, encoder) in enumerate(zip(m_in, encoders)):
            image_batch = image_batches[m]
            # compute embeddings
            _, _, class_mu, class_logvar = encoder(Variable(image_batch))
            class_mu_mm[j] = class_mu
            class_logvar_mm[j] = class_logvar

        # compute the product
        poe_mu, poe_logvar = poe(class_mu_mm, class_logvar_mm, prior_expert=flags.prior_expert)

        # compute outputs
        if reparam_c:
            poe_mu = reparameterize(training=True, mu=poe_mu, logvar=poe_logvar)
        random_style = Variable(torch.randn(flags.batch_size, flags.style_dim).cuda(), requires_grad=False)
        out = decoder(random_style, poe_mu, plot=True)
        out = out

        # classify outputs
        out = out.clamp(0., 1.)
        _, y_hat = torch.max(classifier(out), 1)

        # write classification performance to tensorboard
        if label_ix is None:
            labels = labels_batches[0]
        else:
            labels = labels_batches[0][:, label_ix]
        accuracy = (labels == y_hat).to(torch.float32).detach().cpu().numpy().mean() * 100.
        writer.add_scalars("test/OutputClassification%s" % label_suffix, {
            "%s" % label: accuracy,
        }, epoch)
        yield accuracy


def write_unconditional_generation_to_tensorboard(classifiers, decoders, data, writer, epoch, flags, label="",
                                                  num_gen_samples=1000, content_density=None):
    """
     Writes unconditional generative coherence results to TensorBoard.

    Args:
        classifiers: List of classifiers for all modalities.
        decoders: List of decoders for all modalities.
        data: DataLoader for the test set.
        writer: TensorBoard SummaryWriter.
        epoch: Number of the current training epoch.
        flags: argparse.Namespace with input arguments.
        label: Plot label.
        num_gen_samples: Number of samples to generate.

    Returns:
        None.
    """
    assert len(decoders) == len(classifiers)

    if num_gen_samples is not None:
        n_batches = num_gen_samples // flags.batch_size
    else:
        n_batches = len(data)
    for iteration in range(n_batches):
        # compute outputs and predictions on them
        random_style = Variable(torch.randn(flags.batch_size, flags.style_dim).cuda(), requires_grad=False)
        if content_density is None:
            random_class = Variable(torch.randn(flags.batch_size, flags.class_dim).cuda(), requires_grad=False)
        else:
            random_class = torch.Tensor(content_density.sample(flags.batch_size)[0]).cuda()
        y_hats = torch.zeros(flags.batch_size, len(decoders))
        for m, (decoder, classifier) in enumerate(zip(decoders, classifiers)):
            out = decoder(random_style, random_class, plot=True)
            out = out.clamp(0., 1.)
            _, y_hat = torch.max(classifier(out), 1)
            y_hats[:, m] = y_hat
        # targets should all be the same.
        targets = y_hats[:, 0].clone().detach().unsqueeze(1).expand(flags.batch_size, len(decoders))
        accuracy = (targets == y_hats).all(dim=1).to(torch.float32).mean() * 100.

        # write generative coherence to tensorboard
        writer.add_scalars("test/OutputClassification", {
            "%s" % label: accuracy,
        }, epoch)
        yield accuracy


def write_unconditional_fid_to_tensorboard(m_out, decoder, mm_data, writer, epoch, flags, label="",
        gen_path="tmp/unconditional_generation", test_path="tmp/test_images", num_gen_samples=1000, verbose=False, content_density=None):

    # clean all files from the previous conditional generation
    for f in glob.glob("%s/*" % gen_path):
        os.remove(f)

    # generate and save images
    j = 0
    n_batches = num_gen_samples // flags.batch_size
    for _ in range(n_batches):
        s = Variable(torch.randn([flags.batch_size, flags.style_dim]).cuda(), requires_grad=False)
        if content_density is not None:
            c = torch.Tensor(content_density.sample(flags.batch_size)[0]).cuda()
        else:
            c = Variable(torch.randn([flags.batch_size, flags.class_dim]).cuda(), requires_grad=False)
        out = decoder(s, c, plot=True)
        for i in range(out.size(0)):
            torchvision.utils.save_image(out[i, :, :, :], '{}/{}.png'.format(gen_path, j))
            j += 1

    # compute FID and write it to tensorboard
    try:  # TODO: remove this try-catch block, once it's figured out what causes fid to break
        fid = calculate_fid_given_paths([gen_path, test_path], batch_size=64, dims=2048, cuda=True, verbose=verbose)
    except Exception:
        fid = np.nan
        print("[WARN] manual handling of error that occurred during unconditional fid computation")
    writer.add_scalars("test/FID/unconditional", {
        "%s" % label: fid,
    }, epoch)
    return fid


def write_conditional_fid_to_tensorboard(m_in, m_out, encoders, decoder, mm_data, writer, epoch, flags, label="",
        gen_path="tmp/conditional_generation", test_path="tmp/test_images", num_gen_samples=None, verbose=False, reparam_c=True, digit=None):

    # clean all files from the previous conditional generation
    for f in glob.glob("%s/*" % gen_path):
        os.remove(f)

    k = 0
    loader = cycle(mm_data)
    if num_gen_samples is None:
        num_gen_samples = len(mm_data) * flags.batch_size
    cnt = 0
    while cnt < num_gen_samples:
        # load a mini-batch
        batch = next(loader)
        image_batches = []
        labels_batches = []
        for m in range(flags.num_modalities):
            image_batch_m = batch[m][0]
            label_batch_m = batch[m][1]
            if flags.noisy_inputs:
                image_batch_m = image_batch_m + torch.randn_like(image_batch_m)
            if digit is not None:
                ix = label_batch_m == digit
                image_batch_m = image_batch_m[ix]
                label_batch_m = label_batch_m[ix]
            if flags.cuda:
                image_batch_m = image_batch_m.cuda()
                label_batch_m = label_batch_m.cuda()
            image_batches.append(image_batch_m)
            labels_batches.append(label_batch_m)

        class_mu_mm = Variable(torch.zeros(len(m_in), len(image_batches[0]), flags.class_dim)).cuda()
        class_logvar_mm = Variable(torch.ones(len(m_in), len(image_batches[0]), flags.class_dim)).cuda()
        for j, (m, encoder) in enumerate(zip(m_in, encoders)):
            image_batch = image_batches[m]
            # compute embeddings
            _, _, class_mu, class_logvar = encoder(Variable(image_batch))
            class_mu_mm[j] = class_mu
            class_logvar_mm[j] = class_logvar
        # compute the product
        poe_mu, poe_logvar = poe(class_mu_mm, class_logvar_mm, prior_expert=flags.prior_expert)
        # compute outputs
        if reparam_c:
            poe_mu = reparameterize(training=True, mu=poe_mu, logvar=poe_logvar)
        random_style = Variable(torch.randn(len(image_batches[0]), flags.style_dim).cuda(), requires_grad=False)
        out = decoder(random_style, poe_mu, plot=True)
        out = out
        # save images
        for i in range(out.size(0)):
            torchvision.utils.save_image(out[i, :, :, :].squeeze(0), '{}/{}.png'.format(gen_path, k))
            k += 1
        cnt += len(image_batches[0])

    # compute and log FID
    try:  # TODO: remove this try-catch block, once it's figured out what causes fid to break
        fid = calculate_fid_given_paths([gen_path, test_path], batch_size=64,
            dims=2048, cuda=True, glob_pattern="*.png", verbose=verbose)
    except Exception:
        print("[WARN] manual handling of error that occurred during conditional fid computation")
        fid = np.nan
    writer.add_scalars("test/FID/conditional", {
        "%s" % label: fid,
    }, epoch)
    return fid


def write_tsne_embeddings_to_tensorboard(test, encoders, epoch, writer, flags, label, reparam=True, unimodal_poe=True):
    batch = next(cycle(test))
    M = flags.num_modalities
    fig = plt.figure()

    # add unimodal posteriors
    class_mu_stack = []
    class_lv_stack = []
    z_stack = []
    for m in range(M):
        x = batch[m][0].cuda()
        _, _, class_mu, class_lv = encoders[m](x)
        if unimodal_poe:
            poe_mu, poe_lv = poe(class_mu.unsqueeze(0), class_lv.unsqueeze(0), prior_expert=True)
        else:
            poe_mu, poe_lv = class_mu, class_lv
        if reparam:
            z = reparameterize(training=True, mu=poe_mu, logvar=poe_lv)
        else:
            z = poe_mu
        z_stack.append(z)
        class_mu_stack.append(class_mu)
        class_lv_stack.append(class_lv)

    # add joint posterior
    class_mu_stack_tensor = torch.stack(class_mu_stack)
    class_lv_stack_tensor = torch.stack(class_lv_stack)
    poe_mu, poe_lv = poe(class_mu_stack_tensor, class_lv_stack_tensor, prior_expert=flags.prior_expert)
    if reparam:
        z = reparameterize(training=True, mu=poe_mu, logvar=poe_lv)
    else:
        z = poe_mu
    z_stack.append(z)

    # add prior samples
    z_stack.append(torch.randn_like(z))

    # unroll z_stack and compute TSNE embeddings
    z_stack_unroll = torch.stack(z_stack).view(-1, flags.class_dim)
    z_tsne = TSNE(n_components=2).fit_transform(z_stack_unroll.detach().cpu())

    # plotting
    for m in range(M):
        ix_start = m * flags.batch_size
        ix_end = (m+1) * flags.batch_size
        plt.scatter(z_tsne[ix_start:ix_end, 0], z_tsne[ix_start:ix_end, 1], alpha=0.3, label="$p(z|x_%d)$" % m)
    plt.scatter(z_tsne[-(2 * flags.batch_size):-flags.batch_size, 0],
                z_tsne[-(2 * flags.batch_size):-flags.batch_size, 1], alpha=0.3, label="$p(z|x_0,x_1)$")
    plt.scatter(z_tsne[-flags.batch_size:, 0], z_tsne[-flags.batch_size:, 1], alpha=0.3, label="$p(z)$")
    plt.legend()
    writer.add_image("TSNE/%s" % label, fig2data(fig), epoch, dataformats="HWC")
    plt.clf(); plt.close("all")
