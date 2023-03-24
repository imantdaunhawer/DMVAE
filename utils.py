import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import cycle
from torch.autograd import Variable
from torchvision.transforms import Compose, ToTensor, Lambda
from torch.distributions import Bernoulli, Laplace, Normal

# supported likelihood functions
LIKELIHOOD_DICT = {
    "bernoulli": Bernoulli,
    "laplace": Laplace,
    "normal": Normal
}

# compose a transform configuration
transform_config = Compose([ToTensor()])
transform_with_noise_config = Compose([Lambda(lambda x: (x ^ torch.bernoulli(x, 0.10)).float())])
LOG2PI = float(np.log(2.0 * math.pi))


def poe(mu_in, logvar_in, eps=1e-8, mask_keep=None, prior_expert=False):
    """
    Performs product of experts (PoE) aggregation. Adapted from Wu'18:
    https://github.com/mhw32/multimodal-vae-public

    Args:
        mu_in: Means of posteriors with dimensions [M x batch size x number of latent dimensions].
        logvar_in: Log-variances of posteriors with dimensions [M x batch size x number of latent dimensions].
        eps: Small perturbation for variances.
        mask_keep: Mask identifying which modalities need to be included into the PoE.  By default, all modalities
            are kept.
        prior_expert: Flag indicating whether to use a prior expert in the PoE.

    Returns:
        Aggregated means and log-variances.
    """
    assert mu_in.ndimension() == logvar_in.ndimension() == 3  # M x BS x LATENTS
    mu = mu_in.clone()
    logvar = logvar_in.clone()
    if prior_expert is True:
        tmp = Variable(torch.zeros(mu.shape[1:]).unsqueeze(0).cuda(), requires_grad=False)
        mu = torch.cat([mu, tmp])
        logvar = torch.cat([logvar, tmp])  # use zero logvar because log(1) = 0

    var = torch.exp(logvar) + eps
    # precision of i-th Gaussian expert at point x
    T = 1. / var
    if mask_keep is not None:
        if prior_expert is True:
            mask_keep = np.concatenate((mask_keep.copy(), np.ones((mask_keep.shape[0], 1)).astype(bool)), axis=1)
        mask_keep_repeat = mask_keep.T[:, :, np.newaxis].repeat(logvar.shape[2], axis=2)
        assert T.shape == mask_keep_repeat.shape
        # turn off the indices that should not be kept
        T[(~mask_keep_repeat).nonzero()] = 1e-12
        mu[(~mask_keep_repeat).nonzero()] = 0.
    pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
    pd_var = 1. / torch.sum(T, dim=0)
    pd_logvar = torch.log(pd_var)
    return pd_mu, pd_logvar


def reparameterize(training, mu, logvar):
    """Reparameterization for multivariate Gaussian posteriors.

    Args:
        training: bool, indicating if training or testing.
        mu: location parameters.
        logvar: scale parameters (log of variances).

    Returns:
        Reparameterized representations.
    """
    if training:
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
    else:
        return mu


def weights_init(layer):
    """
    Initializes (in-place) weights of the given torch.nn Module.

    Args:
        layer: torch.nn Module.

    Returns:
        None.
    """
    if isinstance(layer, nn.Conv2d):
        layer.weight.data.normal_(0.0, 0.05)
        if layer.bias is not None:
            layer.bias.data.zero_()
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.normal_(1.0, 0.02)
        if layer.bias is not None:
            layer.bias.data.zero_()
    elif isinstance(layer, nn.Linear):
        layer.weight.data.normal_(0.0, 0.05)
        if layer.bias is not None:
            layer.bias.data.zero_()
    else:
        return ValueError


def calc_kl_divergence(mu1, logvar1, mu2=None, logvar2=None, norm_value=None, clip_value=np.inf):
    """
    Calculates KL divergences between a pair of multivariate Gaussians. If mu2
    and logvar2 are not specified, compare to a Standard Normal distribution.

    Args:
        mu1: Means.
        logvar1: Log-variances.
        mu2: Means of reference distributions. By default, if no reference
        distributions are specified, standard normal distribution is used.
        logvar2: Log-variances of reference distributions.
        norm_value: Normalizing constant for KL divergence.
        clip_value: Clipping value for KL divergence.

    Returns:
        KL divergences between pairs of given distributions, or, alternatively,
        between given distributions and standard Gaussian.
    """
    assert not((mu2 is None and logvar2 is not None) or (mu2 is not None and logvar2 is None))
    if mu2 is None and logvar2 is None:
        kld = -0.5 * torch.sum(1 - logvar1.exp() - mu1.pow(2) + logvar1)
    else:
        kld = -0.5 * (torch.sum(1 - logvar1.exp()/logvar2.exp() - (mu2-mu1).pow(2)/logvar2.exp() - logvar2 + logvar1))
    if norm_value is not None:
        kld = kld / float(norm_value)
    return torch.min(kld, torch.Tensor([clip_value]).cuda()[0])


def gaussian_log_pdf(x, mu, logvar):
    """
    Calculates log-likelihood of data given ~N(mu, exp(logvar))
    NOTE: adapted from MVAE codebase

    Args:
        x: Tensor with the ground truth input.
        mu: Mean.
        logvar: Log-variance.

    Returns:
        Gaussian log-likelihood.
    """
    global LOG2PI
    log_pdf = -0.5 * LOG2PI - logvar / 2. - torch.pow(x - mu, 2) / (2. * torch.exp(logvar))
    return torch.sum(log_pdf, dim=1)


def unit_gaussian_log_pdf(x):
    """
    Calculates log-likelihood of data given ~N(0, 1)
    NOTE: adapted from MVAE codebase

    Args:
        x:  Tensor with the ground truth input.

    Returns:
        Standard normal log-likelihood.
    """
    global LOG2PI
    log_pdf = -0.5 * LOG2PI - math.log(1.) / 2. - torch.pow(x, 2) / 2.
    return torch.sum(log_pdf, dim=1)


def log_mean_exp(x, dim=1):
    """
    Calculates log(1/k * sum(exp(x))).
    NOTE: adapted from MVAE codebase

    Args:
        x: Tensor with samples.
        dim: Which dimension to take the mean over.

    Returns:
        Mean of x.
    """
    m = torch.max(x, dim=dim, keepdim=True)[0]
    return m + torch.log(torch.mean(torch.exp(x - m),
                         dim=dim, keepdim=True))


def log_marginal_estimate(ss_list, image, z, mu, logvar, lik_distribution):
    """
    Estimates log p(x).
    NOTE: adapted from MVAE codebase

    Args:
        ss_list: List of sufficient statistics.
        image: Batch of original observed images.
        z: Samples drawn from variational distribution.
        mu: Means of variational distribution.
        logvar: Log-variances of variational distribution.
        lik_distribution: Lilkelihood function.

    Returns:
        An estimate of log p(x).
    """
    batch_size, n_samples, z_dim = z.size()
    input_dim = image.size(1)

    # repeat target k times
    image_k = image.unsqueeze(1).repeat(1, n_samples, 1)

    # reshape all representations to 2d, i.e. (-1, z_dim)
    z2d = z.view(batch_size * n_samples, z_dim)
    mu2d = mu.view(batch_size * n_samples, z_dim)
    logvar2d = logvar.view(batch_size * n_samples, z_dim)
    image_2d = image_k.view(batch_size * n_samples, input_dim)

    # compute components of the likelihood
    log_p_x_given_z_2d = lik_distribution(*ss_list).log_prob(image_2d).sum(dim=1)
    log_q_z_given_x_2d = gaussian_log_pdf(z2d, mu2d, logvar2d)
    log_p_z_2d = unit_gaussian_log_pdf(z2d)

    # combine components and reshape to (bs, k)
    log_weight_2d = log_p_x_given_z_2d + log_p_z_2d - log_q_z_given_x_2d
    log_weight = log_weight_2d.view(batch_size, n_samples)

    # need to compute normalization constant for weights
    # i.e. log ( mean ( exp ( log_weights ) ) )
    log_p = log_mean_exp(log_weight, dim=1)
    return torch.mean(log_p)


def log_joint_estimate(ss_lists, targets, z_k, mus_cat_k, logvars_cat_k, lik_distributions, scaling_factors=None):
    """
    Estimates log p(x,y).
    NOTE: adapted from MVAE codebase

    Args:
        ss_lists: List of lists of sufficient stats, for each modality.
        targets: Batch of originally observed images.
        z_k: Samples drawn from variational distribution.
        mus_cat_k: Means of variational distribution from all modalities including all importance samples.
        logvars_cat_k: Log-variance of variational distribution from all modalities including all importance samples.
        lik_distributions: Likelihood functions.
        scaling_factors: Scaling factors for modalities. By default, all modalities are weighted equally.

    Returns:
        An estimate of log p(x,y).
    """
    assert len(ss_lists) == len(targets) == len(lik_distributions)
    M = len(targets)
    batch_size, n_samples, z_dim = z_k.size()
    log_px_zs = torch.zeros(M, batch_size * n_samples).cuda()
    for m in range(M):
        target = targets[m]
        ss_list = ss_lists[m]
        dist = lik_distributions[m]
        num_pixels = target.size(1)
        scaling_factor = scaling_factors[m] if scaling_factors else 1.

        # repeat target k times and reshape to 2d, i.e. (-1, z_dim)
        target_k = target.unsqueeze(1).repeat(1, n_samples, 1)
        target_k_2d = target_k.view(batch_size * n_samples, num_pixels)

        # compute and append likelihood
        log_px_z = dist(*ss_list).log_prob(target_k_2d) * scaling_factor
        log_px_zs[m] = log_px_z.sum(-1)  # sum over pixels, keep k

    # compute components of likelihood estimate
    log_joint_zs_2d = log_px_zs.sum(0)  # sum over modalities
    z2d = z_k.view(batch_size * n_samples, z_dim)
    mu2d = mus_cat_k.view(batch_size * n_samples, z_dim)
    logvar2d = logvars_cat_k.view(batch_size * n_samples, z_dim)
    log_q_z_given_x_2d = gaussian_log_pdf(z2d, mu2d, logvar2d)
    log_p_z_2d = unit_gaussian_log_pdf(z2d)

    # combine components of likelihood estimate
    log_weight_2d = log_joint_zs_2d + log_p_z_2d - log_q_z_given_x_2d
    log_weight = log_weight_2d.view(batch_size, n_samples)

    # need to compute normalization constant for weights
    # i.e. log ( mean ( exp ( log_weights ) ) )
    log_p = log_mean_exp(log_weight, dim=1)

    return torch.mean(log_p)


def fig2data(fig):
    """
    Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    Args:
        fig: fig a matplotlib figure

    Returns:
        buf: numpy 3D array of RGB values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (h, w, 3)
    return buf


def compute_tc(tc_tuple, style_mu, style_logvar, content_mu, content_logvar, train=True, dimperm=False):
    """
    Estimates total correlation (TC) between a set of variables and optimizes
    the TCDiscriminator if train=true.
    NOTE: adapted from FactorVAE (https://github.com/1Konny/FactorVAE)

    Args:
        tc_tuple: tuple containing a TCDiscriminator and its optimizer
        style_mu: location parameter of modality-specific Gaussian posterior
        style_logvar: scale parameter (log variance) of modality-specific Gaussian posterior
        content_mu: location parameter of shared Gaussian posterior
        content_logvar: scale parameter (log variance) of shared Gaussian posterior
        train: boolean indicator if training or testing
        dimperm: whether to permute the individual dimensions of the
            modality-specific representation. Default: False.

    Returns:
        A tuple (tc, d_loss), where tc is the estimated total correlation and
        d_loss is the loss of the cross-entropy loss of the discriminator
    """
    # prep
    tc_d, tc_opt = tc_tuple
    num_samples = style_mu.shape[0]
    zeros = torch.zeros(num_samples, dtype=torch.long).cuda()
    ones = torch.ones(num_samples, dtype=torch.long).cuda()
    tc_opt.zero_grad()
    if train is True:
        tc_d.train()
    else:
        tc_d.eval()

    # reparameterize to get representations
    s = reparameterize(training=train, mu=style_mu, logvar=style_logvar)
    c = reparameterize(training=train, mu=content_mu, logvar=content_logvar)

    # permute the second representation
    s_perm = s.clone()
    if dimperm:
        for i in range(s_perm.shape[-1]):
            s_perm[:, i] = s_perm[torch.randperm(num_samples), i]
    else:  # batch-wise permutation, keeping dimensions intact
        s_perm = s_perm[torch.randperm(num_samples)]

    # compute the CEL and backprop within the discriminator
    scores = tc_d(s.data, c.data)
    scores_perm = tc_d(s_perm.data, c.data)
    d_loss = 0.5 * (F.cross_entropy(scores, zeros) + F.cross_entropy(scores_perm, ones))
    # backprop
    if train is True:
        d_loss.backward()
        tc_opt.step()

    # estimate tc
    scores = tc_d(s, c)
    lsm = F.log_softmax(scores, dim=1)
    tc = (lsm[:, 0] - lsm[:, 1]).mean()
    return tc, d_loss


def compute_infomax(projection_head, h1, h2, tau=1.0):
    """
    Estimates the mutual information between a set of variables.
    Automatically uses $K = batch_size - 1$ negative samples.

    Args:
        projection_head: projection head for the MI-estimator. Can be identity.
        h1: torch.Tensor, first representation
        h2: torch.Tensor, second representation
        tau: temperature hyperparameter.

    Returns:
        A tuple (mi, d_loss) where mi is the estimated mutual information and
        d_loss is the cross-entropy loss computed from contrasting
        true vs. permuted pairs.
    """

    # compute cosine similarity matrix C of size 2N * (2N - 1), w/o diagonal elements
    batch_size = h1.shape[0]
    z1 = projection_head(h1)
    z2 = projection_head(h2)
    z1_normalized = F.normalize(z1, dim=-1)
    z2_normalized = F.normalize(z2, dim=-1)
    z = torch.cat([z1_normalized, z2_normalized], dim=0)  # 2N * D
    C = torch.mm(z, z.t().contiguous())  # 2N * 2N
    # remove diagonal elements from C
    mask = ~ torch.eye(2 * batch_size, device=C.device).type(torch.ByteTensor)  # logical_not on identity matrix
    C = C[mask].view(2 * batch_size, -1)  # 2N * (2N - 1)

    # compute loss
    numerator = 2 * torch.sum(z1_normalized * z2_normalized) / tau
    denominator = torch.logsumexp(C / tau, dim=-1).sum()
    loss = (denominator - numerator) / (2 * batch_size)
    return np.nan, loss  # NOTE: Currently returns MI=NaN


def get_10_mm_digit_samples(loader, flags):
    """
    Samples randomly 10 multimodal data points corresponding to the 10 different digits from the given dataset.

    Args:
        loader: DataLoader.
        flags: argparse.Namespace with input arguments.

    Returns:
        10 randomly sampled multimodal data points (corresponding to the 10 different digits).
    """
    n_batches = len(loader)
    batch_num = np.random.choice(a=np.arange(n_batches), size=1)
    loader_ = cycle(loader)
    samples = []
    for b in range(n_batches):
        batch = next(loader_)
        if b == batch_num:
            M = len(batch)

            label_0_batch = batch[0][1]
            if flags.cuda:
                label_0_batch = label_0_batch.cuda()
            ixs = np.zeros((10, ))
            for i in range(10):
                if flags.cuda:
                    ixsi = np.where(label_0_batch.cpu() == i)[0]
                else:
                    ixsi = np.where(label_0_batch == i)[0]
                ixs[i] = ixsi[np.random.randint(0, len(ixsi))]

            for m in range(M):
                samples.append([])
                image_m_batch = batch[m][0]
                label_m_batch = batch[m][1]
                if flags.cuda:
                    image_m_batch = image_m_batch.cuda()
                    label_m_batch = label_m_batch.cuda()
                for ix in ixs:
                    if flags.cuda:
                        samples[m].append(image_m_batch[int(ix)].cpu().numpy())
                    else:
                        samples[m].append(image_m_batch[int(ix)].numpy())
            break
    samples_res = []
    for m in range(M):
        samples_m = samples[m]
        samples_m = Variable(torch.Tensor(samples_m), requires_grad=False)
        samples_m = samples_m.transpose(0, 1)
        samples_m = samples_m.squeeze(2)
        if flags.cuda:
            samples_m = samples_m.cuda()
        if flags.noisy_inputs:
            samples_m = samples_m + torch.randn_like(samples_m)
        samples_res.append(samples_m)

    return samples_res
