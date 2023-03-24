import matplotlib; matplotlib.use('agg')
import os
import random
import json
import torch
import torch.optim as optim
import numpy as np
import logging
import tensorboardX

from torch.autograd import Variable
from tensorboardX import SummaryWriter
from tqdm import tqdm
from itertools import cycle
from networks import (TCDiscriminator, ClassifierCollection,
    InfomaxLinearProjectionHead, InfomaxNonlinearProjectionHead)
from utils import (poe, calc_kl_divergence, reparameterize,
    weights_init, compute_tc, compute_infomax, get_10_mm_digit_samples)
from evaluation import eval_loglikelihoods, eval_generation_qual, eval_generation_clf, eval_generation_fid, eval_tsne
from utils import LIKELIHOOD_DICT
from getters import Getters


def run_epoch(epoch, encoders, decoders, optimizer, data_loader, writer, infomax_projection_head, train=False, flags={},
              classifier_collections=None, lin_classifier_collections=None, content_classifier_collections=None,
              lin_content_classifier_collections=None, tc_tuple=None):
    """
    Runs one epoch for the given dataset.

    Args:
        epoch: Number of the current epoch.
        encoders: List of encoders for all modalities.
        decoders: List of decoders for all modalities.
        optimizer: torch.optim object.
        data_loader: DataLoader.
        writer: TensorBoard SummaryWriter.
        train: Inficates whether this is a training epoch.
        flags: argparse.Namespace with input arguments.
        classifier_collections: Structure with nonlinear latent space classifiers.
        lin_classifier_collections: Structure with linear latent space classifiers.
        content_classifier_collections: Structure with nonlinear shared latent space classifiers.
        lin_content_classifier_collections: Structure with linear shared latent space classifiers.
        tc_tuple: tuple containing a TCDiscriminator and its optimizer

    Returns:
        None.
    """
    M = flags.num_modalities  # number of modalities
    loader = cycle(data_loader)
    name = "train" if train else "test"
    n_iter = len(data_loader)
    pbar = tqdm(range(n_iter), desc="Epoch %d" % epoch, bar_format="{desc}: {percentage:3.0f}%{postfix}")
    pbar_postfix = {}
    for iteration in pbar:
        # setup
        optimizer.zero_grad()
        if train is True:
            for m in range(len(encoders)):
                encoders[m].train(); decoders[m].train()
        else:
            for m in range(len(encoders)):
                encoders[m].eval(); decoders[m].eval()
        if flags.annealing_epochs is not None:
            if epoch < flags.start_annealing:
                annealing_coef = 0
            else:
                annealing_coef = min(
                    (epoch-flags.start_annealing+1) / (flags.annealing_epochs+1), 1.)
            # adjust annealing coef to batches
            if train is True:
                min_coef = min((epoch-flags.start_annealing) / (flags.annealing_epochs+1), 1.)
                annealing_coef = annealing_coef * ((iteration+1) / (n_iter+1))
                annealing_coef = max(min_coef, annealing_coef)
        else:
            annealing_coef = 1.
        writer.add_scalar('%s_debug/AnnealingCoef' % name, annealing_coef, epoch)

        # load a mini-batch
        # augment inputs with Gaussian white noise, if necessary
        batch = next(loader)
        image_batches = []
        if flags.noisy_inputs:
            image_batches_clean = []
        labels_batches = []
        for m in range(M):
            image_batch_m = batch[m][0]
            if flags.noisy_inputs:
                image_batch_m_clean = torch.clone(image_batch_m)
            labels_batch_m = batch[m][1]
            if flags.cuda:
                image_batch_m = image_batch_m.cuda()
                labels_batch_m = labels_batch_m.cuda()
                if flags.noisy_inputs:
                    image_batch_m_clean = image_batch_m_clean.cuda()
            if flags.noisy_inputs:
                # augmenting with the noise
                image_batch_m = image_batch_m + torch.randn_like(image_batch_m)
                image_batches_clean.append(image_batch_m_clean)
            image_batches.append(image_batch_m)
            labels_batches.append(labels_batch_m)

        # init variables
        dict_weighted_loss_terms = {}
        total_loss = torch.Tensor([0.]).cuda()
        mm_class_mu = Variable(torch.zeros(M, flags.batch_size, flags.class_dim)).cuda()
        mm_style_mu = Variable(torch.zeros(M, flags.batch_size, flags.style_dim)).cuda()
        mm_class_logvar = Variable(torch.ones(M, flags.batch_size, flags.class_dim)).cuda()
        mm_style_logvar = Variable(torch.ones(M, flags.batch_size, flags.style_dim)).cuda()

        # modality dropout during training
        if flags.cm_dropout is True and train is True:
            mask_keep = np.random.rand(flags.batch_size, M) > 0.5
        else:
            # NOTE: testing w/o missing modalities
            mask_keep = np.ones([flags.batch_size, M]).astype(bool)

        # do the inference step and store latents
        for m in range(M):
            encoder = encoders[m]
            image_batch = image_batches[m]
            style_mu, style_logvar, class_mu, class_logvar = encoder(Variable(image_batch))
            mm_class_mu[m] = class_mu
            mm_style_mu[m] = style_mu
            mm_class_logvar[m] = class_logvar
            mm_style_logvar[m] = style_logvar

        # compute the poe
        poe_mu_a, poe_logvar_a = poe(mm_class_mu, mm_class_logvar, mask_keep=mask_keep, prior_expert=flags.prior_expert)
        poe_mu_b, poe_logvar_b = poe(mm_class_mu, mm_class_logvar, mask_keep=(~mask_keep), prior_expert=flags.prior_expert)
        poe_mu_full, poe_logvar_full = poe(mm_class_mu, mm_class_logvar, mask_keep=None, prior_expert=flags.prior_expert)
        # below are only used for logging
        poe_kld_a = calc_kl_divergence(poe_mu_a, poe_logvar_a, norm_value=(flags.batch_size))
        poe_kld_full = calc_kl_divergence(poe_mu_a, poe_logvar_a, norm_value=(flags.batch_size))

        for m in range(M):
            if flags.noisy_inputs:
                image_batch = image_batches_clean[m]
            else:
                image_batch = image_batches[m]
            decoder = decoders[m]
            style_mu, style_logvar = mm_style_mu[m], mm_style_logvar[m]
            class_mu, class_logvar = mm_class_mu[m], mm_class_logvar[m]

            # compute log-likelihood
            ix_a = mask_keep[:, m].nonzero()     # no cross_reconstructions
            ix_b = (~mask_keep[:, m]).nonzero()  # again, no cross_reconstructions
            loglik_a = decoder.compute_loglik(image_batch[ix_a], style_mu[ix_a], style_logvar[ix_a],
                poe_mu_a[ix_a], poe_logvar_a[ix_a], reparam=train, freeze_content=flags.freeze_content, reparam_c=flags.reparam_c_for_decoders)
            loglik_b = 0.
            if len(ix_b[0]) > 0:  # handle empty set x_B (e.g., during testing)
                loglik_b = decoder.compute_loglik(image_batch[ix_b], style_mu[ix_b], style_logvar[ix_b],
                    poe_mu_b[ix_b], poe_logvar_b[ix_b], reparam=train, freeze_content=flags.freeze_content, reparam_c=flags.reparam_c_for_decoders)
            if flags.disjoint_partition:
                loglik = loglik_a + loglik_b
            else:
                loglik_joint = decoder.compute_loglik(image_batch, style_mu, style_logvar,
                        poe_mu_full, poe_logvar_full, reparam=train, freeze_content=flags.freeze_content, reparam_c=flags.reparam_c_for_decoders)
                loglik = loglik_joint + loglik_a + loglik_b
            total_loss -= loglik * flags.reconstruction_coefs[m]
            dict_weighted_loss_terms["loglik_m%d" % m] = -loglik * flags.reconstruction_coefs[m]

            # logging of KLD terms
            style_kld = calc_kl_divergence(style_mu, style_logvar, norm_value=(flags.batch_size))
            class_kld = calc_kl_divergence(class_mu, class_logvar, norm_value=(flags.batch_size))
            class_mu_pxprt, class_logvar_pxprt = poe(class_mu.unsqueeze(
                0), class_logvar.unsqueeze(0), prior_expert=flags.prior_expert)
            class_kld_w_pxprt = calc_kl_divergence(
                class_mu_pxprt, class_logvar_pxprt, norm_value=(flags.batch_size))
            writer.add_scalars('%s/MM/LogLik_unweighted' % (name),
                {"M%d/LogLik (unweighted)" % m: loglik}, epoch)
            writer.add_scalars('%s/MM/KLD_unweighted' % (name), {
                'M%d/ContentKLD (unweighted)' % m: class_kld,
                'M%d/ContentKLD_w_pxprt (unweighted)' % m: class_kld_w_pxprt,
                'M%d/StyleKLD (unweighted)' % m: style_kld,
            }, epoch)
            writer.add_scalars('%s_debug/MM/KLD_unweighted' % (name), {
                'M%d/ContentKLD (unweighted)' % m: class_kld,
                'M%d/ContentKLD_w_pxprt (unweighted)' % m: class_kld_w_pxprt,
            }, epoch)

            # run the (frozen) unimodal embeddings through the classifiers
            reparam_list = [True, False]
            for reparam in reparam_list:

                # nonlinear classifiers
                if content_classifier_collections is not None:
                    for cc in content_classifier_collections:
                        labels = labels_batches[0]
                        (pred_class, tmp1, tmp2) = cc.classify(m, labels, class_mu=class_mu_pxprt.data,
                                                              class_logvar=class_logvar_pxprt.data, style_mu=None,
                                                              style_logvar=None, train=train, reparam=reparam)

                        # write classification performance to tensorboard
                        content_accuracy = 100 * (pred_class == labels).float().mean() if pred_class is not None else np.nan
                        scalar_dict = {"ContentAccuracy_c_" + str(m): content_accuracy}
                        label = "%s_c_m_classification/%s_accuracy" % (name, cc.label_name)
                        if reparam:
                            label += "_reparam"
                        writer.add_scalars(label, scalar_dict, epoch)

                # linear classifiers
                if lin_content_classifier_collections is not None:
                    for cc in lin_content_classifier_collections:
                        labels = labels_batches[0]
                        (pred_class, tmp1, tmp2) = cc.classify(m, labels, class_mu=class_mu_pxprt.data,
                                                               class_logvar=class_logvar_pxprt.data, style_mu=None,
                                                               style_logvar=None, train=train, reparam=reparam)

                        # write classification performance to tensorboard
                        content_accuracy = 100 * (pred_class == labels).float().mean() if pred_class is not None else np.nan
                        scalar_dict = {"ContentAccuracy_c_" + str(m): content_accuracy}
                        label = "%s_c_m_lin_classification/%s_accuracy" % (name, cc.label_name)
                        if reparam:
                            label += "_reparam"
                        writer.add_scalars(label, scalar_dict, epoch)

        # compute combined kld
        ix_a_any = (mask_keep).any(axis=1).nonzero()
        ix_b_any = (~mask_keep).any(axis=1).nonzero()
        # compute ckld_prior
        ckld_prior = calc_kl_divergence(poe_mu_a[ix_a_any], poe_logvar_a[ix_a_any], norm_value=(len(ix_a_any[0]))) + \
                     calc_kl_divergence(poe_mu_b[ix_b_any], poe_logvar_b[ix_b_any], norm_value=(len(ix_b_any[0])))
        # compute ckld_cond
        if flags.disjoint_partition:
            ckld_cond = calc_kl_divergence(poe_mu_a[ix_a_any], poe_logvar_a[ix_a_any],
               poe_mu_b[ix_a_any], poe_logvar_b[ix_a_any], norm_value=(len(ix_a_any[0])))
            ckld_cond += calc_kl_divergence(poe_mu_b[ix_b_any], poe_logvar_b[ix_b_any],
                poe_mu_a[ix_b_any], poe_logvar_a[ix_b_any], norm_value=(len(ix_b_any[0])))
            # NOTE: our model requires scaling of loglik by a factor of 2 (unlike MVAE)
            ckld_cond /= 2
            ckld_prior /= 2
        else:
            ckld_cond = calc_kl_divergence(poe_mu_full[ix_a_any], poe_logvar_full[ix_a_any], poe_mu_a[ix_a_any],
                                           poe_logvar_a[ix_a_any], norm_value=len(ix_a_any[0])) + \
                        calc_kl_divergence(poe_mu_full[ix_b_any], poe_logvar_full[ix_b_any], poe_mu_b[ix_b_any],
                                           poe_logvar_b[ix_b_any], norm_value=len(ix_b_any[0]))
        # compute sklds
        sklds_unweighted = [calc_kl_divergence(mm_style_mu[m], mm_style_logvar[m], norm_value=flags.batch_size) for m in range(M)]
        # combine kld loss terms
        if flags.anneal_c_only:
            total_loss += sum([sklds_unweighted[m] * flags.beta_prior_styles[m] for m in range(M)]) + \
                          ckld_cond * annealing_coef * flags.beta_condreg + \
                          ckld_prior * annealing_coef * flags.beta_prior_content
        else:
            total_loss += sum([sklds_unweighted[m] * flags.beta_prior_styles[m] for m in range(M)]) * annealing_coef + \
                          ckld_cond * annealing_coef * flags.beta_condreg + \
                          ckld_prior * annealing_coef * flags.beta_prior_content

        # logging
        for m in range(M):
            dict_weighted_loss_terms["skld_m%d" % m] = sklds_unweighted[m] * flags.beta_prior_styles[m]
            if not flags.anneal_c_only:
                dict_weighted_loss_terms["skld_m%d" % m] *= annealing_coef
        dict_weighted_loss_terms["ckld_prior"] = ckld_prior * annealing_coef * flags.beta_prior_content
        dict_weighted_loss_terms["ckld_cond"] = ckld_cond * annealing_coef * flags.beta_condreg

        writer.add_scalars('%s/MM/KLD_unweighted' % (name), {
            'MM/TotalKLD (unweighted)': ckld_prior + ckld_cond + sum(sklds_unweighted),
            'MM/ckld_prior (unweighted)': ckld_prior,
            'MM/ckld_cond (unweighted)': ckld_cond,
            'MM/skld (unweighted)': sum(sklds_unweighted),
        }, epoch)
        writer.add_scalars('%s_debug/MM/KLD_unweighted' % (name), {
            'MM/poekld_a (unweighted)': poe_kld_a,
            'MM/poekld_full (unweighted)': poe_kld_full,
            'MM/ckld_prior (unweighted)': ckld_prior,
            'MM/ckld_cond (unweighted)': ckld_cond,
            'MM/skld (unweighted)': sum(sklds_unweighted),
        }, epoch)

        # run the (frozen) joint embeddings through the classifiers
        reparam_list = [True, False]
        for reparam in reparam_list:

            # nonlinear classifiers
            if classifier_collections is not None:
                for cc in classifier_collections:
                    for m in range(M):
                        labels = labels_batches[0][ix_a_any]
                        (pred_class, pred_style, pred_concat) = cc.classify(m, labels,
                            class_mu=poe_mu_a[ix_a_any].data, class_logvar=poe_logvar_a[ix_a_any].data,
                            style_mu=mm_style_mu[m, ix_a_any].squeeze(0).data, style_logvar=mm_style_logvar[m, ix_a_any].squeeze(0).data, train=train, reparam=reparam)

                        # write classification performance to tensorboard
                        content_accuracy = 100 * (pred_class == labels).float().mean() if pred_class is not None else np.nan
                        style_accuracy = 100 * (pred_style == labels).float().mean() if pred_style is not None else np.nan
                        concat_accuracy = 100 * \
                            (pred_concat == labels).float().mean() if pred_concat is not None else np.nan
                        label = "%s_classification/%s_accuracy/M%d" % (name, cc.label_name, m)
                        if reparam:
                            label += "_reparam"
                        writer.add_scalars(label, {
                            "ContentAccuracy": content_accuracy,
                            "StyleAccuracy": style_accuracy,
                            "CatAccuracy": concat_accuracy
                        }, epoch)

            # linear classifiers
            if lin_classifier_collections is not None:
                for cc in lin_classifier_collections:
                    for m in range(M):
                        labels = labels_batches[0][ix_a_any]
                        (pred_class, pred_style, pred_concat) = cc.classify(m, labels,
                            class_mu=poe_mu_a[ix_a_any].data, class_logvar=poe_logvar_a[ix_a_any].data,
                            style_mu=mm_style_mu[m, ix_a_any].squeeze(0).data, style_logvar=mm_style_logvar[m, ix_a_any].squeeze(0).data, train=train, reparam=reparam)

                        # write classification performance to tensorboard
                        content_accuracy = 100 * (pred_class == labels).float().mean() if pred_class is not None else np.nan
                        style_accuracy = 100 * (pred_style == labels).float().mean() if pred_style is not None else np.nan
                        concat_accuracy = 100 * \
                            (pred_concat == labels).float().mean() if pred_concat is not None else np.nan
                        label = "%s_lin_classification/%s_accuracy/M%d" % (name, cc.label_name, m)
                        if reparam:
                            label += "_reparam"
                        writer.add_scalars(label, {
                            "ContentAccuracy": content_accuracy,
                            "StyleAccuracy": style_accuracy,
                            "CatAccuracy": concat_accuracy
                        }, epoch)

        # debug-log: write mus and sigmas to tensorboard
        cat_mu = torch.cat((poe_mu_a.view(-1), mm_style_mu.view(-1)))
        cat_logvar = torch.cat((poe_logvar_a.view(-1), mm_style_logvar.view(-1)))

        for m in range(M):
            writer.add_scalars('%s_debug/LatentStatistics_Mu' % name, {
                'M%d/ContentMu' % m: mm_class_mu[m].mean(),
                'M%d/StyleMu' % m: mm_style_mu[m].mean(),
                'MM/PoeMu': poe_mu_a.mean(),
                'MM/CombinedMu': cat_mu.mean(),
            }, epoch)
            writer.add_scalars('%s_debug/LatentStatistics_Logvar' % name, {
                'M%d/ContentLogvar' % m: mm_class_logvar[m].mean(),
                'M%d/StyleLogvar' % m: mm_style_logvar[m].mean(),
                'MM/PoeLogvar': poe_logvar_a.mean(),
                'MM/CombinedLogvar': cat_logvar.mean(),
            }, epoch)
        # save batch statistics (simulates learned means and logvars)
        flags.batch_poe_mu = poe_mu_a.mean()
        flags.batch_poe_logvar = poe_logvar_a.mean()

        # minimize TC(c, s_m) for each modality m
        if tc_tuple is not None:
            for m in range(M):
                ix_m = mask_keep[:, m].nonzero()  # no cross_reconstructions
                if len(ix_m[0]) <= 1:
                    continue  # no contrasting possible with fewer than 2 samples
                if flags.disentangle_style_only:
                    tc_disent, cel_disent = compute_tc(tc_tuple[m], mm_style_mu[m, ix_m].squeeze(0), mm_style_logvar[m, ix_m].squeeze(0),
                        poe_mu_a.data[ix_m], poe_logvar_a.data[ix_m], train=train)
                else:
                    tc_disent, cel_disent = compute_tc(tc_tuple[m], mm_style_mu[m, ix_m], mm_style_logvar[m, ix_m],
                        poe_mu_a[ix_m], poe_logvar_a[ix_m], train=train)
                total_loss += tc_disent * flags.disentanglement_coefs[m]
                writer.add_scalars('%s_debug/InfoMeasures' % (name),
                    {"/M%d_CEL_disent" % m: float(cel_disent)}, epoch)
                writer.add_scalars('%s/MM/InfoMeasures' % (name),
                    {"/M%d_TC_disent" % m: float(tc_disent)}, epoch)
                dict_weighted_loss_terms["tc_disent_m%d" % m] = tc_disent * flags.disentanglement_coefs[m]

        # maximize I(x; c)
        mask_rand = np.random.rand(M) < 0.5
        if mask_rand.sum() == 0:
            mask_rand[np.random.randint(M)] = True  # at least one present modality
        ix_keep = mask_rand.nonzero()
        mu1, lv1 = poe(mm_class_mu[ix_keep], mm_class_logvar[ix_keep], prior_expert=flags.prior_expert)
        if flags.reparam_c_before_infomax:
            if flags.contrast_full_vs_subset is True:
                h1 = reparameterize(training=train, mu=poe_mu_full, logvar=poe_logvar_full)
            else:
                h1 = reparameterize(training=train, mu=poe_mu_a, logvar=poe_logvar_a)
            h2 = reparameterize(training=train, mu=mu1, logvar=lv1)
        else:
            if flags.contrast_full_vs_subset is True:
                h1 = poe_mu_full
            else:
                h1 = poe_mu_a
            h2 = mu1
        mi_info, cel_info = compute_infomax(infomax_projection_head, h1=h1[ix_a_any], h2=h2[ix_a_any])
        total_loss += cel_info * flags.infomax_coef
        writer.add_scalars('%s_debug/InfoMeasures' % (name),
            {"/MM_CEL_shared": float(cel_info)}, epoch)
        writer.add_scalars('%s/MM/InfoMeasures' % (name),
            {"/MM_I_shared": float(mi_info)}, epoch)
        dict_weighted_loss_terms["cel_infomax_m%d" % m] = cel_info * flags.infomax_coef

        # backprop
        pbar_postfix["total_loss"] = total_loss.item()
        pbar.set_postfix(pbar_postfix)
        if torch.isnan(total_loss).any():
            if not train:
                pass
            else:
                print()
                print(total_loss, loglik, ckld_prior, ckld_cond, sum(sklds_unweighted))
                raise ValueError("NaN loss detected during training")
        if train is True:
            total_loss.backward()
            optimizer.step()
        if train and iteration % 100 == 0:
            print("\n\n%-30s %.2f" % ("total_loss", total_loss))
            for k, v in dict_weighted_loss_terms.items():
                print("%-30s | %.2f" % (k, v))


def training_procedure(flags):
    """
    Defines the general training procedure.

    Args:
        flags: argparse.Namespace with input arguments.

    Returns:
        None.
    """
    M = flags.num_modalities

    print("Number of modalities M = " + str(M))

    # set random seed
    if flags.random_seed is not None:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        np.random.seed(flags.random_seed)
        torch.manual_seed(flags.random_seed)
        random.seed(flags.random_seed)
    else:
        print("[WARN] No random seed was set")

    # training
    if torch.cuda.is_available() and not flags.cuda:
        print("[WARN] You have a CUDA device, so you should probably run with --cuda")

    # set up likelihoods
    likelihoods = []
    for l in flags.likelihood_str.split("-"):
        likelihoods.append(LIKELIHOOD_DICT[l])
    if len(likelihoods) == 1:  # assume similar likelihoods if only one was provided
        tmp = likelihoods[0]
        likelihoods = [tmp for _ in range(flags.num_modalities)]
    assert len(likelihoods) == flags.num_modalities

    # model definition
    gtrs = Getters()
    encoders, decoders = gtrs.get_encs_decs(flags, likelihoods)
    for m in range(M):
        encoders[m].apply(weights_init)
        decoders[m].apply(weights_init)
        if flags.cuda:
            encoders[m].cuda()
            decoders[m].cuda()
        # load saved models if load_saved flag is true
        if flags.load_saved:
            print("Loading saved model from checkpoint")
            encoders[m].load_state_dict(torch.load(os.path.join(flags.saved_path, flags.encoder_file + "_%d" % m)))
            decoders[m].load_state_dict(torch.load(os.path.join(flags.saved_path, flags.decoder_file + "_%d" % m)))

    # initialize infomax_projection_head
    if flags.infomax_nonlinear_projection_head:
        infomax_projection_head = InfomaxNonlinearProjectionHead(ndim=flags.class_dim).cuda()
    else:
        infomax_projection_head = InfomaxLinearProjectionHead(ndim=flags.class_dim).cuda()
    infomax_projection_head.apply(weights_init)
    # NOTE: parameters of the projection head are added to optimizer of the whole autoencoder

    # optimizer definition
    params = [p for model in encoders + decoders + [infomax_projection_head] for p in list(model.parameters())]
    optimizer = optim.Adam(params, lr=flags.initial_learning_rate, betas=(flags.beta_1, flags.beta_2))

    if not os.path.exists('%s/checkpoints' % flags.log_dir):
        os.makedirs('%s/checkpoints' % flags.log_dir)

    # load data set and create data loader instance
    print('Loading multimodal dataset...')
    train, test = gtrs.get_data_loaders(batch_size=flags.batch_size, num_modalities=flags.num_modalities,
                                   num_workers=flags.num_workers, shuffle=True, device="cuda" if flags.cuda else "cpu")

    # load global samples
    sample = get_10_mm_digit_samples(test, flags)

    # initialize summary writer
    writer = SummaryWriter(flags.log_dir)
    # save flags to tensorboard
    writer.add_text('flags', json.dumps(vars(flags)), 0)
    print("\nFLAGS:")
    print(json.dumps(vars(flags)), "\n")

    # initialize classifiers and their optimizers
    classifier_collections = []  # len: num_classification_tasks
    for label_ix, (label_name, num_classes) in gtrs.LABEL_DICT.items():
        cc = ClassifierCollection(flags.class_dim, flags.style_dim, num_classes, label_ix, label_name, num_modalities=M)
        cc.initialize_models(lr=flags.initial_learning_rate, betas=(flags.beta_1, flags.beta_2), linear=False)
        classifier_collections.append(cc)

    # Initialize linear classifiers and their optimizers.
    # This is mostly to see if style information does not contain shared information.
    # Shi et al. (2019) use linear classifiers on their latent space; include this for comparability
    lin_classifier_collections = []  # len: num_classification_tasks
    for label_ix, (label_name, num_classes) in gtrs.LABEL_DICT.items():
        lcc = ClassifierCollection(flags.class_dim, flags.style_dim, num_classes, label_ix, label_name,
                                   num_modalities=M)
        lcc.initialize_models(lr=flags.initial_learning_rate, betas=(flags.beta_1, flags.beta_2), linear=True)
        lin_classifier_collections.append(lcc)

    content_classifier_collections = []  # len: num_classification_tasks
    for label_ix, (label_name, num_classes) in gtrs.LABEL_DICT.items():
        cc = ClassifierCollection(flags.class_dim, 0, num_classes, label_ix, label_name, num_modalities=M)
        cc.initialize_models(lr=flags.initial_learning_rate, betas=(flags.beta_1, flags.beta_2), linear=False)
        content_classifier_collections.append(cc)

    lin_content_classifier_collections = []  # len: num_classification_tasks
    for label_ix, (label_name, num_classes) in gtrs.LABEL_DICT.items():
        cc = ClassifierCollection(flags.class_dim, 0, num_classes, label_ix, label_name, num_modalities=M)
        cc.initialize_models(lr=flags.initial_learning_rate, betas=(flags.beta_1, flags.beta_2), linear=True)
        lin_content_classifier_collections.append(cc)

    # initialize tc discriminators for min I(c, s_m) for all modalities
    if flags.style_dim > 0:
        tc_tuple = []
        for m in range(M):
            d_tc = TCDiscriminator(flags.style_dim, flags.class_dim, hidden_size=flags.tc_hidden_size).cuda()
            d_tc.apply(weights_init)
            opt_tc = optim.Adam(d_tc.parameters(), lr=flags.tc_initial_lr, betas=(flags.tc_beta_1, flags.tc_beta_2))
            tc_tuple.append((d_tc, opt_tc))
    else:
        tc_tuple = None

    # initialize pre-trained image classifiers
    img_to_digit_clfs = None
    img_to_digit_clfs = gtrs.get_img_to_digit_clfs(flags)

    # reduce tensorboardx logging level
    tensorboardX.writer.logging.getLogger().setLevel(logging.ERROR)

    # run the training and testing
    for epoch in range(flags.end_epoch):
        print()

        # train
        run_epoch(epoch, encoders, decoders, optimizer, train, writer, infomax_projection_head, train=True, flags=flags,
                  classifier_collections=classifier_collections, lin_classifier_collections=lin_classifier_collections,
                  content_classifier_collections=content_classifier_collections,
                  lin_content_classifier_collections=lin_content_classifier_collections, tc_tuple=tc_tuple)

        # test
        with torch.no_grad():
            run_epoch(epoch, encoders, decoders, optimizer, test, writer, infomax_projection_head, train=False, flags=flags,
                      classifier_collections=classifier_collections,
                      lin_classifier_collections=lin_classifier_collections,
                      content_classifier_collections=content_classifier_collections,
                      lin_content_classifier_collections=lin_content_classifier_collections, tc_tuple=tc_tuple)
            # evaluate test metrics (based on samples during training and more thoroughly after the last epoch)
            num_samples = len(test) * flags.batch_size if (epoch + 1) == flags.end_epoch else flags.batch_size
            num_imp_samples = flags.num_imp_samples if (epoch + 1) == flags.end_epoch else flags.num_imp_samples // 5
            if epoch > 0 and (epoch % flags.eval_freq_likelihood == 0) or ((epoch + 1) == flags.end_epoch):
                eval_loglikelihoods(get_data_loaders=gtrs.get_data_loaders, encoders=encoders, decoders=decoders,
                                    epoch=epoch, writer=writer, flags=flags, num_imp_samples=num_imp_samples)
            if epoch > 0 and ((epoch % flags.eval_freq_generation == 0) or ((epoch + 1) == flags.end_epoch)):
                eval_generation_qual(sample, encoders, decoders, epoch, writer, flags)
                if not flags.noisy_inputs:
                    eval_generation_clf(test, img_to_digit_clfs, encoders, decoders, epoch, writer,
                                        flags, num_samples, reparam_c=False)
                    eval_generation_clf(test, img_to_digit_clfs, encoders, decoders, epoch, writer,
                                        flags, num_samples, reparam_c=True, label_suffix="_reparam")
                else:
                    train_tmp, test_tmp = gtrs.get_data_loaders(batch_size=flags.batch_size,
                                                           num_modalities=flags.num_modalities,
                                                           num_workers=flags.num_workers, shuffle=True,
                                                           device="cuda" if flags.cuda else "cpu", random_noise=True)
                    eval_generation_clf(test_tmp, img_to_digit_clfs, encoders, decoders, epoch,
                                        writer, flags, num_samples, reparam_c=False)
                    eval_generation_clf(test_tmp, img_to_digit_clfs, encoders, decoders, epoch,
                                        writer, flags, num_samples, reparam_c=True, label_suffix="_reparam")
                eval_tsne(test, encoders, epoch, writer, flags)
            if flags.eval_freq_fid != 0 and epoch > 0 and ((epoch % flags.eval_freq_fid == 0) or
                                                           ((epoch + 1) == flags.end_epoch)):
                eval_generation_fid(test, flags.fid_gen_path, flags.fid_test_paths,
                                    encoders, decoders, epoch, writer, flags, num_samples)

        # save checkpoints after every 5 epochs
        if (epoch + 1) % flags.save_freq == 0 or (epoch + 1) == flags.end_epoch:
            for m in range(M):
                torch.save(encoders[m].state_dict(), os.path.join(
                    flags.log_dir, 'checkpoints', flags.encoder_file+"_%d" % m))
                torch.save(decoders[m].state_dict(), os.path.join(
                    flags.log_dir, 'checkpoints', flags.decoder_file+"_%d" % m))
