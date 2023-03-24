from utils_tensorboard import (write_marginal_loglikelihood_to_tensorboard,
                               write_joint_loglikelihood_to_tensorboard,
                               write_conditional_loglikelihood_to_tensorboard,
                               write_reconstructions_to_tensorboard,
                               write_unconditional_sampling_figure_to_tensorboard,
                               write_intramodality_swapping_to_tensorboard,
                               write_crossmodality_swapping_to_tensorboard,
                               write_conditional_generation_to_tensorboard,
                               write_unconditional_generation_to_tensorboard,
                               write_conditional_generation_samples_to_tensorboard,
                               write_conditional_fid_to_tensorboard,
                               write_unconditional_fid_to_tensorboard,
                               write_tsne_embeddings_to_tensorboard,
                               )


def eval_loglikelihoods(get_data_loaders, encoders, decoders, epoch, writer, flags, num_imp_samples,
                        scaling_factors=None):
    """
    Compute marginal, joint, and conditional log-likelihoods on the test set and writes them to TensorBoard.

    Args:
        get_data_loaders: Getter method that returns DataLoaders.
        encoders: List of encoders for all modalities.
        decoders: List of decoders for all modalities.
        epoch: Number of the current training epoch.
        writer: TensorBoard SummaryWriter.
        flags: argparse.Namespace with input arguments.
        num_imp_samples: Number of importance samples.
        scaling_factors: Scaling factors for individual modalities. By default, all modalities are weighted equally.

    Returns:
        None.
    """
    print("\nEvaluate log-likelihoods (K=%d) ..." % num_imp_samples)

    # marginals, i.e. log p(x_i)
    lls = []
    for m in range(flags.num_modalities):
        ll = write_marginal_loglikelihood_to_tensorboard(m, encoders[m], decoders[m], get_data_loaders=get_data_loaders,
                                                         writer=writer, epoch=epoch, flags=flags, label="M%d" % m,
                                                         num_samples=flags.batch_size, num_imp_samples=num_imp_samples)
        lls.append(ll)

    # joint, i.e. log p(x_1, x_2, ..., x_M)
    ll = write_joint_loglikelihood_to_tensorboard(encoders, decoders, get_data_loaders=get_data_loaders,
                                                  writer=writer, epoch=epoch, flags=flags, label="MM",
                                                  num_samples=flags.batch_size,
                                                  num_imp_samples=num_imp_samples,
                                                  scaling_factors=scaling_factors)

    # leave-one-out conditionals, i.e. log p(x_i | x_1, ..., x_{i-1}, x_{i+1}, ..., x_M)
    for m in range(flags.num_modalities):
        conds = list(range(flags.num_modalities))
        conds.remove(m)
        ll = write_conditional_loglikelihood_to_tensorboard(conds, m, encoders, decoders[m],
                                                            get_data_loaders=get_data_loaders, writer=writer,
                                                            epoch=epoch, flags=flags, label="rest->M"+str(m),
                                                            num_samples=flags.batch_size,
                                                            num_imp_samples=num_imp_samples)
        conds = list(range(flags.num_modalities))
        ll = write_conditional_loglikelihood_to_tensorboard(conds, m, encoders, decoders[m],
                                                            get_data_loaders=get_data_loaders,
                                                            writer=writer, epoch=epoch, flags=flags,
                                                            label="MM->M" + str(m) + "'",
                                                            num_samples=flags.batch_size,
                                                            num_imp_samples=num_imp_samples)

    # pairwise conditionals, i.e. log p(x_i | x_j)
    if flags.num_modalities > 2:
        for m_from in range(flags.num_modalities):
            for m_to in range(flags.num_modalities):
                if not m_from == m_to:
                    ll = write_conditional_loglikelihood_to_tensorboard([m_from], m_to, encoders, decoders[m_to],
                                                                        get_data_loaders=get_data_loaders,
                                                                        writer=writer, epoch=epoch, flags=flags,
                                                                        label="M"+str(m_from)+"->M"+str(m_to),
                                                                        num_samples=flags.batch_size,
                                                                        num_imp_samples=num_imp_samples)


def eval_generation_qual(sample, encoders, decoders, epoch, writer, flags):
    """
    Performs conditional and unconditional generation, reconstructions, within- and between-modality swapping of style
    and content. Writes the results to TensorBoard.

    Args:
        sample: Multimodal samples to be used in reconstructions, style and content swapping and conditional generation.
        encoders: List of encoders for all modalities.
        decoders: List of decoders for all modalities.
        epoch: Number of the current training epoch.
        writer: TensorBoard SummaryWriter.
        flags: argparse.Namespace with input arguments.

    Returns:
        None.
    """
    """Evaluate generative performance qualitatively and write the results to tensorboard."""
    print("\nEvaluate generative performance qualitatively...")
    write_reconstructions_to_tensorboard(encoders, decoders, sample, epoch, writer, prior_expert=flags.prior_expert)
    write_unconditional_sampling_figure_to_tensorboard(decoders, flags.class_dim, flags.style_dim, epoch, writer,
                                                       nrows=10, ncols=10)
    for m in range(flags.num_modalities):
        write_intramodality_swapping_to_tensorboard(encoders[m], decoders[m], sample[m], sample[m], epoch, writer,
            num_prior_samples=flags.num_prior_samples, figure_name="Swapping/intra-modality/M%d" % m,
                                                    prior_expert=flags.prior_expert, reparam_c=flags.reparam_c_for_eval)
    for m_0 in range(flags.num_modalities):
        for m_1 in range(m_0 + 1, flags.num_modalities):
            write_crossmodality_swapping_to_tensorboard(encoders[m_0], encoders[m_1], decoders[m_1], sample[m_1],
                                                        sample[m_0], epoch, writer,
                                                        num_prior_samples=flags.num_prior_samples,
                                                        figure_name="Swapping/cross-modality/s" + str(m_1) + "_c" + str(m_0),
                                                        prior_expert=flags.prior_expert,
                                                        reparam_c=flags.reparam_c_for_eval)
            write_crossmodality_swapping_to_tensorboard(encoders[m_1], encoders[m_0], decoders[m_0], sample[m_0],
                                                        sample[m_1], epoch, writer,
                                                        num_prior_samples=flags.num_prior_samples,
                                                        figure_name="Swapping/cross-modality/s" + str(m_0) + "_c" + str(m_1),
                                                        prior_expert=flags.prior_expert,
                                                        reparam_c=flags.reparam_c_for_eval)
            write_conditional_generation_samples_to_tensorboard(encoders[m_1], encoders[m_1], decoders[m_0],
                                                                sample[m_1], epoch, writer, num_samples=10,
                                                                figure_name="Conditional_Generation/M" + str(m_0) + "_given_M" + str(m_1),
                                                                prior_expert=flags.prior_expert,
                                                                reparam_c=flags.reparam_c_for_eval)
            write_conditional_generation_samples_to_tensorboard(encoders[m_0], encoders[m_0], decoders[m_1],
                                                                sample[m_0], epoch, writer, num_samples=10,
                                                                figure_name="Conditional_Generation/M" + str(m_1) + "_given_M" + str(m_0),
                                                                prior_expert=flags.prior_expert,
                                                                reparam_c=flags.reparam_c_for_eval)


def eval_generation_clf(data, clf, encoders, decoders, epoch, writer, flags, num_samples, reparam_c, label_suffix=""):
    """
    Evaluates generative coherence using pre-trained classifiers on the outputs and writes the results to TensorBoard.

    Args:
        data: DataLoader for the test set.
        clf: List with pre-trained classifiers for all modalities.
        encoders: List of encoders for all modalities.
        decoders: List of decoders for all modalities.
        epoch: Number of the current training epoch.
        writer: TensorBoard SummaryWriter.
        flags: argparse.Namespace with input arguments.
        num_samples: Number of samples to generate.

    Returns:
        None.
    """
    print("\nEvaluate generative performance through classification...")
    # self-conditional generation (random style)
    for m in range(flags.num_modalities):
        write_conditional_generation_to_tensorboard(m_in=[m], classifier=clf[m], encoders=[encoders[m]],
                                                    decoder=decoders[m], data=data, writer=writer, epoch=epoch,
                                                    flags=flags, label="M"+str(m)+"->M"+str(m)+"'",
                                                    num_gen_samples=num_samples, reparam_c=reparam_c, label_suffix=label_suffix)
    # leave-one-out mappings
    for m in range(flags.num_modalities):
        conds = list(range(flags.num_modalities))
        conds.remove(m)
        write_conditional_generation_to_tensorboard(conds, clf[m], encoders[:m] + encoders[m+1:], decoders[m], data,
                                                    writer, epoch, flags, label="rest->M"+str(m),
                                                    num_gen_samples=num_samples, reparam_c=reparam_c, label_suffix=label_suffix)
    # if M > 2, look at conditional generation for pairwise mappings (for M = 2, this is the same as LOO mappings)
    if flags.num_modalities > 2:
        for m_from in range(flags.num_modalities):
            for m_to in range(flags.num_modalities):
                if not m_from == m_to:
                    write_conditional_generation_to_tensorboard([m_from], clf[m_to], [encoders[m_from]], decoders[m_to],
                                                                data, writer, epoch, flags,
                                                                label="M" + str(m_from) + "->M" + str(m_to),
                                                                num_gen_samples=num_samples, reparam_c=reparam_c, label_suffix=label_suffix)
    # joint generation
    for m in range(flags.num_modalities):
        write_conditional_generation_to_tensorboard(list(range(flags.num_modalities)), clf[m], encoders, decoders[m],
                                                    data, writer, epoch, flags, label="MM->M"+str(m)+"'",
                                                    num_gen_samples=num_samples, reparam_c=reparam_c, label_suffix=label_suffix)
    write_unconditional_generation_to_tensorboard(clf, decoders, data, writer, epoch, flags, label="z->MM'",
                                                  num_gen_samples=num_samples)


def eval_generation_fid(data, gen_path, test_paths, encoders, decoders, epoch, writer, flags, num_samples):
    """
    Evaluate generative performance quantitatively by using the FID metric and write the results to tensorboard.
    """
    print("\nEvaluate generative performance through FIDs...")
    for m in range(flags.num_modalities):
        write_unconditional_fid_to_tensorboard(m, decoders[m], data, writer, epoch, flags,
                                               label="M" + str(m), gen_path=gen_path, test_path=test_paths[m],
                                               num_gen_samples=num_samples)

    for m in range(flags.num_modalities):
        write_conditional_fid_to_tensorboard(m_in=[m], m_out=m, encoders=[encoders[m]], decoder=decoders[m],
                                             mm_data=data, writer=writer, epoch=epoch, flags=flags,
                                             label="M" + str(m) + "->M" + str(m) + "'", gen_path=gen_path,
                                             test_path=test_paths[m], num_gen_samples=num_samples, reparam_c=flags.reparam_c_for_eval)
        write_conditional_fid_to_tensorboard(m_in=[_ for _ in range(flags.num_modalities)], m_out=m, encoders=encoders,
                                             decoder=decoders[m], mm_data=data, writer=writer, epoch=epoch, flags=flags,
                                             label="MM->M" + str(m), gen_path=gen_path, test_path=test_paths[m],
                                             num_gen_samples=num_samples, reparam_c=flags.reparam_c_for_eval)

    # pairwise mappings
    for m_0 in range(flags.num_modalities):
        for m_1 in range(flags.num_modalities):
            if m_0 != m_1:
                write_conditional_fid_to_tensorboard([m_0], m_1, [encoders[m_0]], decoders[m_1], data, writer, epoch, flags,
                                                     label="M" + str(m_0) + "->M" + str(m_1), gen_path=gen_path,
                                                 test_path=test_paths[m_1], num_gen_samples=num_samples, reparam_c=flags.reparam_c_for_eval)

    # leave-one-out mappings
    if flags.num_modalities > 2:
        for m in range(flags.num_modalities):
            conds = [_ for _ in range(flags.num_modalities)]
            conds.remove(m)
            write_conditional_fid_to_tensorboard(m_in=conds, m_out=m, encoders=encoders[:m] + encoders[m+1:],
                                                 decoder=decoders[m], mm_data=data, writer=writer, epoch=epoch,
                                                 flags=flags, label="rest->M" + str(m), gen_path=gen_path,
                                                 test_path=test_paths[m], num_gen_samples=num_samples, reparam_c=flags.reparam_c_for_eval)


def eval_tsne(test, encoders, epoch, writer, flags):
    """
    Evaluate TSNE embeddings for each modality given a single batch.
    """
    write_tsne_embeddings_to_tensorboard(test, encoders, epoch, writer, flags, "tsne_mean", reparam=True, unimodal_poe=False)
    write_tsne_embeddings_to_tensorboard(test, encoders, epoch, writer, flags, "tsne_reparam", reparam=False, unimodal_poe=False)
    write_tsne_embeddings_to_tensorboard(test, encoders, epoch, writer, flags, "tsne_mean_poe", reparam=True, unimodal_poe=True)
    write_tsne_embeddings_to_tensorboard(test, encoders, epoch, writer, flags, "tsne_reparam_poe", reparam=False, unimodal_poe=True)
